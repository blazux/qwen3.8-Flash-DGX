#!/usr/bin/env bash
# ab-bench — boot-to-evidence A/B between two serving profiles on one GB10 box.
#
#   scripts/ab-bench.sh                          # mtp4-1m vs dflash-1m, 2 blocks, fast suite
#   scripts/ab-bench.sh --blocks 3 --suite full
#   scripts/ab-bench.sh --arms "dflash dflash-1m" --only decode/
#   scripts/ab-bench.sh --dry-run                # print the boots and the runs, touch nothing
#
# The box can hold exactly one of these models at a time, so an A/B is necessarily serial and
# every arm costs a boot (~3-5 min, 75 GiB of weights). Blocks alternate the arm order (AB BA)
# so thermal drift and page-cache warm-up land on both sides instead of on one; the reps within
# a boot are cheap and go to the median.
#
# Both arms are served with PROM_MULTIPROC=1: vLLM only puts prometheus_client into multiprocess
# mode when there is more than one API server, so without it the vllm:spec_decode_* counters stay
# in EngineCore's private registry and never reach /metrics — you would get tok/s with no
# acceptance numbers to explain it. It is the same change on both arms, so it cannot bias the
# delta; it does drop the process_* series from /metrics while the harness runs.
#
# Writes results/ab/<arm>.b<block>.jsonl plus a .meta.json per boot (engine args straight from
# `docker inspect`, KV pool from the boot log, host state). Compare with bench/ab_report.py,
# which fails the run if the two arms differ in anything but --speculative-config.
#
# This stops the running container and uses a separate name for the benchmark.
# The original container is restarted and checked at exit unless --keep succeeds.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLASH="$HERE/flash"
ARMS="mtp4-1m dflash-1m"          # arm = the profile name; first is the report's baseline
BLOCKS=2
SUITE=fast
SEED=20261103                     # identical on every arm, or nothing is matched
REPS_SCALE=1.0
ONLY=""
OUT_DIR="$HERE/results/ab"
KEEP=0
DRYRUN=0
NAME="${NAME:-qwen38-flash}"
PORT="${PORT:-18300}"
export TIMEOUT="${TIMEOUT:-2400}"  # boot ceiling for ./flash wait

while [ $# -gt 0 ]; do
  case "$1" in
    --arms)       ARMS="$2"; shift 2 ;;
    --blocks)     BLOCKS="$2"; shift 2 ;;
    --suite)      SUITE="$2"; shift 2 ;;
    --seed)       SEED="$2"; shift 2 ;;
    --reps-scale) REPS_SCALE="$2"; shift 2 ;;
    --only)       ONLY="$2"; shift 2 ;;
    --out)        OUT_DIR="$2"; shift 2 ;;
    --keep)       KEEP=1; shift ;;
    --dry-run)    DRYRUN=1; shift ;;
    -h|--help)    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "!! unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
done

read -r -a ARM_LIST <<< "$ARMS"
[ "${#ARM_LIST[@]}" -ge 2 ] || { echo "!! --arms needs at least two profiles"; exit 2; }
mkdir -p "$OUT_DIR"

echo ">> A/B: ${ARM_LIST[*]}   blocks=$BLOCKS  suite=$SUITE  seed=$SEED  -> $OUT_DIR"
if [ "$DRYRUN" = 0 ]; then
  echo "   this temporarily stops $NAME while the benchmark uses port $PORT"
fi

# Each arm must be servable before boots start: a missing draft model or an image without patch
# 23 should fail here, not halfway through a 40-minute schedule.
if [ "$DRYRUN" = 0 ]; then
  for p in "${ARM_LIST[@]}"; do
    echo ">> doctor $p"
    "$FLASH" doctor "$p" || { echo "!! '$p' is not ready — run ./flash setup $p first"; exit 1; }
  done
fi

capture_meta() {   # capture_meta <file> <container> — what the container was actually started with
  python3 - "$1" "$2" <<'PY'
import json, os, re, subprocess, sys

out, container = sys.argv[1], sys.argv[2]


def dockerfmt(tmpl):
    return subprocess.run(["docker", "inspect", "-f", tmpl, container],
                          capture_output=True, text=True).stdout.strip()


cmd = json.loads(dockerfmt("{{json .Config.Cmd}}") or "[]")
env = json.loads(dockerfmt("{{json .Config.Env}}") or "[]")
e = dict(kv.split("=", 1) for kv in env if "=" in kv)
# Everything after the snapshot path is the vllm command line, arguments verbatim.
args = cmd[cmd.index("--served-model-name") - 1:] if "--served-model-name" in cmd else cmd


def flag(name):                      # the value in a "--max-model-len 1000000" pair
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
    return None


def num(name):
    v = flag(name)
    return int(v) if v and v.isdigit() else v


try:
    spec = json.loads(flag("--speculative-config") or "null")
except ValueError:
    spec = flag("--speculative-config")

log = subprocess.run(["docker", "logs", container], capture_output=True, text=True).stdout
kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
mc = re.search(r"Maximum concurrency.*?:\s*([\d.]+)x", log)   # both vLLM wordings
mem = {}
try:
    for line in open("/proc/meminfo"):
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable", "SwapCached"):
            mem[k] = int(v.split()[0]) // 1024
except OSError:
    pass
meta = {"vllm_args": args, "ctx": num("--max-model-len"),
           "max_num_seqs": num("--max-num-seqs"),
           "kv_cache_dtype": flag("--kv-cache-dtype"),
           "gpu_memory_utilization": flag("--gpu-memory-utilization"),
           "speculative_config": spec,
           "image": dockerfmt("{{.Config.Image}}"),
           "created": dockerfmt("{{.State.StartedAt}}"),
           "draft_vocab": e.get("VLLM_MTP_DRAFT_VOCAB", "unset"),
           "v2_model_runner": e.get("VLLM_USE_V2_MODEL_RUNNER", "unset"),
           "prefix_cache": "--enable-prefix-caching" in args,
           "kv_cache_tokens": int(kv.group(1).replace(",", "")) if kv else None,
           "max_concurrency": float(mc.group(1)) if mc else None,
           "meminfo_gib": mem, "loadavg": os.getloadavg()}
json.dump(meta, open(out, "w"), indent=1)
print("   KV pool {} tok   ctx {}   draft_vocab {}   v2_runner {}".format(
    meta["kv_cache_tokens"], meta["ctx"], meta["draft_vocab"], meta["v2_model_runner"]))
PY
}

show_boot_failure() {   # show_boot_failure <container> <file> — why the engine gave up
  local container="$1" file="$2"
  docker logs "$container" > "$file" 2>&1 || true
  echo "   full log: $file"
  docker inspect -f '   {{.State.Status}} exit={{.State.ExitCode}} oom_killed={{.State.OOMKilled}}' \
    "$container" 2>/dev/null || echo "   (no container $container to inspect)"
  local restarts
  restarts="$(docker inspect -f '{{.RestartCount}}' "$container" 2>/dev/null || echo 0)"
  if [ "${restarts:-0}" != "0" ]; then
    # serve.sh starts the container with --restart unless-stopped, so a config vLLM refuses
    # becomes an endless re-load of ~73 GiB from NVMe every few minutes — and it keeps going
    # after this script moves on or exits. Only RestartCount says "it has already died": a
    # container that is merely slow to boot sits at 0 and must be left alone.
    echo "   crash-restarting after $restarts attempt(s) — stopping the loop"
    docker update --restart=no "$container" >/dev/null 2>&1 || true
    docker stop -t 20 "$container" >/dev/null 2>&1 || true
  fi
  # ./flash wait prints the TAIL of the log, which is where wait() noticed the engine died,
  # not why. The line that matters is the first exception, so grab those instead.
  python3 - "$file" <<'PY'
import re, sys
try:
    lines = open(sys.argv[1], errors="replace").read().splitlines()
except OSError:
    sys.exit()
shown = set()


def say(i, tag):
    if i not in shown:                  # one line may match several patterns
        shown.add(i)
        print(f"   {tag} {lines[i].strip()[:300]}")


for pat, tag in (
    (r"^(Traceback \(most recent call last\):)$", "traceback:"),
    (r"^[A-Za-z_][\w.]*(Error|Exception|Warning):", "exception:"),
    (r"\bERROR\b.*\b(vllm|engine|Worker)", "error log:"),
    (r"QSA ring|speculative|block size|mamba", "config hint:"),
    (r"out of memory|OOM|KV cache size", "memory:"),
):
    hits = [i for i, l in enumerate(lines) if re.search(pat, l)][:4]
    for i in hits:
        say(i, tag)
    if hits:
        print("   ---")
# vLLM states the context length its KV pool would actually fit. A boot that died on that
# check is a result, not just a failure — so turn it into the number to put in the report.
text = "\n".join(lines)
need = re.search(r"max seq len \((\d+)\), \(?([\d.]+) ?GiB KV cache is needed, which is larger "
                 r"than the available KV cache memory \(([\d.]+) ?GiB", text)
cap = re.search(r"estimated maximum model length is ([\d,]+)", text)
if cap:
    line = f"   ceiling: this arm boots to CTX={int(cap.group(1).replace(',', '')):,}"
    if need:
        short = float(need.group(2)) - float(need.group(3))
        line += f" not {int(need.group(1)):,} — {short:.1f} GiB short of KV"
        pool = 0.0
        try:
            for mem in open("/proc/meminfo"):
                if mem.startswith("MemTotal:"):
                    pool = int(mem.split()[1]) / 1048576       # GiB of the unified pool
        except OSError:
            pass
        util = re.search(r"gpu-memory-utilization=(\d*\.?\d+)", text)
        if pool and util:
            want = float(util.group(1)) + short / pool
            line += (f"; GPU_MEM {float(util.group(1)):.2f} -> {want:.3f} covers it"
                     + (" (past the 0.85 the recipe warns swaps)" if want > 0.85 else ""))
    print(line)
if not shown and not cap:            # nothing matched: at least show where it stopped
    print("   --- no exception found; last lines ---")
    for j in range(max(0, len(lines) - 12), len(lines)):
        say(j, "(tail)")
PY
}

if [ "$DRYRUN" = 0 ]; then
  ORIGINAL_NAME="$NAME"
  ORIGINAL_ID="$(docker container inspect -f '{{.Id}}' "$ORIGINAL_NAME" 2>/dev/null || true)"
  ORIGINAL_RUNNING=0
  if [ -n "$ORIGINAL_ID" ]; then
    ORIGINAL_RUNNING="$(docker container inspect -f '{{.State.Running}}' "$ORIGINAL_ID")"
  fi
  NAME="$ORIGINAL_NAME-ab-$$"
  if docker container inspect "$NAME" >/dev/null 2>&1; then
    echo "!! benchmark container $NAME already exists" >&2
    exit 1
  fi
  restore() {
    local rc=$?
    trap - EXIT
    trap '' INT TERM
    if [ "$rc" = 0 ] && [ "$KEEP" = 1 ]; then
      echo ">> leaving $NAME running (--keep)"
      exit 0
    fi
    if docker container inspect "$NAME" >/dev/null 2>&1; then
      docker rm -f "$NAME" >/dev/null || rc=1
    fi
    if [ "$ORIGINAL_RUNNING" = true ]; then
      echo ">> restoring $ORIGINAL_NAME"
      docker start "$ORIGINAL_ID" >/dev/null &&
        NAME="$ORIGINAL_NAME" "$FLASH" wait || rc=1
    fi
    exit "$rc"
  }
  trap restore EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  if [ "$ORIGINAL_RUNNING" = true ]; then
    docker stop -t 30 "$ORIGINAL_ID" >/dev/null
  fi
  export GB10_BASE_URL="http://127.0.0.1:$PORT/v1"
fi

FAILED=()
for block in $(seq 1 "$BLOCKS"); do
  # ABBA: block 1 runs the arms in order, block 2 reversed, ... so no arm always boots after a
  # freshly-replaced container or always runs at the tail of the session.
  if [ $((block % 2)) -eq 0 ]; then
    order="$(printf '%s\n' "${ARM_LIST[@]}" | tac | tr '\n' ' ')"
  else
    order="${ARM_LIST[*]}"
  fi
  for profile in $order; do
    echo
    echo ">> block $block  arm $profile  (booting)"
    if [ "$DRYRUN" = 1 ]; then
      echo "   DRY: PROM_MULTIPROC=1 $FLASH serve $profile && $FLASH wait"
      echo "   DRY: python3 bench/ab.py --label $profile --block $block --suite $SUITE --seed $SEED --out $OUT_DIR/$profile.b$block.jsonl"
      continue
    fi
    if ! PROM_MULTIPROC=1 NAME="$NAME" PORT="$PORT" "$FLASH" serve "$profile"; then
      echo "   !! serve failed (the container never started)"
      FAILED+=("$profile.b$block:serve"); continue
    fi
    if ! NAME="$NAME" PORT="$PORT" "$FLASH" wait; then
      echo "   !! the server never came up"
      show_boot_failure "$NAME" "$OUT_DIR/$profile.b$block.boot.log"
      FAILED+=("$profile.b$block:boot"); continue
    fi
    capture_meta "$OUT_DIR/$profile.b$block.meta.json" "$NAME"
    [ -n "$ONLY" ] && only=(--only "$ONLY") || only=()
    if ! python3 "$HERE/bench/ab.py" --label "$profile" --block "$block" --suite "$SUITE" \
         --seed "$SEED" --reps-scale "$REPS_SCALE" "${only[@]}" \
         --meta-file "$OUT_DIR/$profile.b$block.meta.json" \
         --out "$OUT_DIR/$profile.b$block.jsonl"; then
      echo "   !! bench/ab.py exited non-zero (see the rows above)"
      FAILED+=("$profile.b$block:bench")
    fi
  done
done

echo
if [ "$DRYRUN" = 1 ]; then
  echo ">> dry run only — nothing was booted or measured"
  exit 0
fi
[ "${#FAILED[@]}" -gt 0 ] && echo "!! incomplete: ${FAILED[*]}"
echo ">> compare with:"
echo "   python3 bench/ab_report.py $OUT_DIR/*.jsonl --baseline ${ARM_LIST[0]}"
echo "   python3 bench/ab_report.py $OUT_DIR/*.jsonl --baseline ${ARM_LIST[0]} --markdown"
[ "${#FAILED[@]}" -gt 0 ] && exit 1
exit 0
