#!/usr/bin/env bash
# Quick check that the server is up, coherent, deterministic, that prefix caching hits,
# and measure prefill + decode.
#   scripts/smoke-test.sh [host:port|http(s)://host:port]
#   FLASH_CA=/path/to/ca.pem for a private CA; FLASH_INSECURE=1 to skip TLS verification.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/api-client.sh"
EP="${1:-${BASE:-${FLASH_API_BASE:-localhost:18300}}}"
case "$EP" in http://*|https://*) BASE="${EP%/}" ;; *) BASE="http://$EP" ;; esac
MODEL="${SERVED_MODEL_NAME:-}"

echo ">> health"
api_curl -sf -m 5 "$BASE/health" >/dev/null && echo "   OK" || { echo "   not ready"; exit 1; }

if [ -z "$MODEL" ]; then
  MODEL="$(api_curl -sf -m 5 "$BASE/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)" || {
    echo "!! could not determine served model from $BASE/v1/models"
    exit 1
  }
fi
echo ">> model: $MODEL"

echo ">> coherence"
api_curl -s -m 120 "$BASE/v1/completions" -H 'Content-Type: application/json' -d \
  "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":12,\"temperature\":0}" \
  | python3 -c 'import json,sys;print("  ",repr(json.load(sys.stdin)["choices"][0]["text"]))'

echo ">> reasoning_effort high (Claude Code's default) on /v1/messages"
code="$(api_curl -s -m 120 -o /dev/null -w '%{http_code}' "$BASE/v1/messages" -H 'Content-Type: application/json' -d \
  "{\"model\":\"$MODEL\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"output_config\":{\"effort\":\"high\"}}" || true)"
case "$code" in
  200) echo "   OK" ;;
  400) echo "   HTTP 400: the chat template rejects effort=high (serve with EFFORT_ALIAS=1)" ;;
  404|405) echo "   skipped: this server has no /v1/messages endpoint (HTTP $code)" ;;
  *) echo "   HTTP ${code:-no response}" ;;
esac

echo ">> prefill (TTFT on a ~8k-token prompt), then determinism + prefix-cache hit on the same prompt"
python3 - "$BASE" "$MODEL" "$SCRIPT_DIR" <<'PY'
import json,sys,time,urllib.request,random
sys.path.insert(0, sys.argv[3])
from api_http import urlopen
base=sys.argv[1]; model=sys.argv[2]; random.seed(1)
words="ledger invoice payroll contract clause annex schedule amount date vendor total net gross tax due paid".split()
prompt=" ".join(random.choice(words)+(str(random.randint(1,9999)) if random.random()<0.2 else "") for _ in range(5800))
prompt+="\n\nQuestion: list three numbers that appear right after the word 'invoice', then name the most frequent word. Answer:"
def hits():
    """vLLM prefix-cache hit counter, or None when /metrics is unavailable."""
    try:
        txt=urlopen(base+"/metrics",timeout=10).read().decode()
    except Exception:
        return None
    tot,seen=0.0,False
    for line in txt.splitlines():
        if line.startswith("vllm:prefix_cache_hits_total"):
            try: tot+=float(line.rsplit(" ",1)[1]); seen=True
            except ValueError: pass
    return tot if seen else None
def run(max_tokens):
    t=time.time()
    req=urllib.request.Request(base+"/v1/completions",
        data=json.dumps({"model":model,"prompt":prompt,"max_tokens":max_tokens,"temperature":0,"logprobs":3}).encode(),
        headers={"Content-Type":"application/json"})
    r=json.load(urlopen(req,timeout=600)); dt=time.time()-t
    c=r["choices"][0]; return r["usage"]["prompt_tokens"], dt, c["text"], c["logprobs"]["top_logprobs"][0]
n,dt,t1,lp1=run(1)
print(f"   {n} tok in {dt:.2f}s  =>  {n/dt:.0f} tok/s prefill (cold)")
h1=hits()
n,dt2,t2,lp2=run(1)
h2=hits()
if h1 is not None and h2 is not None:
    # The wall-clock comparison cannot separate the two effects: with the PLE table on
    # NVMe, a warm page cache alone also speeds up the second pass (HOW-IT-WORKS:139).
    verdict='prefix-cache HIT' if h2>h1 else 'NO prefix-cache hit (serve with PREFIX_CACHE=1?)'
    print(f"   same prompt again: {dt2:.2f}s  =>  {verdict} (hits {h1:.0f} -> {h2:.0f})")
else:
    print(f"   same prompt again: {dt2:.2f}s  =>  timing only: {'consistent with a HIT' if dt2 < dt/2 else 'no speedup'} (no /metrics; a warm page cache alone can halve this)")
same = t1==t2 and all(abs(lp1.get(k,-99)-lp2.get(k,-99))<1e-6 for k in set(lp1)|set(lp2))
print(f"   first-token logprobs identical across runs: {'YES (deterministic)' if same else 'NO (stock top-k kernel? DET_TOPK=0 EXACT_TOPK=0)'}")
PY

echo ">> decode (real answer, greedy — never use ignore_eos with this model)"
python3 - "$BASE" "$MODEL" "$SCRIPT_DIR" <<'PY'
import json,sys,time,urllib.request
sys.path.insert(0, sys.argv[3])
from api_http import urlopen
base=sys.argv[1]; model=sys.argv[2]
msgs=[{"role":"user","content":"Explain in about 300 words how a page cache works and why random reads from an NVMe-backed mmap get faster over time. /no_think"}]
t=time.time()
req=urllib.request.Request(base+"/v1/chat/completions",
    data=json.dumps({"model":model,"messages":msgs,"max_tokens":400,"temperature":0}).encode(),
    headers={"Content-Type":"application/json"})
r=json.load(urlopen(req,timeout=600)); dt=time.time()-t
n=r["usage"]["completion_tokens"]
print(f"   {n} tok in {dt:.2f}s  =>  {n/dt:.1f} tok/s end-to-end (includes TTFT; steady-state decode is higher, see README)")
PY
