#!/usr/bin/env python3
"""Compare bench/ab.py runs: matched-config check, per-cell deltas, acceptance, byte equivalence.

    python3 bench/ab_report.py results/ab/mtp4-1m.b*.jsonl results/ab/dflash-1m.b*.jsonl
    python3 bench/ab_report.py results/ab/*.jsonl --baseline mtp4-1m --markdown   # README table

Files are pooled per arm (meta.label), so an ABBA schedule lands in one column and the block
order shows up as spread rather than as bias. Medians come with the intra-arm spread: a delta
inside that spread is not a result.

Read the first section first — it proves the two arms differ ONLY in the drafter. Engine args
are taken from `docker inspect` at serve time, so a stray SEQS or a forgotten DRAFT_VOCAB=0
turns up here instead of quietly becoming part of the conclusion. Exits 3 when the arms are not
matched (that run measures something else).
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict

# Row keys worth comparing, in report order: (key, label, decimals).
METRICS = [("tok_s", "tok/s", 2), ("aggregate_tok_s", "agg tok/s", 2),
           ("decode_tok_s", "decode tok/s", 2), ("per_stream_tok_s", "tok/s/stream", 2),
           ("prefill_tok_s", "prefill tok/s", 0), ("ttft", "ttft s", 3),
           ("ttft_max", "ttft max s", 3), ("completion_tokens", "out tok", 0)]


def load(paths):
    arms = defaultdict(lambda: dict(rows=defaultdict(list), meta={}, blocks=[]))
    for p in paths:
        meta, cells = None, []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("kind") == "meta":
                    meta = row
                elif row.get("kind") == "cell":
                    cells.append(row)
        if meta is None:
            sys.exit(f"!! {p}: no meta row — not a bench/ab.py output")
        arm = arms[meta["label"]]
        arm["meta"] = meta
        arm["blocks"].append(meta.get("block", 1))
        for row in cells:
            arm["rows"][row["cell"]].append(row)
    return arms


def med(vals):
    return statistics.median(vals) if vals else None


def spread(vals):
    """(max-min)/median — the noise floor a delta has to clear."""
    if len(vals) < 2:
        return None
    m = med(vals)
    return (max(vals) - min(vals)) / m if m else None


def values(arm, cell, key):
    return [r[key] for r in arm["rows"].get(cell, [])
            if r.get(key) is not None and "error" not in r]


def specs(arm, cell=None):
    """Acceptance over one cell (or the whole run): the counters are per-rep deltas."""
    rows = ([r for rs in arm["rows"].values() for r in rs] if cell is None
            else arm["rows"].get(cell, []))
    rows = [r for r in rows if r.get("spec") and "error" not in r]
    if not rows:
        return None
    pos_len = max(len(r["spec"].get("accepted_per_pos") or []) for r in rows)
    return {
        "acceptance_rate": med([r["spec"]["acceptance_rate"] for r in rows
                               if r["spec"]["acceptance_rate"] is not None]),
        "mean_acceptance_length": med([r["spec"]["mean_acceptance_length"] for r in rows]),
        "drafts": sum(r["spec"]["drafts"] for r in rows),
        "per_pos": [med([r["spec"]["accepted_per_pos"][i] for r in rows
                         if len(r["spec"].get("accepted_per_pos") or []) > i])
                    for i in range(pos_len)] if pos_len else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="JSONL files written by bench/ab.py")
    ap.add_argument("--baseline", help="arm to measure against (default: first seen)")
    ap.add_argument("--markdown", action="store_true", help="emit a README-ready table")
    ap.add_argument("--only", default="", help="report only cells whose id contains this")
    args = ap.parse_args()

    arms = load(args.runs)
    names = sorted(arms, key=lambda n: (min(arms[n]["blocks"]), n))
    if args.baseline:
        if args.baseline not in arms:
            sys.exit(f"!! no arm '{args.baseline}' among {', '.join(names)}")
        names = [args.baseline] + [n for n in names if n != args.baseline]
    base = names[0]
    warns = []

    if len({arms[n]["meta"]["seed"] for n in names}) > 1:
        warns.append(f"!! seeds differ ({sorted({arms[n]['meta']['seed'] for n in names})}) — "
                     "the arms saw DIFFERENT prompts, so nothing below is matched")
    if len({arms[n]["meta"]["suite"] for n in names}) > 1:
        warns.append("!! suites differ — only the cells present in both are compared")

    print("== arms")
    for n in names:
        m = arms[n]["meta"]
        ok = sum(1 for rs in arms[n]["rows"].values() for r in rs if "error" not in r)
        bad = sum(1 for rs in arms[n]["rows"].values() for r in rs if "error" in r)
        hits = sum(1 for rs in arms[n]["rows"].values() for r in rs if r.get("cache_hit"))
        print(f"   {n:<14} blocks {sorted(set(arms[n]['blocks']))}  "
              f"suite {m['suite']}  seed {m['seed']}  {ok} measured rep(s), {bad} failed")
        if hits:
            # Every prompt in the matrix is seeded to be unique, so a hit means the prompt
            # builders collided — the prefill and ttft columns of those reps are fiction.
            warns.append(f"!! {n}: {hits} rep(s) had prompt tokens served from the prefix "
                         "cache — those reps measured the cache, not the model")

    print("\n== serving config")
    argsets = {}
    for n in names:
        s = arms[n]["meta"].get("server") or {}
        argsets[n] = s.get("vllm_args") or []
        pool, mc = s.get("kv_cache_tokens"), s.get("max_concurrency")
        pool_str = f"{pool:,}" if isinstance(pool, int) else "-"
        mc_str = f"{mc:.2f}x" if isinstance(mc, (int, float)) else "-"
        print(f"   {n:<14} {s.get('image', '-')}"
              f"\n   {'':<14} ctx {s.get('ctx', '-')}  seqs {s.get('max_num_seqs', '-')}"
              f"  gpu_mem {s.get('gpu_memory_utilization', '-')}"
              f"  kv {s.get('kv_cache_dtype', '-')}"
              f"  draft_vocab {s.get('draft_vocab', '-')}"
              f"\n   {'':<14} KV pool {pool_str} tok  max concurrency {mc_str}"
              f"  spec {s.get('speculative_config', '-')}")
        if isinstance(pool, int) and isinstance(s.get("ctx"), int) and pool < s["ctx"]:
            warns.append(f"!! {n}: the KV pool ({pool:,} tok) is smaller than the advertised "
                         f"context ({s['ctx']:,}) — this recipe cannot actually serve {s['ctx']:,} tokens")
        if s.get("draft_vocab") == "1" and n != base:
            warns.append(f"!! {n} runs with DRAFT_VOCAB=1 (the pruned MTP vocabulary) while "
                         f"{base} does not — that prune is worth ~20% decode on its own")
    if not argsets[base]:
        warns.append("!! no engine args recorded in the meta — run the arms through "
                     "scripts/ab-bench.sh, or verify the match by hand")
    else:
        # --speculative-config is the whole point of the comparison, so it may differ.
        def plain(n):
            """This arm's engine args, minus --speculative-config and its value."""
            out, drop_next = set(), False
            for t in argsets[n]:
                if t == "--speculative-config":
                    drop_next = True
                    continue
                if drop_next:
                    drop_next = False
                    continue
                out.add(t)
            return out
        matched = all(plain(n) == plain(base) for n in names[1:])
        if matched:
            print("   engine args identical apart from --speculative-config: yes")
        else:
            for n in names:
                for tok in sorted(plain(n) - plain(base)):
                    warns.append(f"!! engine arg present in {n} only: {tok}")
                for tok in sorted(plain(base) - plain(n)):
                    warns.append(f"!! engine arg missing from {n} (present in {base}): {tok}")

    def cell_order(c):
        """decode/, gate/, prefill/, conc/ groups in that order, sizes numerically inside them."""
        fam, _, size = c.partition("/")
        digits = size.rstrip("kKmM")
        order = {"decode": 0, "prefill": 1, "conc": 2, "gate": 3}
        return (order.get(fam, 9), fam, int(digits) if digits.isdigit() else 0, size)

    cells = sorted({c for n in names for c in arms[n]["rows"]}, key=cell_order)
    if args.only:
        cells = [c for c in cells if args.only in c]
    missing = [c for c in cells if not all(arms[n]["rows"].get(c) for n in names)]

    if args.markdown:
        print("\nMedians in tok/s (`prefill/` cells: prompt tokens per second to first token);"
              " `conc/` cells: aggregate over all streams). Gate cells are in the text section.\n")
        print("| Cell | " + " | ".join(names) + " | Delta |")
        print("| --- | " + " | ".join(["---:"] * len(names)) + " | ---: |")
        for c in cells:
            if c.startswith("gate/"):
                continue               # an output-equivalence check, not a throughput result
            key = "prefill_tok_s" if c.startswith("prefill/") else ""
            series = [((values(arms[n], c, key) if key else [])
                      or values(arms[n], c, "aggregate_tok_s") or values(arms[n], c, "tok_s"))
                      for n in names]
            if not all(series):
                continue
            ms = [med(v) for v in series]
            delta = f"{(ms[-1] / ms[0] - 1) * 100:+.1f}%" if ms[0] and ms[-1] and len(ms) > 1 else "-"
            print(f"| {c} | " + " | ".join(f"{v:,.1f}" if v else "-" for v in ms) + f" | {delta} |")
    else:
        for c in cells:
            print(f"\n== {c}")
            for key, label, digits in METRICS:
                series = [values(arms[n], c, key) for n in names]
                if not any(series):
                    continue
                line = f"   {label:<16}"
                for i in range(len(names)):
                    m, sp = med(series[i]), spread(series[i])
                    line += (f" {m:>10,.{digits}f}" + (f"±{sp * 100:.0f}% " if sp else "     · ")
                             if m is not None else f" {'-':>10} ·  ")
                if len(names) > 1:
                    a, b = med(series[0]), med(series[-1])
                    line += f"  {(b / a - 1) * 100:+.1f}%" if a and b else "  -"
                print(line + f"   n={max(len(s) for s in series)}")
            sp = [specs(arms[n], c) for n in names]
            if any(sp):
                print("   acceptance      " + "".join(
                    f" {x['acceptance_rate'] * 100:>8.1f}% len {x['mean_acceptance_length']:>4.2f}"
                    if x else f" {'-':>9}      " for x in sp))
                for i, x in enumerate(sp):
                    if x and x["per_pos"]:
                        print(f"      {names[i]:<13} accepted by draft position: "
                              + " ".join(f"{int(v):,}" for v in x["per_pos"]))
            # Identical prompt + temperature 0 must give identical bytes, drafter aside. The
            # prompt depends on (cell, rep) alone, so block is not part of the key — and if two
            # blocks of ONE arm disagree, that is a determinism finding worth printing too.
            hashes = []
            for n in names:
                h = defaultdict(set)
                for r in arms[n]["rows"].get(c, []):
                    if "error" not in r and r.get("text_sha"):
                        h[r["rep"]].add(r["text_sha"])
                hashes.append(h)
            if len(names) > 1 and all(hashes):
                reps = sorted(set(hashes[0]) & set(hashes[-1]))
                if reps:
                    same = [k for k in reps if hashes[0][k] == hashes[-1][k]]
                    # Same arm, same prompt, two boots, two different texts: not deterministic.
                    unstb = [k for k in reps
                             if any(len(h[k]) > 1 for h in (hashes[0], hashes[-1]))]
                    print(f"   output bytes {'identical' if len(same) == len(reps) else 'DIFFER'}"
                          f" on {len(same)}/{len(reps)} matched reps"
                          + (f", and {len(unstb)} rep(s) unstable within an arm" if unstb else ""))
                else:
                    print("   output bytes: no rep measured in both arms — nothing compared")

    print("\n== whole-run acceptance")
    for n in names:
        x = specs(arms[n])
        if not x:
            print(f"   {n:<14} no vllm:spec_decode_* counters — acceptance unmeasured "
                  "(serve with PROM_MULTIPROC=1)")
            continue
        print(f"   {n:<14} {x['acceptance_rate'] * 100:>5.1f}% of draft tokens accepted, "
              f"mean acceptance length {x['mean_acceptance_length']:.2f}, "
              f"{x['drafts']:,} draft steps")

    errs = [(n, c, r) for n in names for c in arms[n]["rows"]
            for r in arms[n]["rows"][c] if "error" in r]
    if errs:
        print(f"\n!! {len(errs)} failed request(s):")
        for n, c, r in errs[:12]:
            print(f"   {n} {c} rep {r.get('rep')}: {r['error']}")
    if missing:
        print(f"\n   cells not measured in every arm (excluded from the deltas): "
              f"{', '.join(missing[:8])}")

    if warns:
        print()
        for w in warns:
            print(w)
        if any(w.startswith("!!") for w in warns):
            sys.exit(3)


if __name__ == "__main__":
    main()
