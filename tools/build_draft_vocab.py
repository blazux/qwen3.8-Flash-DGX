#!/usr/bin/env python3
"""Build the reduced draft vocabulary for the MTP drafter (qwen38-flash-dgx).

The MTP draft head scores all 248,320 vocabulary rows on every draft step; almost all of them
are never the argmax. This picks the N ids the drafter is allowed to propose:
  1. every token that appears in a local text corpus, ranked by count (domain fit),
  2. every special / added token (chat template, tool-call and thinking markers), always,
  3. the first 256 ids (byte fallbacks), always,
  4. the remaining slots filled with the lowest ids not yet chosen (BPE merge order is a
     frequency proxy for Qwen tokenizers).
The target model verifies every drafted token, so a token outside this set can never be
emitted *by the draft*; it simply costs a rejection when the target wants it. Output: a sorted
int32 .npy of ids.

usage: build_draft_vocab.py <tokenizer_dir> <out.npy> [--n 65536] [--corpus path ...]
"""
import argparse, collections, glob, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("tokenizer_dir"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=65536)
ap.add_argument("--corpus", nargs="*", default=[])
a = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.tokenizer_dir, trust_remote_code=True)
V = len(tok)
print(f"tokenizer vocab (incl. added): {V}")

def texts_from(path):
    if os.path.isdir(path):
        for f in glob.glob(os.path.join(path, "**", "*"), recursive=True):
            if os.path.isfile(f) and os.path.getsize(f) < 20_000_000 and f.split(".")[-1] in ("json", "md", "txt", "py", "log", "jsonl"):
                yield from texts_from(f)
        return
    try:
        raw = open(path, "rb").read().decode("utf-8", errors="ignore")
    except Exception:
        return
    if path.endswith(".json"):
        try:
            def walk(o):
                if isinstance(o, str):
                    yield o
                elif isinstance(o, dict):
                    for v in o.values():
                        yield from walk(v)
                elif isinstance(o, list):
                    for v in o:
                        yield from walk(v)
            yield from walk(json.loads(raw)); return
        except Exception:
            pass
    yield raw

counts = collections.Counter()
nchars = 0
for p in a.corpus:
    for t in texts_from(p):
        if len(t) < 20: continue
        nchars += len(t)
        for chunk in (t[i:i+200_000] for i in range(0, len(t), 200_000)):
            counts.update(tok.encode(chunk, add_special_tokens=False))
print(f"corpus: {nchars/1e6:.1f} M chars, {sum(counts.values())/1e6:.2f} M tokens, {len(counts)} distinct ids")

keep = set()
# 2. specials / added tokens
specials = set(tok.all_special_ids)
try:
    specials |= set(int(i) for i in tok.added_tokens_decoder.keys())
except Exception:
    pass
keep |= specials
# 3. byte fallbacks
keep |= set(range(min(256, V)))
# 1. corpus by frequency
for tid, _ in counts.most_common():
    if len(keep) >= a.n: break
    keep.add(int(tid))
n_corpus = len(keep)
# 4. fill with lowest ids
i = 0
while len(keep) < a.n and i < V:
    keep.add(i); i += 1
ids = np.array(sorted(keep), dtype=np.int32)
np.save(a.out, ids)
cov = sum(c for t, c in counts.items() if t in keep) / max(1, sum(counts.values()))
print(f"kept {len(ids)} ids: {len(specials)} special/added, corpus-ranked up to {n_corpus}, filled from id order up to {i}; corpus token coverage {cov*100:.3f}%")
print("saved", a.out)
