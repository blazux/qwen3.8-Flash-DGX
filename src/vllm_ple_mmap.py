"""vllm_ple_mmap — serve the Qwen3.8-Flash-Next N-gram (PLE) table from NVMe via mmap.

Why: the 51B-parameter n-gram table is 47.7 GiB (51.2 GB) in FP8 and vLLM keeps it resident
(GPU, or pinned host RAM with VLLM_PLE_CPU_OFFLOAD). On a DGX Spark / GX10 the
host and the GPU share one 121 GiB pool, so neither fits next to the 78 GiB main
model. But a token only ever touches 16 rows x 160 bytes of that table, so the
table can live on disk and be served through the page cache — exactly what
llama.cpp does with its GGUF mmap.

How: with VLLM_PLE_MMAP=1 this module patches ``Qwen3_8FlashNextNGramEmbedding``:
  * ``__init__`` swaps the 44/95 GiB ``VocabParallelEmbedding`` for a tiny
    placeholder whose ``forward(ids)`` gathers rows from ``np.memmap`` views of the
    checkpoint's ``model-plefp8-*.safetensors`` shards (zero-copy, page-cache backed);
  * ``load_weights`` drops the 128 shard tensors on the floor, keeps the global FP8
    ``weight_scale`` (as ``_offload_weight_scale``, which the untouched
    ``Qwen3_8FlashNextPLELayer._dequantize_embeddings`` already consumes) and opens
    the memmaps.
  * ``forward_impl`` (hashing + lookup) is wrapped in a custom op
    ``vllm::ple_mmap_lookup`` so that (a) torch.compile treats it as opaque — the
    stock version trips an Inductor int64 indexing assert on sm_121 — and (b) it can
    be listed in ``-cc.splitting_ops`` and run OUTSIDE piecewise CUDA graphs: the
    gather is CPU work + a pageable H2D copy, which cannot live inside a capture.
    Use ``-cc.cudagraph_mode=PIECEWISE`` (not FULL*) with the splitting op list in
    serve-flashnext-vllm.sh, or ``--enforce-eager``.
Nothing else in vLLM changes: the n-gram hashing, the short-conv, the dequant path
are the stock ones.

Fast gather hot path (CPU dedup -> persistent pinned staging buffer -> async H2D ->
GPU-side inverse expansion, plus a no-threadpool fast path for decode-sized
batches), bf16/f16 table support, VLLM_PLE_MMAP_DIR and the periodic stats line
were contributed by @Saren-Arterius (github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound).

Knobs (env):
  VLLM_PLE_MMAP=1            enable
  VLLM_PLE_MMAP_WORKERS=32   gather threads (page faults overlap across threads)
  VLLM_PLE_MMAP_CHUNK=2048   rows per gather task
  VLLM_PLE_MMAP_MADVISE=random  madvise on the shard mmaps: random (default, no readahead) or normal
  VLLM_PLE_MMAP_PROMETHEUS=1 0 = do not register the vllm:ple_mmap_* counters
  VLLM_PLE_MMAP_PREWARM=0    1 = stream the whole table once at load to fill the
                             page cache with whatever memory is free (harmless,
                             evictable; ~10 s at 4.7 GB/s)

Install: the Dockerfile copies this file next to vllm and appends
``_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)`` to the end of
``vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py``. See the repo README.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import re
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("vllm.ple_mmap")

ENV_ENABLE = "VLLM_PLE_MMAP"
_FP8_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}
# 16-bit tables need no weight_scale: the stock _dequantize_embeddings is a
# no-op for non-FP8 rows.
_TABLE_DTYPES = {
    **_FP8_DTYPES,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
}


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "0").lower() in ("1", "true", "yes")


def _madvise(mm: np.memmap, kind: str) -> None:
    """Best-effort madvise on the mmap behind a np.memmap (Linux, Python >= 3.8)."""
    try:
        import mmap as _mmap

        flag = {"random": _mmap.MADV_RANDOM, "normal": _mmap.MADV_NORMAL}[kind]
        raw = getattr(mm, "_mmap", None)
        if raw is not None:
            raw.madvise(flag)
            _MADVISED.append(kind)
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.warning("PLE mmap: madvise(%s) failed: %s", kind, exc)


_MADVISED: list[str] = []


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# safetensors header parsing (no dependency on the safetensors package: we need
# raw file offsets, which its Python API does not expose)
# --------------------------------------------------------------------------- #
def parse_safetensors_header(path: str) -> tuple[dict, int]:
    """Return (header_dict, data_start_offset) of a safetensors file."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


class MmapPleTable:
    """Row gather over a table split into ``split_ngram_parts`` shard files.

    ``shards``: {shard_index: (path, absolute_byte_offset, rows)}. Shard ``i``
    holds global rows ``[i*shard_size, i*shard_size + rows)`` (vLLM's
    ``copy_ple_embedding_shard_`` layout).
    """

    def __init__(
        self,
        shards: dict[int, tuple[str, int, int]],
        shard_size: int,
        row_bytes: int,
        torch_dtype: torch.dtype,
        workers: int = 32,
        chunk: int = 2048,
    ) -> None:
        if not shards:
            raise ValueError("no PLE shards")
        self.shard_size = int(shard_size)
        self.row_bytes = int(row_bytes)
        self.torch_dtype = torch_dtype
        self.chunk = max(1, int(chunk))
        self.paths: list[str | None] = [None] * (max(shards) + 1)
        self.mm: list[np.memmap | None] = [None] * (max(shards) + 1)
        self.rows_total = 0
        advise = os.environ.get("VLLM_PLE_MMAP_MADVISE", "random").strip().lower()
        for idx, (path, offset, rows) in shards.items():
            self.paths[idx] = path
            self.mm[idx] = np.memmap(
                path, dtype=np.uint8, mode="r", offset=offset, shape=(rows, row_bytes)
            )
            # Row lookups are 160-byte reads at hashed (random) addresses. Without
            # MADV_RANDOM the kernel's mmap readahead pulls a window of pages around
            # every faulting row and fills the page cache with neighbours that are
            # never used; with it a cold row costs one page. PREWARM reads the file
            # through a separate descriptor, so it is not affected.
            if advise in ("random", "1"):
                _madvise(self.mm[idx], "random")
            self.rows_total += rows
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self.fast_rows = _env_int("VLLM_PLE_MMAP_FAST_ROWS", 512)

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """ids: int64 [N] global row ids -> uint8 [N, row_bytes] (a fresh array)."""
        import time as _time

        t0 = _time.perf_counter()
        try:
            return self._gather(ids)
        finally:
            dt = _time.perf_counter() - t0
            n = int(np.asarray(ids).size)
            _STATS["gather_ms"] += dt * 1e3
            _STATS["rows"] += n
            _STATS["bytes"] += n * self.row_bytes
            _prom_add(gather_s=dt, rows=n, bytes=n * self.row_bytes)

    def _gather(self, ids: np.ndarray) -> np.ndarray:
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        if ids.size <= self.fast_rows:
            # Decode-sized batches: thread-pool dispatch costs more than the
            # reads themselves (~50 tasks for ~65 rows). Gather inline instead.
            if ids.min() < 0 or ids.max() >= self.rows_total:
                raise IndexError(
                    f"PLE row id out of range: [{ids.min()}, {ids.max()}] "
                    f"for {self.rows_total} rows"
                )
            shard = ids // self.shard_size
            local = ids - shard * self.shard_size
            out = np.empty((ids.size, self.row_bytes), dtype=np.uint8)
            for si in np.unique(shard):
                mask = shard == si
                out[mask] = self.mm[si][local[mask]]
            return out
        # Dedupe + sort: repeated n-grams are common, and sorted rows improve
        # locality inside a shard.
        uniq, inverse = np.unique(ids, return_inverse=True)
        if uniq[0] < 0 or uniq[-1] >= self.rows_total:
            raise IndexError(
                f"PLE row id out of range: [{uniq[0]}, {uniq[-1]}] "
                f"for {self.rows_total} rows"
            )
        shard = uniq // self.shard_size
        local = uniq - shard * self.shard_size
        out = np.empty((uniq.size, self.row_bytes), dtype=np.uint8)

        bounds = np.flatnonzero(np.diff(shard)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [uniq.size]))
        tasks: list[tuple[int, int, int]] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            si = int(shard[s])
            for c in range(s, e, self.chunk):
                tasks.append((si, c, min(c + self.chunk, e)))

        def run(task: tuple[int, int, int]) -> None:
            si, a, b = task
            mm = self.mm[si]
            if mm is None:
                raise IndexError(f"PLE shard {si} missing")
            # Fancy indexing on a memmap: page faults do the I/O; NumPy releases
            # the GIL for the copy, so tasks overlap across threads.
            out[a:b] = mm[local[a:b]]

        if len(tasks) == 1:
            run(tasks[0])
        else:
            for _ in self.pool.map(run, tasks):
                pass
        return out[inverse]

    def prewarm(self) -> None:
        """Stream every shard once so the page cache holds as much as it can."""
        block = 64 << 20
        for path, mm in zip(self.paths, self.mm):
            if path is None or mm is None:
                continue
            start = mm.offset
            end = start + mm.shape[0] * mm.shape[1]
            with open(path, "rb", buffering=0) as f:
                pos = start
                while pos < end:
                    n = f.readinto(bytearray(min(block, end - pos)))  # noqa: F841
                    if not n:
                        break
                    pos += n


# --------------------------------------------------------------------------- #
# Placeholder that stands in for VocabParallelEmbedding
# --------------------------------------------------------------------------- #
class _MmapNgramEmbedding(nn.Module):
    """Duck-types the bits of VocabParallelEmbedding the PLE code reads.

    No ``weight`` attribute on purpose: ``Qwen3_8FlashNextPLELayer`` then falls
    back to ``ple_embedding._offload_weight_scale`` for the FP8 scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.org_vocab_size = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.table: MmapPleTable | None = None
        self._zeros_dtype = torch.bfloat16

    def _pinned_buf(self, rows: int, row_bytes: int) -> torch.Tensor | None:
        """Persistent pinned staging buffer for async H2D (grown as needed)."""
        buf = getattr(self, "_pinned", None)
        if buf is None or buf.shape[0] < rows or buf.shape[1] != row_bytes:
            try:
                cap = max(rows + rows // 2, 4096)
                buf = torch.empty((cap, row_bytes), dtype=torch.uint8, pin_memory=True)
            except RuntimeError:  # no CUDA (CPU tests) or pinning unavailable
                buf = None
            self._pinned = buf
        return buf

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        table = self.table
        if table is None:
            # Weights never loaded (e.g. --load-format dummy): keep the plumbing
            # alive with zeros so kernel tests can run without the 48 GiB table.
            return torch.zeros(
                (*ids.shape, self.embedding_dim),
                dtype=self._zeros_dtype,
                device=ids.device,
            )
        ids_np = ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)
        # Dedup on CPU, gather only unique rows, expand on the GPU: fewer disk
        # reads AND fewer H2D bytes (repeated n-grams are the common case).
        uniq, inverse = np.unique(ids_np, return_inverse=True)
        rows = table.gather(uniq)  # uint8 [U, row_bytes], fresh & writable
        u = rows.shape[0]
        buf = self._pinned_buf(u, table.row_bytes) if ids.device.type == "cuda" else None
        if buf is not None:
            buf[:u].numpy()[:] = rows
            dev = buf[:u].to(ids.device, non_blocking=True)
        else:
            dev = torch.from_numpy(rows).to(ids.device)
        inv = torch.from_numpy(inverse.reshape(-1)).to(ids.device, non_blocking=True)
        out = dev.view(table.torch_dtype)[inv]
        return out.reshape(*ids.shape, self.embedding_dim)


# --------------------------------------------------------------------------- #
# Patch
# --------------------------------------------------------------------------- #
def _find_shards(
    model_path: str, layer_idx: int
) -> tuple[dict[int, tuple[str, int, int]], str | None, tuple[str, int, int, str] | None, int | None]:
    """Locate ``layers.<idx>.ple.ple_embedding.ngram_embedding.shard_N.weight``.

    Returns (shards, dtype_str, scale_entry, cols), where scale_entry is
    (path, abs_offset, nbytes, dtype_str) of ``ngram_embedding.weight_scale`` or
    None, and cols is the row width shared by all shards.
    """
    shard_re = re.compile(
        rf"layers\.{layer_idx}\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
    )
    scale_re = re.compile(
        rf"layers\.{layer_idx}\.ple\.ple_embedding\.ngram_embedding\.weight_scale$"
    )
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        files = sorted(
            {
                os.path.join(model_path, fn)
                for name, fn in weight_map.items()
                if shard_re.search(name) or scale_re.search(name)
            }
        )
    else:
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    shards: dict[int, tuple[str, int, int]] = {}
    dtype_str: str | None = None
    scale_entry: tuple[str, int, int, str] | None = None
    cols: int | None = None
    for path in files:
        header, data_start = parse_safetensors_header(path)
        for name, meta in header.items():
            m = shard_re.search(name)
            if m:
                start, end = meta["data_offsets"]
                rows, cols = meta["shape"]
                if dtype_str is None:
                    dtype_str = meta["dtype"]
                elif meta["dtype"] != dtype_str:
                    raise ValueError("PLE shards have mixed dtypes")
                if end - start != rows * cols * _itemsize(dtype_str):
                    raise ValueError(f"PLE shard {name}: size/shape mismatch")
                shards[int(m.group(1))] = (path, data_start + start, rows)
            elif scale_re.search(name):
                start, end = meta["data_offsets"]
                scale_entry = (path, data_start + start, end - start, meta["dtype"])
    return shards, dtype_str, scale_entry, cols


def _itemsize(dtype_str: str) -> int:
    return {
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "U8": 1,
        "I8": 1,
        "BF16": 2,
        "F16": 2,
        "F32": 4,
    }[dtype_str]


def _read_scale(entry: tuple) -> torch.Tensor:
    path, offset, nbytes, dtype_str = entry
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(nbytes)
    if dtype_str == "F32":
        return torch.tensor(struct.unpack("<f", raw[:4])[0], dtype=torch.float32)
    if dtype_str == "BF16":
        u16 = struct.unpack("<H", raw[:2])[0]
        return torch.tensor(u16 << 16, dtype=torch.int32).view(torch.float32).squeeze()
    if dtype_str == "F16":
        return torch.frombuffer(bytearray(raw[:2]), dtype=torch.float16).clone().squeeze()
    raise ValueError(f"unsupported weight_scale dtype {dtype_str}")


_REGISTRY: dict[str, nn.Module] = {}
_OP_NAME = "ple_mmap_lookup"

# Aggregate gather-overhead stats, logged every VLLM_PLE_MMAP_STATS_SEC seconds
# (0 = off). op_ms covers hashing + gather + H2D; gather_ms just the disk reads.
_STATS = {"calls": 0, "op_ms": 0.0, "gather_ms": 0.0, "rows": 0, "bytes": 0}

# The same numbers as monotonic Prometheus counters, so the table's behaviour is
# visible on a dashboard instead of only in the log line below (which is windowed,
# reset every period, and destroyed with the container). Nothing else in this
# recipe exposes how the mmapped table is coping: it is the one component whose
# cost depends on runtime state (page-cache residency) rather than on config, so
# it is exactly the thing worth graphing.
#
# These live in the EngineCore process, not the API server, so they only reach the
# frontend's /metrics when prometheus_client runs in multiprocess mode: the API
# server's MultiProcessCollector then aggregates what every process writes under
# PROMETHEUS_MULTIPROC_DIR. vLLM only turns that on for api_server_count > 1;
# scripts/serve.sh sets it when PROM_MULTIPROC=1 (opt-in). They are created lazily on first use,
# because the env var must be set before the first metric is constructed.
#
# Derived views worth having:
#   rate(vllm:ple_mmap_op_seconds_total[5m])
#     / rate(vllm:ple_mmap_lookup_ops_total[5m])      mean seconds per lookup
#   rate(vllm:ple_mmap_gather_seconds_total[5m])
#     / rate(vllm:ple_mmap_op_seconds_total[5m])      fraction of the op spent on disk
#   rate(vllm:ple_mmap_bytes_total[5m])               NVMe read bandwidth from the table
# The second one is the page-cache health signal: it climbs as the cache is
# squeezed and falls as the hot region settles in.
_PROM: dict[str, object] | None = None
_PROM_TRIED = False


def _prom() -> dict[str, object] | None:
    """Prometheus counters, or None if unavailable. Never raises, tried once."""
    global _PROM, _PROM_TRIED
    if _PROM_TRIED:
        return _PROM
    _PROM_TRIED = True
    if os.environ.get("VLLM_PLE_MMAP_PROMETHEUS", "1").lower() in ("0", "false", "no"):
        return None
    if not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        # The default. vLLM only enables multiprocess metrics for api_server_count > 1,
        # and exporting these costs vLLM its *_created samples, so it is opt-in: say how
        # to turn it on rather than warn about the expected configuration.
        logger.info(
            "PLE mmap: PROMETHEUS_MULTIPROC_DIR is unset, so the vllm:ple_mmap_* counters "
            "stay in this process and will not appear on /metrics while the engine runs "
            "in its own process. scripts/serve.sh exports them with PROM_MULTIPROC=1."
        )
    try:
        from prometheus_client import Counter

        _PROM = {
            "ops": Counter(
                "vllm:ple_mmap_lookup_ops_total",
                "PLE mmap lookups (hash + gather + H2D) executed.",
            ),
            "op_s": Counter(
                "vllm:ple_mmap_op_seconds_total",
                "Cumulative seconds in the PLE mmap lookup op.",
            ),
            "gather_s": Counter(
                "vllm:ple_mmap_gather_seconds_total",
                "Cumulative seconds in the PLE mmap row gather (the disk reads).",
            ),
            "rows": Counter(
                "vllm:ple_mmap_rows_total",
                "Rows gathered from the mmapped PLE table.",
            ),
            "bytes": Counter(
                "vllm:ple_mmap_bytes_total",
                "Bytes read from the mmapped PLE table (page cache or NVMe).",
            ),
        }
        logger.info("PLE mmap: Prometheus counters registered")
    except Exception as exc:  # pragma: no cover - metrics must never break serving
        logger.warning("PLE mmap: Prometheus counters unavailable: %s", exc)
        _PROM = None
    return _PROM


def _prom_add(**kw: float) -> None:
    """Best-effort counter increment; a metrics failure must not fail a request."""
    p = _prom()
    if not p:
        return
    try:
        for key, value in kw.items():
            p[key].inc(value)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        pass
_STATS_LAST = [0.0]
_STATS_SEC = _env_int("VLLM_PLE_MMAP_STATS_SEC", 30)


def _stats_log() -> None:
    import time as _time

    now = _time.monotonic()
    if _STATS_SEC <= 0 or now - _STATS_LAST[0] < _STATS_SEC:
        return
    elapsed = now - _STATS_LAST[0] if _STATS_LAST[0] else float(_STATS_SEC)
    _STATS_LAST[0] = now
    s = _STATS
    if not s["calls"]:
        return
    logger.info(
        "PLE mmap stats (last %.0fs): %d ops, op %.0f ms total (%.2f ms/op), "
        "gather %.0f ms total (%.2f ms/op), %d rows, %.1f MiB read",
        elapsed, s["calls"], s["op_ms"], s["op_ms"] / s["calls"],
        s["gather_ms"], s["gather_ms"] / s["calls"],
        s["rows"], s["bytes"] / 2**20,
    )
    s.update(calls=0, op_ms=0.0, gather_ms=0.0, rows=0, bytes=0)


def _lookup_impl(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    import time as _time

    t0 = _time.perf_counter()
    layer = _REGISTRY[layer_name]
    result = layer._ple_mmap_orig_forward_impl(
        None, input_ids, query_start_loc, ngram_context
    )
    output[: result.shape[0]].copy_(result.to(output.dtype))
    dt = _time.perf_counter() - t0
    _STATS["calls"] += 1
    _STATS["op_ms"] += dt * 1e3
    _prom_add(ops=1, op_s=dt)
    _stats_log()


def _lookup_fake(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


_OP_NAME_IDS = "ple_mmap_lookup_ids"


def _lookup_ids_impl(ngram_ids: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    """v0.29 layout: gather rows for already-hashed ids; output is (N, ngram_heads * head_dim)."""
    import time as _time

    t0 = _time.perf_counter()
    layer = _REGISTRY[layer_name]
    rows = layer.ngram_embedding(ngram_ids)  # (N, heads, head_dim), table dtype (or zeros)
    output.copy_(rows.reshape(rows.shape[0], -1).to(output.dtype))
    dt = _time.perf_counter() - t0
    _STATS["calls"] += 1
    _STATS["op_ms"] += dt * 1e3
    _prom_add(ops=1, op_s=dt)
    _stats_log()


def _lookup_ids_fake(ngram_ids: torch.Tensor, output: torch.Tensor, layer_name: str) -> None:
    return


def _register_op() -> None:
    if hasattr(torch.ops.vllm, _OP_NAME):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name=_OP_NAME,
        op_func=_lookup_impl,
        mutates_args=["output"],
        fake_impl=_lookup_fake,
    )
    direct_register_custom_op(
        op_name=_OP_NAME_IDS,
        op_func=_lookup_ids_impl,
        mutates_args=["output"],
        fake_impl=_lookup_ids_fake,
    )




def _setup_table_v029(self) -> None:
    if self.ngram_embedding.table is not None:
        return
    # VLLM_PLE_MMAP_DIR: serve the table from a different directory than the
    # checkpoint (e.g. an FP8 copy of the table on local NVMe).
    model_path = os.environ.get("VLLM_PLE_MMAP_DIR") or self._ple_mmap_model_path
    if not model_path or not os.path.isdir(model_path):
        raise RuntimeError(
            f"PLE mmap: table path {model_path!r} is not a local directory; "
            "point --model at the downloaded snapshot or set VLLM_PLE_MMAP_DIR"
        )
    m = re.search(r"layers\.(\d+)\.", self._ple_mmap_prefix)
    if not m:
        raise RuntimeError(f"PLE mmap: cannot find layer index in {self._ple_mmap_prefix!r}")
    layer_idx = int(m.group(1))
    shards, dtype_str, scale_entry, cols = _find_shards(model_path, layer_idx)
    if not shards:
        raise RuntimeError(f"PLE mmap: no shard tensors for layer {layer_idx} under {model_path}")
    if cols != self.head_dim:
        raise RuntimeError(f"PLE mmap: shard width {cols} != head_dim {self.head_dim}")
    if dtype_str not in _TABLE_DTYPES:
        raise RuntimeError(f"PLE mmap: unsupported shard dtype {dtype_str}")
    if dtype_str in _FP8_DTYPES and not hasattr(self, "_offload_weight_scale"):
        if scale_entry is None:
            raise RuntimeError("PLE mmap: FP8 shards without ngram_embedding.weight_scale")
        self.register_buffer(
            "_offload_weight_scale",
            _read_scale(scale_entry).to(torch.accelerator.current_accelerator()),
            persistent=False,
        )
    parts = int(self.split_ngram_parts)
    vocab = int(self.ngram_embedding.org_vocab_size)
    shard_size = math.ceil(vocab / parts)
    for idx, (_p, _o, rows) in shards.items():
        expected = max(0, min(shard_size, vocab - idx * shard_size))
        if rows != expected:
            raise RuntimeError(
                f"PLE mmap: shard {idx} has {rows} rows, expected {expected}"
            )
    table = MmapPleTable(
        shards, shard_size, cols * _itemsize(dtype_str), _TABLE_DTYPES[dtype_str],
        workers=_env_int("VLLM_PLE_MMAP_WORKERS", 32),
        chunk=_env_int("VLLM_PLE_MMAP_CHUNK", 2048),
    )
    if _env_int("VLLM_PLE_MMAP_PREWARM", 0):
        logger.info("PLE mmap: prewarming page cache (%.1f GiB)...", table.rows_total * table.row_bytes / 2**30)
        table.prewarm()
    self.ngram_embedding.table = table
    logger.info(
        "PLE mmap: layer %d, %d shards, %d rows x %d B (%.1f GiB on disk), dtype %s, %d workers",
        layer_idx, len(shards), table.rows_total, table.row_bytes,
        table.rows_total * table.row_bytes / 2**30, dtype_str, table.pool._max_workers,
    )


def apply(cls: type) -> None:
    """Patch the n-gram embedding class (pass the class) when enabled.

    Two layouts are supported: the preview image's ``Qwen3_8FlashNextNGramEmbedding``
    (hashing + lookup in ``forward_impl``) and vLLM >= 0.29's ``Qwen4ExpNGramEmbedding``
    (hashing in the ``qwen4_exp_compute_ple_ngram_ids`` op, lookup through a
    ``PLEVocabParallelEmbedding`` whose ``weight_scale`` the PLE layer reads).
    """
    if not enabled():
        return
    if getattr(cls, "_ple_mmap_patched", False):
        return
    if not hasattr(cls, "forward_impl"):
        _apply_v029(cls)
        return
    mod = sys.modules[cls.__module__]
    orig_init = cls.__init__
    orig_load_weights = cls.load_weights

    def __init__(self, config, embedding_dim, ple_dense_layer_id, max_total_tokens,
                 max_num_reqs, prefix, quant_config=None, params_dtype=None):
        # Run the stock constructor (hash buffers, workspaces, ...) with the
        # embedding class swapped for our placeholder so nothing large is
        # allocated. quant_config=None keeps the stock code from selecting an
        # FP8 quant method that would create an FP8 weight parameter.
        real_embedding_cls = mod.VocabParallelEmbedding
        mod.VocabParallelEmbedding = lambda n, d, **_kw: _MmapNgramEmbedding(n, d)
        try:
            orig_init(self, config, embedding_dim, ple_dense_layer_id,
                      max_total_tokens, max_num_reqs, prefix,
                      quant_config=None, params_dtype=params_dtype)
        finally:
            mod.VocabParallelEmbedding = real_embedding_cls
        self._ple_mmap_prefix = prefix
        _REGISTRY[prefix] = self
        self._ple_mmap_model_path = None
        try:
            from vllm.config import get_current_vllm_config
            self._ple_mmap_model_path = get_current_vllm_config().model_config.model
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("PLE mmap: cannot read model path from vllm config: %s", exc)
        if params_dtype is not None:
            self.ngram_embedding._zeros_dtype = params_dtype
        logger.info(
            "PLE mmap: %s -> placeholder embedding (%d rows x %d), table will be mmapped",
            prefix, self.ngram_embedding.org_vocab_size, self.head_dim,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        rest: list[tuple[str, torch.Tensor]] = []
        for name, w in weights:
            if name.startswith("ngram_embedding.shard_") and name.endswith(".weight"):
                loaded.add(name)  # served from disk, never materialised
                continue
            if name == "ngram_embedding.weight_scale":
                self.register_buffer(
                    "_offload_weight_scale",
                    w.detach().to(device=torch.accelerator.current_accelerator()),
                    persistent=False,
                )
                loaded.add(name)
                continue
            rest.append((name, w))
        loaded.update(orig_load_weights(self, rest))
        _setup_table(self)
        return loaded

    def _setup_table(self) -> None:
        if self.ngram_embedding.table is not None:
            return
        # VLLM_PLE_MMAP_DIR: serve the table from a different directory than the
        # checkpoint (e.g. an FP8 copy of the table on local NVMe).
        model_path = os.environ.get("VLLM_PLE_MMAP_DIR") or self._ple_mmap_model_path
        if not model_path or not os.path.isdir(model_path):
            raise RuntimeError(
                f"PLE mmap: table path {model_path!r} is not a local directory; "
                "point --model at the downloaded snapshot or set VLLM_PLE_MMAP_DIR"
            )
        m = re.search(r"layers\.(\d+)\.", self._ple_mmap_prefix)
        if not m:
            raise RuntimeError(f"PLE mmap: cannot find layer index in {self._ple_mmap_prefix!r}")
        layer_idx = int(m.group(1))
        shards, dtype_str, scale_entry, cols = _find_shards(model_path, layer_idx)
        if not shards:
            raise RuntimeError(f"PLE mmap: no shard tensors for layer {layer_idx} under {model_path}")
        if cols != self.head_dim:
            raise RuntimeError(f"PLE mmap: shard width {cols} != head_dim {self.head_dim}")
        if dtype_str not in _TABLE_DTYPES:
            raise RuntimeError(f"PLE mmap: unsupported shard dtype {dtype_str}")
        if dtype_str in _FP8_DTYPES and not hasattr(self, "_offload_weight_scale"):
            if scale_entry is None:
                raise RuntimeError("PLE mmap: FP8 shards without ngram_embedding.weight_scale")
            self.register_buffer(
                "_offload_weight_scale",
                _read_scale(scale_entry).to(torch.accelerator.current_accelerator()),
                persistent=False,
            )
        parts = int(self.split_ngram_parts)
        vocab = int(self.ngram_embedding.org_vocab_size)
        shard_size = math.ceil(vocab / parts)
        for idx, (_p, _o, rows) in shards.items():
            expected = max(0, min(shard_size, vocab - idx * shard_size))
            if rows != expected:
                raise RuntimeError(
                    f"PLE mmap: shard {idx} has {rows} rows, expected {expected}"
                )
        table = MmapPleTable(
            shards, shard_size, cols * _itemsize(dtype_str), _TABLE_DTYPES[dtype_str],
            workers=_env_int("VLLM_PLE_MMAP_WORKERS", 32),
            chunk=_env_int("VLLM_PLE_MMAP_CHUNK", 2048),
        )
        if _env_int("VLLM_PLE_MMAP_PREWARM", 0):
            logger.info("PLE mmap: prewarming page cache (%.1f GiB)...", table.rows_total * table.row_bytes / 2**30)
            table.prewarm()
        self.ngram_embedding.table = table
        logger.info(
            "PLE mmap: layer %d, %d shards, %d rows x %d B (%.1f GiB on disk), dtype %s, %d workers",
            layer_idx, len(shards), table.rows_total, table.row_bytes,
            table.rows_total * table.row_bytes / 2**30, dtype_str, table.pool._max_workers,
        )

    def forward_impl(self, hidden_states, input_ids, query_start_loc, ngram_context,
                     output_buffer=None):
        del hidden_states, output_buffer
        num_tokens = input_ids.reshape(-1).shape[0]
        table = self.ngram_embedding.table
        dtype = table.torch_dtype if table is not None else self.ngram_embedding._zeros_dtype
        output = torch.empty(
            (num_tokens, self.embedding_dim), dtype=dtype, device=input_ids.device
        )
        getattr(torch.ops.vllm, _OP_NAME)(
            input_ids, query_start_loc, ngram_context, output, self._ple_mmap_prefix
        )
        return output

    _register_op()
    cls._ple_mmap_orig_forward_impl = cls.forward_impl
    cls.forward_impl = forward_impl
    cls.__init__ = __init__
    cls.load_weights = load_weights
    cls._setup_table = _setup_table
    cls._ple_mmap_patched = True
    logger.info("PLE mmap patch applied to %s.%s", cls.__module__, cls.__name__)


def _apply_v029(cls: type) -> None:
    """vLLM >= 0.29 layout (``vllm/models/qwen4_exp``)."""
    mod = sys.modules[cls.__module__]
    orig_init = cls.__init__
    orig_load_weights = cls.load_weights
    embed_attr = "PLEVocabParallelEmbedding"
    if not hasattr(mod, embed_attr):
        raise RuntimeError(f"PLE mmap: {mod.__name__} has no {embed_attr}; layout not recognised")

    def __init__(self, config, embedding_dim, ple_dense_layer_id, max_total_tokens,
                 max_num_reqs, prefix, layer_name, quant_config=None, params_dtype=None):
        real_cls = getattr(mod, embed_attr)
        setattr(mod, embed_attr, lambda n, d, **_kw: _MmapNgramEmbedding(n, d))
        try:
            orig_init(self, config, embedding_dim, ple_dense_layer_id, max_total_tokens,
                      max_num_reqs, prefix, layer_name, quant_config=None,
                      params_dtype=params_dtype)
        finally:
            setattr(mod, embed_attr, real_cls)
        self._ple_mmap_prefix = prefix
        _REGISTRY[prefix] = self
        self._ple_mmap_model_path = None
        try:
            from vllm.config import get_current_vllm_config
            self._ple_mmap_model_path = get_current_vllm_config().model_config.model
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("PLE mmap: cannot read model path from vllm config: %s", exc)
        if params_dtype is not None:
            self.ngram_embedding._zeros_dtype = params_dtype
        logger.info(
            "PLE mmap (v0.29 layout): %s -> placeholder embedding (%d rows x %d), table will be mmapped",
            prefix, self.ngram_embedding.org_vocab_size, self.head_dim,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        rest: list[tuple[str, torch.Tensor]] = []
        dev = torch.accelerator.current_accelerator()
        for name, w in weights:
            if name.startswith("ngram_embedding.shard_") and name.endswith(".weight"):
                loaded.add(name)  # served from disk, never materialised
                continue
            if name == "ngram_embedding.weight_scale":
                scale = w.detach().to(device=dev)
                self.register_buffer("_offload_weight_scale", scale, persistent=False)
                # Qwen4ExpPLELayer._get_embedding_weight_scale reads ngram_embedding.weight_scale
                self.ngram_embedding.weight_scale = scale
                loaded.add(name)
                continue
            rest.append((name, w))
        loaded.update(orig_load_weights(self, rest))
        self._setup_table()
        if getattr(self.ngram_embedding, "weight_scale", None) is None and hasattr(self, "_offload_weight_scale"):
            self.ngram_embedding.weight_scale = self._offload_weight_scale
        return loaded

    def forward(self, input_ids, query_start_loc, ngram_context):
        ngram_ids = input_ids.new_empty((input_ids.shape[0], self.ngram_heads), dtype=torch.long)
        torch.ops.vllm.qwen4_exp_compute_ple_ngram_ids(
            input_ids, query_start_loc, ngram_context, ngram_ids, self.layer_name
        )
        table = self.ngram_embedding.table
        dtype = table.torch_dtype if table is not None else self.ngram_embedding._zeros_dtype
        output = torch.empty((ngram_ids.shape[0], self.embedding_dim), dtype=dtype, device=input_ids.device)
        getattr(torch.ops.vllm, _OP_NAME_IDS)(ngram_ids, output, self._ple_mmap_prefix)
        return output

    _register_op()
    cls.__init__ = __init__
    cls.load_weights = load_weights
    cls.forward = forward
    cls._setup_table = _setup_table_v029
    cls._ple_mmap_patched = True
    logger.info("PLE mmap patch (v0.29 layout) applied to %s.%s", cls.__module__, cls.__name__)
