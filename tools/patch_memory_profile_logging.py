#!/usr/bin/env python3
"""Add startup memory diagnostics to a disposable vLLM image (issue #51).

Only adds logging to MemorySnapshot.measure and memory_profiling; it does not
change allocation, profiling or KV-budget decisions. Supports the v0.30/v0.31
anchors and refuses unexpected source layouts before writing anything.
"""

import ast
from pathlib import Path
import sys


MARKER = "# qwen38-memory-diagnostics"
HELPERS = '''

# qwen38-memory-diagnostics
def _qwen38_proc_memory(path, fields):
    from pathlib import Path
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return {}
    result = {}
    for line in lines:
        key, _, value = line.partition(":")
        parts = value.split()
        if key in fields and parts and parts[0].isdigit():
            result[key] = int(parts[0]) * (1024 if parts[1:] == ["kB"] else 1)
    return result


def _qwen38_memory_log(event, values):
    import json
    import os
    import traceback
    from importlib.metadata import PackageNotFoundError, version
    record = {"event": event, "pid": os.getpid(), "time": time.time(), **values}
    record["host_bytes"] = _qwen38_proc_memory("/proc/meminfo", {
        "MemTotal", "MemFree", "MemAvailable", "Cached", "SReclaimable",
        "AnonPages", "Shmem", "Unevictable", "Mlocked", "SwapTotal", "SwapFree",
    })
    record["process_bytes"] = _qwen38_proc_memory("/proc/self/status", {
        "VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap",
    })
    record["process_rollup_bytes"] = _qwen38_proc_memory("/proc/self/smaps_rollup", {
        "Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Locked",
    })
    record["callers"] = [frame.name for frame in traceback.extract_stack(limit=7)[:-1]]
    for package in ("vllm", "torch", "triton"):
        try:
            record[package + "_version"] = version(package)
        except PackageNotFoundError:
            pass
    logger.info("qwen38-memory %s", json.dumps(record, sort_keys=True))
'''


def instrument(source):
    if MARKER in source:
        raise ValueError("memory diagnostics already installed")
    replacements = (
        (
            "        self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)\n",
            "        self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)\n"
            "        _qwen38_raw_cuda_free = self.free_memory\n",
        ),
        (
            "        self.timestamp = time.time()\n",
            "        self.timestamp = time.time()\n"
            "        _qwen38_memory_log(\"snapshot\", {\n"
            "            \"raw_cuda_free_bytes\": _qwen38_raw_cuda_free,\n"
            "            \"free_bytes\": self.free_memory,\n"
            "            \"total_bytes\": self.total_memory,\n"
            "            \"torch_peak_bytes\": self.torch_peak,\n"
            "            \"torch_allocated_bytes\": self.torch_allocated,\n"
            "            \"torch_reserved_bytes\": self.torch_memory,\n"
            "            \"non_torch_bytes\": self.non_torch_memory,\n"
            "        })\n",
        ),
        (
            "    result.non_kv_cache_memory = result.total_consumed + result.transient_peak_headroom\n",
            "    result.non_kv_cache_memory = result.total_consumed + result.transient_peak_headroom\n"
            "    _qwen38_memory_log(\"profile_result\", {\n"
            "        \"initial_free_bytes\": result.before_create.free_memory,\n"
            "        \"before_profile_free_bytes\": result.before_profile.free_memory,\n"
            "        \"after_profile_free_bytes\": result.after_profile.free_memory,\n"
            "        \"weights_bytes\": result.weights_memory,\n"
            "        \"total_consumed_bytes\": result.total_consumed,\n"
            "        \"non_torch_increase_bytes\": result.non_torch_increase,\n"
            "        \"transient_peak_headroom_bytes\": result.transient_peak_headroom,\n"
            "        \"non_kv_cache_bytes\": result.non_kv_cache_memory,\n"
            "    })\n",
        ),
    )
    for old, new in replacements:
        count = source.count(old)
        if count != 1:
            raise ValueError(f"unexpected mem_utils.py: expected one anchor, got {count}: {old.strip()}")
        source = source.replace(old, new, 1)
    source += HELPERS
    ast.parse(source)
    return source


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: patch_memory_profile_logging.py <site-packages>")
    target = Path(sys.argv[1]) / "vllm/utils/mem_utils.py"
    result = instrument(target.read_text())
    target.write_text(result)
    print("Startup memory logging installed; allocation and budget logic unchanged")


if __name__ == "__main__":
    main()
