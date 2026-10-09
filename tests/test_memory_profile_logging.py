#!/usr/bin/env python3
"""CPU tests for optional startup instrumentation: no torch or CUDA imports."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


TOOL = Path(__file__).resolve().parents[1] / "tools/patch_memory_profile_logging.py"
spec = importlib.util.spec_from_file_location("memory_logging", TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# The same accounting anchors as vLLM's MemorySnapshot and memory_profiling.
SOURCE = '''class Snapshot:
    def measure(self):
        device = None
        self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)
        self.free_memory = 80
        self.torch_peak = 25
        self.torch_allocated = 20
        self.torch_memory = 30
        self.non_torch_memory = self.total_memory - self.free_memory - self.torch_memory
        self.timestamp = time.time()

def finish(result):
    result.non_kv_cache_memory = result.total_consumed + result.transient_peak_headroom
'''


class MemoryLoggingTest(unittest.TestCase):
    def test_measurements_and_budget_unchanged(self):
        instrumented = ast.parse(module.instrument(SOURCE))
        # Test the added calls with a recorder instead of the /proc logger.
        tree = ast.Module(body=instrumented.body[:2], type_ignores=[])
        records = []
        ns = {
            "torch": SimpleNamespace(accelerator=SimpleNamespace(get_memory_info=lambda _: (50, 100))),
            "time": SimpleNamespace(time=lambda: 123),
            "_qwen38_memory_log": lambda event, values: records.append((event, values)),
        }
        exec(compile(tree, "instrumented", "exec"), ns)
        snap = ns["Snapshot"]()
        snap.measure()
        self.assertEqual((snap.free_memory, snap.non_torch_memory, snap.timestamp), (80, -10, 123))
        self.assertEqual(records[0][1]["raw_cuda_free_bytes"], 50)
        self.assertEqual(records[0][1]["free_bytes"], 80)
        result = SimpleNamespace(
            total_consumed=75, transient_peak_headroom=5, weights_memory=70,
            non_torch_increase=-2, before_create=SimpleNamespace(free_memory=100),
            before_profile=SimpleNamespace(free_memory=30),
            after_profile=SimpleNamespace(free_memory=25),
        )
        ns["finish"](result)
        self.assertEqual(result.non_kv_cache_memory, 80)
        self.assertEqual(records[1][1]["non_torch_increase_bytes"], -2)

    def test_refuses_unexpected_or_already_instrumented_source(self):
        for source in (SOURCE.replace("self.timestamp = time.time()", "self.timestamp = 0"),
                       SOURCE + SOURCE, module.instrument(SOURCE)):
            with self.subTest(source=source[:30]), self.assertRaises(ValueError):
                module.instrument(source)

    def test_proc_fields_are_filtered_and_converted_to_bytes(self):
        ns = {"time": SimpleNamespace(time=lambda: 123)}
        exec(module.HELPERS, ns)
        with patch.object(Path, "read_text", return_value="VmRSS: 42 kB\nName: secret\nVmSwap: 0 kB\n"):
            self.assertEqual(ns["_qwen38_proc_memory"]("unused", {"VmRSS", "VmSwap"}),
                             {"VmRSS": 42 * 1024, "VmSwap": 0})
        with patch.object(Path, "read_text", side_effect=PermissionError):
            self.assertEqual(ns["_qwen38_proc_memory"]("unused", {"VmRSS"}), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
