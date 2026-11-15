"""Guard compatibility and opt-in semantics without importing torch/vLLM."""
import ast
import importlib.util
import os
import tempfile
import unittest
from math import lcm
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ORIGINAL = HERE/'fixtures/prefix_cache/abstract_v033.py'
spec = importlib.util.spec_from_file_location('patcher', ROOT/'src/prefix_cache_boundaries/apply_patch.py')
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


class CheckpointPatchTests(unittest.TestCase):
    def test_source_guard_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'abstract.py'
            path.write_bytes(ORIGINAL.read_bytes())
            patcher.apply(path)
            first = path.read_bytes()
            patcher.apply(path)
            self.assertEqual(first,path.read_bytes())
            path.write_bytes(first+b'\n# upstream changed\n')
            before = path.read_bytes()
            with self.assertRaises(SystemExit):
                patcher.apply(path)
            self.assertEqual(before,path.read_bytes())

    def test_capability_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'abstract.py'
            path.write_bytes(ORIGINAL.read_bytes())
            patcher.apply(path)
            tree = ast.parse(path.read_text())
        func = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_mamba_prefill_checkpoint_alignment')
        class Layer:
            def __init__(self, alignment): self.alignment=alignment
            def get_prefill_checkpoint_alignment(self): return self.alignment
        namespace={'os':os,'VllmConfig':object,'MambaBase':object,'lcm':lcm,
                   'get_layers_from_vllm_config':lambda config,kind:config}
        exec(compile(ast.Module(body=[func],type_ignores=[]),'abstract.py','exec'),namespace)
        alignment=namespace[func.name]
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(alignment({'gdn':Layer(64),'ple':Layer(1)}),64)
            self.assertIsNone(alignment({'gdn':Layer(64),'other':Layer(None)}))
        with patch.dict(os.environ,{'VLLM_DGX_SPLIT_PREFILL':'1'}):
            self.assertIsNone(alignment({'gdn':Layer(64),'ple':Layer(1)}))
        with patch.dict(os.environ,{'VLLM_DGX_SPLIT_PREFILL':'0'}):
            self.assertEqual(alignment({'gdn':Layer(64),'ple':Layer(1)}),64)


if __name__ == '__main__':
    unittest.main()
