#!/usr/bin/env python3
"""Checkpoint preparation filesystem regression tests (issue #50).

Run: python3 tests/test_prepare_hybrid.py
Uses synthetic checkpoints and stand-ins for Docker and the FP8 converter;
no model download, GPU, third-party Python package or Docker daemon is needed.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare-hybrid.sh"
INDEX = "model.safetensors.index.json"


class PrepareHybridTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prepare hybrid ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.repo = self.cache / "hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4"
        self.source = self.repo / "snapshots/latest"
        self.source.mkdir(parents=True)
        self.hybrid = self.source.with_name("latest-fp8hybrid")
        self.blobs = self.repo / "blobs"
        self.blobs.mkdir()
        (self.repo / "refs").mkdir()
        (self.repo / "refs/main").write_text("latest")
        for name in (
            "config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "chat_template.jinja",
            "preprocessor_config.json", "hf_quant_config.json",
        ):
            (self.source / name).write_text("{}")
        self.original_index = json.dumps({"weight_map": {
            "dense.weight": "dense.safetensors",
            "expert.weight": "expert.safetensors",
        }}).encode()
        (self.blobs / "index").write_bytes(self.original_index)
        for name in ("dense.safetensors", "expert.safetensors"):
            (self.blobs / name).write_bytes(b"original synthetic shard")
            (self.source / name).symlink_to("../../blobs/" + name)

        tools = self.root / "tools"
        tools.mkdir()
        # Exercise the converter's filesystem contract: rewrite a shard and the
        # index, leaving untouched shards and the original snapshot alone.
        (tools / "fp8_convert.py").write_text('''import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
index = root / "model.safetensors.index.json"
assert not index.is_symlink(), "index must be detached from the shared blob"
if os.environ.get("FAKE_CONVERSION_FAIL"):
    sys.exit(3)
data = json.loads(index.read_text())
shard = root / "dense.safetensors"
shard.rename(root / "dense.safetensors.bf16.bak")
shard.write_bytes(b"converted synthetic shard")
data["weight_map"]["dense.weight_scale_inv"] = "dense.safetensors"
index.write_text(json.dumps(data))
''')
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        docker = bin_dir / "docker"
        docker.write_text('''#!/usr/bin/env python3
import os, shlex, subprocess, sys
args = sys.argv[1:]
assert args[0] == "run" and args[-2] == "-c", args
mounts = {}
for i, arg in enumerate(args):
    if arg == "-v":
        host, container, *_ = args[i + 1].split(":")
        mounts[container] = host
command = args[-1].replace("/hf/", mounts["/hf"] + "/")
command = command.replace("/tools/", shlex.quote(os.environ["FAKE_TOOLS"] + "/"))
sys.exit(subprocess.run(["bash", "-c", command]).returncode)
''')
        docker.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "HF_CACHE": str(self.cache),
            "MODEL": "nvidia/Qwen3.8-Flash-Next-NVFP4",
            "IMAGE": "synthetic-image",
            "FAKE_TOOLS": str(tools),
        }
        self.env.pop("FAKE_CONVERSION_FAIL", None)

    def prepare(self, **overrides):
        return subprocess.run(
            ["bash", str(SCRIPT)], cwd=SCRIPT.parents[1],
            env={**self.env, **overrides}, text=True, capture_output=True,
        )

    def assert_prepared(self):
        result = self.prepare()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.hybrid / ".prepared").is_file())
        self.assertFalse((self.hybrid / INDEX).is_symlink())
        self.assertFalse(os.path.samefile(self.hybrid / INDEX, self.source / INDEX))
        converted = json.loads((self.hybrid / INDEX).read_text())
        self.assertIn("dense.weight_scale_inv", converted["weight_map"])
        self.assertEqual((self.source / INDEX).read_bytes(), self.original_index)
        self.assertEqual((self.blobs / "index").read_bytes(), self.original_index)
        self.assertEqual((self.source / "dense.safetensors").read_bytes(),
                         b"original synthetic shard")
        self.assertEqual((self.hybrid / "dense.safetensors").read_bytes(),
                         b"converted synthetic shard")
        self.assertTrue((self.hybrid / "expert.safetensors").is_symlink())
        self.assertEqual((self.hybrid / "expert.safetensors").read_bytes(),
                         b"original synthetic shard")
        # A second preparation must reuse the completed checkpoint.
        again = self.prepare(FAKE_CONVERSION_FAIL="1")
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertIn("already prepared", again.stdout)

    def test_relative_symlink_index(self):
        (self.source / INDEX).symlink_to("../../blobs/index")
        self.assert_prepared()

    def test_regular_index(self):
        (self.source / INDEX).write_bytes(self.original_index)
        self.assert_prepared()

    def test_hardlinked_index(self):
        (self.source / INDEX).hardlink_to(self.blobs / "index")
        self.assert_prepared()

    def test_retry_after_failed_conversion(self):
        (self.source / INDEX).write_bytes(self.original_index)
        failed = self.prepare(FAKE_CONVERSION_FAIL="1")
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse((self.hybrid / ".prepared").exists())
        self.assertEqual((self.source / INDEX).read_bytes(), self.original_index)
        self.assert_prepared()


if __name__ == "__main__":
    unittest.main(verbosity=2)
