#!/usr/bin/env python3
"""Launch/setup regressions for #63/#64; synthetic cache and Docker, no GPU needed."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LaunchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="flash-launch-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / "cache"
        self.log = self.root / "docker.jsonl"
        self.cap = self.root / "capability"
        self.cap.write_text("1")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        docker = self.bin / "docker"
        docker.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["FAKE_DOCKER_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
cap = Path(os.environ["FAKE_CAP"])
if args[:2] == ["image", "inspect"]:
    if "-f" in args:
        print(cap.read_text() if "qwen38.dflash" in args[args.index("-f") + 1] else "v0.32")
elif args[0] == "build":
    cap.write_text("1")
elif args[0] == "inspect":
    print("running exit=0 oom=false" if "exit=" in args[2] else "running")
elif args[0] == "info":
    print("nvidia")
elif args[0] == "version":
    print("test")
elif args[0] in ("ps", "rm", "run", "logs"):
    pass
else:
    raise SystemExit("Unexpected Docker command: " + repr(args))
''')
        docker.chmod(0o755)
        for name, body in [("sleep", "exit 0"), ("nvidia-smi", "echo GB10")]:
            p = self.bin / name
            p.write_text("#!/bin/sh\n" + body + "\n")
            p.chmod(0o755)
        for model in ("nvidia/Qwen3.8-Flash-Next-NVFP4", "incoai/Qwen3.8-Flash-Next-DFlash2"):
            repo = self.cache / "hub" / ("models--" + model.replace("/", "--"))
            (repo / "refs").mkdir(parents=True)
            (repo / "refs/main").write_text("testrev")
            snap = repo / "snapshots/testrev"
            snap.mkdir(parents=True)
            files = {
                "config.json": {}, "generation_config.json": {},
                "tokenizer.json": {}, "tokenizer_config.json": {},
                "chat_template.jinja": {}, "preprocessor_config.json": {},
                "hf_quant_config.json": {},
                "model.safetensors.index.json": {"weight_map": {"weight": "model.safetensors"}},
                "model.safetensors": {}, ".prepared": {},
            }
            for d in (snap, snap.with_name("testrev-fp8hybrid")):
                d.mkdir(exist_ok=True)
                for name, value in files.items():
                    (d / name).write_text(json.dumps(value))
        # Keep host overrides out of the fixture, including profile defaults.
        self.env = {k: os.environ[k] for k in ("HOME", "LANG") if k in os.environ}
        self.env.update({"PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
                         "HF_CACHE": str(self.cache), "IMAGE": "test-image",
                         "XDG_STATE_HOME": str(self.root / "state"),
                         "FAKE_DOCKER_LOG": str(self.log), "FAKE_CAP": str(self.cap),
                         "EFFORT_ALIAS": "0", "DRY": "0"})

    def run_script(self, script, *args, **env):
        return subprocess.run(["bash", str(ROOT / script), *args],
                              env={**self.env, **env}, text=True, capture_output=True,
                              cwd=self.root, timeout=20)

    def calls(self):
        return [json.loads(s) for s in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_docker_options_reach_docker_not_vllm(self):
        r = self.run_script("flash", "serve", "default",
                            "DOCKER_EXTRA=-v /srv/certs:/certs:ro -e PRESENT=1",
                            "EXTRA=--ssl-certfile /certs/vllm.crt")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        args = next(x for x in self.calls() if x[0] == "run")
        image = args.index("test-image")
        self.assertIn("/srv/certs:/certs:ro", args[:image])
        self.assertIn("PRESENT=1", args[:image])
        self.assertNotIn("PRESENT=1", args[image:])
        self.assertIn("--ssl-certfile", args[image:])

    def test_dry_run_shows_override_and_starts_nothing(self):
        r = self.run_script("flash", "serve", "default", "DOCKER_EXTRA=-e PRESENT=1", DRY="1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("DOCKER_EXTRA=-e PRESENT=1", r.stdout)
        self.assertFalse(any(x[0] in ("rm", "run") for x in self.calls()))

    def test_direct_serve_no_shell_or_glob_expansion(self):
        (self.root / "glob-match").touch()
        r = self.run_script("scripts/serve.sh", MODE="nvfp4", DRAFT_MODEL="",
                            DOCKER_EXTRA="-e LITERAL=*\n-e VALUE=$(touch_owned)")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        args = next(x for x in self.calls() if x[0] == "run")
        self.assertIn("LITERAL=*", args)
        self.assertIn("VALUE=$(touch_owned)", args)

    def test_default_options_remain_empty(self):
        r = self.run_script("flash", "serve", "default")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        args = next(x for x in self.calls() if x[0] == "run")
        self.assertNotIn("", args)
        spec = json.loads(args[args.index("--speculative-config") + 1])
        self.assertEqual(spec, {"method": "mtp", "num_speculative_tokens": 2, "max_model_len": 500000})

    def test_same_base_missing_capability_rebuilds_once(self):
        self.cap.write_text("")
        for _ in range(2):
            r = self.run_script("flash", "setup", "dflash")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(sum(x[0] == "build" for x in self.calls()), 1)
        self.assertFalse(any(x[0] in ("rm", "run") for x in self.calls()))

    def test_default_setup_does_not_require_dflash(self):
        self.cap.write_text("")
        r = self.run_script("flash", "setup", "default")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(any(x[0] == "build" for x in self.calls()))

    def test_missing_capability_blocks_both_launch_paths_before_removal(self):
        self.cap.write_text("")
        for script, args in [("flash", ("serve", "dflash")), ("scripts/serve.sh", ())]:
            with self.subTest(script=script):
                r = self.run_script(script, *args, DRAFT_MODEL="incoai/Qwen3.8-Flash-Next-DFlash2")
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("lacks DFlash support", r.stdout + r.stderr)
        self.assertFalse(any(x[0] in ("rm", "run") for x in self.calls()))

    def test_capable_dflash_launch(self):
        r = self.run_script("flash", "serve", "dflash")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        args = next(x for x in self.calls() if x[0] == "run")
        spec = json.loads(args[args.index("--speculative-config") + 1])
        self.assertEqual(spec["method"], "dflash")
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER=1", args)

    def test_doctor_reports_missing_capability(self):
        self.cap.write_text("")
        r = self.run_script("flash", "doctor", "dflash")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("lacks the DFlash capability label", r.stdout)


if __name__ == "__main__":
    unittest.main()
