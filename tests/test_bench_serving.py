#!/usr/bin/env python3
"""Benchmark integrity against actual HTTP/SSE fixtures, without a model."""
import http.server
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

SPEC = importlib.util.spec_from_file_location(
    "bench_serving", Path(__file__).resolve().parents[1] / "tools/bench_serving.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def sse(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


TEXT = {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}
FINISH = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
USAGE = {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
DONE = b"data: [DONE]\n\n"


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(self.server.payload)))
        self.end_headers()
        self.wfile.write(self.server.payload)

    def do_GET(self):
        body = json.dumps({"data": [{"id": "fixture-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class StreamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, payload):
        self.server.payload = payload
        return bench.request(self.base, {"stream": True})

    def test_complete_stream_with_separate_usage(self):
        result = self.request(sse(TEXT) + sse(FINISH) + sse(USAGE) + DONE)
        self.assertEqual(result["text"], "hello")
        self.assertEqual(result["usage"]["completion_tokens"], 2)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertIsNotNone(result["ttft_s"])

    def test_output_limit_is_retained_for_reporting(self):
        finish = {"choices": [{"delta": {}, "finish_reason": "length"}]}
        result = self.request(sse(TEXT) + sse(finish) + sse(USAGE) + DONE)
        self.assertEqual(result["finish_reason"], "length")

    def test_http_200_stream_error_is_not_a_success(self):
        for payload in (sse({"error": {"message": "engine failed"}}) + DONE,
                        sse(TEXT) + sse(FINISH) + sse(USAGE)
                        + sse({"error": {"message": "engine failed"}}) + DONE):
            with self.subTest(payload=payload), self.assertRaisesRegex(RuntimeError, "stream error"):
                self.request(payload)

    def test_eof_and_missing_terminal_fields_are_not_successes(self):
        for payload in (b"", sse(TEXT), sse(TEXT) + sse(FINISH) + sse(USAGE),
                        sse(TEXT) + sse(USAGE) + DONE, sse(TEXT) + sse(FINISH) + DONE):
            with self.subTest(payload=payload), self.assertRaisesRegex(RuntimeError, "incomplete stream"):
                self.request(payload)

    def test_invalid_completion_count_is_not_a_success(self):
        for count in (None, -1, "2", True):
            usage = {"usage": {"completion_tokens": count}}
            with self.subTest(count=count), self.assertRaisesRegex(RuntimeError, "invalid stream usage"):
                self.request(sse(TEXT) + sse(FINISH) + sse(usage) + DONE)

    def test_empty_generation_with_valid_usage_is_reported(self):
        usage = {"usage": {"prompt_tokens": 10, "completion_tokens": 0}}
        result = self.request(sse(FINISH) + sse(usage) + DONE)
        self.assertEqual(result["text"], "")
        self.assertIsNone(result["ttft_s"])
        self.assertIsNone(result["decode_tokens_per_s"])

    def test_cli_preserves_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "baseline.jsonl"
            output.write_text('previous measurements\n')
            result = subprocess.run([sys.executable, str(SPEC.origin),
                                     '--base-url', self.base, '--seconds', '1',
                                     '--label', 'fixture', '--output', str(output)],
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('FileExistsError', result.stderr)
            self.assertEqual(output.read_text(), 'previous measurements\n')


if __name__ == "__main__":
    unittest.main()
