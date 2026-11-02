#!/usr/bin/env python3
"""Issue #65: actual HTTP/TLS clients against loopback fixtures, Docker simulated."""
import http.server
import importlib.util
import json
import os
from pathlib import Path
import ssl
import subprocess
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("api_endpoint", ROOT / "scripts/api-endpoint.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def container(port="8000", host="0.0.0.0", published="18443", args=None):
    return {"Args": args or [], "NetworkSettings": {
        "Ports": {port + "/tcp": [{"HostIp": host, "HostPort": published}]}}}


class EndpointTest(unittest.TestCase):
    def test_http_published_port(self):
        self.assertEqual(module.endpoint(container()), "http://127.0.0.1:18443")

    def test_https_both_flag_forms(self):
        for args in (["--ssl-certfile", "/certs/cert.pem"], ["--ssl-certfile=/certs/cert.pem"]):
            self.assertEqual(module.endpoint(container(args=args)), "https://127.0.0.1:18443")

    def test_explicit_bind_and_custom_internal_port(self):
        c = container(port="9000", host="192.0.2.1", args=["--port=9000"])
        self.assertEqual(module.endpoint(c), "http://192.0.2.1:18443")

    def test_ipv6(self):
        self.assertEqual(module.endpoint(container(host="::")), "http://[::1]:18443")
        self.assertEqual(module.endpoint(container(host="::1")), "http://[::1]:18443")

    def test_unpublished_port_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "FLASH_API_BASE"):
            module.endpoint({"Args": [], "NetworkSettings": {"Ports": {}}})


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/metrics":
            body = ("vllm:num_requests_running 0\nvllm:prefix_cache_hits_total 1\n"
                    "vllm:prefix_cache_queries_total 2\n").encode()
        else:
            body = json.dumps({"data": [{"id": "fixture-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = {"choices": [{"text": "Paris", "logprobs": {"top_logprobs": [{"Paris": -0.1}]}}],
                "usage": {"prompt_tokens": 8000, "completion_tokens": 12}}
        body = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="flash-api-")
        cls.root = Path(cls.tmp.name)
        cls.cert, cls.key = cls.root / "cert.pem", cls.root / "key.pem"
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                        "-days", "1", "-subj", "/CN=localhost", "-addext",
                        "subjectAltName=DNS:localhost,IP:127.0.0.1", "-keyout", str(cls.key),
                        "-out", str(cls.cert)], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        cls.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.https = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cls.cert, cls.key)
        cls.https.socket = ctx.wrap_socket(cls.https.socket, server_side=True)
        for server in (cls.http, cls.https):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        bin_dir = cls.root / "bin"
        bin_dir.mkdir()
        docker = bin_dir / "docker"
        docker.write_text('''#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
if args[0] == "inspect":
    print("running" if "-f" in args else os.environ["FAKE_CONTAINER"])
elif args[0] in ("logs", "ps"):
    pass
else:
    raise SystemExit("Unexpected Docker command: " + repr(args))
''')
        docker.chmod(0o755)
        cls.env = {k: os.environ[k] for k in ("HOME", "LANG", "PATH") if k in os.environ}
        cls.env.update({"PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                        "XDG_STATE_HOME": str(cls.root / "state"), "PORT": "1", "TIMEOUT": "0"})

    @classmethod
    def tearDownClass(cls):
        for server in (cls.http, cls.https):
            server.shutdown()
            server.server_close()
        cls.tmp.cleanup()

    def run_client(self, command, tls=True, **extra):
        server = self.https if tls else self.http
        c = container(published=str(server.server_port), args=["--ssl-certfile=/certs/server.pem"] if tls else [])
        env = {**self.env, "FAKE_CONTAINER": json.dumps([c]), **extra}
        return subprocess.run(["bash", str(ROOT / command[0]), *command[1:]], env=env,
                              text=True, capture_output=True, timeout=20)

    def assert_ok(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_wait_discovers_tls_port_with_ca(self):
        r = self.run_client(["flash", "wait"], FLASH_CA=str(self.cert))
        self.assert_ok(r)
        self.assertIn(f"https://127.0.0.1:{self.https.server_port}/v1", r.stdout)
        self.assertIn("fixture-model", r.stdout)

    def test_status_reads_https_metrics_without_startup_logs(self):
        r = self.run_client(["flash", "status"], FLASH_CA=str(self.cert))
        self.assert_ok(r)
        self.assertIn("API answering at https://", r.stdout)
        self.assertIn("prefix-cache hit rate: 50%", r.stdout)

    def test_flash_test_uses_ca_for_curl_and_python(self):
        r = self.run_client(["flash", "test"], FLASH_CA=str(self.cert))
        self.assert_ok(r)
        self.assertIn("YES (deterministic)", r.stdout)
        self.assertIn("tok/s end-to-end", r.stdout)

    def test_insecure_is_explicit_and_works_for_both_clients(self):
        self.assert_ok(self.run_client(["flash", "test"], FLASH_INSECURE="1"))
        r = self.run_client(["flash", "test"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not ready", r.stdout)

    def test_ca_fallback_used_by_both_clients(self):
        self.assert_ok(self.run_client(["flash", "test"], CURL_CAINFO=str(self.cert)))

    def test_http_commands_still_work_at_published_port(self):
        for cmd in ("wait", "status", "test"):
            with self.subTest(command=cmd):
                self.assert_ok(self.run_client(["flash", cmd], tls=False))

    def test_direct_smoke_accepts_full_url_and_legacy_host_port(self):
        self.assert_ok(self.run_client(["scripts/smoke-test.sh", f"https://localhost:{self.https.server_port}"],
                                       FLASH_CA=str(self.cert)))
        self.assert_ok(self.run_client(["scripts/smoke-test.sh", f"127.0.0.1:{self.http.server_port}"], tls=False))

    def test_explicit_url_overrides_discovery(self):
        self.assert_ok(self.run_client(["flash", "test"], tls=False,
                                       FLASH_API_BASE=f"https://localhost:{self.https.server_port}/",
                                       FLASH_CA=str(self.cert)))


if __name__ == "__main__":
    unittest.main()
