#!/usr/bin/env python3
"""Resolve the host API URL from docker inspect JSON, without executing its args."""
import json
import sys


def endpoint(container):
    args = container.get("Args", [])
    port, cert = "8000", ""
    for i, arg in enumerate(args):
        for flag in ("--port", "--ssl-certfile"):
            value = None
            if arg == flag and i + 1 < len(args):
                value = args[i + 1]
            elif arg.startswith(flag + "="):
                value = arg.split("=", 1)[1]
            if value is not None:
                if flag == "--port":
                    port = value
                else:
                    cert = value
    bindings = (container.get("NetworkSettings", {}).get("Ports") or {}).get(port + "/tcp") or []
    if not bindings:
        raise ValueError("no published API port; set FLASH_API_BASE to the reachable URL")
    # Prefer IPv4 when Docker publishes both families; keep explicit bind addresses.
    binding = next((b for b in bindings if ":" not in b.get("HostIp", "")), bindings[0])
    host = binding.get("HostIp") or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host:
        host = "[" + host + "]"
    return ("https" if cert else "http") + "://" + host + ":" + binding["HostPort"]


if __name__ == "__main__":
    try:
        print(endpoint(json.load(sys.stdin)[0]))
    except (ValueError, KeyError, IndexError, TypeError) as error:
        sys.exit("Cannot determine API endpoint: " + str(error))
