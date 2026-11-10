#!/usr/bin/env python3
"""Small tool/structured-JSON regression smoke. Does not execute model tools."""
import argparse
import json
import urllib.request
from pathlib import Path
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--base-url", default="http://localhost:18300")
    a = p.parse_args()
    model = json.load(urllib.request.urlopen(a.base_url + "/v1/models", timeout=10))["data"][0]["id"]
    tool = {"type": "function", "function": {"name": "lookup_ticket", "description": "Look up a ticket by numeric ID",
            "parameters": {"type": "object", "properties": {"ticket_id": {"type": "integer"}},
                           "required": ["ticket_id"], "additionalProperties": False}}}
    schema = {"type": "object", "properties": {"ticket_id": {"type": "integer"}, "status": {"type": "string", "enum": ["open"]}},
              "required": ["ticket_id", "status"], "additionalProperties": False}
    with a.output.open("x") as out:
        for rep in range(3):
            for case in ("tool", "json"):
                body = dict(model=model, temperature=0, max_tokens=1024,
                            chat_template_kwargs={"enable_thinking": False}, messages=[{"role": "user", "content":
                            "Look up ticket 731 using the tool. /no_think" if case == "tool" else
                            "Return a JSON object for ticket 731 with status open. /no_think"}])
                if case == "tool":
                    body.update(tools=[tool], tool_choice={"type": "function", "function": {"name": "lookup_ticket"}})
                else:
                    body["response_format"] = {"type": "json_schema", "json_schema": {"name": "ticket", "strict": True, "schema": schema}}
                row = dict(case=case, repetition=rep, label=a.label)
                try:
                    req = urllib.request.Request(a.base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                    data = json.load(urllib.request.urlopen(req, timeout=180))
                    msg = data["choices"][0]["message"]
                    if case == "tool":
                        calls = msg.get("tool_calls", [])
                        passed = len(calls) == 1 and calls[0]["function"]["name"] == "lookup_ticket" and json.loads(calls[0]["function"]["arguments"]) == {"ticket_id": 731}
                    else:
                        passed = json.loads(msg["content"]) == {"ticket_id": 731, "status": "open"}
                    row.update(ok=True, passed=passed, response=data)
                except Exception as exc:
                    row.update(ok=False, passed=False, error=str(exc))
                row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(json.dumps({k: v for k, v in row.items() if k != "response"}), flush=True)


if __name__ == "__main__":
    main()
