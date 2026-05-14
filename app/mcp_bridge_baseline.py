from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class MCPBridgeBaseline:
    def __init__(self, base_url: str = "http://127.0.0.1:8765") -> None:
        self.base_url = base_url.rstrip("/")

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                obj = json.loads(raw)
                return obj if isinstance(obj, dict) else {"ok": False, "error": "invalid_json_object"}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return {"ok": False, "error": f"http_error:{e.code}", "body": body}
        except Exception as e:
            return {"ok": False, "error": f"request_error:{e}"}

    def execute_tool_call(self, tool_call: dict[str, Any]) -> dict[str, str]:
        call_id = tool_call.get("id", "call_unknown")
        fn = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        name = fn.get("name", "unknown")
        args_raw = fn.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except Exception:
            args = {}
        res = self._post_json("/mcp/tools/call", {"name": name, "arguments": args})
        payload = res.get("result") if res.get("ok") else {"status": "error", "message": res.get("error", "mcp_call_failed")}
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": json.dumps(payload, ensure_ascii=False),
        }

    def execute_tool_calls(self, assistant_message: dict[str, Any]) -> list[dict[str, str]]:
        calls = assistant_message.get("tool_calls", []) if isinstance(assistant_message, dict) else []
        out: list[dict[str, str]] = []
        for tc in calls:
            if isinstance(tc, dict):
                out.append(self.execute_tool_call(tc))
        return out

