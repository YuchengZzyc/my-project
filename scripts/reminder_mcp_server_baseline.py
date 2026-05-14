from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reminder_service import ReminderService
from app.storage import JSONReminderStorage
from app.tool_registry import get_tools


class CallBody(BaseModel):
    name: str
    arguments: dict[str, Any] = {}


def build_app(storage_path: str) -> FastAPI:
    app = FastAPI(title="Reminder MCP Baseline Server")
    service = ReminderService(JSONReminderStorage(storage_path))
    tool_map = {
        "create_reminder": service.create_reminder,
        "query_reminder": service.query_reminder,
        "update_reminder": service.update_reminder,
        "delete_reminder": service.delete_reminder,
    }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/mcp/tools/list")
    def tools_list() -> dict[str, Any]:
        return {"ok": True, "tools": get_tools()}

    @app.post("/mcp/tools/call")
    def tools_call(req: CallBody) -> dict[str, Any]:
        fn = tool_map.get(req.name)
        if fn is None:
            return {"ok": False, "error": f"unknown_tool:{req.name}"}
        try:
            result = fn(**req.arguments)
            return {"ok": True, "result": result}
        except TypeError as e:
            return {"ok": True, "result": {"status": "error", "state": False, "message": f"invalid_arguments:{e}"}}
        except Exception as e:
            return {"ok": True, "result": {"status": "error", "state": False, "message": f"tool_execution_error:{e}"}}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local MCP baseline server for reminder tools.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--storage-path", default="data/reminders_mcp_baseline.json")
    args = parser.parse_args()

    app = build_app(storage_path=args.storage_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

