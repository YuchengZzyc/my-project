from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.mcp_bridge_baseline import MCPBridgeBaseline
from app.tool_registry import get_tools

logger = logging.getLogger("web_mcp_baseline")

SYSTEM_PROMPT = (
    "You are XiaoNuan, a warm and patient companion assistant. "
    "Only call reminder tools when user explicitly asks to create/query/update/delete reminders. "
    "For normal chatting, reply naturally without tools. Never fabricate tool execution result."
)

HTML = """
<!doctype html>
<html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MCP Baseline Chat</title>
<style>
body{margin:0;background:#0b1220;color:#e5e7eb;font-family:Segoe UI,system-ui}
.w{max-width:980px;margin:0 auto;height:100vh;display:flex;flex-direction:column;padding:14px}
.h{padding:10px;border:1px solid #334155;border-radius:12px;background:#111827}
.c{flex:1;overflow:auto;padding:12px 0;display:flex;flex-direction:column;gap:10px}
.m{max-width:78%;padding:10px 12px;border-radius:12px;white-space:pre-wrap;border:1px solid #334155}
.u{align-self:flex-end;background:#1d4ed8}.a{align-self:flex-start;background:#1f2937}.d{align-self:flex-start;background:#0b3b2e;font-family:Consolas,monospace;font-size:12px}
.b{display:flex;gap:8px;border:1px solid #334155;background:#111827;padding:10px;border-radius:12px}
input{flex:1;padding:10px;border-radius:10px;border:1px solid #475569;background:#0f172a;color:#e5e7eb}
button{padding:10px 14px;border:none;border-radius:10px;background:#22c55e;color:#052e16;font-weight:700}
</style></head>
<body><div class="w"><div class="h"><b>MCP Baseline Chat</b></div><div id="c" class="c"></div><div class="b"><input id="i" placeholder="提醒我明天下午打篮球"/><button id="s">Send</button></div></div>
<script>
const c=document.getElementById('c'),i=document.getElementById('i'),s=document.getElementById('s');
const sid=localStorage.getItem('sid_mcp')||(crypto.randomUUID?crypto.randomUUID():Math.random().toString(36).slice(2)); localStorage.setItem('sid_mcp',sid);
function add(t,k){const d=document.createElement('div');d.className='m '+k;d.textContent=t;c.appendChild(d);c.scrollTop=c.scrollHeight;}
async function run(){const t=i.value.trim(); if(!t)return; i.value=''; add(t,'u'); s.disabled=true;
try{const r=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({session_id:sid,message:t})}); const d=await r.json();
if(d.first_model_output)add('[first_model_output]\\n'+d.first_model_output,'d');
if(d.tool_result)add('[tool_result]\\n'+d.tool_result,'d');
add(d.reply||'(empty)','a');
}catch(e){add('request failed: '+e,'a')}finally{s.disabled=false}}
s.onclick=run; i.addEventListener('keydown',e=>{if(e.key==='Enter')run();});
</script></body></html>
"""


def extract_json(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def parse_tool_call_output(text: str) -> dict[str, Any] | None:
    obj = extract_json(text)
    if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
        return {"role": "assistant", "content": None, "tool_calls": obj["tool_calls"]}
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.S)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
    except Exception:
        return None
    if not isinstance(call, dict):
        return None
    name = call.get("name")
    arguments = call.get("arguments", {})
    if not isinstance(name, str) or not name.strip():
        return None
    args_text = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name.strip(), "arguments": args_text},
            }
        ],
    }


def assistant_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    raw = msg.get("_raw_preview")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "[empty model output]"


class HFAssistant:
    def __init__(self, model_path: str, adapter_path: str | None = None, max_new_tokens: int = 384) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.max_new_tokens = max_new_tokens

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        txt = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        logger.info("model_called prompt_tokens=%s output_chars=%s", inputs["input_ids"].shape[1], len(txt))
        tc = parse_tool_call_output(txt)
        if tc:
            tc["_raw_preview"] = txt[:500]
            return tc
        obj = extract_json(txt)
        if isinstance(obj, dict):
            content = obj.get("content")
            if isinstance(content, str) and content.strip():
                return {"role": "assistant", "content": content.strip(), "_raw_preview": txt[:500]}
        return {"role": "assistant", "content": txt, "_raw_preview": txt[:500]}


class RequestBody(BaseModel):
    session_id: str
    message: str


def build_app(model_path: str, adapter_path: str | None, mcp_base_url: str) -> FastAPI:
    app = FastAPI(title="Web MCP Baseline")
    model_callable = HFAssistant(model_path=model_path, adapter_path=adapter_path)
    mcp = MCPBridgeBaseline(base_url=mcp_base_url)
    sessions: dict[str, list[dict[str, Any]]] = {}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.post("/api/chat")
    def chat(req: RequestBody) -> JSONResponse:
        history = sessions.get(req.session_id, [])
        tools = get_tools()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": req.message}]
        first_model_output = ""
        tool_result = ""

        for _ in range(3):
            assistant = model_callable(messages, tools)
            if not first_model_output:
                first_model_output = str(assistant.get("_raw_preview") or assistant.get("content") or "")
            messages.append(assistant)
            if not assistant.get("tool_calls"):
                sessions[req.session_id] = messages[1:]
                return JSONResponse({"reply": assistant_text(assistant), "first_model_output": first_model_output, "tool_result": tool_result})
            tool_msgs = mcp.execute_tool_calls(assistant)
            if tool_msgs:
                tool_result = tool_msgs[0]["content"]
            messages.extend(tool_msgs)

        sessions[req.session_id] = messages[1:]
        return JSONResponse({"reply": "Stopped at max tool rounds.", "first_model_output": first_model_output, "tool_result": tool_result})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent MCP + Qwen baseline web chat.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--mcp-base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8018)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = build_app(model_path=args.model_path, adapter_path=args.adapter_path, mcp_base_url=args.mcp_base_url)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

