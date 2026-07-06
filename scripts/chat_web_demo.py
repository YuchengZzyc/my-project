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
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reminder_service import ReminderService
from app.storage import JSONReminderStorage
from app.tool_executor import ToolExecutor
from app.tool_registry import get_tools


SYSTEM_PROMPT = (
    "You are XiaoNuan, a warm and patient companion assistant. "
    "Only call reminder tools when user explicitly asks to create/query/update/delete reminders. "
    "For normal chatting, reply naturally without tools. "
    "Never fabricate tool execution result."
)

logger = logging.getLogger("chat_web_demo")


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Reminder Tool Chat</title>
  <style>
    :root{--bg:#0f172a;--panel:#111827;--text:#e5e7eb;--muted:#94a3b8;--user:#1d4ed8;--bot:#1f2937;--tool:#0b3b2e;--border:#334155;}
    *{box-sizing:border-box}
    body{margin:0;background:radial-gradient(circle at 20% 20%, #1e293b 0, #0f172a 40%, #020617 100%);color:var(--text);font-family: "Segoe UI",system-ui,sans-serif;}
    .wrap{max-width:980px;margin:0 auto;height:100vh;display:flex;flex-direction:column;padding:16px;}
    .head{padding:10px 14px;border:1px solid var(--border);border-radius:14px;background:rgba(17,24,39,.8);backdrop-filter:blur(4px)}
    .chat{flex:1;overflow:auto;padding:16px 4px;display:flex;flex-direction:column;gap:12px}
    .msg{max-width:78%;padding:12px 14px;border-radius:14px;line-height:1.45;border:1px solid var(--border);white-space:pre-wrap}
    .u{align-self:flex-end;background:linear-gradient(135deg,#1e40af,#1d4ed8)}
    .a{align-self:flex-start;background:var(--bot)}
    .t{align-self:flex-start;background:var(--tool);font-family:Consolas,monospace;font-size:13px}
    .bar{display:flex;gap:10px;border:1px solid var(--border);padding:10px;border-radius:14px;background:rgba(17,24,39,.9)}
    input{flex:1;border:1px solid #475569;background:#0b1220;color:var(--text);border-radius:10px;padding:10px 12px}
    button{border:none;background:#22c55e;color:#052e16;padding:10px 14px;border-radius:10px;font-weight:700;cursor:pointer}
    .muted{color:var(--muted);font-size:12px}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <div><strong>Reminder Tool Chat</strong></div>
      <div class="muted">Normal chat UI. Tool calls run internally.</div>
    </div>
    <div id="chat" class="chat"></div>
    <div class="bar">
      <input id="inp" placeholder="Type message, e.g. remind me tomorrow at 4pm to take medicine" />
      <button id="send">Send</button>
    </div>
  </div>
<script>
const chat = document.getElementById('chat');
const inp = document.getElementById('inp');
const send = document.getElementById('send');
const sid = localStorage.getItem('sid') || (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2));
localStorage.setItem('sid', sid);

function add(text, cls){
  const d=document.createElement('div');
  d.className='msg '+cls;
  d.textContent=text;
  chat.appendChild(d);
  chat.scrollTop=chat.scrollHeight;
}

async function run(){
  const text=inp.value.trim();
  if(!text) return;
  inp.value='';
  add(text,'u');
  send.disabled=true;
  try{
    const res=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({session_id:sid,message:text})});
    const data=await res.json();
    if(data.first_model_output){
      add('[first_model_output]\\n'+data.first_model_output,'t');
    }
    if(data.tool_result){
      add('[tool_result]\\n'+data.tool_result,'t');
    }
    add(data.reply || '(empty)','a');
  }catch(e){
    add('Request failed: '+e,'a');
  }finally{
    send.disabled=false;
  }
}
send.onclick=run;
inp.addEventListener('keydown',e=>{if(e.key==='Enter')run();});
</script>
</body>
</html>
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
    # Format A: OpenAI-style assistant JSON containing tool_calls
    obj = extract_json(text)
    if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
        msg = dict(obj)
        msg["role"] = "assistant"
        msg["content"] = None
        return msg

    # Format B: Qwen XML function-call tag
    # <tool_call><function=query_reminder><parameter=task>...</parameter></function></tool_call>
    qwen_xml = re.search(
        r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
        text,
        flags=re.S,
    )
    if qwen_xml:
        name = qwen_xml.group(1).strip()
        body = qwen_xml.group(2)
        arguments = {
            match.group(1).strip(): match.group(2).strip()
            for match in re.finditer(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", body, flags=re.S)
        }
        if name and arguments:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                    }
                ],
            }

    # Format C: custom JSON tag
    # <tool_call>{"name":"query_reminder","arguments":{...}}</tool_call>
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
    if isinstance(arguments, str):
        args_text = arguments
    else:
        args_text = json.dumps(arguments, ensure_ascii=False)
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


def messages_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for message in messages:
        msg = {k: v for k, v in message.items() if not k.startswith("_")}
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            msg["tool_calls"] = []
            for tool_call in tool_calls:
                tc = dict(tool_call)
                fn = dict(tc.get("function") or {})
                arguments = fn.get("arguments")
                if isinstance(arguments, str):
                    try:
                        decoded = json.loads(arguments)
                    except json.JSONDecodeError:
                        decoded = {}
                    fn["arguments"] = decoded if isinstance(decoded, dict) else {}
                tc["function"] = fn
                msg["tool_calls"].append(tc)
        rendered.append(msg)
    return rendered


class HFAssistant:
    def __init__(self, model_path: str, adapter_path: str | None = None, max_new_tokens: int = 384) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.max_new_tokens = max_new_tokens

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        template_messages = messages_for_chat_template(messages)
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    template_messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(
                    template_messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"TOOLS:\n{json.dumps(tools, ensure_ascii=False)}\n\n"
                f"MESSAGES:\n{json.dumps(template_messages, ensure_ascii=False)}\n\n"
                "Answer naturally. Use tool call only when needed."
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,     # 采样
                # do_sample=True,
                # temperature=0.7,    # 数值越高回答越有创造性,do_sample开了才有用,但回复会变长
                pad_token_id=self.tokenizer.pad_token_id,
            )
        txt = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        logger.info("model_called prompt_tokens=%s output_chars=%s", inputs["input_ids"].shape[1], len(txt))
        tc_msg = parse_tool_call_output(txt)
        if tc_msg:
            tc_msg["_raw_preview"] = txt[:500]
            return tc_msg
        obj = extract_json(txt)
        if not obj:
            return {"role": "assistant", "content": txt, "_raw_preview": txt[:500]}
        content = obj.get("content")
        if isinstance(content, str) and content.strip():
            return {"role": "assistant", "content": content.strip(), "_raw_preview": txt[:500]}
        return {"role": "assistant", "content": txt, "_raw_preview": txt[:500]}


class RequestBody(BaseModel):
    session_id: str
    message: str


def assistant_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    raw = msg.get("_raw_preview")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "[empty model output]"


def build_app(model_path: str, adapter_path: str | None, storage_path: str, stateless: bool = False) -> FastAPI:
    app = FastAPI()
    storage = JSONReminderStorage(storage_path)
    service = ReminderService(storage)
    executor = ToolExecutor(service)
    model_callable = HFAssistant(model_path=model_path, adapter_path=adapter_path)
    sessions: dict[str, list[dict[str, Any]]] = {}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.post("/api/chat")
    def chat(req: RequestBody) -> JSONResponse:
        try:
            logger.info("chat request session_id=%s user=%s", req.session_id, req.message)
            history = [] if stateless else sessions.get(req.session_id, [])
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
                    if not stateless:
                        sessions[req.session_id] = messages[1:]
                    return JSONResponse(
                        {
                            "reply": assistant_text(assistant),
                            "first_model_output": first_model_output,
                            "tool_result": tool_result,
                        }
                    )
                tool_msgs = executor.execute_tool_calls(assistant)
                for t in tool_msgs:
                    tool_result = t["content"]
                messages.extend(tool_msgs)

            if not stateless:
                sessions[req.session_id] = messages[1:]
            return JSONResponse(
                {
                    "reply": "Stopped at max tool rounds.",
                    "first_model_output": first_model_output,
                    "tool_result": tool_result,
                }
            )
        except Exception as exc:
            logger.exception("chat request failed")
            return JSONResponse(
                {
                    "reply": f"Backend error: {exc}",
                    "first_model_output": "",
                    "tool_result": "",
                    "error": str(exc),
                },
                status_code=500,
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Web chat demo with tool-use trace.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--storage-path", default="data/reminders.json")
    parser.add_argument("--stateless", action="store_true", help="Disable session memory; each request is single-turn.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = build_app(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        storage_path=args.storage_path,
        stateless=args.stateless,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
