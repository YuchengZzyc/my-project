from __future__ import annotations

import argparse
import json
import logging
import re
import sys
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


SYSTEM_PROMPT = "Extract device-control intent and slots. Return JSON only."
REQUIRED_LABEL_KEYS = {
    "matched",
    "capability_id",
    "capability",
    "intent",
    "slots",
    "missing_slots",
    "confidence",
}

logger = logging.getLogger("device_intent_web_demo")


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Device Intent Demo</title>
  <style>
    :root{--bg:#0b1020;--panel:#121a2b;--panel2:#172033;--text:#e7edf7;--muted:#8fa1bd;--accent:#2dd4bf;--user:#1f6feb;--trace:#10271f;--border:#2b3b55;--warn:#f59e0b;--ok:#22c55e;}
    *{box-sizing:border-box}
    body{margin:0;background:#0b1020;color:var(--text);font-family:"Segoe UI",system-ui,sans-serif;}
    .wrap{max-width:1280px;margin:0 auto;height:100vh;display:flex;flex-direction:column;padding:16px;}
    .head{padding:12px 14px;border:1px solid var(--border);border-radius:8px;background:var(--panel)}
    .title{font-weight:700}
    .muted{color:var(--muted);font-size:12px;margin-top:3px}
    .main{flex:1;min-height:0;display:grid;grid-template-columns:minmax(360px,430px) 1fr;gap:14px;padding:14px 0}
    .state{min-height:0;overflow:auto;border:1px solid var(--border);border-radius:8px;background:var(--panel);padding:12px}
    .stateGrid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .tile{border:1px solid var(--border);border-radius:8px;background:var(--panel2);padding:10px;min-height:80px}
    .tileWide{grid-column:1 / -1}
    .label{color:var(--muted);font-size:12px;margin-bottom:6px}
    .value{font-size:20px;font-weight:700;line-height:1.2;overflow-wrap:anywhere}
    .small{font-size:13px;color:var(--text);overflow-wrap:anywhere}
    .meter{height:8px;background:#0b1220;border-radius:999px;overflow:hidden;margin-top:8px;border:1px solid #26364f}
    .fill{height:100%;background:var(--accent);width:0%}
    .pill{display:inline-block;padding:3px 7px;border-radius:999px;background:#0b1220;border:1px solid var(--border);font-size:12px;color:var(--muted);margin-top:6px}
    .on{color:var(--ok)}
    .off{color:var(--warn)}
    .chatPane{min-height:0;display:flex;flex-direction:column}
    .chat{flex:1;min-height:0;overflow:auto;display:flex;flex-direction:column;gap:12px}
    .msg{max-width:86%;padding:12px 14px;border-radius:8px;line-height:1.45;border:1px solid var(--border);white-space:pre-wrap}
    .u{align-self:flex-end;background:var(--user)}
    .a{align-self:flex-start;background:var(--panel2)}
    .t{align-self:flex-start;background:var(--trace);font-family:Consolas,monospace;font-size:13px}
    .bar{display:flex;gap:10px;border:1px solid var(--border);padding:10px;border-radius:8px;background:var(--panel)}
    input{flex:1;border:1px solid #3b4d6b;background:#0b1220;color:var(--text);border-radius:6px;padding:10px 12px}
    button{border:none;background:#2dd4bf;color:#042f2e;padding:10px 14px;border-radius:6px;font-weight:700;cursor:pointer}
    button:disabled{opacity:.55;cursor:not-allowed}
    @media(max-width:880px){.main{grid-template-columns:1fr}.stateGrid{grid-template-columns:1fr}.msg{max-width:100%}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <div class="title">Device Intent Demo</div>
      <div class="muted">13 capability simulator. Recognized intents update the device panel; raw JSON stays visible for debugging.</div>
    </div>
    <div class="main">
      <section class="state">
        <div class="stateGrid">
          <div class="tile">
            <div class="label">Volume</div>
            <div class="value"><span id="volumeText">50</span>%</div>
            <div class="meter"><div id="volumeFill" class="fill"></div></div>
          </div>
          <div class="tile">
            <div class="label">Brightness</div>
            <div class="value"><span id="brightnessText">60</span>%</div>
            <div class="meter"><div id="brightnessFill" class="fill"></div></div>
          </div>
          <div class="tile">
            <div class="label">Do Not Disturb</div>
            <div id="dndText" class="value off">Off</div>
          </div>
          <div class="tile">
            <div class="label">Security</div>
            <div id="securityText" class="value off">Disarmed</div>
          </div>
          <div class="tile">
            <div class="label">Door Lock</div>
            <div id="doorText" class="value on">Locked</div>
          </div>
          <div class="tile">
            <div class="label">Call</div>
            <div id="callText" class="value">Idle</div>
            <div id="callTarget" class="pill">No target</div>
          </div>
          <div class="tile">
            <div class="label">Wallpaper</div>
            <div id="wallpaperText" class="value">default</div>
          </div>
          <div class="tile">
            <div class="label">Messages</div>
            <div id="messageText" class="value">0</div>
            <div id="messageMeta" class="pill">Unread</div>
          </div>
          <div class="tile tileWide">
            <div class="label">Monitor</div>
            <div id="monitorText" class="value">Closed</div>
            <div id="monitorMeta" class="small">Scene: front_door · Channel: 1</div>
          </div>
          <div class="tile tileWide">
            <div class="label">Alarm Records</div>
            <div id="alarmText" class="value">Not queried</div>
            <div id="alarmMeta" class="small">Sample records: 2</div>
          </div>
          <div class="tile tileWide">
            <div class="label">Last Action</div>
            <div id="lastActionText" class="small">Ready</div>
          </div>
        </div>
      </section>
      <section class="chatPane">
        <div id="chat" class="chat"></div>
        <div class="bar">
          <input id="inp" placeholder="Try: Set the volume to 60 / Unlock the door / Show unread SMS" />
          <button id="send">Send</button>
        </div>
      </section>
    </div>
  </div>
<script>
const chat = document.getElementById('chat');
const inp = document.getElementById('inp');
const send = document.getElementById('send');
function add(text, cls){
  const d=document.createElement('div');
  d.className='msg '+cls;
  d.textContent=text;
  chat.appendChild(d);
  chat.scrollTop=chat.scrollHeight;
}
function pct(n){ return Math.max(0, Math.min(100, Number(n)||0)); }
function renderState(s){
  if(!s) return;
  document.getElementById('volumeText').textContent=pct(s.volume);
  document.getElementById('volumeFill').style.width=pct(s.volume)+'%';
  document.getElementById('brightnessText').textContent=pct(s.brightness);
  document.getElementById('brightnessFill').style.width=pct(s.brightness)+'%';
  const dnd=document.getElementById('dndText');
  dnd.textContent=s.dnd_enabled?'On':'Off';
  dnd.className='value '+(s.dnd_enabled?'on':'off');
  const sec=document.getElementById('securityText');
  sec.textContent=s.security_armed?'Armed':'Disarmed';
  sec.className='value '+(s.security_armed?'on':'off');
  const door=document.getElementById('doorText');
  door.textContent=s.door_locked?'Locked':'Unlocked';
  door.className='value '+(s.door_locked?'on':'off');
  document.getElementById('callText').textContent=s.call?.status || 'idle';
  document.getElementById('callTarget').textContent=s.call?.target || 'No target';
  document.getElementById('wallpaperText').textContent=s.wallpaper || 'default';
  const unread=(s.messages?.unread_sms||0)+(s.messages?.unread_notifications||0);
  document.getElementById('messageText').textContent=unread;
  document.getElementById('messageMeta').textContent=`SMS ${s.messages?.unread_sms||0} · Notifications ${s.messages?.unread_notifications||0}`;
  document.getElementById('monitorText').textContent=s.monitor?.enabled ? (s.monitor?.mode || 'open') : 'Closed';
  document.getElementById('monitorMeta').textContent=`Scene: ${s.monitor?.scene || 'front_door'} · Channel: ${s.monitor?.channel || 1}`;
  document.getElementById('alarmText').textContent=s.alarm_records?.last_query || 'Not queried';
  document.getElementById('alarmMeta').textContent=`Sample records: ${s.alarm_records?.sample_count ?? 0}`;
  document.getElementById('lastActionText').textContent=s.last_action || 'Ready';
}
async function loadState(){
  try{
    const res=await fetch('/api/state');
    const data=await res.json();
    renderState(data.device_state);
  }catch(e){}
}
async function run(){
  const text=inp.value.trim();
  if(!text) return;
  inp.value='';
  add(text,'u');
  send.disabled=true;
  try{
    const res=await fetch('/api/intent',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({message:text})});
    const data=await res.json();
    if(data.raw_model_output) add('[raw_model_output]\\n'+data.raw_model_output,'t');
    if(data.parsed_intent) add('[parsed_intent]\\n'+JSON.stringify(data.parsed_intent,null,2),'t');
    if(data.postprocess_result) add('[postprocess_result]\\n'+JSON.stringify(data.postprocess_result,null,2),'t');
    if(data.device_state) renderState(data.device_state);
    add(data.reply || '(empty)','a');
  }catch(e){
    add('Request failed: '+e,'a');
  }finally{
    send.disabled=false;
  }
}
send.onclick=run;
inp.addEventListener('keydown',e=>{if(e.key==='Enter')run();});
loadState();
</script>
</body>
</html>
"""


class RequestBody(BaseModel):
    message: str


def build_training_prompt(user_message: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    return "\n".join(
        [
            "<TOOLS>",
            "[]",
            "</TOOLS>",
            "<SYSTEM>",
            system_prompt,
            "</SYSTEM>",
            "<USER>",
            user_message,
            "</USER>",
            "<ASSISTANT>",
        ]
    )


def extract_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if "</ASSISTANT>" in cleaned:
        cleaned = cleaned.split("</ASSISTANT>", 1)[0].strip()
    if cleaned.startswith("<ASSISTANT>"):
        cleaned = cleaned[len("<ASSISTANT>") :].strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def normalize_intent_label(obj: dict[str, Any]) -> dict[str, Any] | None:
    if not REQUIRED_LABEL_KEYS.issubset(obj):
        return None
    slots = obj.get("slots")
    missing_slots = obj.get("missing_slots")
    return {
        "matched": bool(obj.get("matched")),
        "capability_id": obj.get("capability_id"),
        "capability": obj.get("capability"),
        "intent": obj.get("intent"),
        "slots": slots if isinstance(slots, dict) else {},
        "missing_slots": missing_slots if isinstance(missing_slots, list) else [],
        "confidence": obj.get("confidence", 0.0),
    }


def parse_intent_output(text: str) -> dict[str, Any] | None:
    obj = extract_json(text)
    if obj is None:
        return None
    return normalize_intent_label(obj)


def default_device_state() -> dict[str, Any]:
    return {
        "volume": 50,
        "brightness": 60,
        "dnd_enabled": False,
        "security_armed": False,
        "door_locked": True,
        "call": {"status": "idle", "target": None},
        "wallpaper": "default",
        "monitor": {"enabled": False, "mode": "idle", "scene": "front_door", "channel": 1},
        "messages": {"unread_sms": 3, "unread_notifications": 5, "last_view": None},
        "alarm_records": {"last_query": None, "sample_count": 2},
        "last_action": "Ready",
    }


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(state, ensure_ascii=False))


def clamp_level(value: Any, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(0, min(100, number))


def apply_level(current: int, slots: dict[str, Any]) -> int:
    adjustment = slots.get("adjustment")
    level = slots.get("level")
    if isinstance(level, str):
        lowered = level.lower()
        if lowered == "max":
            return 100
        if lowered == "min":
            return 0
    if level is not None:
        return clamp_level(level, current)
    if adjustment == "up":
        return clamp_level(current + 10, current)
    if adjustment == "down":
        return clamp_level(current - 10, current)
    return current


class SimulatedDeviceExecutor:
    def __init__(self) -> None:
        self.state = default_device_state()

    def execute(self, intent_label: dict[str, Any]) -> dict[str, Any]:
        if not intent_label.get("matched"):
            self.state["last_action"] = "No supported device-control intent detected."
            return {"status": "skipped", "reason": "not_matched", "state": clone_state(self.state), "changes": []}

        missing_slots = intent_label.get("missing_slots") or []
        if missing_slots:
            self.state["last_action"] = "Missing required slots: " + ", ".join(str(item) for item in missing_slots)
            return {"status": "missing_slots", "missing_slots": missing_slots, "state": clone_state(self.state), "changes": []}

        before = clone_state(self.state)
        intent = intent_label.get("intent")
        slots = intent_label.get("slots") or {}
        changes = self.apply_intent(intent, slots)
        event = {
            "event": "device_intent_postprocess",
            "capability_id": intent_label.get("capability_id"),
            "capability": intent_label.get("capability"),
            "intent": intent,
            "slots": slots,
            "changes": changes,
            "state": clone_state(self.state),
        }
        print(json.dumps(event, ensure_ascii=False), flush=True)
        return {"status": "printed", "before": before, **event}

    def apply_intent(self, intent: str | None, slots: dict[str, Any]) -> list[str]:
        changes: list[str] = []
        if intent == "call_manager":
            self.state["call"] = {"status": "dialing", "target": "property_manager"}
            self.state["last_action"] = "Calling property manager."
            changes.append("call.status")
        elif intent == "call_contact":
            target = slots.get("contact") or "unknown_contact"
            self.state["call"] = {"status": "dialing", "target": target}
            self.state["last_action"] = f"Calling {target}."
            changes.append("call.status")
        elif intent == "answer_call":
            self.state["call"]["status"] = "answered"
            self.state["last_action"] = "Incoming call answered."
            changes.append("call.status")
        elif intent == "unlock":
            self.state["door_locked"] = False
            self.state["last_action"] = "Door unlocked."
            changes.append("door_locked")
        elif intent == "end_or_reject_call":
            action = slots.get("action")
            self.state["call"]["status"] = "rejected" if action == "reject" else "idle"
            self.state["last_action"] = "Call rejected." if action == "reject" else "Call ended."
            changes.append("call.status")
        elif intent == "change_wallpaper":
            wallpaper_type = slots.get("wallpaper_type") or "default"
            self.state["wallpaper"] = str(wallpaper_type)
            self.state["last_action"] = f"Wallpaper changed to {wallpaper_type}."
            changes.append("wallpaper")
        elif intent == "set_dnd":
            self.state["dnd_enabled"] = bool(slots.get("enabled"))
            self.state["last_action"] = "Do not disturb enabled." if self.state["dnd_enabled"] else "Do not disturb disabled."
            changes.append("dnd_enabled")
        elif intent == "set_volume":
            self.state["volume"] = apply_level(int(self.state["volume"]), slots)
            self.state["last_action"] = f"Volume set to {self.state['volume']}%."
            changes.append("volume")
        elif intent == "set_brightness":
            self.state["brightness"] = apply_level(int(self.state["brightness"]), slots)
            self.state["last_action"] = f"Brightness set to {self.state['brightness']}%."
            changes.append("brightness")
        elif intent == "set_security_mode":
            self.state["security_armed"] = bool(slots.get("enabled"))
            self.state["last_action"] = "Security armed." if self.state["security_armed"] else "Security disarmed."
            changes.append("security_armed")
        elif intent == "query_alarm_records":
            time_range = slots.get("time_range") or "all"
            self.state["alarm_records"]["last_query"] = str(time_range)
            self.state["last_action"] = f"Queried alarm records: {time_range}."
            changes.append("alarm_records.last_query")
        elif intent == "monitor_control":
            action = slots.get("action") or "view"
            if action == "open":
                self.state["monitor"]["enabled"] = True
                self.state["monitor"]["mode"] = "live"
            elif action == "close":
                self.state["monitor"]["enabled"] = False
                self.state["monitor"]["mode"] = "closed"
            elif action == "playback":
                self.state["monitor"]["enabled"] = True
                self.state["monitor"]["mode"] = "playback"
            elif action == "switch_channel":
                self.state["monitor"]["enabled"] = True
                self.state["monitor"]["mode"] = "live"
                self.state["monitor"]["channel"] = int(slots.get("channel") or self.state["monitor"]["channel"])
            else:
                self.state["monitor"]["enabled"] = True
                self.state["monitor"]["mode"] = "live"
            if slots.get("scene"):
                self.state["monitor"]["scene"] = str(slots["scene"])
            self.state["last_action"] = f"Monitor action: {action}."
            changes.append("monitor")
        elif intent == "view_messages":
            message_type = slots.get("message_type") or "all"
            self.state["messages"]["last_view"] = str(message_type)
            if message_type == "sms":
                self.state["messages"]["unread_sms"] = 0
            elif message_type == "notification":
                self.state["messages"]["unread_notifications"] = 0
            else:
                self.state["messages"]["unread_sms"] = 0
                self.state["messages"]["unread_notifications"] = 0
            self.state["last_action"] = f"Viewed messages: {message_type}."
            changes.append("messages")
        else:
            self.state["last_action"] = f"Unsupported parsed intent: {intent}."
        return changes


PrintOnlyDeviceExecutor = SimulatedDeviceExecutor


class DeviceIntentModel:
    def __init__(self, model_path: str, adapter_path: str | None = None, max_new_tokens: int = 160) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.max_new_tokens = max_new_tokens

    def __call__(self, user_message: str) -> str:
        prompt = build_training_prompt(user_message)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        logger.info("model_called prompt_tokens=%s output_chars=%s", inputs["input_ids"].shape[1], len(text))
        return text


def reply_for_result(raw_output: str, parsed: dict[str, Any] | None, postprocess_result: dict[str, Any] | None) -> str:
    if parsed is None:
        return raw_output or "No model output."
    if not parsed.get("matched"):
        return "No device-control intent detected. Treating this as normal output."
    if postprocess_result and postprocess_result.get("status") == "missing_slots":
        return "Device intent recognized, but required slots are missing: " + ", ".join(postprocess_result.get("missing_slots", []))
    return f"Device intent recognized: {parsed.get('intent')}. The simulator panel has been updated."


def build_app(model_path: str, adapter_path: str | None, max_new_tokens: int = 160) -> FastAPI:
    app = FastAPI()
    model = DeviceIntentModel(model_path=model_path, adapter_path=adapter_path, max_new_tokens=max_new_tokens)
    executor = SimulatedDeviceExecutor()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse({"device_state": clone_state(executor.state)})

    @app.post("/api/intent")
    def intent(req: RequestBody) -> JSONResponse:
        try:
            raw_output = model(req.message)
            parsed = parse_intent_output(raw_output)
            postprocess_result = executor.execute(parsed) if parsed else None
            device_state = postprocess_result.get("state") if postprocess_result else clone_state(executor.state)
            return JSONResponse(
                {
                    "reply": reply_for_result(raw_output, parsed, postprocess_result),
                    "raw_model_output": raw_output,
                    "parsed_intent": parsed,
                    "postprocess_result": postprocess_result,
                    "device_state": device_state,
                }
            )
        except Exception as exc:
            logger.exception("intent request failed")
            return JSONResponse({"reply": f"Backend error: {exc}", "error": str(exc)}, status_code=500)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Web demo for device intent recognition and print-only postprocess.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8019)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = build_app(model_path=args.model_path, adapter_path=args.adapter_path, max_new_tokens=args.max_new_tokens)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
