from __future__ import annotations

import argparse
import json
import logging
import time
import uuid

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn


SYSTEM_PROMPT = (
    "You are XiaoNuan, a warm and knowledgeable companion for elderly users. "
    "Your role is to chat with them, answer their questions, and keep them company. "
    "Speak naturally and gently, like a caring, well-informed friend — not a doctor, "
    "not a dispatcher, not a secretary. "

    # ── 对话策略 ──
    "When a user shares a worry, discomfort, or sadness, your first and most "
    "important job is to acknowledge their feeling and sit with them in the moment. "
    "Offer comfort and companionship first. "
    "Only if the situation sounds genuinely serious may you gently suggest they "
    "consider talking to a doctor — phrase it softly, e.g. 'It might be worth "
    "checking with your doctor about this' or 'Would it help to talk to someone?'. "
    "Never use commanding language like 'Do not', 'You must', or 'Call immediately'. "
    "Most health mentions are users wanting to be heard, not redirected. "
    "Do not let 'see a doctor' become your default reply. "

    # ── 表达多样性 ──
    "Vary your wording, sentence structure, and closing in every response. "
    "Do not reuse the same phrases or templates across conversations. "
    "Each reply should feel fresh, personal, and spontaneous. "

    # ── 知识问答 ──
    "When the user asks a factual question, answer it clearly and accurately "
    "in plain, easy-to-understand language. "
    "If you are unsure about something, be honest and say so."

    # ── 回答长度（重要）──
    "Keep your reply concise — aim for 2 to 4 short paragraphs at most. "
    "Go straight to the point; skip unnecessary greetings, filler words, "
    "transitional phrases, and repeating what the user already said. "
    "Be warm and patient, but do not pad your answer. "
    "If a topic can be covered in two sentences, use two sentences — not ten."
)

logger = logging.getLogger("chat_web_demo")


# ──────────────────────────────────────────────────────────────────────────────
#  Frontend — warm & elderly-friendly UI  (marked.js + thinking timer + sidebar)
# ──────────────────────────────────────────────────────────────────────────────
HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>XiaoNuan Agent</title>
  <!-- marked.js for Markdown rendering -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    /* ── Design Tokens — warm & soft ── */
    :root {
      --bg:          #fdfbf7;
      --sidebar:     #f5f0e8;
      --panel:       #ffffff;
      --surface:     #faf7f2;
      --border:      #e8e0d4;
      --border-hi:   #d4c9b8;
      --text:        #3d3329;
      --text-soft:   #6b5d4f;
      --muted:       #9a8b78;
      --accent:      #5b9bd5;
      --accent-soft: #eaf4fc;
      --accent-glow: rgba(91,155,213,.12);
      --green:       #6dab86;
      --green-soft:  #edf7f0;
      --user-bubble: #5b9bd5;
      --bot-bubble:  #ffffff;
      --radius:      18px;
      --radius-sm:   12px;
      --radius-xs:   8px;
      --shadow-sm:   0 2px 8px rgba(61,51,41,.06);
      --shadow-md:   0 4px 16px rgba(61,51,41,.08);
      --shadow-lg:   0 8px 32px rgba(61,51,41,.10);
      --font:        "PingFang SC","Hiragino Sans GB","Microsoft YaHei",
                     "Segoe UI",system-ui,sans-serif;
    }

    *{box-sizing:border-box;margin:0;padding:0}
    html,body{height:100%;font-family:var(--font);background:var(--bg);color:var(--text);overflow:hidden}

    /* ── Layout ── */
    .app{display:flex;height:100vh;width:100vw}

    /* ═══════════════════════════════════════════
       Sidebar
       ═══════════════════════════════════════════ */
    .sidebar{
      width:250px;min-width:250px;
      background:var(--sidebar);
      border-right:1px solid var(--border);
      display:flex;flex-direction:column;
      overflow:hidden;
    }
    .sidebar-header{
      padding:20px 18px 16px;
      border-bottom:1px solid var(--border);
    }
    .sidebar-header .logo{
      width:42px;height:42px;border-radius:14px;
      background:linear-gradient(135deg,#7db8e8,#5b9bd5);
      display:flex;align-items:center;justify-content:center;
      font-size:20px;color:#fff;box-shadow:0 3px 12px rgba(91,155,213,.18);
      margin-bottom:10px;
    }
    .sidebar-header h2{font-size:16px;font-weight:700;color:var(--text)}
    .sidebar-header .sub{font-size:12px;color:var(--muted);margin-top:3px}

    .history-list{flex:1;overflow-y:auto;padding:10px 10px}

    .history-empty{
      padding:16px 12px;font-size:12.5px;color:var(--muted);
      text-align:center;line-height:1.6;
    }

    .history-item{
      padding:10px 12px;border-radius:var(--radius-sm);
      font-size:13px;line-height:1.45;cursor:pointer;
      color:var(--text-soft);transition:all .2s;
      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
      border:1px solid transparent;margin-bottom:3px;
      background:transparent;
    }
    .history-item:hover{
      background:var(--panel);border-color:var(--border);
      box-shadow:var(--shadow-sm);
    }
    .history-item.active{
      background:var(--accent-soft);border-color:var(--accent);
      color:var(--accent);font-weight:600;
    }
    .history-item .idx{
      display:inline-block;width:22px;height:22px;border-radius:7px;
      background:var(--accent);color:#fff;font-size:10.5px;
      font-weight:700;text-align:center;line-height:22px;
      margin-right:8px;vertical-align:middle;
    }

    .sidebar-footer{
      padding:10px 16px;border-top:1px solid var(--border);
      font-size:11px;color:var(--muted);text-align:center;
    }

    /* ═══════════════════════════════════════════
       Main Chat Area
       ═══════════════════════════════════════════ */
    .main{flex:1;display:flex;flex-direction:column;min-width:0;background:var(--bg)}

    .chat-header{
      padding:14px 24px;border-bottom:1px solid var(--border);
      background:rgba(253,251,247,.85);backdrop-filter:blur(16px);
      display:flex;align-items:center;gap:14px;
    }
    .chat-header .avatar{
      width:40px;height:40px;border-radius:14px;
      background:linear-gradient(135deg,#7db8e8,#5b9bd5);
      display:flex;align-items:center;justify-content:center;
      font-size:18px;color:#fff;flex-shrink:0;
      box-shadow:0 3px 10px rgba(91,155,213,.18);
    }
    .chat-header .info h3{font-size:15px;font-weight:700;color:var(--text)}
    .chat-header .info p{font-size:12px;color:var(--muted)}
    .chat-header .status{
      margin-left:auto;display:flex;align-items:center;gap:6px;
      font-size:12px;color:var(--green);font-weight:600;
    }
    .chat-header .status .dot{
      width:7px;height:7px;border-radius:50%;background:var(--green);
      box-shadow:0 0 6px rgba(90,158,111,.4);
    }

    /* ── Messages ── */
    .chat{
      flex:1;overflow-y:auto;padding:28px 24px;
      display:flex;flex-direction:column;gap:22px;
    }

    .msg-row{display:flex;flex-direction:column;gap:5px}
    .msg-row.user{align-items:flex-end}
    .msg-row.bot{align-items:flex-start}

    .msg-label{
      font-size:11.5px;color:var(--muted);padding:0 8px;margin-bottom:2px;
      font-weight:500;
    }

    .msg{
      max-width:74%;padding:14px 18px;border-radius:var(--radius);
      line-height:1.65;font-size:15px;word-break:break-word;
    }
    .msg.user{
      background:var(--user-bubble);color:#fff;
      border-bottom-right-radius:6px;
      box-shadow:0 3px 12px rgba(91,155,213,.15);
    }
    .msg.bot{
      background:var(--bot-bubble);color:var(--text);
      border:1px solid var(--border);
      border-bottom-left-radius:6px;
      box-shadow:var(--shadow-md);
      position:relative;
    }

    /* ── Markdown content inside bot bubble ── */
    .msg.bot p{margin:0 0 .55em}
    .msg.bot p:last-child{margin-bottom:0}
    .msg.bot ul,.msg.bot ol{padding-left:1.4em;margin:.3em 0 .55em}
    .msg.bot li{margin-bottom:.25em}
    .msg.bot strong{color:var(--accent);font-weight:700}
    .msg.bot em{color:var(--text-soft)}
    .msg.bot h1,.msg.bot h2,.msg.bot h3{
      margin:.5em 0 .35em;font-weight:700;color:var(--text);
    }
    .msg.bot h1{font-size:1.15em}.msg.bot h2{font-size:1.08em}.msg.bot h3{font-size:1.02em}
    .msg.bot blockquote{
      border-left:3px solid var(--accent);padding:.3em .8em;
      margin:.5em 0;color:var(--text-soft);
      background:var(--accent-soft);border-radius:0 var(--radius-xs) var(--radius-xs) 0;
    }
    .msg.bot code{
      background:#f0ebe3;padding:2px 6px;border-radius:5px;
      font-size:.9em;color:var(--accent);
    }
    .msg.bot pre{
      background:#f5f0e8;padding:12px 14px;border-radius:var(--radius-xs);
      overflow-x:auto;margin:.5em 0;font-size:.88em;
    }
    .msg.bot pre code{background:none;padding:0;color:var(--text)}
    .msg.bot hr{border:none;border-top:1px solid var(--border);margin:.6em 0}
    .msg.bot a{color:var(--accent);text-decoration:underline}

    /* ── Collapsible bot message ── */
    .msg.bot.collapsible{
      max-height:280px;overflow:hidden;
      transition:max-height .45s cubic-bezier(.4,0,.2,1);
    }
    .msg.bot.collapsible::after{
      content:"";position:absolute;bottom:0;left:0;right:0;
      height:52px;
      background:linear-gradient(transparent 0%,var(--bot-bubble) 85%);
      pointer-events:none;transition:opacity .35s;
    }
    .msg.bot.expanded::after{opacity:0}

    .toggle-btn{
      display:inline-flex;align-items:center;gap:5px;
      background:var(--panel);color:var(--accent);
      border:1px solid var(--border-hi);border-radius:20px;
      padding:7px 18px;font-size:13px;font-weight:600;
      cursor:pointer;margin-top:4px;transition:all .2s;
      box-shadow:var(--shadow-sm);
    }
    .toggle-btn:hover{
      background:var(--accent);color:#fff;border-color:var(--accent);
      box-shadow:0 3px 12px rgba(91,155,213,.18);
    }

    /* ── Thinking card ── */
    .thinking-card{
      display:inline-flex;align-items:center;gap:10px;
      background:var(--panel);border:1px solid var(--border);
      border-radius:var(--radius);padding:12px 20px;
      box-shadow:var(--shadow-md);
    }
    .thinking-card .dots span{
      display:inline-block;width:8px;height:8px;border-radius:50%;
      background:var(--accent);animation:blink 1.4s infinite;
    }
    .thinking-card .dots span:nth-child(2){animation-delay:.2s}
    .thinking-card .dots span:nth-child(3){animation-delay:.4s}
    @keyframes blink{0%,80%,100%{opacity:.25;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
    .thinking-card .timer{font-size:13px;color:var(--muted);font-weight:500}

    /* ── Input Bar ── */
    .input-area{
      padding:14px 24px 16px;border-top:1px solid var(--border);
      background:rgba(253,251,247,.9);backdrop-filter:blur(16px);
    }
    .input-wrap{
      display:flex;gap:12px;border:2px solid var(--border);
      border-radius:var(--radius);padding:8px 10px 8px 18px;
      background:var(--panel);transition:border-color .2s,box-shadow .2s;
      box-shadow:var(--shadow-sm);
    }
    .input-wrap:focus-within{
      border-color:var(--accent);box-shadow:0 0 0 4px var(--accent-glow);
    }
    .input-wrap input{
      flex:1;border:none;background:transparent;color:var(--text);
      font-size:15px;outline:none;font-family:var(--font);
    }
    .input-wrap input::placeholder{color:var(--muted)}
    .input-wrap button{
      border:none;background:var(--user-bubble);color:#fff;
      padding:10px 22px;border-radius:var(--radius-sm);
      font-weight:700;font-size:14.5px;cursor:pointer;
      transition:all .2s;white-space:nowrap;
      box-shadow:0 3px 10px rgba(91,155,213,.18);
    }
    .input-wrap button:hover{background:#4a8ac4;box-shadow:0 4px 14px rgba(91,155,213,.25)}
    .input-wrap button:disabled{opacity:.45;cursor:not-allowed;box-shadow:none}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar{width:6px;height:6px}
    ::-webkit-scrollbar-track{background:transparent}
    ::-webkit-scrollbar-thumb{background:var(--border-hi);border-radius:4px}
    ::-webkit-scrollbar-thumb:hover{background:var(--muted)}
  </style>
</head>
<body>
<div class="app">

  <!-- ═══════════════ Sidebar ═══════════════ -->
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="logo">X</div>
      <h2>XiaoNuan</h2>
      <div class="sub">Your warm companion</div>
    </div>
    <div id="historyList" class="history-list">
      <div class="history-empty">No conversations yet.<br>Start chatting!</div>
    </div>
    <div class="sidebar-footer">Click a question to jump</div>
  </aside>

  <!-- ═══════════════ Main ═══════════════ -->
  <main class="main">
    <div class="chat-header">
      <div class="avatar">X</div>
      <div class="info">
        <h3>XiaoNuan</h3>
        <p>Warm companion · Always here</p>
      </div>
      <div class="status"><span class="dot"></span> Online</div>
    </div>

    <div id="chat" class="chat"></div>

    <div class="input-area">
      <div class="input-wrap">
        <input id="inp" placeholder="Type your message…" autocomplete="off" />
        <button id="send">Send</button>
      </div>
    </div>
  </main>
</div>

<script>
/* ═══════════════════════════════════════════
   State
   ═══════════════════════════════════════════ */
const chat      = document.getElementById('chat');
const inp       = document.getElementById('inp');
const sendBtn   = document.getElementById('send');
const historyEl = document.getElementById('historyList');
const sid       = localStorage.getItem('sid') || (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2));
localStorage.setItem('sid', sid);

const historyEntries = [];
let   msgCounter     = 0;

/* Configure marked.js */
if (typeof marked !== 'undefined') {
  marked.setOptions({ breaks: true, gfm: true });
}

/* ═══════════════════════════════════════════
   Auto-scroll helper — always scrolls to bottom
   ═══════════════════════════════════════════ */
function scrollToBottom() {
  requestAnimationFrame(() => {
    chat.scrollTop = chat.scrollHeight;
  });
}

/* ═══════════════════════════════════════════
   Sidebar
   ═══════════════════════════════════════════ */
function renderSidebar() {
  if (historyEntries.length === 0) {
    historyEl.innerHTML = '<div class="history-empty">No conversations yet.<br>Start chatting!</div>';
    return;
  }
  historyEl.innerHTML = '';
  historyEntries.forEach((entry, i) => {
    const div = document.createElement('div');
    div.className = 'history-item';
    div.dataset.id = entry.id;
    div.innerHTML = `<span class="idx">${i + 1}</span>${escapeHtml(entry.question)}`;
    div.onclick = () => scrollToMessage(entry.id);
    historyEl.appendChild(div);
  });
}

function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function scrollToMessage(id) {
  document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
  const item = document.querySelector(`.history-item[data-id="${id}"]`);
  if (item) {
    item.classList.add('active');
    item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  const userEl = document.querySelector(`.msg-row[data-id="${id}"]`);
  if (userEl) userEl.scrollIntoView({ block: 'start', behavior: 'smooth' });
}

/* ═══════════════════════════════════════════
   Add message bubble
   ═══════════════════════════════════════════ */
function addMessage(role, content, id, useMarkdown) {
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;
  if (id) row.dataset.id = id;

  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = role === 'user' ? 'You' : 'XiaoNuan';

  const bubble = document.createElement('div');
  bubble.className = 'msg ' + role;

  if (useMarkdown && typeof marked !== 'undefined') {
    bubble.innerHTML = marked.parse(content);
  } else {
    bubble.textContent = content;
  }

  row.appendChild(label);
  row.appendChild(bubble);
  chat.appendChild(row);
  scrollToBottom();

  return { row, bubble };
}

/* ═══════════════════════════════════════════
   Collapse logic
   ═══════════════════════════════════════════ */
function setupCollapsible(bubble) {
  requestAnimationFrame(() => {
    const fullHeight = bubble.scrollHeight;
    if (fullHeight <= 300) return;

    bubble.classList.add('collapsible');
    bubble.style.maxHeight = '280px';

    const btn = document.createElement('button');
    btn.className = 'toggle-btn';
    btn.innerHTML = 'Read more <span>▾</span>';
    btn.onclick = () => {
      const expanding = !bubble.classList.contains('expanded');
      bubble.classList.toggle('expanded');
      if (expanding) {
        bubble.style.maxHeight = fullHeight + 'px';
        btn.innerHTML = 'Show less <span>▴</span>';
      } else {
        bubble.style.maxHeight = '280px';
        btn.innerHTML = 'Read more <span>▾</span>';
        setTimeout(() => {
          const row = bubble.closest('.msg-row');
          if (row) row.scrollIntoView({ block: 'start', behavior: 'smooth' });
        }, 100);
      }
    };

    const row = bubble.closest('.msg-row');
    row.parentNode.insertBefore(btn, row.nextSibling);
  });
}

/* ═══════════════════════════════════════════
   Thinking timer
   ═══════════════════════════════════════════ */
function showThinking() {
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.id = 'thinkingRow';

  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = 'XiaoNuan';

  const card = document.createElement('div');
  card.className = 'thinking-card';
  card.innerHTML = `
    <span class="dots"><span></span><span></span><span></span></span>
    <span class="timer" id="thinkingTimer">Thinking… (0s)</span>
  `;

  row.appendChild(label);
  row.appendChild(card);
  chat.appendChild(row);
  scrollToBottom();

  /* start timer */
  let seconds = 0;
  const timerEl = card.querySelector('#thinkingTimer');
  const interval = setInterval(() => {
    seconds++;
    if (timerEl) timerEl.textContent = `Thinking… (${seconds}s)`;
  }, 1000);

  return {
    stop() { clearInterval(interval); },
    remove() { row.remove(); }
  };
}

/* ═══════════════════════════════════════════
   Send message
   ═══════════════════════════════════════════ */
async function run() {
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  sendBtn.disabled = true;

  const id = 'msg-' + (++msgCounter);

  /* user bubble (plain text) */
  addMessage('user', text, id, false);

  /* record in sidebar */
  historyEntries.push({ id, question: text });
  renderSidebar();
  setTimeout(() => {
    document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
    const item = document.querySelector(`.history-item[data-id="${id}"]`);
    if (item) item.classList.add('active');
  }, 0);

  /* show thinking card with timer */
  const thinking = showThinking();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ session_id: sid, message: text }),
    });
    const data = await res.json();
    const reply = data.reply || '(empty)';

    /* remove thinking card */
    thinking.stop();
    thinking.remove();

    /* render bot bubble with Markdown */
    const { bubble } = addMessage('bot', reply, id, true);
    setupCollapsible(bubble);

  } catch (e) {
    thinking.stop();
    thinking.remove();
    addMessage('bot', 'Request failed: ' + e, id, false);
  } finally {
    sendBtn.disabled = false;
    inp.focus();
  }
}

sendBtn.onclick = run;
inp.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.isComposing) run(); });
inp.focus();
</script>
</body>
</html>
"""


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

    def generate(self, messages: list[dict]) -> str:
        """调用模型生成一段纯文本回复，不包含任何工具逻辑。"""
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"MESSAGES:\n{json.dumps(messages, ensure_ascii=False)}\n\n"
                "Answer naturally and accurately."
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.15,
                top_p=0.85,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        txt = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        logger.info("model_called prompt_tokens=%s output_chars=%s", inputs["input_ids"].shape[1], len(txt))
        return txt


class RequestBody(BaseModel):
    session_id: str
    message: str


# ── OpenAI-compatible schemas ──────────────────────────────────────────
class OAI_Message(BaseModel):
    role: str
    content: str


class OAI_ChatRequest(BaseModel):
    model: str = "qwen35_4b_lora"
    messages: list[OAI_Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None
    stream: bool | None = False


OAI_SYSTEM_PROMPT = SYSTEM_PROMPT  # same prompt, exported for OpenAI callers


def build_app(model_path: str, adapter_path: str | None, stateless: bool = False) -> FastAPI:
    app = FastAPI()
    assistant = HFAssistant(model_path=model_path, adapter_path=adapter_path)
    sessions: dict[str, list[dict]] = {}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.post("/api/chat")
    def chat(req: RequestBody) -> JSONResponse:
        try:
            logger.info("chat request session_id=%s user=%s", req.session_id, req.message)
            history = [] if stateless else sessions.get(req.session_id, [])
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": req.message}]

            reply = assistant.generate(messages)

            # 更新会话历史
            if not stateless:
                history.append({"role": "user", "content": req.message})
                history.append({"role": "assistant", "content": reply})
                sessions[req.session_id] = history

            return JSONResponse({"reply": reply})
        except Exception as exc:
            logger.exception("chat request failed")
            return JSONResponse(
                {"reply": f"Backend error: {exc}", "error": str(exc)},
                status_code=500,
            )

    # ── OpenAI-compatible endpoints ───────────────────────────────────
    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{
                "id": "qwen35_4b_lora",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "xiaonuan",
            }]
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: OAI_ChatRequest):
        try:
            request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            messages = [{"role": m.role, "content": m.content} for m in req.messages]
            if not any(m["role"] == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": OAI_SYSTEM_PROMPT})

            logger.info("oai_request=%s model=%s msgs=%d", request_id, req.model, len(messages))

            reply = assistant.generate(messages)

            logger.info("oai_request=%s done chars=%d", request_id, len(reply))

            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        except Exception as exc:
            logger.exception("oai_request failed")
            return JSONResponse(
                {"error": {"message": str(exc), "type": "internal_error"}},
                status_code=500,
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Web chat demo for Q&A.")
    parser.add_argument("--model-path", required=True, help="Path to the base model directory.")
    parser.add_argument("--adapter-path", default=None, help="Optional path to a LoRA adapter checkpoint.")
    parser.add_argument("--stateless", action="store_true", help="Disable session memory; each request is single-turn.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = build_app(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        stateless=args.stateless,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
