from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.tool_registry import get_tools


SYSTEM_PROMPT = (
    "You are a reliable assistant for reminder tool use. "
    "Use reminder tools when needed. Never fabricate tool results."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


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
        msg = dict(obj)
        msg["role"] = "assistant"
        msg["content"] = None
        return msg

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


class HFAssistant:
    def __init__(self, model_path: str, adapter_path: str | None = None, max_new_tokens: int = 256) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        # device_map="auto" 会自动将模型加载到可用的设备上 反而把模型切碎了
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).cuda()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.max_new_tokens = max_new_tokens

    def run_once(self, user_prompt: str) -> tuple[dict[str, Any] | None, str]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
        tools = get_tools()
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                # print(prompt)
                # return None, ""
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return parse_tool_call_output(text), text


def _norm(v: Any) -> str:
    return " ".join(str(v).strip().lower().split())


def compare_args(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[int, int, list[str]]:
    checked = 0
    ok = 0
    notes: list[str] = []
    for k, ev in expected.items():
        if ev is None:
            continue
        checked += 1
        if k not in actual:
            notes.append(f"missing:{k}")
            continue
        av = actual.get(k)
        if isinstance(ev, str) and isinstance(av, str):
            if _norm(ev) == _norm(av):
                ok += 1
            else:
                notes.append(f"mismatch:{k}")
        else:
            if ev == av:
                ok += 1
            else:
                notes.append(f"mismatch:{k}")
    return ok, checked, notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tool call + arguments correctness without backend execution.")
    parser.add_argument("--cases-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--smoke-test",action="store_true",help="Only run one inference and exit.")
    args = parser.parse_args()

    rows = load_jsonl(args.cases_jsonl)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    model = HFAssistant(model_path=args.model_path, adapter_path=args.adapter_path)

    if args.smoke_test:
        print("[Smoke Test] Model loaded successfully.")
        tool_msg, raw = model.run_once(rows[0]["prompt"])
        print("=" * 60)
        print("Prompt:")
        print(rows[0]["prompt"])
        print("=" * 60)
        print("Model Output:")
        print(raw)
        print("=" * 60)
        print("Parsed Tool Call:")
        print(tool_msg)
        return

    total = 0
    tool_call_expected = 0
    tool_call_correct = 0
    tool_name_correct = 0
    args_json_ok = 0
    arg_match_total = 0
    arg_match_ok = 0
    details: list[dict[str, Any]] = []

    for row in rows:
        total += 1
        ev = row.get("eval", {})
        should_call = bool(ev.get("should_call_tool", True))
        expected_tool = ev.get("expected_tool_name")
        expected_args = ev.get("expected_args") or {}

        tool_msg, raw = model.run_once(str(row.get("prompt", "")))
        got_call = tool_msg is not None and bool(tool_msg.get("tool_calls"))
        if should_call:
            tool_call_expected += 1

        tool_ok = False
        name_ok = False
        args_ok = False
        arg_ok = 0
        arg_checked = 0
        arg_notes: list[str] = []

        if should_call and got_call:
            tool_ok = True
            tool_call_correct += 1
            tc = tool_msg["tool_calls"][0]
            name = tc.get("function", {}).get("name")
            if name == expected_tool:
                name_ok = True
                tool_name_correct += 1
            args_raw = tc.get("function", {}).get("arguments", "{}")
            try:
                actual_args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                if isinstance(actual_args, dict):
                    args_ok = True
                    args_json_ok += 1
                    arg_ok, arg_checked, arg_notes = compare_args(expected_args, actual_args)
                    arg_match_ok += arg_ok
                    arg_match_total += arg_checked
                else:
                    arg_notes.append("arguments_not_object")
            except Exception:
                arg_notes.append("arguments_json_invalid")

        if not should_call and not got_call:
            tool_ok = True
            tool_call_correct += 1

        details.append(
            {
                "id": row.get("id"),
                "scenario": row.get("scenario"),
                "should_call_tool": should_call,
                "expected_tool": expected_tool,
                "tool_called": got_call,
                "tool_call_ok": tool_ok,
                "tool_name_ok": name_ok,
                "args_json_ok": args_ok,
                "arg_match_ok": arg_ok,
                "arg_match_checked": arg_checked,
                "arg_notes": arg_notes,
                "raw_output_preview": raw[:500],
            }
        )

    summary = {
        "total": total,
        "tool_call_accuracy": (tool_call_correct / total) if total else 0.0,
        "tool_name_accuracy_overall": (tool_name_correct / total) if total else 0.0,
        "args_json_rate_overall": (args_json_ok / total) if total else 0.0,
        "arg_match_rate": (arg_match_ok / arg_match_total) if arg_match_total else 0.0,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
