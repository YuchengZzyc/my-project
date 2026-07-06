from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def load_tokenizer(model_path: str) -> Any:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    tools = row.get("tools", [])
    system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
    user = messages[1]["content"] if len(messages) > 1 and messages[1].get("role") == "user" else ""
    return (
        f"System:\n{system}\n\n"
        f"Tools:\n{json.dumps(tools, ensure_ascii=False)}\n\n"
        "User:\n"
        f"{user}\n\n"
        "Return assistant next turn only. If a tool is needed, output OpenAI-style "
        'assistant tool_calls JSON with content:null. Otherwise output natural text.'
    )


def extract_tool_name(text: str) -> str | None:
    m = re.search(r'"name"\s*:\s*"([a-z_]+)"', text)
    return m.group(1) if m else None


def should_call_tool(row: dict[str, Any]) -> bool:
    scenario = row.get("metadata", {}).get("scenario", "")
    return not str(scenario).startswith("no_tool")


def expected_tool_name(row: dict[str, Any]) -> str | None:
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            try:
                return msg["tool_calls"][0]["function"]["name"]
            except Exception:
                return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick tool-use capability eval script.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_path = cfg["model"]["name_or_path"]
    adapter_path = cfg["model"].get("adapter_path")
    data_file = cfg["data"]["test_file"]
    max_new_tokens = cfg["eval"]["max_new_tokens"]
    limit = cfg["eval"].get("limit", 100)
    temperature = cfg["eval"].get("temperature", 0.0)
    output_report = Path(cfg["output"]["report_file"])

    tokenizer = load_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,torch_dtype=torch.float16).to("cuda")
    if adapter_path:
        ap = Path(adapter_path)
        if not ap.exists():
            raise FileNotFoundError(f"adapter_path not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.to("cuda")
    model.eval()

    rows = load_jsonl(data_file)[:limit]

    total = len(rows)
    expected_tool = 0
    predicted_tool = 0
    correct_tool_name = 0
    correct_no_tool = 0
    details: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        prompt = build_prompt(row)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_name = extract_tool_name(gen)
        expect_call = should_call_tool(row)
        expect_name = expected_tool_name(row)

        if expect_call:
            expected_tool += 1
        if pred_name:
            predicted_tool += 1
        if expect_call and pred_name and pred_name == expect_name:
            correct_tool_name += 1
        if (not expect_call) and (pred_name is None):
            correct_no_tool += 1

        details.append(
            {
                "index": idx,
                "scenario": row.get("metadata", {}).get("scenario"),
                "expect_call": expect_call,
                "expect_tool": expect_name,
                "pred_tool": pred_name,
                "raw_output_preview": gen[:500],
            }
        )

    tool_call_recall = (correct_tool_name / expected_tool) if expected_tool else 0.0
    tool_call_precision = (correct_tool_name / predicted_tool) if predicted_tool else 0.0
    no_tool_acc = (correct_no_tool / max(total - expected_tool, 1)) if total else 0.0

    report = {
        "total": total,
        "expected_tool_samples": expected_tool,
        "predicted_tool_samples": predicted_tool,
        "tool_name_correct": correct_tool_name,
        "tool_call_recall": tool_call_recall,
        "tool_call_precision": tool_call_precision,
        "no_tool_accuracy": no_tool_acc,
        "model_path": model_path,
        "adapter_path": adapter_path,
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps({"summary": report, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
