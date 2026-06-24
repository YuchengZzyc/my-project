from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

FORBIDDEN_PHRASES: list[str] = [
    "take another dose",
    "take an extra pill",
    "increase the dose",
    "stop taking it",
    "switch medicine",
    "take half a tablet",
    "decide yourself",
    "leftover medicine is fine",
    "start antibiotics yourself",
]

SAFETY_KEYWORDS: list[str] = [
    "pharmacist",
    "doctor",
    "healthcare provider",
    "family member",
    "pill organizer",
    "pill box",
    "medication record",
    "prescription label",
    "pill bottle",
]


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(prompt_dir: str | Path = PROMPT_DIR) -> tuple[str, str]:
    system_path = Path(prompt_dir) / "teacher_system.txt"
    user_path = Path(prompt_dir) / "teacher_user.txt"
    return (
        system_path.read_text(encoding="utf-8").strip(),
        user_path.read_text(encoding="utf-8").strip(),
    )


# ---------------------------------------------------------------------------
# Env / Config helpers
# ---------------------------------------------------------------------------

def load_env_file(path: str | Path) -> dict[str, str]:
    target = Path(path)
    if not target.exists():
        return {}
    values: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8-sig").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_required(name: str, values: list[str | None]) -> str:
    for value in values:
        if value and str(value).strip():
            return str(value).strip()
    raise ValueError(f"missing required setting: {name}")


# ---------------------------------------------------------------------------
# API client (OpenAI-compatible)
# ---------------------------------------------------------------------------

class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        endpoint: str = "chat.completions",
        timeout: int = 120,
        auth_header: str = "Authorization",
        auth_scheme: str = "bearer",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.auth_header = auth_header
        self.auth_scheme = auth_scheme

    def _auth_value(self) -> str:
        if self.auth_scheme == "raw":
            return self.api_key
        return f"Bearer {self.api_key}"

    def chat(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int) -> str:
        if self.endpoint == "responses":
            url = f"{self.base_url}/responses"
            payload = {
                "model": self.model,
                "input": messages,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        else:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                self.auth_header: self._auth_value(),
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
                content_encoding = (resp.headers.get("Content-Encoding") or "").lower()
                if "gzip" in content_encoding or body[:2] == b"\x1f\x8b":
                    body = gzip.decompress(body)
                raw = body.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            err_body = e.read()
            if err_body[:2] == b"\x1f\x8b":
                err_body = gzip.decompress(err_body)
            err_text = err_body.decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTPError: status={e.code}, body={err_text[:1000]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"URLError: {e}") from e

        obj = json.loads(raw)
        if isinstance(obj.get("choices"), list) and obj["choices"]:
            content = obj["choices"][0].get("message", {}).get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        if isinstance(obj.get("output_text"), str) and obj["output_text"].strip():
            return obj["output_text"].strip()
        if isinstance(obj.get("output"), list):
            parts: list[str] = []
            for block in obj["output"]:
                if not isinstance(block, dict):
                    continue
                for content in block.get("content", []) or []:
                    if isinstance(content, dict) and isinstance(content.get("text"), str):
                        parts.append(content["text"])
            joined = "\n".join(parts).strip()
            if joined:
                return joined
        raise RuntimeError(f"Model response did not contain readable text. keys={list(obj.keys())[:20]}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def index_cards(cards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c["card_id"]: c for c in cards if "card_id" in c}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_negated(lower: str, phrase: str, start_pos: int) -> bool:
    """Check if forbidden phrase at start_pos is negated (advised against, not advised)."""
    before = lower[max(0, start_pos - 40):start_pos].strip().rstrip(".,!?;")
    for neg in ("don't", "do not", "avoid", "never", "don’t", "please don't", "please do not"):
        if before.endswith(neg) or before.endswith(neg + " "):
            return True
    return False


def validate_assistant_response(text: str) -> str | None:
    """Check assistant response for safety rules.

    - Must not be empty.
    - Must not *advise* dangerous actions (negated advice is OK).
    - Must contain at least one safety keyword.
    """
    lower = text.lower().strip()
    if not lower:
        return "empty response"

    # --- Forbidden phrase check (negation-aware) ---
    for phrase in FORBIDDEN_PHRASES:
        if phrase not in lower:
            continue
        idx = 0
        while True:
            idx = lower.find(phrase, idx)
            if idx == -1:
                break
            if not _is_negated(lower, phrase, idx):
                return f"forbidden: suggests '{phrase}'"
            idx += len(phrase)

    # --- Safety keyword check ---
    if not any(kw in lower for kw in SAFETY_KEYWORDS):
        return "missing safety keyword"

    return None


SYSTEM_MESSAGE_CONTENT = (
    "You are an eldercare companion AI. Respond warmly, clearly, and naturally. "
    "Acknowledge emotions before offering practical advice. "
    "Never diagnose medical conditions or recommend changing medication without professional guidance."
)


def build_metadata(card: dict[str, Any]) -> dict[str, str]:
    return {
        "scene": str(card.get("scene", "")),
        "sub_scene": str(card.get("sub_scene", "")),
        "risk_level": str(card.get("risk_level", "")),
        "emotion": str(card.get("emotion", "")),
        "emotional_need": str(card.get("emotional_need", "")),
        "safety_challenge": str(card.get("safety_challenge", "")),
    }


def build_sft_sample(
    card: dict[str, Any],
    user_prompt: str,
    assistant_response: str,
    sample_index: int,
) -> dict[str, Any]:
    return {
        "task": "eldercare_companion_sft",
        "schema_version": "eldercare_v1",
        "card_id": card.get("card_id", ""),
        "sample_id": f"sft_{sample_index:06d}",
        "metadata": build_metadata(card),
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE_CONTENT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ],
    }


# ---------------------------------------------------------------------------
# Teacher LLM call
# ---------------------------------------------------------------------------

def generate_assistant_response(
    client: OpenAICompatibleClient,
    teacher_system: str,
    teacher_user_template: str,
    card: dict[str, Any],
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    validation_counter: Counter[str] | None = None,
) -> str:
    card_json = json.dumps(card, ensure_ascii=False, indent=2)
    teacher_user = teacher_user_template.format(card_json=card_json, user_prompt=user_prompt)

    messages = [
        {"role": "system", "content": teacher_system},
        {"role": "user", "content": teacher_user},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            raw = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
            response = raw.strip()

            err = validate_assistant_response(response)
            if err is None:
                return response

            print(f"  Retry {attempt} reason: {err}", flush=True)
            if validation_counter is not None:
                validation_counter[err] += 1

            time.sleep(min(2.0, 0.4 * attempt))
        except Exception as exc:
            print(f"  Retry {attempt} exception: {exc}", flush=True)
            time.sleep(min(2.0, 0.4 * attempt))

    raise RuntimeError(
        f"generate_assistant_response failed after retries for card={card.get('card_id', '?')}"
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(sft_samples: list[dict[str, Any]], report_path: Path) -> dict[str, Any]:
    risk_counter: Counter[str] = Counter()
    sub_scene_counter: Counter[str] = Counter()
    card_counter: Counter[str] = Counter()

    for sample in sft_samples:
        meta = sample.get("metadata", {})
        risk_counter[str(meta.get("risk_level", "unknown"))] += 1
        sub_scene_counter[str(meta.get("sub_scene", "unknown"))] += 1
        card_counter[str(sample.get("card_id", "unknown"))] += 1

    report = {
        "total_sft_samples": len(sft_samples),
        "cards_used": len(card_counter),
        "risk_level_distribution": dict(sorted(risk_counter.items())),
        "sub_scene_distribution": dict(sorted(sub_scene_counter.items())),
        "samples_per_card": dict(sorted(card_counter.items())),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

GLOBAL_SAMPLE_COUNTER: int = 0


def generate_sft(
    cards_path: Path,
    users_path: Path,
    output_path: Path,
    client: OpenAICompatibleClient,
    teacher_system: str,
    teacher_user_template: str,
    report_path: Path | None = None,
    workers: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 256,
    max_retries: int = 3,
) -> dict[str, Any]:
    global GLOBAL_SAMPLE_COUNTER

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and index
    all_cards = load_jsonl(cards_path)
    all_users = load_jsonl(users_path)
    cards_by_id = index_cards(all_cards)
    print(f"loaded {len(all_cards)} cards, {len(all_users)} user prompts", flush=True)

    # Build jobs: user prompts that have a matching card
    jobs: list[dict[str, Any]] = []
    for user_entry in all_users:
        cid = user_entry.get("card_id", "")
        card = cards_by_id.get(cid)
        if card is None:
            print(f"[warn] no card found for card_id={cid}, skipping", flush=True)
            continue
        jobs.append({
            "card": card,
            "user_prompt": user_entry.get("user_prompt", ""),
            "style_note": user_entry.get("style_note", ""),
            "card_id": cid,
        })

    print(
        f"starting SFT generation: {len(jobs)} jobs, workers={max(1, int(workers))}",
        flush=True,
    )

    completed = 0
    errors = 0
    all_samples: list[dict[str, Any]] = []
    validation_counter: Counter[str] = Counter()

    _counter_lock = threading.Lock()

    def _run(job: dict[str, Any]) -> dict[str, Any]:
        global GLOBAL_SAMPLE_COUNTER
        card = job["card"]
        up = job["user_prompt"]

        response = generate_assistant_response(
            client=client,
            teacher_system=teacher_system,
            teacher_user_template=teacher_user_template,
            card=card,
            user_prompt=up,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            validation_counter=validation_counter,
        )

        with _counter_lock:
            GLOBAL_SAMPLE_COUNTER += 1
            idx = GLOBAL_SAMPLE_COUNTER

        return build_sft_sample(card, up, response, idx)

    with output_path.open("w", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [pool.submit(_run, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                try:
                    sample = future.result()
                    completed += 1
                    all_samples.append(sample)
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    f.flush()
                    cid = sample.get("card_id", "?")
                    sid = sample.get("sample_id", "?")
                    sub = sample.get("metadata", {}).get("sub_scene", "?")
                    print(
                        f"[progress] {completed}/{len(jobs)} sample_id={sid} card={cid} sub_scene={sub}",
                        flush=True,
                    )
                except Exception as exc:
                    completed += 1
                    errors += 1
                    print(f"[error] job failed: {exc}", flush=True)

    # Print validation failure summary
    if validation_counter:
        print("Validation Summary:", flush=True)
        for reason, count in validation_counter.most_common():
            print(f"  {reason} {' ' * max(1, 40 - len(reason))} {count}", flush=True)

    print(
        f"generation done: total={len(jobs)} completed={completed} errors={errors}",
        flush=True,
    )

    if report_path is not None:
        write_report(all_samples, report_path)

    return {"total_jobs": len(jobs), "completed": completed, "errors": errors}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate medication SFT dataset via Teacher LLM."
    )
    parser.add_argument("--cards", type=Path, required=True, help="Prompt cards JSONL.")
    parser.add_argument("--users", type=Path, required=True, help="User prompts JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Output SFT JSONL.")
    parser.add_argument("--report", type=Path, default=None, help="Output stats JSON.")
    parser.add_argument("--api-env", type=str, default="configs/data/api_generation.env")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--endpoint", choices=["chat.completions", "responses"], default="chat.completions")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--prompt-dir",
        type=str,
        default=str(PROMPT_DIR),
        help="Directory containing prompt txt files.",
    )
    args = parser.parse_args()

    teacher_system, teacher_user_template = load_prompts(args.prompt_dir)

    env_file = load_env_file(args.api_env)
    base_url = resolve_required("base_url", [
        args.base_url,
        env_file.get("DISTILL_API_BASE_URL"),
        env_file.get("OPENAI_BASE_URL"),
        os.getenv("DISTILL_API_BASE_URL"),
        os.getenv("OPENAI_BASE_URL"),
    ])
    api_key = resolve_required("api_key", [
        args.api_key,
        env_file.get("DISTILL_API_KEY"),
        env_file.get("OPENAI_API_KEY"),
        os.getenv("DISTILL_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
    ])
    model = resolve_required("model", [
        args.model,
        env_file.get("DISTILL_API_MODEL"),
        env_file.get("OPENAI_MODEL"),
        os.getenv("DISTILL_API_MODEL"),
        os.getenv("OPENAI_MODEL"),
    ])

    client = OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        endpoint=args.endpoint,
        timeout=args.timeout_sec,
    )

    result = generate_sft(
        cards_path=args.cards,
        users_path=args.users,
        output_path=args.output,
        client=client,
        teacher_system=teacher_system,
        teacher_user_template=teacher_user_template,
        report_path=args.report,
        workers=args.workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()