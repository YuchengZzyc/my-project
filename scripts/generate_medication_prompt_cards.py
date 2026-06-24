from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import random
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

VALID_SUB_SCENES: set[str] = {
    "missed_or_uncertain_dose",
    "double_dose",
    "increase_dose",
    "stop_medication",
    "expired_medicine",
    "leftover_medicine",
    "mix_medications",
    "supplement_conflict",
    "unknown_medication",
    "medicine_not_working",
    "side_effect",
    "sharing_medicine",
    "wrong_medication",
    "wrong_dosage",
    "wrong_timing",
    "afraid_to_tell_family",
    "avoid_doctor",
    "wait_and_see",
    "self_adjust_medication",
    "forgets_medication_name",
}

VALID_RISK_LEVELS: set[str] = {"low", "medium", "high"}

REQUIRED_STRING_FIELDS: list[str] = [
    "card_id", "scene", "sub_scene", "risk_level", "elder_profile",
    "emotion", "emotional_need", "emotional_support_strategy",
    "expression_style", "safety_challenge", "life_detail",
    "prompt_card_description",
]
REQUIRED_LIST_FIELDS: list[str] = [
    "missing_info", "critical_safety_points", "forbidden_tone",
    "expected_assistant_behavior",
]
REQUIRED_BOOL_FIELDS: list[str] = ["requires_tool", "must_refuse_or_redirect"]


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(prompt_dir: str | Path = PROMPT_DIR) -> tuple[str, str]:
    system_path = Path(prompt_dir) / "medication_prompt_card_system.txt"
    user_path = Path(prompt_dir) / "medication_prompt_card_user.txt"
    system_prompt = system_path.read_text(encoding="utf-8").strip()
    user_prompt = user_path.read_text(encoding="utf-8").strip()
    return system_prompt, user_prompt


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
# Validation
# ---------------------------------------------------------------------------

MEDICATION_RULES: dict[str, list[str]] = {
    "increase_dose": [
        "do not increase dosage",
        "consult healthcare provider",
    ],
    "stop_medication": [
        "do not stop medication",
        "consult healthcare provider",
    ],
    "double_dose": [
        "avoid taking another dose",
    ],
}


def validate_prompt_card(obj: dict[str, Any]) -> str | None:
    """Validate structure, field types, and content. Returns error message or None."""
    if not isinstance(obj, dict):
        return "response is not a dict"

    # --- string fields ---
    for field in REQUIRED_STRING_FIELDS:
        if field not in obj:
            return f"missing field: {field}"
        if not isinstance(obj[field], str):
            return f"field {field} must be str, got {type(obj[field]).__name__}"

    # --- list fields ---
    for field in REQUIRED_LIST_FIELDS:
        if field not in obj:
            return f"missing field: {field}"
        if not isinstance(obj[field], list):
            return f"field {field} must be list, got {type(obj[field]).__name__}"

    # --- bool fields ---
    for field in REQUIRED_BOOL_FIELDS:
        if field not in obj:
            return f"missing field: {field}"
        if not isinstance(obj[field], bool):
            return f"field {field} must be bool, got {type(obj[field]).__name__}"

    # --- content checks ---
    if obj["scene"] != "medication_safety":
        return f"scene must be 'medication_safety', got {obj['scene']!r}"

    if obj["risk_level"] not in VALID_RISK_LEVELS:
        return f"risk_level must be one of {sorted(VALID_RISK_LEVELS)}, got {obj['risk_level']!r}"

    sub_scene = obj["sub_scene"].strip()
    if not sub_scene:
        return "sub_scene must not be empty"
    if sub_scene not in VALID_SUB_SCENES:
        return f"sub_scene {sub_scene!r} is not in valid set"

    if not obj["safety_challenge"].strip():
        return "safety_challenge must not be empty"

    desc = obj["prompt_card_description"].strip()
    if not desc:
        return "prompt_card_description must not be empty"
    if len(desc) < 30:
        return f"prompt_card_description too short ({len(desc)} chars, need >= 30)"

    if not obj["life_detail"].strip():
        return "life_detail must not be empty"

    if not obj["elder_profile"].strip():
        return "elder_profile must not be empty"

    if len(obj["expected_assistant_behavior"]) == 0:
        return "expected_assistant_behavior must not be empty"

    if len(obj["critical_safety_points"]) == 0:
        return "critical_safety_points must not be empty"

    return None


def validate_medication_rules(obj: dict[str, Any]) -> str | None:
    """Check medication-specific safety rules based on sub_scene."""
    sub_scene = obj.get("sub_scene", "")
    required_items = MEDICATION_RULES.get(sub_scene)
    if required_items is None:
        return None  # no special rules for this sub_scene

    safety_points = [str(p).strip().lower() for p in obj.get("critical_safety_points", [])]
    missing: list[str] = []
    for item in required_items:
        if not any(item in sp for sp in safety_points):
            missing.append(item)

    if missing:
        return (
            f"sub_scene={sub_scene!r} requires critical_safety_points to contain: "
            f"{missing}, but got {obj.get('critical_safety_points', [])}"
        )
    return None


# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------

def make_fingerprint(card: dict[str, Any]) -> str:
    """Fields that must be unique across cards."""
    parts = [
        str(card.get("sub_scene", "")),
        str(card.get("safety_challenge", "")),
        str(card.get("life_detail", "")),
        str(card.get("elder_profile", "")),
        str(card.get("emotion", "")),
    ]
    return "||".join(parts)


# ---------------------------------------------------------------------------
# Card generation
# ---------------------------------------------------------------------------

def card_id(idx: int) -> str:
    return f"med_card_{idx:04d}"


def clean_json_response(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return raw


def generate_prompt_card(
    client: OpenAICompatibleClient,
    system_prompt: str,
    user_prompt: str,
    idx: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Card index: {idx}\n\n"
                f"{user_prompt}\n\n"
                f"This is card #{idx}. Make it unique. "
                f"Use card_id \"{card_id(idx)}\". "
                "Ensure the life_detail is original and not a copy of previous cards."
            ),
        },
    ]

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
            cleaned = clean_json_response(raw)
            obj = json.loads(cleaned)
            err = validate_prompt_card(obj)
            if err is not None:
                last_err = RuntimeError(f"validate_prompt_card failed (attempt {attempt}): {err}")
                continue
            err = validate_medication_rules(obj)
            if err is not None:
                last_err = RuntimeError(f"validate_medication_rules failed (attempt {attempt}): {err}")
                continue
            obj["card_id"] = card_id(idx)
            return obj
        except json.JSONDecodeError as exc:
            last_err = exc
            time.sleep(min(2.0, 0.4 * attempt))
        except Exception as exc:
            last_err = exc
            time.sleep(min(2.0, 0.4 * attempt))
    raise RuntimeError(f"generate_prompt_card failed after retries: {last_err}")


# ---------------------------------------------------------------------------
# Job builder
# ---------------------------------------------------------------------------

def build_jobs(total_samples: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    jobs: list[dict[str, Any]] = []
    for idx in range(1, total_samples + 1):
        temperature = rng.uniform(0.75, 1.05)
        jobs.append({"idx": idx, "temperature": temperature})
    return jobs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(output_path: Path, report_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    risk_counter: Counter[str] = Counter()
    sub_scene_counter: Counter[str] = Counter()
    emotion_counter: Counter[str] = Counter()
    expression_style_counter: Counter[str] = Counter()
    tool_counter: Counter[str] = Counter()
    refuse_counter: Counter[str] = Counter()
    life_details: list[str] = []
    duplicate_life_details: list[str] = []

    for row in rows:
        risk_counter[str(row.get("risk_level", "unknown"))] += 1
        sub_scene_counter[str(row.get("sub_scene", "unknown"))] += 1
        emotion_counter[str(row.get("emotion", "unknown"))] += 1
        expression_style_counter[str(row.get("expression_style", "unknown"))] += 1
        tool_counter[str(row.get("requires_tool", False))] += 1
        refuse_counter[str(row.get("must_refuse_or_redirect", False))] += 1
        ld = row.get("life_detail", "")
        if ld:
            if ld in life_details:
                duplicate_life_details.append(ld)
            life_details.append(ld)

    report = {
        "total_cards": len(rows),
        "risk_level_distribution": dict(sorted(risk_counter.items())),
        "sub_scene_distribution": dict(sorted(sub_scene_counter.items())),
        "emotion_distribution": dict(sorted(emotion_counter.items())),
        "expression_style_distribution": dict(sorted(expression_style_counter.items())),
        "requires_tool_distribution": dict(sorted(tool_counter.items())),
        "must_refuse_distribution": dict(sorted(refuse_counter.items())),
        "duplicate_life_detail_count": len(duplicate_life_details),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Generation orchestration
# ---------------------------------------------------------------------------

def generate_cards(
    output_path: Path,
    client: OpenAICompatibleClient,
    system_prompt: str,
    user_prompt: str,
    report_path: Path | None = None,
    samples: int = 50,
    seed: int = 42,
    workers: int = 8,
    temperature: float = 0.9,
    max_tokens: int = 512,
    max_retries: int = 3,
    dedupe_retries: int = 2,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(samples, seed)
    seen_lock = threading.Lock()
    seen_fingerprints: set[str] = set()

    def _run_job(job: dict[str, Any]) -> dict[str, Any]:
        idx = job["idx"]
        t = max(0.2, min(1.2, temperature + job["temperature"] - 0.9))
        attempts = max(1, dedupe_retries + 1)
        for attempt in range(attempts):
            card = generate_prompt_card(
                client=client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                idx=idx,
                temperature=t + attempt * 0.05,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
            fp = make_fingerprint(card)
            with seen_lock:
                if fp not in seen_fingerprints or attempt == attempts - 1:
                    seen_fingerprints.add(fp)
                    return card
        return card

    completed = 0
    errors = 0
    total_jobs = len(jobs)
    print(f"starting generation: total={total_jobs}, workers={max(1, int(workers))}", flush=True)

    with output_path.open("w", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [pool.submit(_run_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                try:
                    card = future.result()
                    completed += 1
                    f.write(json.dumps(card, ensure_ascii=False) + "\n")
                    f.flush()
                    print(
                        f"[progress] completed={completed}/{total_jobs} "
                        f"card_id={card.get('card_id', '?')} "
                        f"sub_scene={card.get('sub_scene', '?')}",
                        flush=True,
                    )
                except Exception as exc:
                    completed += 1
                    errors += 1
                    print(f"[error] job failed: {exc}", flush=True)

    print(f"generation done: total={total_jobs} completed={completed} errors={errors}", flush=True)

    if report_path is not None:
        write_report(output_path, report_path)
    return total_jobs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate medication safety prompt cards via Teacher LLM."
    )
    parser.add_argument("--api-env", type=str, default="configs/data/api_generation.env")
    parser.add_argument("--output", type=Path, default=Path("data/medication_prompt_cards.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/medication_prompt_cards.stats.json"))
    parser.add_argument("--samples", type=int, default=50, help="Number of prompt cards to generate.")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--endpoint", choices=["chat.completions", "responses"], default="chat.completions")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dedupe-retries", type=int, default=2)
    parser.add_argument(
        "--prompt-dir",
        type=str,
        default=str(PROMPT_DIR),
        help="Directory containing prompt txt files.",
    )
    args = parser.parse_args()

    # Load prompts from external files
    system_prompt, user_prompt = load_prompts(args.prompt_dir)

    # Load API config
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

    count = generate_cards(
        output_path=args.output,
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        report_path=args.report,
        samples=args.samples,
        seed=args.seed,
        workers=args.workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        dedupe_retries=args.dedupe_retries,
    )
    print(json.dumps({"output": str(args.output), "samples": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()