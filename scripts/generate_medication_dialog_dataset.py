from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
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

# ---------------------------------------------------------------------------
# Sub-scene definitions
# ---------------------------------------------------------------------------

MEDICATION_SUB_SCENES: list[str] = [
    "increase_dose",
    "stop_medication",
    "take_extra_dose",
    "feel_unwell_after_medication",
    "use_leftover_medicine",
    "mix_medications",
    "unknown_medicine_name",
    "medicine_with_supplements",
    "change_medication_due_to_no_effect",
]

# ---------------------------------------------------------------------------
# Forbidden patterns
# ---------------------------------------------------------------------------

USER_FORBIDDEN_PATTERNS: list[str] = [
    "according to your medication history",
    "according to today's record",
    "based on your record",
    # "did i take",
    # "did i forget",
    # "have i taken",
    # "already take my medicine",
    # "already take my pill",
    # "already took my medicine",
    # "already took my pill",
    # "today's dose",
    # "this morning's dose",
    # "did i miss",
    # "did i already",
    # "have i already",
]

ASSISTANT_FORBIDDEN_PATTERNS: list[str] = [
    "increase the dose",
    "increase your dose",
    "increase your dosage",
    "take an extra pill",
    "take another dose",
    "take another pill",
    "stop taking it",
    "stop your medication",
    "stop taking your medicine",
    "switch medicine",
    "switch medication",
    "try my neighbor's",
    "decide yourself",
    "decide on your own",
    "according to your medication history",
    "according to today's record",
    "based on your record",
]

NEGATION_WORDS: list[str] = [
    "don't", "do not", "never", "avoid", "please don't", "please do not",
    "shouldn't", "should not", "must not", "mustn't", "no", "not"
]


def _is_negated(lower: str, pattern: str, start: int) -> bool:
    before = lower[max(0, start - 30):start].strip().rstrip(".,!?;: ")
    for neg in NEGATION_WORDS:
        clean_before = before.rstrip(",")
        if clean_before.endswith(neg):
            return True
    return False


def contains_forbidden(text: str, patterns: list[str], check_negation: bool = True) -> str | None:
    lower = text.lower().strip()

    for pat in patterns:
        if pat not in lower:
            continue

        idx = 0
        while True:
            idx = lower.find(pat, idx)
            if idx == -1:
                break

            # 检查是否是否定句（如：don't, never, avoid）
            if check_negation and _is_negated(lower, pat, idx):
                # 如果是"否定 + 禁词"（例如：don't take leftover），则允许通过，不视为违规
                pass  # 不做任何处理，直接跳过后续的 return
            else:
                # 如果是肯定句（例如：take leftover），则视为违规，直接返回该禁词
                return pat

            # 【关键修正】将指针移动放在 if-else 外面，确保每次循环后指针都会前进
            idx += len(pat)

    return None


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(prompt_dir: str | Path = PROMPT_DIR) -> tuple[str, str]:
    system_path = Path(prompt_dir) / "medication_dialog_system.txt"
    user_path = Path(prompt_dir) / "medication_dialog_user.txt"
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
# API client
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
        if not isinstance(obj, dict):
            raise RuntimeError(
                f"API response is not a JSON object: type={type(obj).__name__}, raw={raw[:200]}"
            )
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

def count_words(text: str) -> int:
    return len(text.strip().split())


def validate_dialog(dialog: dict[str, Any]) -> str | None:
    """Validate a single dialogue. Returns error message or None."""
    if not isinstance(dialog, dict):
        return "response is not a dict"

    messages = dialog.get("messages", [])
    if len(messages) < 2:
        return f"expected 2 messages, got {len(messages)}"

    user_msg = messages[0]
    assistant_msg = messages[1]

    if user_msg.get("role") != "user":
        return "first message role must be 'user'"
    if assistant_msg.get("role") != "assistant":
        return "second message role must be 'assistant'"

    user_text = user_msg.get("content", "")
    assistant_text = assistant_msg.get("content", "")

    # User validation
    user_words = count_words(user_text)
    if user_words < 5:
        return f"user too short: {user_words} words"
    if user_words > 45:
        return f"user too long: {user_words} words"

    forbidden_user = contains_forbidden(user_text, USER_FORBIDDEN_PATTERNS, check_negation=False)
    if forbidden_user:
        return f"user forbidden pattern: '{forbidden_user}'"

    # Assistant validation
    assistant_words = count_words(assistant_text)
    if assistant_words < 12:
        return f"assistant too short: {assistant_words} words"
    if assistant_words > 60:
        return f"assistant too long: {assistant_words} words"

    forbidden_assistant = contains_forbidden(assistant_text, ASSISTANT_FORBIDDEN_PATTERNS, check_negation=True)
    if forbidden_assistant:
        return f"assistant forbidden pattern: '{forbidden_assistant}'"

    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def find_json_array(text: str) -> list[dict[str, Any]] | None:
    """Find a JSON array anywhere in text using regex."""
    import re
    for m in re.finditer(r"\[[\s\S]*\]", text):
        candidate = m.group()
        try:
            arr = json.loads(candidate)
            if isinstance(arr, list) and len(arr) > 0 and all(isinstance(item, dict) for item in arr):
                return arr
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def find_json_objects(text: str) -> list[str]:
    """Extract all JSON object strings from text using line-by-line parsing."""
    results: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not (line.startswith("{") and line.endswith("}")):
            continue
        if '"messages"' not in line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                results.append(line)
        except json.JSONDecodeError:
            pass
    return results


def clean_json_response(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return raw


def parse_json_objects(text: str) -> list[dict[str, Any]]:
    """Parse one or more JSON objects from model output."""
    cleaned = clean_json_response(text)

    # Strategy 1: find a JSON array anywhere in the text
    arr = find_json_array(cleaned)
    if arr is not None:
        return arr

    # Strategy 2: try whole text as JSON array
    if cleaned.startswith("["):
        try:
            arr = json.loads(cleaned)
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass

    # Strategy 3: line-by-line extraction
    return find_json_objects(cleaned)


# ---------------------------------------------------------------------------
# Dialog ID counter
# ---------------------------------------------------------------------------

_dialog_counter_lock = threading.Lock()
_dialog_counter = 0


def next_dialog_id() -> str:
    global _dialog_counter
    with _dialog_counter_lock:
        _dialog_counter += 1
        return f"med_{_dialog_counter:05d}"


# ---------------------------------------------------------------------------
# Generation per sub-scene
# ---------------------------------------------------------------------------

def generate_dialogs_for_scene(
    client: OpenAICompatibleClient,
    system_prompt: str,
    user_template: str,
    sub_scene: str,
    count: int,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    validation_counter: Counter[str],
) -> list[dict[str, Any]]:
    all_valid: list[dict[str, Any]] = []

    for attempt in range(1, max_retries + 1):
        remaining = count - len(all_valid)
        if remaining <= 0:
            break

        request_count = min(batch_size, remaining)

        # Build user prompt: ask for exactly request_count dialogues
        user_prompt = user_template.replace("{sub_scene}", sub_scene).replace("{count}", str(request_count))

        # Build messages with dynamic retry hint
        sys_content = system_prompt
        if attempt > 1:
            sys_content += (
                "\n\nPrevious response was not valid JSON. "
                "Return JSON only. Do not output any explanation. "
                "Do not include Markdown. "
                "The first character must be '[' and the last must be ']'."
            )

        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
            parsed = parse_json_objects(raw)

            if not parsed:
                reason = "no valid JSON found"
                print(f"  Retry {attempt} [{sub_scene}] reason: {reason}", flush=True)
                validation_counter[f"parse: {reason}"] += 1
                time.sleep(min(2.0, 0.4 * attempt))
                continue

            # Per-dialogue validation — don't discard the whole batch
            new_valid = 0
            rejected = 0
            for obj in parsed:
                if len(all_valid) >= count:
                    break
                err = validate_dialog(obj)
                if err is None:
                    obj["dialogue_id"] = next_dialog_id()
                    obj["scene"] = "medication_safety"
                    obj["sub_scene"] = sub_scene
                    all_valid.append(obj)
                    new_valid += 1
                else:
                    rejected += 1
                    print(f"  Retry {attempt} [{sub_scene}] validation: {err}", flush=True)
                    validation_counter[f"validation: {err}"] += 1

            if rejected > 0 and new_valid == 0:
                # All items in this batch were rejected
                reason = f"{rejected} items rejected, 0 valid"
                print(f"  Retry {attempt} [{sub_scene}] reason: {reason}", flush=True)
                validation_counter[reason] += 1
                time.sleep(min(2.0, 0.4 * attempt))
                continue

            if new_valid > 0:
                print(
                    f"  Retry {attempt} [{sub_scene}] batch={request_count} "
                    f"valid={new_valid} rejected={rejected} "
                    f"(total so far: {len(all_valid)}/{count})",
                    flush=True,
                )
                # Continue without sleeping — next batch is fresh
                continue

            # No items at all in the batch
            reason = "no valid entries this attempt"
            print(f"  Retry {attempt} [{sub_scene}] reason: {reason}", flush=True)
            validation_counter[reason] += 1
            time.sleep(min(2.0, 0.4 * attempt))

        except Exception as exc:
            print(f"  Retry {attempt} [{sub_scene}] exception: {exc}", flush=True)
            validation_counter[f"exception: {type(exc).__name__}"] += 1
            time.sleep(min(2.0, 0.4 * attempt))

    # Graceful degradation — return whatever we got
    if len(all_valid) < count:
        print(
            f"  [partial] [{sub_scene}] target={count} got={len(all_valid)} "
            f"(missing {count - len(all_valid)})",
            flush=True,
        )

    return all_valid


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    dialogs: list[dict[str, Any]],
    report_path: Path,
    samples_per_scene: int,
) -> dict[str, Any]:
    scene_counter: Counter[str] = Counter()
    user_word_counts: list[int] = []
    assistant_word_counts: list[int] = []

    for d in dialogs:
        sub = d.get("sub_scene", "unknown")
        scene_counter[sub] += 1
        msgs = d.get("messages", [])
        if len(msgs) >= 2:
            user_word_counts.append(count_words(msgs[0].get("content", "")))
            assistant_word_counts.append(count_words(msgs[1].get("content", "")))

    sub_scene_distribution: dict[str, int] = {}
    missing = 0
    for scene in MEDICATION_SUB_SCENES:
        got = scene_counter.get(scene, 0)
        sub_scene_distribution[scene] = got
        if got < samples_per_scene:
            missing += samples_per_scene - got

    report = {
        "scene": "medication_safety",
        "samples_per_scene": samples_per_scene,
        "sub_scene_distribution": sub_scene_distribution,
        "missing": missing,
        "total_generated": len(dialogs),
        "user_word_length": {
            "min": min(user_word_counts) if user_word_counts else 0,
            "max": max(user_word_counts) if user_word_counts else 0,
            "avg": round(sum(user_word_counts) / max(len(user_word_counts), 1), 1),
        },
        "assistant_word_length": {
            "min": min(assistant_word_counts) if assistant_word_counts else 0,
            "max": max(assistant_word_counts) if assistant_word_counts else 0,
            "avg": round(sum(assistant_word_counts) / max(len(assistant_word_counts), 1), 1),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_dialogs(
    output_path: Path,
    client: OpenAICompatibleClient,
    system_prompt: str,
    user_template: str,
    report_path: Path | None = None,
    samples_per_scene: int = 10,
    batch_size: int = 5,
    workers: int = 3,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    max_retries: int = 50,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_counter: Counter[str] = Counter()

    all_dialogs: list[dict[str, Any]] = []
    completed = 0
    errors = 0

    def _run(sub_scene: str) -> list[dict[str, Any]]:
        return generate_dialogs_for_scene(
            client=client,
            system_prompt=system_prompt,
            user_template=user_template,
            sub_scene=sub_scene,
            count=samples_per_scene,
            batch_size=batch_size,
            temperature=temperature + (hash(sub_scene) % 5) * 0.05,
            max_tokens=max_tokens,
            max_retries=max_retries,
            validation_counter=validation_counter,
        )

    print(
        f"starting dialog generation: {len(MEDICATION_SUB_SCENES)} sub-scenes, "
        f"{samples_per_scene} each, total={len(MEDICATION_SUB_SCENES) * samples_per_scene}",
        flush=True,
    )

    with output_path.open("w", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {pool.submit(_run, s): s for s in MEDICATION_SUB_SCENES}
            for future in concurrent.futures.as_completed(futures):
                scene_name = futures[future]
                try:
                    dialogs = future.result()
                    completed += 1
                    all_dialogs.extend(dialogs)
                    for d in dialogs:
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")
                    f.flush()
                    print(
                        f"[progress] completed={completed}/{len(MEDICATION_SUB_SCENES)} "
                        f"sub_scene={scene_name} dialogs={len(dialogs)}",
                        flush=True,
                    )
                except Exception as exc:
                    completed += 1
                    errors += 1
                    print(f"[error] sub_scene={scene_name} failed: {exc}", flush=True)

    if validation_counter:
        print("Validation Summary:", flush=True)
        for reason, count in validation_counter.most_common():
            print(f"  {reason} {' ' * max(1, 50 - len(reason))} {count}", flush=True)

    print(
        f"generation done: sub_scenes={completed} errors={errors} total_dialogs={len(all_dialogs)}",
        flush=True,
    )

    if report_path is not None:
        write_report(all_dialogs, report_path, samples_per_scene)

    return {
        "sub_scenes": completed,
        "errors": errors,
        "total_dialogs": len(all_dialogs),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate medication safety dialog dataset (user + assistant one-turn)."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--report", type=Path, default=None, help="Output stats JSON path.")
    parser.add_argument("--api-env", type=str, default="configs/data/api_generation.env")
    parser.add_argument("--samples-per-scene", type=int, default=10, help="Dialogues per sub-scene.")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of dialogues to request per API call.")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--endpoint", choices=["chat.completions", "responses"], default="chat.completions")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=50)
    parser.add_argument(
        "--prompt-dir",
        type=str,
        default=str(PROMPT_DIR),
        help="Directory containing prompt txt files.",
    )
    args = parser.parse_args()

    system_prompt, user_template = load_prompts(args.prompt_dir)

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

    result = generate_dialogs(
        output_path=args.output,
        client=client,
        system_prompt=system_prompt,
        user_template=user_template,
        report_path=args.report,
        samples_per_scene=args.samples_per_scene,
        batch_size=args.batch_size,
        workers=args.workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()