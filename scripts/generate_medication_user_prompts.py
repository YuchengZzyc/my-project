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

VALID_STYLE_NOTES: set[str] = {
    "hesitant", "forgetful", "complaining", "asking_indirectly",
    "embarrassed", "rambling", "uncertain", "worried",
    "confused", "self_doubting", "repeating", "trailing_off",
}


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(prompt_dir: str | Path = PROMPT_DIR) -> tuple[str, str]:
    system_path = Path(prompt_dir) / "medication_user_prompt_system.txt"
    user_path = Path(prompt_dir) / "medication_user_prompt_user.txt"
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

def validate_user_prompt(obj: dict[str, Any]) -> str | None:
    """Validate a single user prompt entry. Returns error message or None."""
    if not isinstance(obj, dict):
        return "response is not a dict"

    # card_id
    card_id = obj.get("card_id")
    if not card_id or not isinstance(card_id, str) or not card_id.strip():
        return "card_id must be a non-empty string"

    # user_prompt
    prompt = obj.get("user_prompt")
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        return "user_prompt must be a non-empty string"
    word_count = len(prompt.strip().split())
    if word_count < 5:
        return f"user_prompt too short ({word_count} words, need >= 5)"
    if word_count > 30:
        return f"user_prompt too long ({word_count} words, need <= 30)"

    # style_note
    style_note = obj.get("style_note")
    if not style_note or not isinstance(style_note, str) or not style_note.strip():
        return "style_note must be a non-empty string"

    return None


# ---------------------------------------------------------------------------
# Prompt Card loading and sampling
# ---------------------------------------------------------------------------

def load_prompt_cards(input_path: Path) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cards.append(json.loads(line))
    return cards


def sample_cards(cards: list[dict[str, Any]], num_cards: int, seed: int) -> list[dict[str, Any]]:
    """Pick `num_cards` cards from different sub_scenes."""
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        sub = card.get("sub_scene", "unknown")
        groups.setdefault(sub, []).append(card)

    selected: list[dict[str, Any]] = []
    all_sub_scenes = list(groups.keys())
    rng.shuffle(all_sub_scenes)

    for sub in all_sub_scenes:
        if len(selected) >= num_cards:
            break
        card = rng.choice(groups[sub])
        selected.append(card)

    # Fallback: if not enough distinct sub_scenes, fill from remaining cards
    if len(selected) < num_cards:
        remaining = [c for c in cards if c not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: num_cards - len(selected)])

    return selected


# ---------------------------------------------------------------------------
# User prompt generation
# ---------------------------------------------------------------------------

def clean_json_response(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return raw


def parse_multi_json(text: str) -> list[dict[str, Any]]:
    """Parse one or more JSON objects from model output."""
    cleaned = clean_json_response(text)
    results: list[dict[str, Any]] = []
    # Try as JSON array first
    if cleaned.startswith("["):
        try:
            arr = json.loads(cleaned)
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass
    # Try line-by-line JSON objects
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            pass
    return results


def generate_user_prompts_for_card(
    client: OpenAICompatibleClient,
    system_prompt: str,
    user_prompt_template: str,
    card: dict[str, Any],
    samples_per_card: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    card_id_str = card.get("card_id", "unknown")
    card_json = json.dumps(card, ensure_ascii=False, indent=2)

    user_prompt = user_prompt_template.format(
        samples_per_card=samples_per_card,
        card_id=card_id_str,
    )
    full_user_content = f"Prompt Card:\n{card_json}\n\n{user_prompt}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_user_content},
    ]

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
            parsed = parse_multi_json(raw)

            if not parsed:
                last_err = RuntimeError(f"no valid JSON found in response (attempt {attempt})")
                continue

            # Validate each entry
            valid: list[dict[str, Any]] = []
            for obj in parsed:
                err = validate_user_prompt(obj)
                if err is None:
                    valid.append(obj)

            if len(valid) >= samples_per_card:
                return valid[:samples_per_card]

            last_err = RuntimeError(
                f"only {len(valid)}/{samples_per_card} valid entries (attempt {attempt})"
            )

        except Exception as exc:
            last_err = exc
            time.sleep(min(2.0, 0.4 * attempt))

    raise RuntimeError(
        f"generate_user_prompts_for_card({card_id_str}) failed after retries: {last_err}"
    )


# ---------------------------------------------------------------------------
# Dedup within a card
# ---------------------------------------------------------------------------

def dedup_entries(
    entries: list[dict[str, Any]],
    max_entries: int,
    card_id_str: str,
) -> list[dict[str, Any]]:
    """Deduplicate entries from same card: style_note must be unique."""
    seen_styles: set[str] = set()
    seen_prompts: set[str] = set()
    result: list[dict[str, Any]] = []

    for entry in entries:
        style = (entry.get("style_note") or "").strip().lower()
        prompt = (entry.get("user_prompt") or "").strip().lower()
        if style in seen_styles:
            continue
        if prompt in seen_prompts:
            continue
        seen_styles.add(style)
        seen_prompts.add(prompt)
        result.append(entry)
        if len(result) >= max_entries:
            break

    # Fill remaining with non-duplicate style entries
    if len(result) < max_entries:
        for entry in entries:
            if entry in result:
                continue
            style = (entry.get("style_note") or "").strip().lower()
            if style in seen_styles:
                continue
            seen_styles.add(style)
            result.append(entry)
            if len(result) >= max_entries:
                break

    return result[:max_entries]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(output_path: Path, report_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    card_counter: Counter[str] = Counter()
    style_counter: Counter[str] = Counter()
    word_lengths: list[int] = []

    for row in rows:
        card_counter[str(row.get("card_id", "unknown"))] += 1
        style_counter[str(row.get("style_note", "unknown"))] += 1
        prompt = row.get("user_prompt", "")
        if isinstance(prompt, str):
            word_lengths.append(len(prompt.strip().split()))

    total = len(rows)
    avg_words = round(sum(word_lengths) / max(len(word_lengths), 1), 1)
    min_words = min(word_lengths) if word_lengths else 0
    max_words = max(word_lengths) if word_lengths else 0

    report = {
        "total_user_prompts": total,
        "cards_used": len(card_counter),
        "style_note_distribution": dict(sorted(style_counter.items())),
        "prompts_per_card": dict(sorted(card_counter.items())),
        "word_length": {"min": min_words, "max": max_words, "avg": avg_words},
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Generation orchestration
# ---------------------------------------------------------------------------

def generate_user_prompts(
    input_path: Path,
    output_path: Path,
    client: OpenAICompatibleClient,
    system_prompt: str,
    user_prompt_template: str,
    report_path: Path | None = None,
    cards: int = 3,
    samples_per_card: int = 10,
    seed: int = 42,
    workers: int = 4,
    temperature: float = 0.9,
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_cards = load_prompt_cards(input_path)
    print(f"loaded {len(all_cards)} prompt cards from {input_path}", flush=True)

    selected_cards = sample_cards(all_cards, cards, seed)
    print(f"selected {len(selected_cards)} cards from different sub_scenes:", flush=True)
    for c in selected_cards:
        print(
            f"  card_id={c.get('card_id', '?')} sub_scene={c.get('sub_scene', '?')}",
            flush=True,
        )

    entries_lock = threading.Lock()
    all_entries: list[dict[str, Any]] = []
    completed = 0
    errors = 0
    total_jobs = len(selected_cards)

    def _run(card: dict[str, Any]) -> list[dict[str, Any]]:
        card_id_str = card.get("card_id", "unknown")
        try:
            entries = generate_user_prompts_for_card(
                client=client,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                card=card,
                samples_per_card=samples_per_card,
                temperature=temperature + random.uniform(-0.1, 0.1),
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
            # Dedup within this card's entries
            entries = dedup_entries(entries, samples_per_card, card_id_str)
            # Ensure card_id is set
            for e in entries:
                e["card_id"] = card_id_str
            return entries
        except Exception as exc:
            raise RuntimeError(f"card {card_id_str} failed: {exc}") from exc

    print(
        f"starting user prompt generation: cards={total_jobs} samples_per_card={samples_per_card} total_expected={total_jobs * samples_per_card}",
        flush=True,
    )

    with output_path.open("w", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {pool.submit(_run, card): card for card in selected_cards}
            for future in concurrent.futures.as_completed(futures):
                card = futures[future]
                try:
                    entries = future.result()
                    completed += 1
                    with entries_lock:
                        all_entries.extend(entries)
                    for entry in entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f.flush()
                    print(
                        f"[progress] completed={completed}/{total_jobs} "
                        f"card_id={card.get('card_id', '?')} "
                        f"entries={len(entries)}",
                        flush=True,
                    )
                except Exception as exc:
                    completed += 1
                    errors += 1
                    print(f"[error] card_id={card.get('card_id', '?')} failed: {exc}", flush=True)

    print(
        f"generation done: cards={completed} errors={errors} "
        f"total_entries={len(all_entries)}",
        flush=True,
    )

    if report_path is not None:
        write_report(output_path, report_path)

    return {"total_cards": completed, "total_entries": len(all_entries), "errors": errors}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate natural elderly user utterances from medication prompt cards."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to prompt cards JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--report", type=Path, default=None, help="Output stats JSON path.")
    parser.add_argument("--api-env", type=str, default="configs/data/api_generation.env")
    parser.add_argument("--cards", type=int, default=3, help="Number of prompt cards to process.")
    parser.add_argument("--samples-per-card", type=int, default=10, help="User prompts per card.")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--endpoint", choices=["chat.completions", "responses"], default="chat.completions")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1024)
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

    # Load prompts from external files
    system_prompt, user_prompt_template = load_prompts(args.prompt_dir)

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

    result = generate_user_prompts(
        input_path=args.input,
        output_path=args.output,
        client=client,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        report_path=args.report,
        cards=args.cards,
        samples_per_card=args.samples_per_card,
        seed=args.seed,
        workers=args.workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()