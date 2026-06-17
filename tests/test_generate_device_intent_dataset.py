from __future__ import annotations

import json

from scripts.generate_device_intent_dataset import generate_dataset, scenario_counts


class AlwaysFailClient:
    def chat(self, *args, **kwargs):
        raise TimeoutError("simulated timeout")


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def test_generate_device_intent_dataset_offline(tmp_path):
    out = tmp_path / "device_intent_dataset.jsonl"
    report = tmp_path / "device_intent_dataset.stats.json"
    count = generate_dataset(output_path=out, client=None, report_path=report, offline=True, workers=1, user_language="english")

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert count == len(rows)
    assert count > 100

    capability_ids = set()
    matched_values = set()
    scenarios = set()
    for row in rows:
        assert row["task"] == "device_intent_slot_extraction"
        assert row["schema_version"] == "device_intent_v2"
        assert "tools" not in row
        assert all("tool_calls" not in message for message in row["messages"])
        scenarios.add(row["scenario"])

        payload = json.loads(row["messages"][-1]["content"])
        assert set(payload) == {
            "matched",
            "capability_id",
            "capability",
            "intent",
            "slots",
            "missing_slots",
            "confidence",
        }
        matched_values.add(payload["matched"])
        if payload["capability_id"] is not None:
            capability_ids.add(payload["capability_id"])
        utterance = row["messages"][1]["content"]
        if payload["intent"] == "set_volume" and payload["slots"].get("adjustment") == "up":
            assert "set to" not in utterance.lower()
            assert "volume up" in utterance.lower()
        if payload["intent"] == "set_brightness" and payload["slots"].get("adjustment") == "up":
            assert "set to" not in utterance.lower()
            assert "brighter" in utterance.lower()
        for message in row["messages"]:
            assert not has_cjk(message["content"])

    assert set(range(1, 14)).issubset(capability_ids)
    assert matched_values == {True, False}
    assert "no_intent_device_words" in scenarios

    stats = json.loads(report.read_text(encoding="utf-8"))
    assert stats["total"] == count
    negative_ratio = stats["matched_false"] / stats["total"]
    assert 0.18 <= negative_ratio <= 0.22
    assert set(str(i) for i in range(1, 14)).issubset(stats["capability_distribution"])


def test_generate_device_intent_dataset_custom_sample_count(tmp_path):
    out = tmp_path / "device_intent_dataset_60.jsonl"
    count = generate_dataset(output_path=out, client=None, offline=True, workers=1, samples=60, user_language="english")

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    labels = [json.loads(row["messages"][-1]["content"]) for row in rows]

    assert count == 60
    assert len(rows) == 60
    assert any(label["matched"] is True for label in labels)
    assert any(label["matched"] is False for label in labels)
    assert sum(scenario_counts(60)) == 60


def test_generate_device_intent_dataset_appends_by_default(tmp_path):
    out = tmp_path / "device_intent_dataset_append.jsonl"
    report = tmp_path / "device_intent_dataset_append.stats.json"

    first_count = generate_dataset(output_path=out, client=None, offline=True, workers=1, samples=3, user_language="english")
    second_count = generate_dataset(output_path=out, client=None, report_path=report, offline=True, workers=1, samples=2, user_language="english")

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    stats = json.loads(report.read_text(encoding="utf-8"))

    assert first_count == 3
    assert second_count == 2
    assert len(rows) == 5
    assert stats["total"] == 5


def test_generate_device_intent_dataset_can_overwrite(tmp_path):
    out = tmp_path / "device_intent_dataset_overwrite.jsonl"

    generate_dataset(output_path=out, client=None, offline=True, workers=1, samples=3, user_language="english")
    count = generate_dataset(output_path=out, client=None, offline=True, workers=1, samples=2, user_language="english", append=False)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    assert count == 2
    assert len(rows) == 2


def test_generate_device_intent_dataset_mixed_user_language_keeps_json_english(tmp_path):
    out = tmp_path / "device_intent_dataset_mixed.jsonl"
    count = generate_dataset(output_path=out, client=None, offline=True, workers=1, samples=40, user_language="mixed")

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assistant_payloads = [row["messages"][-1]["content"] for row in rows]
    user_texts = [row["messages"][1]["content"] for row in rows]

    assert count == 40
    assert any(has_cjk(text) for text in user_texts)
    assert any(not has_cjk(text) for text in user_texts)
    assert all(not has_cjk(payload) for payload in assistant_payloads)


def test_generate_device_intent_dataset_falls_back_on_api_error(tmp_path):
    out = tmp_path / "device_intent_dataset_fallback.jsonl"
    report = tmp_path / "device_intent_dataset_fallback.stats.json"
    count = generate_dataset(
        output_path=out,
        client=AlwaysFailClient(),
        report_path=report,
        offline=False,
        workers=2,
        samples=5,
        user_language="english",
        max_retries=1,
        dedupe_retries=0,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    stats = json.loads(report.read_text(encoding="utf-8"))

    assert count == 5
    assert len(rows) == 5
    assert all(row["generation_source"] == "offline_fallback" for row in rows)
    assert all("generation_error" in row for row in rows)
    assert stats["generation_error_count"] == 5
