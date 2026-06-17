from __future__ import annotations

import json

from scripts.export_device_intent_data import export_jsonl


def test_export_device_intent_data_has_no_reminder_tools(tmp_path):
    out = tmp_path / "device_intent.jsonl"
    count = export_jsonl(out)

    assert count > 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == count

    capability_ids = set()
    for row in rows:
        assert row["task"] == "device_intent_slot_extraction"
        assert "tools" not in row
        assert all("tool_calls" not in message for message in row["messages"])

        assistant = row["messages"][-1]
        payload = json.loads(assistant["content"])
        assert set(payload) == {
            "capability_id",
            "capability",
            "intent",
            "slots",
            "missing_slots",
            "confidence",
        }
        capability_ids.add(payload["capability_id"])

    assert set(range(1, 14)).issubset(capability_ids)
    assert None in capability_ids
