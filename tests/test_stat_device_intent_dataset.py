from __future__ import annotations

import json

from scripts.stat_device_intent_dataset import build_report, load_jsonl


def row(scenario: str, user: str, label: dict) -> dict:
    return {
        "task": "device_intent_slot_extraction",
        "schema_version": "device_intent_v2",
        "scenario": scenario,
        "generation_source": "test",
        "messages": [
            {"role": "system", "content": "Extract device-control intent and slots. Return JSON only."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
        ],
    }


def test_build_report_counts_scenarios_and_negative_ratio():
    rows = [
        row(
            "volume_adjust",
            "Turn the volume up.",
            {
                "matched": True,
                "capability_id": 8,
                "capability": "Adjust volume",
                "intent": "set_volume",
                "slots": {"adjustment": "up"},
                "missing_slots": [],
                "confidence": 0.9,
            },
        ),
        row(
            "no_intent_daily_chat",
            "The weather is nice today.",
            {
                "matched": False,
                "capability_id": None,
                "capability": None,
                "intent": None,
                "slots": {},
                "missing_slots": [],
                "confidence": 0.0,
            },
        ),
        row(
            "wallpaper_missing_type",
            "Change the wallpaper.",
            {
                "matched": True,
                "capability_id": 6,
                "capability": "Change wallpaper",
                "intent": "change_wallpaper",
                "slots": {},
                "missing_slots": ["wallpaper_type"],
                "confidence": 0.82,
            },
        ),
    ]

    report = build_report(rows)

    assert report["total"] == 3
    assert report["matched_true"] == 2
    assert report["matched_false"] == 1
    assert report["negative_ratio"] == 1 / 3
    assert report["scenario_distribution"] == {
        "no_intent_daily_chat": 1,
        "volume_adjust": 1,
        "wallpaper_missing_type": 1,
    }
    assert report["scene_distribution"] == {
        "06 更换壁纸": 1,
        "08 调节音量": 1,
        "负例-普通闲聊": 1,
    }
    assert report["detailed_scene_distribution"] == {
        "06 更换壁纸-缺少壁纸类型": 1,
        "08 调节音量": 1,
        "负例-普通闲聊": 1,
    }
    assert report["scene_examples"]["08 调节音量"] == ["Turn the volume up."]
    assert report["scene_examples"]["06 更换壁纸"] == ["Change the wallpaper."]
    assert report["capability_distribution"]["8"] == 1
    assert report["capability_distribution"]["None"] == 1
    assert report["missing_slot_distribution"] == {"wallpaper_type": 1}


def test_load_jsonl_and_duplicate_utterance_count(tmp_path):
    path = tmp_path / "dataset.jsonl"
    sample = row(
        "unsupported_device_control",
        "Turn on Bluetooth.",
        {
            "matched": False,
            "capability_id": None,
            "capability": None,
            "intent": None,
            "slots": {},
            "missing_slots": [],
            "confidence": 0.0,
        },
    )
    path.write_text(json.dumps(sample) + "\n" + json.dumps(sample) + "\n", encoding="utf-8")

    report = build_report(load_jsonl(path))

    assert report["total"] == 2
    assert report["exact_duplicate_utterance_count"] == 1
    assert report["top_duplicate_utterances"] == {"Turn on Bluetooth.": 2}
