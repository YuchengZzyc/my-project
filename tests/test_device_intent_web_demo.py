from __future__ import annotations

import json

from scripts.device_intent_web_demo import (
    PrintOnlyDeviceExecutor,
    SimulatedDeviceExecutor,
    build_training_prompt,
    parse_intent_output,
    reply_for_result,
)


def test_build_training_prompt_matches_adapter_format():
    prompt = build_training_prompt("Turn the volume up.")

    assert prompt.startswith("<TOOLS>\n[]\n</TOOLS>")
    assert "<SYSTEM>\nExtract device-control intent and slots. Return JSON only.\n</SYSTEM>" in prompt
    assert "<USER>\nTurn the volume up.\n</USER>" in prompt
    assert prompt.endswith("<ASSISTANT>")


def test_parse_intent_output_extracts_json_with_assistant_tag():
    raw = (
        '{"matched":true,"capability_id":8,"capability":"Adjust volume",'
        '"intent":"set_volume","slots":{"adjustment":"up"},'
        '"missing_slots":[],"confidence":0.94}</ASSISTANT>'
    )

    parsed = parse_intent_output(raw)

    assert parsed == {
        "matched": True,
        "capability_id": 8,
        "capability": "Adjust volume",
        "intent": "set_volume",
        "slots": {"adjustment": "up"},
        "missing_slots": [],
        "confidence": 0.94,
    }


def test_executor_prints_only_for_matched_intent(capsys):
    executor = SimulatedDeviceExecutor()
    label = {
        "matched": True,
        "capability_id": 8,
        "capability": "Adjust volume",
        "intent": "set_volume",
        "slots": {"adjustment": "up"},
        "missing_slots": [],
        "confidence": 0.94,
    }

    result = executor.execute(label)
    printed = json.loads(capsys.readouterr().out)

    assert result["status"] == "printed"
    assert result["state"]["volume"] == 60
    assert printed["event"] == "device_intent_postprocess"
    assert printed["intent"] == "set_volume"
    assert printed["slots"] == {"adjustment": "up"}
    assert printed["state"]["volume"] == 60


def test_executor_skips_not_matched(capsys):
    executor = PrintOnlyDeviceExecutor()

    result = executor.execute({"matched": False})

    assert result["status"] == "skipped"
    assert result["reason"] == "not_matched"
    assert result["state"]["volume"] == 50
    assert capsys.readouterr().out == ""


def test_executor_sets_volume_level(capsys):
    executor = SimulatedDeviceExecutor()
    result = executor.execute(
        {
            "matched": True,
            "capability_id": 8,
            "capability": "Adjust volume",
            "intent": "set_volume",
            "slots": {"adjustment": "set", "level": 60},
            "missing_slots": [],
            "confidence": 0.9,
        }
    )

    assert result["state"]["volume"] == 60
    assert result["changes"] == ["volume"]
    assert "Volume set to 60%" in result["state"]["last_action"]
    capsys.readouterr()


def test_executor_missing_slots_does_not_change_state(capsys):
    executor = SimulatedDeviceExecutor()
    result = executor.execute(
        {
            "matched": True,
            "capability_id": 6,
            "capability": "Change wallpaper",
            "intent": "change_wallpaper",
            "slots": {},
            "missing_slots": ["wallpaper_type"],
            "confidence": 0.8,
        }
    )

    assert result["status"] == "missing_slots"
    assert result["state"]["wallpaper"] == "default"
    assert capsys.readouterr().out == ""


def test_reply_for_parse_failure_uses_raw_output():
    assert reply_for_result("hello", None, None) == "hello"
