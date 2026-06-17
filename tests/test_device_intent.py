from __future__ import annotations

from app.device_intent import build_device_intent_label, missing_required_slots, parse_device_command


def test_parse_all_13_device_capabilities():
    cases = [
        ("帮我呼叫管理员", 1, "call_manager", {"contact": "property_manager"}),
        ("给家人打电话", 2, "call_contact", {"contact": "family"}),
        ("接听电话", 3, "answer_call", {}),
        ("通话中帮我开锁", 4, "unlock", {"method": "remote", "during_call": True}),
        ("拒接这个电话", 5, "end_or_reject_call", {"action": "reject"}),
        ("换成风景壁纸", 6, "change_wallpaper", {"wallpaper_type": "风景"}),
        ("打开免打扰", 7, "set_dnd", {"enabled": True}),
        ("提高音量", 8, "set_volume", {"adjustment": "up"}),
        ("把亮度调暗", 9, "set_brightness", {"adjustment": "down"}),
        ("开启布防", 10, "set_security_mode", {"enabled": True}),
        ("查一下今天的告警记录", 11, "query_alarm_records", {"time_range": "今天"}),
        ("切换到3通道", 12, "monitor_control", {"action": "switch_channel", "channel": 3}),
        ("看一下未读短信", 13, "view_messages", {"message_type": "sms", "unread_only": True}),
    ]

    for text, capability_id, intent, slots in cases:
        label = build_device_intent_label(text)
        assert label["capability_id"] == capability_id
        assert label["intent"] == intent
        assert label["slots"] == slots
        assert label["missing_slots"] == []


def test_parse_contact_missing_slot():
    parsed = parse_device_command("帮我打电话")

    assert parsed.intent == "call_contact"
    assert missing_required_slots(parsed.intent, parsed.slots) == ["contact"]


def test_parse_monitor_missing_action():
    parsed = parse_device_command("门口监控")

    assert parsed.intent == "monitor_control"
    assert parsed.slots == {"scene": "门口"}
    assert missing_required_slots(parsed.intent, parsed.slots) == ["action"]


def test_parse_no_device_intent():
    label = build_device_intent_label("今天外面下雨了")

    assert label["capability_id"] is None
    assert label["capability"] is None
    assert label["intent"] is None
    assert label["slots"] == {}
