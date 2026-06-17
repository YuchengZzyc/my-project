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


SYSTEM_PROMPT = "Extract device-control intent and slots. Return JSON only."

INFERENCE_SYSTEM_PROMPT = (
    "You are a device-control intent classifier and slot extractor. Return JSON only. "
    "Supported capabilities are IDs 1..13: call property manager, call contact, answer incoming call, "
    "unlock door, reject or hang up call, change wallpaper, toggle do not disturb, adjust volume, "
    "adjust screen brightness, arm or disarm security, query alarm records, monitor control, and "
    "view notifications or SMS. If no supported capability matches, return matched=false with null "
    "capability_id, null capability, null intent, empty slots, empty missing_slots, and confidence 0.0. "
    "All string values in the assistant JSON must be normalized English."
)


CAPABILITY_NAMES: dict[int, str] = {
    1: "Call property manager",
    2: "Call contact",
    3: "Answer incoming call",
    4: "Unlock door",
    5: "Reject or hang up call",
    6: "Change wallpaper",
    7: "Toggle do not disturb",
    8: "Adjust volume",
    9: "Adjust screen brightness",
    10: "Arm or disarm security",
    11: "Query alarm records",
    12: "Monitor control",
    13: "View notifications or SMS",
}


def label(
    matched: bool,
    capability_id: int | None,
    intent: str | None,
    slots: dict[str, Any] | None = None,
    missing_slots: list[str] | None = None,
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {
        "matched": matched,
        "capability_id": capability_id,
        "capability": CAPABILITY_NAMES.get(capability_id) if capability_id is not None else None,
        "intent": intent,
        "slots": slots or {},
        "missing_slots": missing_slots or [],
        "confidence": confidence if matched else 0.0,
    }


SCENARIOS: list[dict[str, Any]] = [
    {"scenario": "call_manager_property", "n": 12, "label": label(True, 1, "call_manager", {"contact": "property_manager"}, confidence=0.95), "notes": "Contact the property manager, building manager, or community manager."},
    {"scenario": "call_contact_family", "n": 12, "label": label(True, 2, "call_contact", {"contact": "family"}), "notes": "Call a family contact."},
    {"scenario": "call_contact_frequent", "n": 10, "label": label(True, 2, "call_contact", {"contact": "frequent_contacts"}), "notes": "Call a frequent contact."},
    {"scenario": "call_contact_service", "n": 10, "label": label(True, 2, "call_contact", {"contact": "service_phone"}), "notes": "Call a service phone number."},
    {"scenario": "call_contact_missing", "n": 8, "label": label(True, 2, "call_contact", {}, ["contact"], 0.8), "notes": "The user asks to make a call but does not say who to call."},
    {"scenario": "answer_call", "n": 10, "label": label(True, 3, "answer_call", confidence=0.95), "notes": "Answer an incoming call by voice."},
    {"scenario": "unlock_remote", "n": 10, "label": label(True, 4, "unlock", {"method": "remote"}, confidence=0.95), "notes": "Remote door unlock."},
    {"scenario": "unlock_during_call", "n": 8, "label": label(True, 4, "unlock", {"method": "remote", "during_call": True}, confidence=0.95), "notes": "Unlock the door during an active call."},
    {"scenario": "reject_call", "n": 8, "label": label(True, 5, "end_or_reject_call", {"action": "reject"}, confidence=0.95), "notes": "Reject an incoming call."},
    {"scenario": "hangup_call", "n": 8, "label": label(True, 5, "end_or_reject_call", {"action": "hangup"}, confidence=0.95), "notes": "Hang up the current call."},
    {
        "scenario": "wallpaper_type",
        "n": 14,
        "label": label(True, 6, "change_wallpaper", {"wallpaper_type": "landscape"}),
        "slot_variants": [{"wallpaper_type": x} for x in ["landscape", "default", "family", "holiday", "cartoon", "solid_color", "dark", "light"]],
        "notes": "Change the screen wallpaper to a specific type.",
    },
    {"scenario": "wallpaper_missing_type", "n": 6, "label": label(True, 6, "change_wallpaper", {}, ["wallpaper_type"], 0.82), "notes": "The user wants to change wallpaper but does not specify the type."},
    {"scenario": "dnd_on", "n": 8, "label": label(True, 7, "set_dnd", {"enabled": True}), "notes": "Turn on do not disturb mode."},
    {"scenario": "dnd_off", "n": 8, "label": label(True, 7, "set_dnd", {"enabled": False}), "notes": "Turn off do not disturb mode."},
    {
        "scenario": "volume_adjust",
        "n": 18,
        "label": label(True, 8, "set_volume", {"adjustment": "up"}),
        "slot_variants": [{"adjustment": "up"}, {"adjustment": "down"}, {"adjustment": "set", "level": "max"}, {"adjustment": "set", "level": "min"}, {"adjustment": "set", "level": 0}, {"adjustment": "set", "level": 60}],
        "notes": "Set, increase, decrease, maximize, minimize, mute, or set numeric volume.",
    },
    {
        "scenario": "brightness_adjust",
        "n": 18,
        "label": label(True, 9, "set_brightness", {"adjustment": "up"}),
        "slot_variants": [{"adjustment": "up"}, {"adjustment": "down"}, {"adjustment": "set", "level": "max"}, {"adjustment": "set", "level": "min"}, {"adjustment": "set", "level": 70}],
        "notes": "Set, brighten, dim, maximize, minimize, or set numeric screen brightness.",
    },
    {"scenario": "security_on", "n": 8, "label": label(True, 10, "set_security_mode", {"enabled": True}), "notes": "Arm or turn on home security."},
    {"scenario": "security_off", "n": 8, "label": label(True, 10, "set_security_mode", {"enabled": False}), "notes": "Disarm or turn off home security."},
    {
        "scenario": "alarm_query",
        "n": 14,
        "label": label(True, 11, "query_alarm_records", {"time_range": "today"}),
        "slot_variants": [{"time_range": x} for x in ["today", "yesterday", "the day before yesterday", "last week", "last three days", "this month", "last month"]],
        "notes": "Query alarm records with an optional time range.",
    },
    {
        "scenario": "monitor_control",
        "n": 20,
        "label": label(True, 12, "monitor_control", {"action": "view", "scene": "front_door"}),
        "slot_variants": [{"action": "open", "scene": "front_door"}, {"action": "close", "scene": "living_room"}, {"action": "view", "scene": "bedroom"}, {"action": "playback"}, {"action": "switch_channel", "channel": 3}],
        "notes": "Open or close monitoring, view a scene, play back video, or switch channel.",
    },
    {"scenario": "monitor_missing_action", "n": 6, "label": label(True, 12, "monitor_control", {"scene": "front_door"}, ["action"], 0.8), "notes": "The user mentions a monitor scene but omits the action."},
    {
        "scenario": "view_messages",
        "n": 14,
        "label": label(True, 13, "view_messages", {"message_type": "all"}),
        "slot_variants": [{"message_type": "sms", "unread_only": True}, {"message_type": "notification", "unread_only": True}, {"message_type": "all", "unread_only": True}, {"message_type": "all"}],
        "notes": "View new messages, notifications, or SMS.",
    },
    {"scenario": "unsupported_device_control", "n": 12, "label": label(False, None, None), "notes": "The user asks for an action, but it is outside the 13 supported device capabilities, such as sending a message, booking travel, opening Bluetooth, or ordering food."},
    {"scenario": "ambiguous_device_control", "n": 8, "label": label(False, None, None), "notes": "The user gives an underspecified control command such as turning it off, but no supported device or capability can be identified safely."},
    {"scenario": "no_intent_daily_chat", "n": 14, "label": label(False, None, None), "notes": "Ordinary chat that must not trigger a device-control capability."},
    {"scenario": "no_intent_device_words", "n": 14, "label": label(False, None, None), "notes": "Contains words like sound, phone, door, brightness, delivery, or message but is not a control request."},
    {"scenario": "no_intent_general_question", "n": 12, "label": label(False, None, None), "notes": "General question or companionship request."},
]


USER_STYLES = [
    "direct command",
    "polite request",
    "elderly spoken wording",
    "short urgent phrase",
    "context first then request",
    "family member speaking on behalf of an older adult",
]

SURFACE_FORMS = [
    "one short sentence",
    "natural spoken sentence with a filler word",
    "imperative command",
    "question-like request",
    "sentence with one extra context clause",
]

DISTRACTOR_CONTEXTS = [
    "no extra context",
    "mentions current call state",
    "mentions being unable to see or hear clearly",
    "mentions family, property, or community context",
    "mentions time, but time is not a slot unless the scenario needs it",
    "mentions device words as background only for no-intent cases",
]

USER_LANGUAGES = ["english", "chinese"]


OFFLINE_TEMPLATES = {
    "call_manager_property": ["Please call the property manager.", "Get the building manager for me.", "Can you contact the community manager?"],
    "call_contact_family": ["Call my family.", "Please phone my family contact."],
    "call_contact_frequent": ["Call my frequent contact.", "Please dial my usual contact."],
    "call_contact_service": ["Call the service number.", "Please dial the service phone."],
    "call_contact_missing": ["Please make a call for me.", "I want to call someone."],
    "answer_call": ["Answer the call.", "The phone is ringing, pick it up for me."],
    "unlock_remote": ["Unlock the door remotely.", "Open the door lock for me."],
    "unlock_during_call": ["Unlock the door while I am on the call.", "Open the door during this call."],
    "reject_call": ["Reject this incoming call.", "Do not answer this call."],
    "hangup_call": ["Hang up the phone.", "End the current call."],
    "wallpaper_type": ["Change the wallpaper to {wallpaper_type}.", "Set the screen wallpaper to {wallpaper_type}."],
    "wallpaper_missing_type": ["Change the wallpaper for me.", "Switch the screen wallpaper."],
    "dnd_on": ["Turn on do not disturb.", "Enable do not disturb mode."],
    "dnd_off": ["Turn off do not disturb.", "Cancel do not disturb mode."],
    "volume_adjust": ["Turn the volume {direction}.", "Set the volume to {level_text}."],
    "brightness_adjust": ["Make the screen {direction}.", "Set the brightness to {level_text}."],
    "security_on": ["Arm the security system.", "Turn on home security."],
    "security_off": ["Disarm the security system.", "Turn off home security."],
    "alarm_query": ["Check the alarm records for {time_range}.", "See whether there were any alarms {time_range}."],
    "monitor_control": ["{action_text} the {scene_text} monitor.", "Switch to channel {channel_text}."],
    "monitor_missing_action": ["The front door monitor.", "The living room camera."],
    "view_messages": ["Show unread SMS messages.", "Open the new notifications."],
    "unsupported_device_control": ["Send a message to Alex.", "Book me a flight to Tokyo.", "Order dinner for me.", "Turn on Bluetooth."],
    "ambiguous_device_control": ["Turn it off.", "Open that for me.", "Make it lower.", "Switch it back."],
    "no_intent_daily_chat": ["The weather is nice today.", "I went out for a walk earlier."],
    "no_intent_device_words": ["The hallway was noisy today.", "I forgot my keys and the wind near the door was strong.", "The delivery person left the package by the front door.", "I hope the delivery person leaves the package by the front door."],
    "no_intent_general_question": ["What should I eat for dinner?", "Please keep me company for a while."],
}


ZH_OFFLINE_TEMPLATES = {
    "call_manager_property": ["帮我联系物业管理员。", "呼叫一下楼栋管理员。", "麻烦找一下物管。"],
    "call_contact_family": ["给家人打个电话。", "帮我拨一下家人的电话。"],
    "call_contact_frequent": ["拨打常用联系人。", "帮我联系常用联系人。"],
    "call_contact_service": ["帮我打服务电话。", "拨一下服务电话。"],
    "call_contact_missing": ["帮我打个电话。", "我想拨个电话。"],
    "answer_call": ["接听电话。", "来电话了，帮我接一下。"],
    "unlock_remote": ["帮我远程开门。", "打开门锁。"],
    "unlock_during_call": ["通话中帮我开锁。", "边通话边开门。"],
    "reject_call": ["拒接这个来电。", "别接这个电话。"],
    "hangup_call": ["把电话挂了。", "结束当前通话。"],
    "wallpaper_type": ["把壁纸换成{wallpaper_type}。", "把屏幕壁纸设成{wallpaper_type}。"],
    "wallpaper_missing_type": ["帮我换个壁纸。", "屏幕壁纸换一下。"],
    "dnd_on": ["打开免打扰。", "开启勿扰模式。"],
    "dnd_off": ["关闭免打扰。", "取消勿扰模式。"],
    "volume_adjust": ["把声音调{direction}。", "音量设为{level_text}。"],
    "brightness_adjust": ["把屏幕调{direction}。", "亮度设为{level_text}。"],
    "security_on": ["开启布防。", "打开安防。"],
    "security_off": ["撤防。", "关闭安防。"],
    "alarm_query": ["查一下{time_range}的告警记录。", "看看{time_range}有没有报警。"],
    "monitor_control": ["{action_text}{scene_text}监控。", "切换到{channel_text}。"],
    "monitor_missing_action": ["门口监控。", "客厅摄像头。"],
    "view_messages": ["看一下未读短信。", "查看新通知。"],
    "unsupported_device_control": ["给Alex发个消息。", "帮我订一张去东京的机票。", "帮我点个晚饭。", "打开蓝牙。"],
    "ambiguous_device_control": ["把它关掉。", "帮我打开那个。", "调低一点。", "切回去。"],
    "no_intent_daily_chat": ["今天天气挺好。", "我刚才出去散步了。"],
    "no_intent_device_words": ["楼道里今天声音有点大。", "我忘了带钥匙，门口风很大。", "快递员把包裹放在门口了。", "希望快递员把包裹放在门口。"],
    "no_intent_general_question": ["你觉得晚饭吃什么好？", "陪我聊聊天吧。"],
}

ZH_OFFLINE_TEMPLATES.update(
    {
        "call_manager_property": ["\u5e2e\u6211\u8054\u7cfb\u7269\u4e1a\u7ba1\u7406\u5458\u3002", "\u547c\u53eb\u4e00\u4e0b\u697c\u680b\u7ba1\u7406\u5458\u3002"],
        "call_contact_family": ["\u7ed9\u5bb6\u4eba\u6253\u4e2a\u7535\u8bdd\u3002"],
        "call_contact_frequent": ["\u62e8\u6253\u5e38\u7528\u8054\u7cfb\u4eba\u3002"],
        "call_contact_service": ["\u5e2e\u6211\u6253\u670d\u52a1\u7535\u8bdd\u3002"],
        "call_contact_missing": ["\u5e2e\u6211\u6253\u4e2a\u7535\u8bdd\u3002"],
        "answer_call": ["\u63a5\u542c\u7535\u8bdd\u3002"],
        "unlock_remote": ["\u5e2e\u6211\u8fdc\u7a0b\u5f00\u95e8\u3002"],
        "unlock_during_call": ["\u901a\u8bdd\u4e2d\u5e2e\u6211\u5f00\u9501\u3002"],
        "reject_call": ["\u62d2\u63a5\u8fd9\u4e2a\u6765\u7535\u3002"],
        "hangup_call": ["\u628a\u7535\u8bdd\u6302\u4e86\u3002"],
        "wallpaper_type": ["\u628a\u58c1\u7eb8\u6362\u6210{wallpaper_type}\u3002"],
        "wallpaper_missing_type": ["\u5e2e\u6211\u6362\u4e2a\u58c1\u7eb8\u3002"],
        "dnd_on": ["\u6253\u5f00\u514d\u6253\u6270\u3002"],
        "dnd_off": ["\u5173\u95ed\u514d\u6253\u6270\u3002"],
        "volume_adjust": ["\u628a\u58f0\u97f3\u8c03{direction}\u3002", "\u97f3\u91cf\u8bbe\u4e3a{level_text}\u3002"],
        "brightness_adjust": ["\u628a\u5c4f\u5e55\u8c03{direction}\u3002", "\u4eae\u5ea6\u8bbe\u4e3a{level_text}\u3002"],
        "security_on": ["\u5f00\u542f\u5e03\u9632\u3002"],
        "security_off": ["\u64a4\u9632\u3002"],
        "alarm_query": ["\u67e5\u4e00\u4e0b{time_range}\u7684\u544a\u8b66\u8bb0\u5f55\u3002"],
        "monitor_control": ["{action_text}{scene_text}\u76d1\u63a7\u3002", "\u5207\u6362\u5230{channel_text}\u3002"],
        "monitor_missing_action": ["\u95e8\u53e3\u76d1\u63a7\u3002"],
        "view_messages": ["\u770b\u4e00\u4e0b\u672a\u8bfb\u77ed\u4fe1\u3002"],
        "unsupported_device_control": ["\u7ed9Alex\u53d1\u4e2a\u6d88\u606f\u3002", "\u5e2e\u6211\u8ba2\u4e00\u5f20\u53bb\u4e1c\u4eac\u7684\u673a\u7968\u3002"],
        "ambiguous_device_control": ["\u628a\u5b83\u5173\u6389\u3002", "\u5e2e\u6211\u6253\u5f00\u90a3\u4e2a\u3002"],
        "no_intent_daily_chat": ["\u4eca\u5929\u5929\u6c14\u633a\u597d\u3002"],
        "no_intent_device_words": ["\u5feb\u9012\u5458\u628a\u5305\u88f9\u653e\u5728\u95e8\u53e3\u4e86\u3002"],
        "no_intent_general_question": ["\u4f60\u89c9\u5f97\u665a\u996d\u5403\u4ec0\u4e48\u597d\uff1f"],
    }
)


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


def label_for_scenario(scenario: dict[str, Any], occurrence_idx: int) -> dict[str, Any]:
    out = json.loads(json.dumps(scenario["label"], ensure_ascii=False))
    variants = scenario.get("slot_variants") or []
    if variants:
        out["slots"] = variants[occurrence_idx % len(variants)]
    return out


def choose_user_language(user_language: str, occurrence_idx: int, rng: random.Random) -> str:
    if user_language == "mixed":
        return USER_LANGUAGES[(occurrence_idx + rng.randrange(len(USER_LANGUAGES))) % len(USER_LANGUAGES)]
    return user_language


def build_diversity_profile(
    scenario: dict[str, Any],
    label_data: dict[str, Any],
    occurrence_idx: int,
    rng: random.Random,
    user_language: str,
) -> dict[str, str]:
    style = USER_STYLES[(occurrence_idx + rng.randrange(len(USER_STYLES))) % len(USER_STYLES)]
    surface = SURFACE_FORMS[(occurrence_idx + rng.randrange(len(SURFACE_FORMS))) % len(SURFACE_FORMS)]
    context = DISTRACTOR_CONTEXTS[(occurrence_idx + rng.randrange(len(DISTRACTOR_CONTEXTS))) % len(DISTRACTOR_CONTEXTS)]
    language = choose_user_language(user_language, occurrence_idx, rng)
    if not label_data.get("matched"):
        context = "must not express a device-control request; may mention device words only as life context"
    if label_data.get("missing_slots"):
        context = f"must omit these slots: {', '.join(label_data['missing_slots'])}"
    return {
        "style": style,
        "surface_form": surface,
        "context": context,
        "language": language,
        "avoid": "avoid labels, explanations, assistant wording, and near-duplicate phrasing",
    }


def format_offline_prompt(
    scenario: str,
    label_data: dict[str, Any],
    occurrence_idx: int,
    rng: random.Random,
    user_language: str = "english",
) -> str:
    use_chinese = user_language == "chinese"
    templates = ZH_OFFLINE_TEMPLATES if use_chinese else OFFLINE_TEMPLATES
    template = rng.choice(templates.get(scenario, ["请帮我处理一下。"] if use_chinese else ["Please handle this."]))
    slots = label_data.get("slots") or {}
    if label_data.get("matched") is False:
        return template

    if label_data.get("intent") in {"set_volume", "set_brightness"}:
        noun = "volume" if label_data.get("intent") == "set_volume" else "brightness"
        if slots.get("adjustment") == "up":
            text = ("把声音调大一点。" if use_chinese else "Turn the volume up.") if label_data.get("intent") == "set_volume" else ("把屏幕调亮一点。" if use_chinese else "Make the screen brighter.")
        elif slots.get("adjustment") == "down":
            text = ("把声音调低一点。" if use_chinese else "Turn the volume down.") if label_data.get("intent") == "set_volume" else ("把屏幕调暗一点。" if use_chinese else "Dim the screen.")
        else:
            level = slots.get("level")
            level_text = ({"max": "最大", "min": "最小", 0: "静音"} if use_chinese else {"max": "maximum", "min": "minimum", 0: "mute"}).get(level, str(level))
            text = f"把{('音量' if label_data.get('intent') == 'set_volume' else '亮度')}设为{level_text}。" if use_chinese else f"Set the {noun} to {level_text}."
    elif label_data.get("intent") == "monitor_control" and slots.get("action") == "switch_channel":
        text = f"切换到{slots.get('channel', 1)}通道。" if use_chinese else f"Switch to channel {slots.get('channel', 1)}."
    else:
        action_text_map = {
            "open": "Open",
            "close": "Close",
            "view": "View",
            "playback": "Play back",
            "switch_channel": "Switch",
        }
        zh_action_text_map = {
            "open": "打开",
            "close": "关闭",
            "view": "查看",
            "playback": "回放",
            "switch_channel": "切换",
        }
        action_text = (zh_action_text_map if use_chinese else action_text_map).get(slots.get("action"), "查看" if use_chinese else "View")
        english_scene_map = {"front_door": "front door", "living_room": "living room", "bedroom": "bedroom"}
        zh_scene_map = {"front_door": "\u95e8\u53e3", "living_room": "\u5ba2\u5385", "bedroom": "\u5367\u5ba4"}
        raw_scene = str(slots.get("scene", "front_door"))
        if use_chinese:
            scene_text = zh_scene_map.get(raw_scene, "\u95e8\u53e3")
        else:
            scene_text = english_scene_map.get(raw_scene, raw_scene.replace("_", " "))
        channel_text = str(slots.get("channel", 1))
        wallpaper_type = str(slots.get("wallpaper_type", "landscape")).replace("_", " ")
        if use_chinese:
            wallpaper_type = {
                "landscape": "风景",
                "default": "默认",
                "family": "家庭",
                "holiday": "节日",
                "cartoon": "卡通",
                "solid_color": "纯色",
                "dark": "深色",
                "light": "浅色",
            }.get(str(slots.get("wallpaper_type")), wallpaper_type)
        time_range = str(slots.get("time_range", "today"))
        if use_chinese:
            time_range = {
                "today": "今天",
                "yesterday": "昨天",
                "the day before yesterday": "前天",
                "last week": "最近一周",
                "last three days": "最近三天",
                "this month": "这个月",
                "last month": "上个月",
            }.get(time_range, time_range)
        text = template.format(
            wallpaper_type=wallpaper_type,
            time_range=time_range,
            action_text=action_text,
            scene_text=scene_text,
            channel_text=channel_text,
        )

    if occurrence_idx % 3 == 1:
        if use_chinese:
            return f"麻烦你{text}"
        return f"Please, {text[0].lower()}{text[1:]}"
    if occurrence_idx % 3 == 2:
        if use_chinese:
            return f"{text.rstrip('。')}，现在就弄。"
        return f"{text.rstrip('.')} right now."
    return text


def clean_generated_utterance(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("utterance"), str):
            raw = obj["utterance"]
    except Exception:
        pass
    raw = raw.strip().strip('"').strip("'").strip()
    for prefix in ["User:", "Utterance:", "utterance:"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].strip()
    return " ".join(raw.split())


def generate_utterance(
    client: OpenAICompatibleClient,
    scenario: dict[str, Any],
    label_data: dict[str, Any],
    diversity_profile: dict[str, str],
    occurrence_idx: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> str:
    prompt = [
        {
            "role": "system",
            "content": (
                "Generate user utterances for a device-control intent and slot extraction dataset. "
                "Return JSON only: {\"utterance\":\"...\"}. "
                "Do not output labels or explanations. "
                "The utterance must be natural and suitable for an older adult or home device assistant setting."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Scenario: {scenario['scenario']}\n"
                f"Notes: {scenario['notes']}\n"
                f"Fixed label: {json.dumps(label_data, ensure_ascii=False)}\n"
                f"Sample index: {occurrence_idx}\n\n"
                "Diversity profile:\n"
                f"- style: {diversity_profile['style']}\n"
                f"- surface_form: {diversity_profile['surface_form']}\n"
                f"- context: {diversity_profile['context']}\n"
                f"- user_language: {diversity_profile['language']}\n"
                f"- avoid: {diversity_profile['avoid']}\n\n"
                "Generate exactly one user utterance in the requested user_language. It must match the fixed label exactly. "
                "All string values in the fixed label are already normalized English and must not be changed. "
                "If matched=false, it must be normal speech that is not one of the 13 device-control requests. "
                "It may mention device-related words, but it must not ask the assistant to control a device. "
                "If missing_slots is non-empty, the utterance must intentionally omit those slots."
            ),
        },
    ]
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            utterance = clean_generated_utterance(client.chat(prompt, temperature=temperature, max_tokens=max_tokens))
            if utterance:
                return utterance
        except Exception as exc:
            last_err = exc
            time.sleep(min(2.0, 0.4 * attempt))
    raise RuntimeError(f"generate_utterance failed after retries: {last_err}")


def build_training_sample(utterance: str, label_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "device_intent_slot_extraction",
        "schema_version": "device_intent_v2",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": utterance},
            {"role": "assistant", "content": json.dumps(label_data, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def scenario_counts(total_samples: int | None = None) -> list[int]:
    base_counts = [int(scenario["n"]) for scenario in SCENARIOS]
    if total_samples is None:
        return base_counts
    if total_samples <= 0:
        raise ValueError("--samples must be positive when provided")

    base_total = sum(base_counts)
    raw = [(count / base_total) * total_samples for count in base_counts]
    counts = [int(value) for value in raw]

    if total_samples >= len(SCENARIOS):
        counts = [max(1, count) for count in counts]

    remainder = total_samples - sum(counts)
    fractions = sorted(((raw[i] - int(raw[i]), i) for i in range(len(raw))), reverse=remainder >= 0)
    while remainder != 0:
        changed = False
        for _, idx in fractions:
            if remainder > 0:
                counts[idx] += 1
                remainder -= 1
                changed = True
            elif counts[idx] > (1 if total_samples >= len(SCENARIOS) else 0):
                counts[idx] -= 1
                remainder += 1
                changed = True
            if remainder == 0:
                break
        if not changed:
            break
    return counts


def build_jobs(rng: random.Random, total_samples: int | None = None, user_language: str = "mixed") -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    idx = 0
    counts = scenario_counts(total_samples)
    for scenario, scenario_count in zip(SCENARIOS, counts):
        for occurrence_idx in range(scenario_count):
            idx += 1
            label_data = label_for_scenario(scenario, occurrence_idx)
            jobs.append(
                {
                    "idx": idx,
                    "scenario": scenario,
                    "occurrence_idx": occurrence_idx,
                    "label": label_data,
                    "diversity_profile": build_diversity_profile(scenario, label_data, occurrence_idx, rng, user_language),
                    "temperature": rng.uniform(0.75, 1.05),
                }
            )
    return jobs


def write_dataset_report(output_path: Path, report_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [json.loads(row["messages"][-1]["content"]) for row in rows]
    utterances = [row["messages"][1]["content"] for row in rows]
    matched_counter = Counter(str(item["matched"]) for item in labels)
    capability_counter = Counter(str(item["capability_id"]) for item in labels)
    scenario_counter = Counter(str(row.get("scenario")) for row in rows)
    duplicate_utterances = {utterance: count for utterance, count in Counter(utterances).items() if count > 1}
    missing_slot_counter = Counter()
    for item in labels:
        for slot in item.get("missing_slots", []):
            missing_slot_counter[slot] += 1

    total = len(rows)
    negative = matched_counter.get("False", 0)
    source_counter = Counter(str(row.get("generation_source", "unknown")) for row in rows)
    error_count = sum(1 for row in rows if row.get("generation_error"))
    report = {
        "total": total,
        "matched_true": matched_counter.get("True", 0),
        "matched_false": negative,
        "negative_ratio": (negative / total) if total else 0.0,
        "capability_distribution": dict(sorted(capability_counter.items())),
        "scenario_distribution": dict(sorted(scenario_counter.items())),
        "missing_slot_distribution": dict(sorted(missing_slot_counter.items())),
        "exact_duplicate_utterance_count": sum(duplicate_utterances.values()) - len(duplicate_utterances),
        "exact_duplicate_utterances": duplicate_utterances,
        "generation_source_distribution": dict(sorted(source_counter.items())),
        "generation_error_count": error_count,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_existing_utterances(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    utterances: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        messages = row.get("messages")
        if isinstance(messages, list) and len(messages) > 1:
            content = messages[1].get("content") if isinstance(messages[1], dict) else None
            if isinstance(content, str) and content.strip():
                utterances.add(" ".join(content.split()))
    return utterances


def count_existing_rows(output_path: Path) -> int:
    if not output_path.exists():
        return 0
    return sum(1 for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip())


def generate_dataset(
    output_path: Path,
    client: OpenAICompatibleClient | None,
    report_path: Path | None = None,
    samples: int | None = None,
    user_language: str = "mixed",
    seed: int = 42,
    workers: int = 1,
    temperature: float = 0.9,
    max_tokens: int = 160,
    max_retries: int = 3,
    dedupe_retries: int = 2,
    offline: bool = False,
    append: bool = True,
) -> int:
    rng = random.Random(seed)
    jobs = build_jobs(rng, total_samples=samples, user_language=user_language)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = count_existing_rows(output_path) if append else 0
    seen_utterances = load_existing_utterances(output_path) if append else set()
    seen_lock = threading.Lock()

    def _run_job(job: dict[str, Any]) -> dict[str, Any]:
        scenario = job["scenario"]
        label_data = job["label"]
        generation_source = "offline" if offline or client is None else "api"
        generation_error: str | None = None
        local_rng = random.Random(seed + int(job["idx"]) * 1009)
        if offline or client is None:
            utterance = format_offline_prompt(scenario["scenario"], label_data, job["occurrence_idx"], local_rng, job["diversity_profile"]["language"])
        else:
            utterance = ""
            attempts = max(1, int(dedupe_retries) + 1)
            try:
                for attempt in range(attempts):
                    candidate = generate_utterance(
                        client=client,
                        scenario=scenario,
                        label_data=label_data,
                        diversity_profile=job["diversity_profile"],
                        occurrence_idx=job["occurrence_idx"] + attempt * 1000,
                        temperature=max(0.2, min(1.2, temperature + job["temperature"] - 0.9 + attempt * 0.05)),
                        max_tokens=max_tokens,
                        max_retries=max_retries,
                    )
                    normalized = " ".join(candidate.split())
                    with seen_lock:
                        if normalized not in seen_utterances or attempt == attempts - 1:
                            seen_utterances.add(normalized)
                            utterance = candidate
                            break
            except Exception as exc:
                generation_source = "offline_fallback"
                generation_error = str(exc)
                utterance = format_offline_prompt(scenario["scenario"], label_data, job["occurrence_idx"], local_rng, job["diversity_profile"]["language"])

        sample = build_training_sample(utterance, label_data)
        sample["_idx"] = job["idx"]
        sample["scenario"] = scenario["scenario"]
        sample["generation_source"] = generation_source
        if generation_error:
            sample["generation_error"] = generation_error
        return sample

    completed = 0
    errors = 0
    total_jobs = len(jobs)
    mode = "a" if append else "w"
    print(
        f"starting generation: total={total_jobs}, existing={existing_rows}, mode={mode}, workers={max(1, int(workers))}, offline={offline}",
        flush=True,
    )
    with output_path.open(mode, encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [pool.submit(_run_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                sample = future.result()
                completed += 1
                if sample.get("generation_error"):
                    errors += 1
                    print(
                        f"[warn] idx={sample['_idx']} scenario={sample['scenario']} used offline fallback: {sample['generation_error']}",
                        flush=True,
                    )
                out = dict(sample)
                out.pop("_idx", None)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"[progress] completed={completed}/{total_jobs} written_this_run={completed} file_rows={existing_rows + completed} errors={errors} scenario={sample['scenario']}",
                    flush=True,
                )

    if report_path is not None:
        write_dataset_report(output_path, report_path)
    return len(jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diverse English device intent + slot extraction training data.")
    parser.add_argument("--output", type=Path, default=Path("data/device_intent_dataset.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/device_intent_dataset.stats.json"))
    parser.add_argument("--samples", type=int, default=None, help="Total number of samples to generate. Defaults to the built-in balanced scenario counts.")
    parser.add_argument("--user-language", choices=["english", "chinese", "mixed"], default="mixed", help="Language for generated user utterances. Assistant JSON string values stay normalized English.")
    parser.add_argument("--api-env", type=str, default="configs/data/api_generation.env")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--endpoint", choices=["chat.completions", "responses"], default="chat.completions")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dedupe-retries", type=int, default=2, help="Additional API retries when an exact duplicate utterance is generated.")
    parser.add_argument("--offline", action="store_true", help="Use deterministic local English templates instead of an API.")
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=True, help="Append to the output JSONL by default. Use --no-append to overwrite.")
    args = parser.parse_args()

    client: OpenAICompatibleClient | None = None
    if not args.offline:
        env_file = load_env_file(args.api_env)
        base_url = resolve_required("base_url", [args.base_url, env_file.get("DISTILL_API_BASE_URL"), env_file.get("OPENAI_BASE_URL"), os.getenv("DISTILL_API_BASE_URL"), os.getenv("OPENAI_BASE_URL")])
        api_key = resolve_required("api_key", [args.api_key, env_file.get("DISTILL_API_KEY"), env_file.get("OPENAI_API_KEY"), os.getenv("DISTILL_API_KEY"), os.getenv("OPENAI_API_KEY")])
        model = resolve_required("model", [args.model, env_file.get("DISTILL_API_MODEL"), env_file.get("OPENAI_MODEL"), os.getenv("DISTILL_API_MODEL"), os.getenv("OPENAI_MODEL")])
        client = OpenAICompatibleClient(base_url=base_url, api_key=api_key, model=model, endpoint=args.endpoint, timeout=args.timeout_sec)

    count = generate_dataset(
        output_path=args.output,
        client=client,
        report_path=args.report,
        samples=args.samples,
        user_language=args.user_language,
        seed=args.seed,
        workers=args.workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        dedupe_retries=args.dedupe_retries,
        offline=args.offline,
        append=args.append,
    )
    print(json.dumps({"output": str(args.output), "samples": count, "offline": args.offline, "append": args.append}, ensure_ascii=False))


if __name__ == "__main__":
    main()
