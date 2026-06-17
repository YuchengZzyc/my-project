from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CAPABILITY_SCENE_NAMES = {
    "1": "01 呼叫管理员",
    "2": "02 呼叫联系人",
    "3": "03 接听来电",
    "4": "04 开锁",
    "5": "05 拒接/挂断",
    "6": "06 更换壁纸",
    "7": "07 切换勿扰模式",
    "8": "08 调节音量",
    "9": "09 调节屏幕亮度",
    "10": "10 布防/撤防",
    "11": "11 查询告警记录",
    "12": "12 监控管理",
    "13": "13 查看通知/短信",
}

NEGATIVE_SCENE_NAMES = {
    "unsupported_device_control": "负例-不支持的控制请求",
    "ambiguous_device_control": "负例-模糊控制请求",
    "no_intent_daily_chat": "负例-普通闲聊",
    "no_intent_device_words": "负例-含设备词但非控制",
    "no_intent_general_question": "负例-普通问题/陪伴",
}

DETAILED_SCENE_NAMES = {
    "call_manager_property": "01 呼叫管理员-物业/管理员",
    "call_contact_family": "02 呼叫联系人-家人",
    "call_contact_frequent": "02 呼叫联系人-常用联系人",
    "call_contact_service": "02 呼叫联系人-服务电话",
    "call_contact_missing": "02 呼叫联系人-缺少联系人槽位",
    "answer_call": "03 接听来电",
    "unlock_remote": "04 开锁-远程开锁",
    "unlock_during_call": "04 开锁-通话中开锁",
    "reject_call": "05 拒接/挂断-拒接",
    "hangup_call": "05 拒接/挂断-挂断",
    "wallpaper_type": "06 更换壁纸-指定类型",
    "wallpaper_missing_type": "06 更换壁纸-缺少壁纸类型",
    "dnd_on": "07 切换勿扰模式-开启",
    "dnd_off": "07 切换勿扰模式-关闭",
    "volume_adjust": "08 调节音量",
    "brightness_adjust": "09 调节屏幕亮度",
    "security_on": "10 布防/撤防-布防",
    "security_off": "10 布防/撤防-撤防",
    "alarm_query": "11 查询告警记录",
    "monitor_control": "12 监控管理",
    "monitor_missing_action": "12 监控管理-缺少动作槽位",
    "view_messages": "13 查看通知/短信",
    **NEGATIVE_SCENE_NAMES,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row must be a JSON object")
        rows.append(row)
    return rows


def get_assistant_label(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {row_index}: missing messages list")
    assistant = messages[-1]
    if not isinstance(assistant, dict):
        raise ValueError(f"row {row_index}: last message must be an object")
    content = assistant.get("content")
    if not isinstance(content, str):
        raise ValueError(f"row {row_index}: assistant content must be a JSON string")
    try:
        label = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"row {row_index}: assistant content is not valid JSON: {exc}") from exc
    if not isinstance(label, dict):
        raise ValueError(f"row {row_index}: assistant content must decode to an object")
    return label


def user_utterance(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and len(messages) > 1 and isinstance(messages[1], dict):
        content = messages[1].get("content")
        if isinstance(content, str):
            return content
    return ""


def scene_name(row: dict[str, Any], label: dict[str, Any]) -> str:
    if label.get("matched") is True:
        capability_id = str(label.get("capability_id"))
        return CAPABILITY_SCENE_NAMES.get(capability_id, f"未知正例能力-{capability_id}")
    scenario = str(row.get("scenario", "__missing__"))
    return NEGATIVE_SCENE_NAMES.get(scenario, f"负例-其他/{scenario}")


def detailed_scene_name(row: dict[str, Any], label: dict[str, Any]) -> str:
    scenario = str(row.get("scenario", "__missing__"))
    if scenario in DETAILED_SCENE_NAMES:
        return DETAILED_SCENE_NAMES[scenario]
    return scene_name(row, label)


def add_example(examples: dict[str, list[str]], key: str, utterance: str, limit: int) -> None:
    if limit <= 0 or not utterance:
        return
    bucket = examples.setdefault(key, [])
    if len(bucket) < limit and utterance not in bucket:
        bucket.append(utterance)


def build_report(rows: list[dict[str, Any]], top_duplicates: int = 20, examples_per_scene: int = 2) -> dict[str, Any]:
    labels = [get_assistant_label(row, idx) for idx, row in enumerate(rows, start=1)]
    total = len(rows)
    matched_counter = Counter(str(label.get("matched")) for label in labels)
    negative = matched_counter.get("False", 0)
    scenario_counter = Counter(str(row.get("scenario", "__missing__")) for row in rows)
    scene_counter: Counter[str] = Counter()
    detailed_scene_counter: Counter[str] = Counter()
    scene_examples: dict[str, list[str]] = {}
    detailed_scene_examples: dict[str, list[str]] = {}
    capability_counter = Counter(str(label.get("capability_id")) for label in labels)
    intent_counter = Counter(str(label.get("intent")) for label in labels)
    source_counter = Counter(str(row.get("generation_source", "unknown")) for row in rows)
    error_count = sum(1 for row in rows if row.get("generation_error"))

    missing_slot_counter: Counter[str] = Counter()
    for label in labels:
        missing_slots = label.get("missing_slots")
        if isinstance(missing_slots, list):
            for slot in missing_slots:
                missing_slot_counter[str(slot)] += 1

    utterances = []
    for row, label in zip(rows, labels):
        utterance = user_utterance(row)
        utterances.append(utterance)
        scene = scene_name(row, label)
        detailed_scene = detailed_scene_name(row, label)
        scene_counter[scene] += 1
        detailed_scene_counter[detailed_scene] += 1
        add_example(scene_examples, scene, utterance, examples_per_scene)
        add_example(detailed_scene_examples, detailed_scene, utterance, examples_per_scene)

    duplicates = {
        utterance: count
        for utterance, count in Counter(utterances).most_common()
        if utterance and count > 1
    }
    duplicate_items = list(duplicates.items())[: max(0, top_duplicates)]

    return {
        "total": total,
        "matched_true": matched_counter.get("True", 0),
        "matched_false": negative,
        "negative_ratio": (negative / total) if total else 0.0,
        "scene_distribution": dict(sorted(scene_counter.items())),
        "detailed_scene_distribution": dict(sorted(detailed_scene_counter.items())),
        "scene_examples": dict(sorted(scene_examples.items())),
        "detailed_scene_examples": dict(sorted(detailed_scene_examples.items())),
        "scenario_distribution": dict(sorted(scenario_counter.items())),
        "capability_distribution": dict(sorted(capability_counter.items())),
        "intent_distribution": dict(sorted(intent_counter.items())),
        "missing_slot_distribution": dict(sorted(missing_slot_counter.items())),
        "generation_source_distribution": dict(sorted(source_counter.items())),
        "generation_error_count": error_count,
        "exact_duplicate_utterance_count": sum(duplicates.values()) - len(duplicates),
        "top_duplicate_utterances": dict(duplicate_items),
    }


def print_table(title: str, data: dict[str, int]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not data:
        print("(empty)")
        return
    key_width = max(len(str(key)) for key in data)
    for key, value in data.items():
        print(f"{key:<{key_width}}  {value}")


def print_examples(title: str, examples: dict[str, list[str]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not examples:
        print("(empty)")
        return
    for key, values in examples.items():
        print(f"{key}:")
        for value in values:
            print(f"  - {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize scenario distribution for a device intent JSONL dataset.")
    parser.add_argument("--input", type=Path, default=Path("data/device_intent_dataset.jsonl"))
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.add_argument("--top-duplicates", type=int, default=20)
    parser.add_argument("--examples-per-scene", type=int, default=2)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    report = build_report(rows, top_duplicates=args.top_duplicates, examples_per_scene=args.examples_per_scene)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"total: {report['total']}")
        print(f"matched_true: {report['matched_true']}")
        print(f"matched_false: {report['matched_false']}")
        print(f"negative_ratio: {report['negative_ratio']:.4f}")
        print_table("scene_distribution", report["scene_distribution"])
        print_table("detailed_scene_distribution", report["detailed_scene_distribution"])
        print_examples("scene_examples", report["scene_examples"])
        print_table("scenario_distribution", report["scenario_distribution"])
        print_table("capability_distribution", report["capability_distribution"])
        print_table("intent_distribution", report["intent_distribution"])
        print_table("missing_slot_distribution", report["missing_slot_distribution"])
        print_table("generation_source_distribution", report["generation_source_distribution"])
        print(f"\ngeneration_error_count: {report['generation_error_count']}")
        print(f"exact_duplicate_utterance_count: {report['exact_duplicate_utterance_count']}")


if __name__ == "__main__":
    main()
