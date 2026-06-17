from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.device_intent import DEVICE_CAPABILITIES, build_device_intent_label


SYSTEM_PROMPT = (
    "你是设备控制意图识别与槽位抽取器。"
    "只输出JSON，不要调用工具。"
    "支持的能力只有编号1到13的设备能力：呼叫管理员、呼叫联系人、接听来电、开锁、拒接/挂断、"
    "更换壁纸、切换勿扰模式、调节音量、调节屏幕亮度、布防/撤防、查询告警记录、监控管理、查看通知/短信。"
    "输出字段必须包含 capability_id、capability、intent、slots、missing_slots、confidence。"
    "如果不是这些设备控制请求，capability_id、capability、intent 为 null，slots 为空对象。"
)


DEVICE_INTENT_CASES: list[str] = [
    "帮我呼叫管理员",
    "联系一下物业管理员",
    "给家人打电话",
    "帮我拨打常用联系人",
    "帮我拨打服务电话",
    "呼叫张阿姨",
    "接听电话",
    "来电时帮我语音接听",
    "拒接这个电话",
    "不要接这个来电",
    "把电话挂了",
    "结束通话",
    "帮我开门",
    "远程开锁",
    "通话中帮我开锁",
    "换成风景壁纸",
    "把壁纸改成深色",
    "切换成家庭壁纸",
    "打开免打扰",
    "关闭勿扰模式",
    "把声音调大一点",
    "提高音量",
    "声音调低一点",
    "音量调到60",
    "音量调到最大",
    "设置成静音",
    "屏幕调亮一点",
    "把亮度调暗",
    "亮度调到80",
    "屏幕亮度最亮",
    "开启布防",
    "撤防",
    "关闭安防",
    "查一下今天的告警记录",
    "看看最近一周有没有报警",
    "查询上个月的告警",
    "打开门口监控",
    "关闭客厅监控",
    "查看卧室场景",
    "回放昨天的录像",
    "切换到3通道",
    "看一下未读短信",
    "打开新通知",
    "查看新消息和通知",
]


NO_INTENT_CASES: list[str] = [
    "今天天气不错",
    "我有点想聊天",
    "明天可能去散步",
    "你觉得晚饭吃什么好",
    "我刚才看了电视",
]


def build_sample(text: str) -> dict[str, Any]:
    return {
        "task": "device_intent_slot_extraction",
        "schema_version": "device_intent_v1",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {
                "role": "assistant",
                "content": json.dumps(build_device_intent_label(text), ensure_ascii=False, separators=(",", ":")),
            },
        ],
    }


def generate_samples() -> list[dict[str, Any]]:
    samples = [build_sample(text) for text in DEVICE_INTENT_CASES]
    samples.extend(build_sample(text) for text in NO_INTENT_CASES)
    return samples


def export_jsonl(output_path: Path) -> int:
    samples = generate_samples()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples) + "\n",
        encoding="utf-8",
    )
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export device-control intent and slot extraction data.")
    parser.add_argument("--output", type=Path, default=Path("data/device_intent_data.jsonl"))
    parser.add_argument("--print-capabilities", action="store_true")
    args = parser.parse_args()

    if args.print_capabilities:
        print(json.dumps(DEVICE_CAPABILITIES, ensure_ascii=False, indent=2))
        return

    count = export_jsonl(args.output)
    print(json.dumps({"output": str(args.output), "samples": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
