from __future__ import annotations

import argparse
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainConfig:
    model_name_or_path: str
    train_file: str
    eval_file: str | None
    output_dir: str
    max_length: int
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    logging_steps: int
    save_steps: int
    eval_steps: int
    warmup_ratio: float
    weight_decay: float
    bf16: bool
    fp16: bool
    gradient_checkpointing: bool
    finetune_type: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    use_dora: bool
    use_4bit: bool
    seed: int
    system_key: str
    debug_sample: bool = False
    debug_index: int | None = None


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainConfig(
        model_name_or_path=raw["model"]["name_or_path"],
        train_file=raw["data"]["train_file"],
        eval_file=raw["data"].get("eval_file"),
        output_dir=raw["output"]["dir"],
        max_length=raw["train"]["max_length"],
        learning_rate=raw["train"]["learning_rate"],
        num_train_epochs=raw["train"]["num_train_epochs"],
        per_device_train_batch_size=raw["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=raw["train"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=raw["train"]["gradient_accumulation_steps"],
        logging_steps=raw["train"]["logging_steps"],
        save_steps=raw["train"]["save_steps"],
        eval_steps=raw["train"]["eval_steps"],
        warmup_ratio=raw["train"]["warmup_ratio"],
        weight_decay=raw["train"]["weight_decay"],
        bf16=raw["train"]["bf16"],
        fp16=raw["train"]["fp16"],
        gradient_checkpointing=raw["train"]["gradient_checkpointing"],
        finetune_type=raw["peft"]["finetune_type"],
        lora_r=raw["peft"]["lora_r"],
        lora_alpha=raw["peft"]["lora_alpha"],
        lora_dropout=raw["peft"]["lora_dropout"],
        lora_target_modules=raw["peft"]["target_modules"],
        use_dora=raw["peft"].get("use_dora", False),
        use_4bit=raw["peft"].get("use_4bit", False),
        seed=raw["train"]["seed"],
        system_key=raw["data"].get("system_key", "messages"),
        debug_sample=raw.get("debug", {}).get("print_sample", False),
        debug_index=raw.get("debug", {}).get("sample_index"),
    )


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def load_tokenizer(model_path: str) -> Any:
    # Qwen2.5 local snapshots often work with fast tokenizer only.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def to_text_sample(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    tools = row.get("tools", [])

    lines = ["<TOOLS>", json.dumps(tools, ensure_ascii=False), "</TOOLS>"]
    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant" and msg.get("tool_calls"):
            lines.append("<ASSISTANT_TOOL_CALL>")
            lines.append(json.dumps(msg, ensure_ascii=False))
            lines.append("</ASSISTANT_TOOL_CALL>")
            continue
        content = msg.get("content")
        if role == "tool":
            lines.append("<TOOL>")
            lines.append(json.dumps(msg, ensure_ascii=False))
            lines.append("</TOOL>")
        else:
            lines.append(f"<{role.upper()}>")
            lines.append("" if content is None else str(content))
            lines.append(f"</{role.upper()}>")
    return "\n".join(lines)


def print_debug_sample(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    sample_index: int | None,
) -> None:
    if not rows:
        print("[debug] no rows available")
        return
    idx = sample_index if sample_index is not None else random.randint(0, len(rows) - 1)
    idx = max(0, min(idx, len(rows) - 1))
    row = rows[idx]
    text = to_text_sample(row)
    encoded = tokenizer(text, truncation=True, max_length=max_length, padding=False)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    batch = collator([{"input_ids": encoded["input_ids"], "attention_mask": encoded.get("attention_mask")}])
    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    visible_label_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    decoded_labels_visible_text = tokenizer.decode(visible_label_ids, skip_special_tokens=False)
    label_non_ignore_count = sum(1 for x in labels if x != -100)

    has_tool_call_tag = "<ASSISTANT_TOOL_CALL>" in text
    has_tool_calls_key = '"tool_calls"' in text
    has_function_name = '"name"' in text
    has_function_arguments = '"arguments"' in text
    has_tool_call_tag_in_labels = "<ASSISTANT_TOOL_CALL>" in decoded_labels_visible_text
    has_function_name_in_labels = '"name"' in decoded_labels_visible_text
    has_function_arguments_in_labels = '"arguments"' in decoded_labels_visible_text

    print("\n[debug] ===== Serialized Training Sample =====")
    print(f"[debug] sample_index={idx}")
    print(f"[debug] has_tool_call_tag={has_tool_call_tag}")
    print(f"[debug] has_tool_calls_key={has_tool_calls_key}")
    print(f"[debug] has_function_name={has_function_name}")
    print(f"[debug] has_function_arguments={has_function_arguments}")
    print(f"[debug] token_count={len(encoded['input_ids'])}")
    print(f"[debug] label_non_ignore_count={label_non_ignore_count}")
    print(f"[debug] has_tool_call_tag_in_labels={has_tool_call_tag_in_labels}")
    print(f"[debug] has_function_name_in_labels={has_function_name_in_labels}")
    print(f"[debug] has_function_arguments_in_labels={has_function_arguments_in_labels}")
    print("[debug] ---- serialized text ----")
    print(text[:4000])
    print("[debug] ---- tokenized->decoded preview ----")
    print(decoded[:1200])
    print("[debug] ---- decoded_labels_visible_text preview ----")
    print(decoded_labels_visible_text[:2000])
    print("[debug] =====================================\n")


def build_dataset(tokenizer: Any, rows: list[dict[str, Any]], max_length: int) -> Dataset:
    texts = [to_text_sample(r) for r in rows]
    ds = Dataset.from_dict({"text": texts})

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return ds.map(tokenize, batched=True, remove_columns=["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven LoRA/DoRA training entry.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--debug-sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--debug-index", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)

    tokenizer = load_tokenizer(cfg.model_name_or_path)

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if cfg.bf16:
        model_kwargs["dtype"] = torch.bfloat16
    elif cfg.fp16:
        model_kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if cfg.finetune_type.lower() in {"lora", "dora", "qlora"}:
        want_dora = cfg.use_dora or cfg.finetune_type.lower() == "dora"
        lora_kwargs: dict[str, Any] = {
            "task_type": TaskType.CAUSAL_LM,
            "r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "target_modules": cfg.lora_target_modules,
            "bias": "none",
        }
        supports_use_dora = "use_dora" in inspect.signature(LoraConfig).parameters
        if want_dora and supports_use_dora:
            lora_kwargs["use_dora"] = True
        elif want_dora and not supports_use_dora:
            raise RuntimeError(
                "Current peft version does not support DoRA (missing LoraConfig.use_dora). "
                "Please upgrade peft (pip install -U peft) or switch to LoRA config."
            )

        peft_cfg = LoraConfig(**lora_kwargs)
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    train_rows = load_jsonl(cfg.train_file)
    eval_rows = load_jsonl(cfg.eval_file) if cfg.eval_file else []

    debug_sample = cfg.debug_sample if args.debug_sample is None else args.debug_sample
    debug_index = cfg.debug_index if args.debug_index is None else args.debug_index
    if debug_sample:
        print_debug_sample(
            rows=train_rows,
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            sample_index=debug_index,
        )
        print("\n[debug] Sample check finished. Skip training.")
        # return 返回 输入--debug -sample时，不进行训练
        return

    train_ds = build_dataset(tokenizer, train_rows, cfg.max_length)
    eval_ds = build_dataset(tokenizer, eval_rows, cfg.max_length) if eval_rows else None

    ta_kwargs: dict[str, Any] = {
        "output_dir": cfg.output_dir,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "eval_steps": cfg.eval_steps,
        "save_strategy": "steps",
        "weight_decay": cfg.weight_decay,
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "report_to": [],
        "seed": cfg.seed,
    }
    eval_value = "steps" if eval_ds is not None else "no"
    ta_sig = inspect.signature(TrainingArguments)
    if "warmup_ratio" in ta_sig.parameters:
        ta_kwargs["warmup_ratio"] = cfg.warmup_ratio
    elif "warmup_steps" in ta_sig.parameters:
        total_steps_est = max(
            int(
                (len(train_ds) / max(cfg.per_device_train_batch_size, 1))
                / max(cfg.gradient_accumulation_steps, 1)
                * cfg.num_train_epochs
            ),
            1,
        )
        ta_kwargs["warmup_steps"] = int(total_steps_est * cfg.warmup_ratio)
    if "evaluation_strategy" in ta_sig.parameters:
        ta_kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in ta_sig.parameters:
        ta_kwargs["eval_strategy"] = eval_value

    training_args = TrainingArguments(**ta_kwargs)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print(f"Training done. output_dir={cfg.output_dir}")


if __name__ == "__main__":
    main()
