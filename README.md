# Reminder Tool-Use Training Workflow

## 1) Dataset

- Cleaned dataset: `data/training_data_llm.cleaned.jsonl`
- Cleaner script: `scripts/clean_training_data_llm.py`

```bash
python scripts/export_training_data_llm.py  --api-env configs/data/api_generation.env --output data/training_data_llm_onlytool.jsonl --n 1000 --endpoint responses
```

## 2) Config-Driven LoRA/DoRA Training

Train with LoRA:

```bash
python scripts/train_adapter.py --config configs/train/qwen25_15b_lora.yaml
```

Train with DoRA:

```bash
python scripts/train_adapter.py --config configs/train/qwen25_15b_dora.yaml
```

Only edit YAML to switch backbone, dataset, output path, and PEFT settings.

## 3) Quick Tool-Use Evaluation

```bash
python scripts/eval_tooluse_model.py --config configs/eval/qwen25_15b_tooluse_eval.yaml
```

The report is written to the `output.report_file` configured in YAML.

##  4) web
```bash
python scripts/chat_web_demo.py --model-path "E:/LLM/Qwen/Qwen2.5-1.5B-Instruct" --adapter-path "outputs/qwen25_15b_dora/checkpoint-580" 
```

mcp baseline:
```bash
python scripts/reminder_mcp_server_baseline.py --host 127.0.0.1 --port 8765
python scripts/web_mcp_baseline.py --model-path "E:/LLM/Qwen/Qwen2.5-1.5B-Instruct" --adapter-path "outputs/qwen25_15b_dora/checkpoint-580" --mcp-base-url "http://127.0.0.1:8765" --host 127.0.0.1 --port 8018
```