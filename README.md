# Reminder Tool-Use Training Workflow

## 1) Dataset

- Cleaned dataset: `data/training_data_llm.cleaned.jsonl`
- Cleaner script: `scripts/clean_training_data_llm.py`

Device intent + slot extraction data, without reminder tools or `tool_calls`:

```bash
python scripts/export_device_intent_data.py --output data/device_intent_data.jsonl
pytest -q tests/test_device_intent.py tests/test_export_device_intent_data.py
```

Generate a richer device intent dataset with the same pattern as the benchmark prompt generator:

```bash
python scripts/generate_device_intent_dataset.py --api-env configs/data/api_generation.env --output data/device_intent_dataset.jsonl --report data/device_intent_dataset.stats.json --samples 1000 --user-language mixed --dedupe-retries 2
```

`--samples` is the number of rows added by this run. The generator appends to `--output` by default; use `--no-append` to overwrite and rebuild the file. `--dedupe-retries 2` may make extra API calls when a generated utterance is duplicated. If one API item times out, that row falls back to a local template, records `generation_error`, and the rest of the dataset continues. Use `--user-language english`, `--user-language chinese`, or `--user-language mixed`; assistant JSON string values are always normalized English.

The generator keeps non-intent negative examples near one fifth of the dataset. The default offline distribution is `238` matched device-control rows and `60` negative rows (`20.1%` negative). With `--samples 1000`, the expected distribution is `799` matched rows and `201` negative rows.

For local verification without an API:

```bash
python scripts/generate_device_intent_dataset.py --offline --no-append --output data/device_intent_dataset.jsonl --report data/device_intent_dataset.stats.json
pytest -q tests/test_generate_device_intent_dataset.py
```

Summarize an existing device intent dataset:

```bash
python scripts/stat_device_intent_dataset.py --input data/device_intent_dataset.jsonl --format text
python scripts/stat_device_intent_dataset.py --input data/device_intent_dataset.jsonl --format text --examples-per-scene 3
python scripts/stat_device_intent_dataset.py --input data/device_intent_dataset.jsonl --report data/device_intent_dataset.stats.json
```

The report includes concrete scene distributions such as `08 调节音量`, `06 更换壁纸`, `04 开锁`, and negative scene groups, plus example utterances per scene.

The assistant label is JSON with:

```json
{"matched":true,"capability_id":8,"capability":"Adjust volume","intent":"set_volume","slots":{"adjustment":"up"},"missing_slots":[],"confidence":0.9}
```


python generate_benchmark_prompts_structured.py --seed-md benchmark/seed.md --output benchmark/prompts_structured.jsonl
python build_tool_eval_cases_from_structured_prompts.py --prompts-jsonl benchmark/prompts_structured.jsonl --output benchmark/tool_eval_cases_structured.jsonl
python init_tool_eval_dataset_dynamic_time.py --cases-jsonl benchmark/tool_eval_cases_structured.jsonl --out-dir benchmark/eval_runs --with-sqlite --overwrite --now 2026-05-26T09:00:00+08:00


python init_tool_eval_dataset_dynamic_time.py --cases-jsonl benchmark/tool_eval_cases_structured.jsonl --out-dir benchmark/eval_runs --with-sqlite --overwrite --now 2026-05-26T09:00:00+08:00
python scripts/run_tool_eval_from_dataset.py --cases-jsonl benchmark/tool_eval_cases_structured.jsonl --eval-dir benchmark/eval_runs --model-path "..." --adapter-path "..." --report benchmark/reports/qwen_580.json


python scripts/clean_training_data_llm.py --input data/training_data2.jsonl --output data/training_data2.cleaned.jsonl --rejects data/training_data2.rejects.jsonl --report data/training_data2.clean_report.json

python scripts/split_jsonl_dataset.py --input data/training_data2.cleaned.jsonl --train-output data/training_data2.train.jsonl --val-output data/training_data2.val.jsonl --val-ratio 0.1 --seed 42 --stratify-by-scenario


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

API few-shot tool-call parameter evaluation:

```bash
python scripts/run_toolcall_param_eval_api.py --api-env configs/data/api_generation.env --cases-jsonl benchmark/toolcall_param_cases.jsonl --max-samples 20 --report benchmark/reports/toolcall_param_eval_api.json
```

Or use the example YAML config:

```bash
python scripts/run_toolcall_param_eval_api.py --config configs/eval/toolcall_param_api_eval.example.yaml
```

The API endpoint should be OpenAI-compatible and provide `/chat/completions`. The script injects the four canonical reminder tools from `app.tool_registry` into the prompt and uses four few-shot examples by default. Add `--native-tools` when the provider supports OpenAI `tools` payloads.

##  4) web
```bash
python scripts/chat_web_demo.py --model-path "E:/LLM/Qwen/Qwen2.5-1.5B-Instruct" --adapter-path "outputs/qwen25_15b_dora/checkpoint-580" 
python scripts/chat_web_demo.py --model-path "E:/LLM/Qwen/Qwen2.5-1.5B-Instruct" --adapter-path "outputs/qwen25_15b_dora/checkpoint-800" --stateless 

Device intent recognition web demo with a simulated GUI and print trace:

```bash
python scripts/device_intent_web_demo.py --model-path "E:/LLM/Qwen/Qwen2.5-3B-Instruct" --adapter-path "outputs/qwen25_3b_lora/checkpoint-6260"
```

The demo uses the same `<SYSTEM>/<USER>/<ASSISTANT>` serialization format as `scripts/train_adapter.py`. If the model output parses to `matched=true` and has no missing slots, the backend prints a postprocess event to the terminal and updates the simulated device panel for the 13 supported capabilities. If the output is `matched=false`, no postprocess is executed. If JSON parsing fails, the raw model output is shown as the normal response.


```

mcp baseline:
```bash
python scripts/reminder_mcp_server_baseline.py --host 127.0.0.1 --port 8765
python scripts/web_mcp_baseline.py --model-path "E:/LLM/Qwen/Qwen2.5-1.5B-Instruct" --adapter-path "outputs/qwen25_15b_dora/checkpoint-580" --mcp-base-url "http://127.0.0.1:8765" --host 127.0.0.1 --port 8018
```

## 5) Local popup reminder watcher

Run one poll:

```bash
python scripts/run_reminder_notifier.py --once
```

Run continuously (default every 5 seconds):

```bash
python scripts/run_reminder_notifier.py --poll-interval 5
```



意图识别
python scripts/generate_device_intent_dataset.py --api-env configs/data/api_generation.env --output data/device_intent_dataset.jsonl --report data/device_intent_dataset.stats.json --samples 4000  --user-language english --dedupe-retries 2

python scripts/split_jsonl_dataset.py --input data/device_intent_dataset.jsonl --train-output data/device_intent_dataset.train.jsonl --val-output data/device_intent_dataset.val.jsonl --val-ratio 0.1 --seed 42 --no-stratify-by-scenario

python scripts/stat_device_intent_dataset.py --input data/device_intent_dataset.jsonl --format text


python scripts/train_adapter.py --config configs/train/qwen25_15b_lora_intent.yaml   

python scripts/device_intent_web_demo.py --model-path "E:/LLM/Qwen/Qwen2.5-1.5B-Instruct" --adapter-path ".\outputs\qwen25_15b_device_intent_lora\checkpoint-1000\"
