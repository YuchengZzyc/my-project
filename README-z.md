训练数据由大模型构建，包括我们的benchmark都是，这两个部分都比较重要，训练的多样性，tool use的格式的正确与否都很重要。



只需要在 configs/data/api_generation.env 配置相关的api和baseurl就可以生成数据了

```
python scripts/export_training_data_llm.py  --api-env configs/data/api_generation.env --output data/training_data_llm_onlytool.jsonl --n 1000 --endpoint responses
```

划分数据集

```
python scripts/split_jsonl_dataset.py --input data/trainData_v2.0.jsonl --train-output data/trainData_v2.0.train.jsonl --val-output data/trainData_v2.0.val.jsonl --val-ratio 0.1 --seed 42 --no-stratify-by-scenario
```

训练模型

```
python scripts/train_adapter.py --config configs/train/qwen35_4b_lora_v2.0.yaml
```

手动测评（开个聊天框，对话测试）

```
python scripts/chat_web_demo.py --model-path "E:\LLM\Qwen\Qwen3.5-4B" --adapter-path "E:\Yc_Zzzz\MADM\madm-llm\outputs\qwen35_4b_lora_v2.0"
```

训练后就是评测了，目前benchmark也是用模型生成的

```
python generate_benchmark_prompts_structured.py --seed-md benchmark/seed.md --output benchmark/prompts_structured.jsonl
```

最后就是评测了

```
python scripts/run_toolcall_param_eval.py --cases-jsonl benchmark/toolcall_param_cases.jsonl --model-path "E:\LLM\Qwen\Qwen3.5-4B" --adapter-path ".\outputs\qwen35_4b_lora_v1.4_stage2\checkpoint-202" --report benchmark/reports/toolcall_param_eval_Qwen35_4b_lora_v1.4_stage2.json

```

调用api的评测，可以用更大的模型，但是边缘部署的话可能就不太行，因为太大了

python scripts/run_toolcall_param_eval_api.py --api-env configs/data/api_generation.env --cases-jsonl benchmark/toolcall_param_cases.jsonl  --max-samples 300 --report benchmark/reports/toolcall_param_eval_api.json                  

暴露端口

cloudflared tunnel --url http://localhost:8009 --no-autoupdate

激活环境

conda activate llamafactory

API测试

python scripts/Test_API.py
