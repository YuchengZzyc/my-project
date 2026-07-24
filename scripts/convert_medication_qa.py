import pandas as pd
import json
import os

# 1. 配置路径
# 请根据你实际下载的 parquet 文件路径进行修改
parquet_path = r"E:\DataSets\MedicationQA\data\train-00000-of-00001-7427a10e891759be.parquet"
output_jsonl_path = r"E:\DataSets\MedicationQA\data\medication_qa_dialog_en.jsonl"

# 2. 定义专业的英文 System Prompt
SYSTEM_PROMPT = (
    "You are a professional medical health assistant with expertise in pharmacology and pathology. "
    "Please provide accurate, evidence-based, and easy-to-understand answers to users' health inquiries. "
    "IMPORTANT: Your responses are for educational and informational purposes only and cannot replace "
    "professional medical diagnosis, treatment, or prescriptions."
)


def convert_to_jsonl(parquet_file, jsonl_file):
    # 读取 parquet 文件
    df = pd.read_parquet(parquet_file)

    # 兼容大小写列名
    if 'question' in df.columns and 'answer' in df.columns:
        df.rename(columns={'question': 'Question', 'answer': 'Answer'}, inplace=True)
    elif 'Question' not in df.columns or 'Answer' not in df.columns:
        raise ValueError("Parquet file does not contain 'Question' and 'Answer' columns.")

    # 转换为对话格式的字典列表
    dialog_data = []
    for _, row in df.iterrows():
        # 确保问题和答案都是字符串，且去除首尾空白
        question = str(row["Question"]).strip()
        answer = str(row["Answer"]).strip()

        message = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        dialog_data.append(message)

    # 写入 JSONL 文件
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for entry in dialog_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✅ Conversion complete! Processed {len(dialog_data)} records.")
    print(f"📁 File saved to: {jsonl_file}")


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    convert_to_jsonl(parquet_path, output_jsonl_path)