"""
评估训练好的模型：使用evidence和问题测评是否能正确回答
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

# 配置
BASE_MODEL_PATH = "/home/cz/python/knowledge_base/models/Qwen2.5-7B-Instruct"
LORA_ADAPTER_PATH = "/home/cz/python/finance_sft/output/qwen25_finance_lora"
DATA_PATH = "/home/cz/python/finance_sft/data/financebench_train_sharegpt.jsonl"

SYSTEM_PROMPT = "You are a professional financial analyst assistant. Answer the question based ONLY on the evidence provided below. Cite specific numbers and facts from the evidence when available. Do not make up information not present in the evidence."


def load_model():
    """加载模型和tokenizer"""
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    print("加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print("加载LoRA适配器...")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
    model.eval()

    return model, tokenizer


def build_prompt(evidence_texts, question):
    """构建输入提示"""
    # 组合所有evidence
    evidence_combined = "\n\n---\n\n".join(evidence_texts)

    # Qwen chat template 格式
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nQuestion: {question}\n\nEvidence:\n{evidence_combined}\n\nPlease answer the question based on the evidence above.<|im_end|>\n<|im_start|>assistant\n"

    return prompt


def extract_answer(response, prompt):
    """从模型回复中提取答案部分"""
    # 找到assistant回复的开始位置
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    return response.strip()


def normalize_text(text):
    """文本归一化用于比较"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def evaluate_sample(model, tokenizer, evidence_texts, question, gold_answer):
    """评估单个样本"""
    prompt = build_prompt(evidence_texts, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"DEBUG - Full response length: {len(response)}")
    print(f"DEBUG - Response starts with: {response[:200]}")

    # 提取模型的回答
    model_answer = extract_answer(response, prompt)

    # 比较答案
    gold_normalized = normalize_text(gold_answer)
    model_normalized = normalize_text(model_answer)

    # 简单的包含检查
    is_correct = gold_normalized in model_normalized or model_normalized in gold_normalized

    return {
        "question": question[:100] + "..." if len(question) > 100 else question,
        "gold_answer": gold_answer,
        "model_answer": model_answer,
        "is_correct": is_correct
    }


def run_evaluation():
    """运行评估"""
    # 加载模型
    model, tokenizer = load_model()

    # 加载数据
    print(f"\n加载数据: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print(f"共 {len(data)} 条样本\n")

    # 评估
    results = []
    correct = 0

    for i, sample in enumerate(data):
        question = sample.get("question", "") or ""
        evidence_texts = sample.get("evidence_texts", [])
        gold_answer = sample.get("gold_answer", sample.get("conversations", [{}])[-1].get("value", ""))

        # 如果question为空，尝试从evidence_texts提取
        if not question and evidence_texts:
            # 从第一个evidence中提取Question
            match = re.search(r'Question:\s*(.*?)(?:\n\nEvidence:|$)', evidence_texts[0], re.DOTALL)
            if match:
                question = match.group(1).strip()

        if not question or not gold_answer:
            print(f"样本 {i+1}: 跳过（无问题或无标准答案）")
            continue

        print(f"评估样本 {i+1}/{len(data)}...")

        result = evaluate_sample(model, tokenizer, evidence_texts, question, gold_answer)
        results.append(result)

        if result["is_correct"]:
            correct += 1
            print(f"  ✓ 正确")
        else:
            print(f"  ✗ 错误")

        # 打印前3个样本的详细结果
        if i < 3 or not result["is_correct"]:
            print(f"  问题: {result['question'][:80]}...")
            print(f"  标准答案: {result['gold_answer'][:80]}...")
            print(f"  模型答案: {result['model_answer'][:80]}...")
            print()

    # 打印总结
    print("\n" + "="*60)
    print("评估总结")
    print("="*60)
    accuracy = correct / len(results) * 100 if results else 0
    print(f"总样本数: {len(results)}")
    print(f"正确数: {correct}")
    print(f"准确率: {accuracy:.2f}%")

    # 保存详细结果
    output_path = "/home/cz/python/finance_sft/evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": len(results),
                "correct": correct,
                "accuracy": accuracy
            },
            "details": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_path}")

    return results


if __name__ == "__main__":
    run_evaluation()
