#!/usr/bin/env python3
"""
使用 SFT 模型对 FinanceBench 数据进行评测

Usage:
    # 1. 启动 SFT 模型服务
    cd model_serving && python scripts/serve.py &

    # 2. 运行评测
    USE_SFT_SERVICE=true python scripts/eval_financebench_sft.py \
        --dataset evals/data/financebench_v1_subset_3docs.jsonl \
        --output evals/reports/run_sft_financebench.json
"""
import argparse
import json
import sys
import os
from pathlib import Path

# 确保能导入 knowledge_base 的模块
ROOT_KB = Path(__file__).resolve().parent.parent.parent / "knowledge_base"
sys.path.insert(0, str(ROOT_KB))


def main():
    parser = argparse.ArgumentParser(description="使用 SFT 模型对 FinanceBench 评测")
    parser.add_argument("--dataset", required=True, help="数据集路径")
    parser.add_argument("--output", required=True, help="输出路径")
    parser.add_argument("--top-k", type=int, default=5, help="检索chunk数")
    parser.add_argument("--workers", type=int, default=1, help="并发数")
    args = parser.parse_args()

    use_sft = os.getenv("USE_SFT_SERVICE", "false").lower() in {"1", "true", "yes", "on"}

    print("=" * 60)
    print("FinanceBench Evaluation with SFT Model")
    print("=" * 60)
    print(f"USE_SFT_SERVICE: {use_sft}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print("=" * 60)

    if not use_sft:
        print("\n[ERROR] 请设置环境变量 USE_SFT_SERVICE=true")
        print("示例: USE_SFT_SERVICE=true python scripts/eval_financebench_sft.py ...")
        sys.exit(1)

    # 使用 RAG 层的评测脚本
    from evals.scripts.run_financebench_eval import main as run_fb_eval

    sys.argv = [
        "run_financebench_eval",
        "--dataset", args.dataset,
        "--output", args.output,
        "--top-k", str(args.top_k),
        "--workers", str(args.workers),
    ]

    run_fb_eval()


if __name__ == "__main__":
    main()
