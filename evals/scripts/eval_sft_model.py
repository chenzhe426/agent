#!/usr/bin/env python3
"""
使用 SFT 模型进行 RAG 评测

当 USE_SFT_SERVICE=true 时，RAG 层会调用 SFT 模型服务来生成答案

Usage:
    # 1. 先启动 SFT 模型服务
    cd model_serving && python scripts/serve.py

    # 2. 设置环境变量并运行评测
    USE_SFT_SERVICE=true python -m evals.scripts.run_eval \
        --dataset evals/data/financebench_v1_subset_3docs.jsonl \
        --config evals/configs/baseline.yaml \
        --output evals/reports/run_sft_$(date +%Y%m%d_%H%M%S).json
"""
import argparse
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="使用 SFT 模型进行 RAG 评测")
    parser.add_argument(
        "--dataset",
        required=True,
        help="评测数据集路径",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出报告路径",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发数",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索chunk数",
    )
    args = parser.parse_args()

    # 检查环境变量
    use_sft = os.getenv("USE_SFT_SERVICE", "false").lower() in {"1", "true", "yes", "on"}
    sft_url = os.getenv("SFT_SERVICE_BASE_URL", "http://localhost:8001")

    print("=" * 60)
    print("RAG + SFT Model Evaluation")
    print("=" * 60)
    print(f"USE_SFT_SERVICE: {use_sft}")
    print(f"SFT_SERVICE_BASE_URL: {sft_url}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print("=" * 60)

    if not use_sft:
        print("WARNING: USE_SFT_SERVICE is not set to true")
        print("Set: export USE_SFT_SERVICE=true")
        print()

    # 使用现有的 run_eval 脚本
    from evals.scripts import run_eval

    # 动态设置 sys.argv 以便复用 run_eval 的参数解析
    sys.argv = [
        "run_eval",
        "--dataset", args.dataset,
        "--config", "evals/configs/baseline.yaml",
        "--output", args.output,
        "--workers", str(args.workers),
    ]

    run_eval.main()


if __name__ == "__main__":
    main()
