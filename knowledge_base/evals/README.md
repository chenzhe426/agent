# Evals – Knowledge Base Evaluation System

## 概述

可重复的参数回归测试系统，覆盖检索层和回答层，支持 SFT 数据导出。

## 目录结构

```
evals/
├── datasets/
│   ├── kb_eval_seed.jsonl     # 种子评测数据
│   └── financebench/           # FinanceBench 评测集
├── configs/
│   └── baseline.yaml          # 默认配置
├── reports/                   # 生成的报告
├── scripts/
│   ├── run_eval.py            # 评测运行
│   ├── run_financebench_eval.py  # FinanceBench 评测
│   ├── score_eval.py          # 评分
│   ├── compare_runs.py        # 回归对比
│   ├── gold/
│   │   ├── suggest_gold.py   # AI 推荐 gold
│   │   ├── review_candidates.py  # 查看候选 chunks
│   │   └── apply_gold.py     # 回填标注
│   └── analysis/
│       └── export_sft.py      # 导出 SFT 格式
└── utils/
    ├── dataset.py      # 数据 schema + 加载
    ├── adapters.py     # 内部/API 调用适配
    ├── scorer.py       # 评分逻辑
    └── report.py       # 报告生成
```

## 数据格式

每条样本为 JSONL 一行：

| 字段 | 说明 |
|------|------|
| `id` | 样本唯一 ID |
| `task_type` | `factoid` / `yesno` / `refuse` / `clarify` |
| `question.user_query` | 用户原始问题 |
| `question.conversation_history` | 多轮对话历史 |
| `retrieval.label_status` | 标注状态 |
| `retrieval.gold_chunk_ids` | 标准 Chunk ID（检索评测用） |
| `retrieval.gold_doc_ids` | 兜底 Doc ID |
| `answer.gold_answer` | 标准答案 |
| `answer.must_include` | 必须包含的关键词 |
| `answer.must_not_include` | 禁止包含的关键词 |
| `evaluation.expected_behavior` | `answer` / `refuse` / `clarify` |

### label_status

| 状态 | 检索评测 | 回答评测 |
|------|---------|---------|
| `unlabeled` | 跳过 | 是 |
| `labeled_doc` | 是（doc 级） | 是 |
| `labeled_chunk` | 是（chunk 级） | 是 |
| `unanswerable` | 跳过 | 是（refuse 评测） |

## 评测指标

### 检索层

| 指标 | 说明 |
|------|------|
| Hit@K | Top-K 是否命中任意 gold chunk |
| Recall@5 | Top-5 命中占总数比例 |
| MRR | 首个命中位置倒数均值 |

### 回答层

| Label | 条件 |
|-------|------|
| `exact` | 归一化后与 gold_answer 完全一致（≥85% token 重叠） |
| `partial` | `must_include` 满足 ≥50% 且无违规 |
| `wrong` | 既非 exact 也非 partial |
| `refuse_correct` | 期望拒答且答案含拒答语义 |
| `refuse_wrong` | 期望拒答但给出确定答案 |
| `clarify_correct` | 期望澄清且答案含澄清意图 |
| `clarify_wrong` | 期望澄清但未澄清 |

## 运行评测

```bash
# 基础评测
python evals/scripts/run_eval.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --config evals/configs/baseline.yaml \
    --output evals/reports/run_baseline.json

# FinanceBench 评测
python evals/scripts/run_financebench_eval.py \
    --config evals/configs/baseline.yaml \
    --output evals/reports/fb_eval.json

# 回归对比
python evals/scripts/compare_runs.py \
    --base evals/reports/run_baseline.json \
    --new evals/reports/run_exp.json
```

## Gold 标注流程

```bash
# 1. 运行评测
python evals/scripts/run_eval.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --config evals/configs/baseline.yaml \
    --output evals/reports/run_debug.json

# 2. AI 推荐 gold（可选）
python evals/scripts/gold/suggest_gold.py \
    --report evals/reports/run_debug.json \
    --only-need-labeling \
    --output evals/reports/suggestions.json

# 3. 查看候选 chunks
python evals/scripts/gold/review_candidates.py \
    --report evals/reports/run_debug.json \
    --only-need-labeling

# 4. 回填标注
python evals/scripts/gold/apply_gold.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --case kb_0001 \
    --gold-chunk 1 --gold-chunk 10 \
    --label-status labeled_chunk

# 5. 重新评测
python evals/scripts/run_eval.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --config evals/configs/baseline.yaml \
    --output evals/reports/run_after_labeling.json
```

## 配置说明

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `mode` | `internal`（直接调 Python）或 `api`（HTTP） | `internal` |
| `top_k` | 检索返回的 chunk 数量 | `5` |
| `normalize_text` | 答案比较前是否归一化 | `true` |
| `use_doc_level_fallback` | chunk 匹配失败时用 doc ID 兜底 | `true` |
| `api.base_url` | API 模式下的 FastAPI 地址 | `http://127.0.0.1:8000` |

## SFT 导出

```bash
python evals/scripts/analysis/export_sft.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --output evals/reports/sft_data.jsonl
```

## 设计原则

- **规则优先** – 所有评分逻辑规则-based，无 LLM 依赖，便于 CI 集成
- **零侵入** – 通过 `EvalAdapter` 封装，不修改主服务代码
- **可读报告** – JSON 供程序消费，Markdown 供人工审查
