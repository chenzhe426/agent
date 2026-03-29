# Knowledge Base System

本地知识库 RAG 系统，支持**文档导入→索引→检索→问答→验证→Agent 自主推理**全流程，以及配套的**评测系统**、**SFT 训练**和**模型服务**。

---

## 目录

- [系统架构总览](#系统架构总览)
- [子系统概览](#子系统概览)
- [快速上手](#快速上手)
- [Knowledge Base (RAG)](#knowledge-base-rag)
  - [技术栈](#技术栈)
  - [完整 Pipeline](#完整-pipeline)
  - [模块详解](#模块详解)
  - [数据库 Schema](#数据库-schema)
  - [API 接口](#api-接口)
  - [Agent 系统](#agent-系统)
  - [工具列表](#工具列表)
  - [配置说明](#配置说明)
- [Evals (评测系统)](#evals-评测系统)
  - [目录结构](#评测目录结构)
  - [EvalSample 数据格式](#evalsample-数据格式)
  - [评测指标](#评测指标)
  - [评测流程](#评测流程)
  - [Gold 标注流程](#gold-标注流程)
  - [评测配置](#评测配置)
- [Model Serving (SFT 服务)](#model-serving-sft-服务)
  - [目录结构](#serving-目录结构)
  - [API 接口](#serving-api-接口)
  - [LoRA 动态切换](#lora-动态切换)
  - [启动服务](#启动-sft-服务)
- [Finance SFT (训练)](#finance-sft-训练)
  - [目录结构](#sft-目录结构)
  - [训练流程](#训练流程)
- [端到端流程](#端到端流程)
- [版本历史](#版本历史)

---

## 系统架构总览

```
/home/cz/agent/
├── knowledge_base/              # ─────────── RAG 核心 ───────────
│   ├── app/
│   │   ├── api.py            # FastAPI (17 路由)
│   │   ├── main.py           # CLI (12 子命令)
│   │   ├── ingestion/        # 文档解析 (PDF/DOCX/TXT)
│   │   ├── services/         # 索引服务 (分块/向量/重排/记忆)
│   │   ├── retrieval/        # 混合检索 (5 路召回)
│   │   ├── qa/              # 问答生成 (verify + refine)
│   │   ├── tools/           # Agent Tools (11 个)
│   │   ├── agent/           # LangGraph Agent (5 节点)
│   │   └── db/              # MySQL + Qdrant
│   └── scripts/
│       └── export_sft_data.py  # SFT 数据导出
│
├── evals/                     # ─────────── 评测系统 ───────────
│   ├── scripts/
│   │   ├── run_eval.py          # 通用评测运行
│   │   ├── run_financebench_eval.py  # FinanceBench 评测
│   │   ├── score_eval.py        # 多层评分 (doc/page/section/evidence/answer)
│   │   ├── enrich_with_evidence.py  # 回填 gold evidence 文本
│   │   ├── gold/                # AI 辅助标注 → 人工审核 → 回填
│   │   ├── analysis/            # SFT 导出 / 回归对比
│   │   └── train_sft.py         # 多后端 SFT 训练入口
│   ├── utils/
│   │   ├── dataset.py          # EvalSample schema + load_dataset
│   │   ├── adapters.py         # EvalAdapter (internal/API 双模式)
│   │   ├── scorer.py           # RetrievalScorer + AnswerScorer
│   │   └── report.py           # JSON/Markdown 报告构建
│   ├── configs/
│   │   ├── baseline.yaml       # 基础评测配置
│   │   ├── retrieval_v4.yaml    # V4 pipeline 评测配置
│   │   └── sft_llamafactory.yaml  # SFT 训练配置
│   └── training/              # 训练输出
│
├── model_serving/             # ─────────── SFT 模型服务 ───────────
│   ├── service/
│   │   ├── app.py            # FastAPI (Ollama 兼容)
│   │   └── model.py          # SFTModel 单例 (LoRA 动态加载)
│   ├── scripts/
│   │   ├── serve.py          # 服务入口
│   │   ├── eval_financebench_sft.py  # SFT 模型评测包装
│   │   └── check_model.py    # 健康检查
│   └── configs/
│       └── serve_config.yaml  # 模型 + LoRA + 服务配置
│
└── finance_sft/               # ─────────── SFT 训练 ───────────
    ├── configs/
    │   └── train.yaml        # LlamaFactory 训练配置
    ├── data/                  # 训练数据 (ShareGPT)
    ├── scripts/
    │   └── eval_model.py    # 直接评测 (无需 serving)
    └── output/
        └── qwen25_finance_lora/  # 训练产出的 LoRA adapter
```

---

## 子系统概览

| 子系统 | 入口 | 职责 |
|--------|------|------|
| **Knowledge Base** | `knowledge_base/app/api.py` + `main.py` | 文档导入 → 解析 → 索引 → 检索 → 问答 → Agent 推理 |
| **Evals** | `evals/scripts/run_eval.py` | 检索/问答质量评测、数据标注、SFT 数据导出 |
| **Model Serving** | `model_serving/scripts/serve.py` | SFT 模型 HTTP 服务，支持 Ollama 兼容接口 + LoRA 动态切换 |
| **Finance SFT** | `finance_sft/scripts/eval_model.py` | LoRA SFT 训练 + 直接评测 (无需 serving) |

---

## 快速上手

### 1. 初始化 Knowledge Base

```bash
cd /home/cz/agent/knowledge_base

# 初始化数据库
python -m app.main init-db

# 导入文档
python -m app.main import data/

# 索引
python -m app.main index-all

# 启动 API
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### 2. 运行评测

```bash
cd /home/cz/agent

# 评测 (internal 模式)
python evals/scripts/run_eval.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --config evals/configs/baseline.yaml \
    --output evals/reports/run_baseline.json

# 评分 (多层指标)
python evals/scripts/score_eval.py \
    --run evals/reports/run_baseline.jsonl \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --output evals/reports/run_baseline_scored.json
```

### 3. 导出 SFT 数据

```bash
python evals/scripts/analysis/export_sft.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --from-evidence \
    --output evals/reports/sft_data.jsonl
```

### 4. 训练 LoRA

```bash
python evals/scripts/train_sft.py \
    --backend llamafactory \
    --data evals/reports/sft_data.jsonl \
    --output finance_sft/output/qwen25_finance_lora
```

### 5. 启动 SFT 服务

```bash
cd /home/cz/agent/model_serving

# 设置 LoRA adapter 路径
export SFT_LORA_PATH=/home/cz/agent/finance_sft/output/qwen25_finance_lora

# 启动服务 (端口 8001)
python scripts/serve.py
```

### 6. 用 SFT 模型评测

```bash
# 方式 A: 通过 model_serving 包装脚本
USE_SFT_SERVICE=true \
SFT_SERVICE_BASE_URL=http://localhost:8001 \
python model_serving/scripts/eval_financebench_sft.py \
    --output evals/reports/run_sft.json

# 方式 B: 直接评测 (无需服务)
python finance_sft/scripts/eval_model.py
```

---

## Knowledge Base (RAG)

### 技术栈

| 组件 | 技术 |
|------|------|
| API 框架 | FastAPI |
| LLM | Ollama (qwen3:8b) |
| Embedding | Ollama (nomic-embed-text, 768 维) |
| 向量检索 | Qdrant |
| 关键词检索 | MySQL FULLTEXT + BM25 |
| 数据库 | MySQL |
| Agent 框架 | LangGraph + LangChain Core Tools |
| 文档解析 | PyMuPDF (PDF), python-docx (DOCX), 内置 (TXT) |

---

### 完整 Pipeline

#### Pipeline 1: 文档导入 → 索引

```
文件 (PDF/DOCX/TXT)
    │
    ▼
ingestion/pipeline.py: parse_document()
    ├── detectors.py: detect_file_type()
    ├── parsers/pdf_parser.py: PyMuPDF 多线程解析
    ├── parsers/docx_parser.py: Docling + python-docx 双引擎
    └── parsers/text_parser.py: 智能结构化
    │
    ▼
services/document_service.py: import_single_document()
    │
    ▼
services/chunk_service.py: index_document()
    ├── split_blocks_into_chunks()   # 金融感知分块
    │   ├── text_chunk
    │   ├── table_chunk
    │   └── table_linearized_chunk
    ├── _build_finance_search_text()  # 金融搜索文本模板
    ├── diff by chunk_hash           # 增量索引
    ├── get_embeddings_batch()       # Ollama 批量 embedding
    │
    ▼
MySQL (document_chunks) + Qdrant (vectors)
```

#### Pipeline 2: 检索 → 重排

```
用户 query
    │
    ▼
retrieval/service.py: retrieve_chunks()
    │
    ├─ query_understanding.py: enhance_financial_query()
    │   └── 金融 query 扩展 (ratio → component terms)
    │
    ▼
多路召回 (并行)
    ├─ _vector_recall_from_qdrant()     # ANN 向量检索
    ├─ _lexical_recall_from_db()         # MySQL FULLTEXT
    ├─ _keyword_recall_from_candidates()  # 关键词打分
    └─ _secondary_financial_recall()     # 金融二次召回
    │
    ▼
_merge_recall_candidates()              # 去重合并
    │
    ▼
rerank.py: _rerank_hybrid_candidates()
    └── 权重融合 (embedding 0.45 + keyword 0.20 + title 0.15 + section 0.10 + BM25 0.10)
    └── 15+ 信号: numeric_density, table_like, query_aware, anti_noise...
    │
    ▼
diversity.py
    ├─ _deduplicate_candidates()        # Jaccard 去重
    ├─ _expand_neighbor_chunks()       # 邻接 chunk 扩展
    └─ _cap_page_duplicates()          # 页内去重 (max 5/page)
    │
    ▼
返回 top_k chunks
```

#### Pipeline 3: 问答 → 验证 → 修正

```
用户问题 + session_id
    │
    ▼
qa/pipeline.py: answer_question()
    │
    ├─ get_chat_history() + rewrite_query_with_history()
    │
    ▼
retrieve_chunks()                        # 检索 chunks
    │
    ▼
assemble_context()                       # 组装上下文
    │
    ▼
_build_answer_prompt() / _build_structured_answer_prompt()
    │
    ▼
chat_completion()                      # LLM 生成
    │
    ▼
[V4] verify_answer()                    # 答案验证
    │   启发式门控 → LLM verifier (中置信度)
    │   返回: is_supported, support_level, numeric_consistency,
    │         citation_adequate, failure_reasons, missing_requirements
    │
    ▼
[V4] refine_answer()                   # 自修正 (默认关闭)
    │   触发: is_supported=False OR numeric_consistency=False
    │
    ▼
返回: answer + sources + confidence + verification + refine
```

#### Pipeline 4: Agent 自主推理

```
question → _reasoning_node()
    │   LLM 结构化 JSON {thought, action, action_input}
    │   System prompt 注入 entity_memory + learning_memory
    │     ├─ entity_memory: session 级实体提及历史
    │     └─ learning_memory: 向量记忆(相似问题) + 工具链记忆
    │
    ├─ [kb_answer_question] → tool_execution → verification → refine
    ├─ [高风险工具] → 暂停等待 confirm
    └─ [action=final] → final_answer

_tool_execution_node: 执行工具 → track_entities() 提取实体
_verification_node: verify_answer() (仅 ANSWER_TOOL)
_refine_node: refine_answer() (验证失败时)
_final_answer_node: 组装最终答案
验证成功(is_supported=True) → memory_service.store_success() 存储记忆
```

---

### 模块详解

#### ingestion/ — 文档解析

| 文件 | 职责 |
|------|------|
| `pipeline.py` | `parse_document()` 单文件; `parse_documents_from_folder()` 并行 (ProcessPoolExecutor) |
| `detectors.py` | 文件类型检测: pdf/docx/text/unknown |
| `loaders.py` | 编码 cascade: UTF-8 → GBK → GB18030 → Latin-1 |
| `normalizers.py` | 文本清洗、连字符修复、小块合并 |
| `quality.py` | PDF 块质量评分 |
| `parsers/pdf_parser.py` | PyMuPDF 多线程; 表格检测; 标题识别 |
| `parsers/docx_parser.py` | Docling + python-docx 双引擎 |
| `parsers/text_parser.py` | 智能段落重建; Markdown/中文章节标题检测 |

#### services/ — 核心服务

| 文件 | 职责 |
|------|------|
| `document_service.py` | `import_single_document()` / `import_documents()` |
| `chunk_service.py` | `index_document()` 7 阶段; 金融分块; 增量 diff |
| `indexing_orchestrator.py` | parse + 可选 index 编排 |
| `vector_store.py` | Qdrant SDK; `ensure_collection()`; `upsert_chunks()` 200/batch |
| `llm_service.py` | Ollama 封装: embedding (单条/批量/LRU) + chat (自动降级) |
| `reranker_service.py` | LLM answerability-first 重排 |
| `verifier_service.py` | 诊断式答案验证 |
| `refine_service.py` | 自我修正 (默认关闭) |
| `memory_vector_store.py` | Qdrant 向量记忆 (kb_memory 集合); 存储/召回成功问答案例 |
| `memory_service.py` | 统一记忆服务 facade; `store_success()` / `retrieve_context()` |

#### retrieval/ — 检索召回

| 文件 | 职责 |
|------|------|
| `service.py` | `retrieve_chunks()` 入口; standard / multistage 自动选择 |
| `multistage.py` | V3 多阶段: section 级 embedding → section 召回 → chunk 召回 |
| `recall.py` | 5 种召回: vector / lexical / keyword / BM25 / 金融扩展 |
| `rerank.py` | 权重融合; 金融 query 特殊处理 |
| `diversity.py` | Jaccard 去重; 邻接扩展; 页内去重 |
| `query_understanding.py` | 意图分类; 金融 query 自动扩展 |
| `signals.py` | 12 个重排信号 |

#### qa/ — 问答生成

| 文件 | 职责 |
|------|------|
| `pipeline.py` | `answer_question()` 完整流程 |
| `session.py` | `summarize_document()`; `get_chat_history()`; 自动摘要 |
| `prompts.py` | text/structured prompt; JSON 解析; citation 增强 |
| `context.py` | `assemble_context()`; 历史格式化 |

#### agent/ — 自主 Agent

| 文件 | 职责 |
|------|------|
| `graph.py` | LangGraph StateGraph; 5 节点; entity_memory + learning_memory 注入 |
| `service.py` | `agent_ask()` 同步; `agent_ask_stream()` SSE; `handle_agent_confirmation()`; 自动存储记忆 |
| `agent.py` | 11 个 LangChain Tool 对象 |
| `confirmation.py` | Human-in-loop 确认; `_pending_confirmations` 内存存储 |
| `llm.py` | `get_chat_llm()` |

#### db/ — 数据库

| 文件 | 职责 |
|------|------|
| `schema.py` | 7 张表建表语句 |
| `connection.py` | PyMySQL 连接池 |
| `bootstrap.py` | `init_db()` / `reset_database()` |
| `migrations.py` | 5 个增量迁移 (001-005) |
| `repositories/entity_repository.py` | Agent 实体记忆 (session 级) |
| `repositories/tool_chain_repository.py` | Agent 工具链记忆 (跨 session); 存储/召回成功工具调用序列 |

---

### 数据库 Schema

#### documents
```sql
id BIGINT PK AUTO_INCREMENT
title VARCHAR(255) NOT NULL
content LONGTEXT
raw_text LONGTEXT
summary TEXT
source VARCHAR(1024)
source_type VARCHAR(64)
file_path VARCHAR(1024)
file_type VARCHAR(128)
mime_type VARCHAR(255)
lang VARCHAR(64)
author VARCHAR(255)
published_at DATETIME
content_hash VARCHAR(64)
block_count INT DEFAULT 0
blocks_json JSON
metadata_json JSON
tags_json JSON
status VARCHAR(64) DEFAULT 'active'
created_at TIMESTAMP, updated_at TIMESTAMP
INDEX: title, status, source_type, file_type, content_hash
```

#### document_chunks
```sql
id BIGINT PK AUTO_INCREMENT
document_id BIGINT FK → documents(id) ON DELETE CASCADE
chunk_index INT NOT NULL
chunk_text LONGTEXT NOT NULL
search_text LONGTEXT
lexical_text LONGTEXT
embedding LONGTEXT
chunk_type VARCHAR(64)
title VARCHAR(255)
section_title VARCHAR(255)
section_path JSON
page_start INT, page_end INT
block_start_index INT, block_end_index INT
token_count INT
chunk_hash VARCHAR(64)
metadata_json JSON
embedding_model VARCHAR(255)
created_at TIMESTAMP, updated_at TIMESTAMP
UNIQUE KEY: (document_id, chunk_index)
FULLTEXT INDEX: ft_chunks_lexical (lexical_text, search_text, title, section_title)
```

#### chat_sessions
```sql
session_id VARCHAR(128) PK
title VARCHAR(255)
user_id VARCHAR(128)
metadata JSON
last_message_at TIMESTAMP
created_at TIMESTAMP, updated_at TIMESTAMP
```

#### chat_messages
```sql
id BIGINT PK AUTO_INCREMENT
session_id VARCHAR(128) FK → chat_sessions(session_id) ON DELETE CASCADE
role VARCHAR(32) NOT NULL  -- user / assistant / system
message LONGTEXT NOT NULL
citations JSON
metadata JSON
created_at TIMESTAMP
```

#### agent_entities (Agent 实体记忆)
```sql
id BIGINT PK AUTO_INCREMENT
session_id VARCHAR(128) NOT NULL
entity_type VARCHAR(64) NOT NULL  -- document/concept/metric/person/company/other
entity_key VARCHAR(512) NOT NULL  -- 归一化实体名
display_name VARCHAR(512)
reference_count INT DEFAULT 1
first_seen_at TIMESTAMP, last_seen_at TIMESTAMP
metadata JSON
INDEX: (session_id), (entity_type, entity_key), (reference_count DESC)
```

#### agent_tool_chains (Agent 工具链记忆)
```sql
id BIGINT PK AUTO_INCREMENT
question_hash VARCHAR(64) UNIQUE NOT NULL  -- SHA256(normalized_question)[:16]
question_text TEXT NOT NULL
tool_sequence JSON NOT NULL               -- [{tool, args}, ...]
session_id VARCHAR(128)
success_count INT DEFAULT 1              -- 该工具链成功次数
last_used TIMESTAMP                      -- 最近使用时间
created_at TIMESTAMP
INDEX: (success_count DESC), (last_used)
```

---

### API 接口

#### FastAPI (knowledge_base/app/api.py)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/import/folder` | 批量导入文件夹 |
| POST | `/import/file` | 导入单个文档 |
| POST | `/index` | 为文档建立索引 |
| POST | `/ask` | 直接 RAG 问答 |
| POST | `/summary` | 文档摘要 |
| POST | `/chat/session` | 创建会话 |
| GET | `/chat/{session_id}` | 获取会话历史 |
| POST | `/agent/ask` | Agent 模式问答 (同步) |
| POST | `/agent/ask/stream` | Agent 模式问答 (SSE 流式) |
| POST | `/agent/confirm` | 确认/拒绝 Agent 高风险操作 |
| GET | `/demo` | 演示页面 |

#### CLI (knowledge_base/app/main.py)

```bash
python -m app.main <command>

Commands:
  init-db        初始化数据库
  reset-db      重置数据库 [--drop] [--yes]
  import         导入文件夹
  import-file   导入单个文件
  list-docs     列出所有文档
  index         索引文档 (指定 ID)
  index-all     并行索引所有文档
  chunks        查看文档的 chunks
  ask           提问 (单次)
  summary       文档摘要
  chat-history  查看会话历史
  chat          交互式聊天 REPL
```

---

### Agent 系统

#### Graph 节点

```
reasoning → tool_execution → verification → refine
     ↑              ↓                        ↓
     └──────────────┴──────────→ final_answer
```

| 节点 | 触发条件 | 作用 |
|------|----------|------|
| `_reasoning_node` | 每次循环 | LLM 决策; 高风险工具挂起 |
| `_tool_execution_node` | action 是工具 | 执行工具; track_entities() |
| `_verification_node` | ANSWER_TOOL 且 AGENT_ENABLE_VERIFIER=True | verify_answer() |
| `_refine_node` | verification.is_supported=False | refine_answer() |
| `_final_answer_node` | action=final 或验证通过 | 组装最终答案 |

#### 高风险工具确认流程

```
reasoning: 高风险操作 → pending_confirmation
    ↓ (SSE: confirmation_required)
Client → POST /agent/confirm {session_id, confirmed: true/false}
    ↓
handle_agent_confirmation()
    ↓ confirmed=true → 执行工具 → reasoning 继续
    ↓ confirmed=false → 跳过 → reasoning 继续
```

#### SSE 事件类型

```json
{"type": "start", "question": "...", "session_id": "..."}
{"type": "reasoning", "step": 1, "thought": "...", "action": "...", "observation": "..."}
{"type": "tool_call", "tool": "...", "args": {...}, "result": {...}}
{"type": "verification", "verification_result": {...}}
{"type": "refine", "refine_result": {...}}
{"type": "confirmation_required", "tool": "...", "args": {...}, "reason": "..."}
{"type": "final", "answer": "..."}
{"type": "done", "session_id": "..."}
```

---

### 工具列表

Agent 可调用的 11 个工具:

| 工具名 | 输入 | 底层调用 |
|--------|------|----------|
| `kb_answer_question` | question, session_id, top_k, response_mode... | qa/pipeline.py |
| `kb_search_knowledge_base` | query, top_k, include_full_text | retrieval/service.py |
| `kb_generate_answer` | question, context, history_text, response_mode | llm_service.chat_completion() |
| `kb_assemble_context` | hits, max_chunks | qa/context.py |
| `kb_rewrite_query` | question, session_id, use_history | qa/pipeline.py |
| `kb_summarize_document` | document_id | qa/session.py |
| `kb_get_chat_history` | session_id, limit | qa/session.py |
| `kb_create_chat_session` | session_id, title, metadata | db/chat_repository.py |
| `kb_import_file` | file_path | services/document_service.py |
| `kb_import_folder` | folder | services/document_service.py |
| `kb_index_document` | document_id, chunk_size, overlap | services/chunk_service.py |

---

### 配置说明

#### 环境变量 (knowledge_base/.env)

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_TIMEOUT=120

# MySQL
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=knowledge_base

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=kb_chunks
EMBEDDING_DIM=768

# V4 Pipeline
V4_ENABLE_ANSWER_VERIFIER=true      # 默认开启验证
V4_ENABLE_SELF_REFINE=false         # 默认关闭自修正
V4_LLM_RERANK_TOP_N=8
V4_LLM_RERANK_WEIGHT=0.40

# Agent
AGENT_ENABLE_VERIFIER=true

# Agent Memory / Learning
VECTOR_MEMORY_ENABLED=true               # 向量记忆 (Qdrant kb_memory 集合)
VECTOR_MEMORY_TOP_K=3                    # 相似案例召回数
TOOL_CHAIN_MEMORY_ENABLED=true           # 工具链记忆 (MySQL agent_tool_chains 表)
TOOL_CHAIN_MEMORY_MIN_SUCCESS=2         # 最小成功次数才计入记忆
QDRANT_MEMORY_COLLECTION=kb_memory      # 记忆 Qdrant collection 名
```

---

## Evals (评测系统)

### 评测目录结构

```
evals/
├── datasets/
│   ├── kb_eval_seed.jsonl       # 种子评测数据 (EvalSample JSONL)
│   └── financebench/             # FinanceBench 原始数据
├── configs/
│   ├── baseline.yaml             # 基础评测 (internal, top_k=5)
│   ├── retrieval_v4.yaml          # V4 pipeline (LLM rerank + verifier)
│   └── sft_llamafactory.yaml     # LlamaFactory SFT 训练配置
├── scripts/
│   ├── run_eval.py              # 通用评测运行
│   ├── run_financebench_eval.py  # FinanceBench 专用评测 (带缓存)
│   ├── score_eval.py            # 多层评分
│   ├── enrich_with_evidence.py   # 回填 gold evidence 文本
│   ├── train_sft.py            # 多后端 SFT 训练
│   ├── gold/
│   │   ├── suggest_gold.py     # AI 辅助推荐 gold chunk
│   │   ├── review_candidates.py  # 候选 chunk 人工审核
│   │   ├── find_gold_chunks.py  # MySQL FULLTEXT 找 gold
│   │   └── apply_gold.py        # 标注回填到数据集
│   └── analysis/
│       ├── export_sft.py        # SFT 数据导出
│       └── compare_runs.py      # 回归对比
├── utils/
│   ├── dataset.py               # EvalSample schema + load_dataset
│   ├── adapters.py             # EvalAdapter (internal/API 双模式)
│   ├── scorer.py               # RetrievalScorer + AnswerScorer
│   └── report.py               # JSON/Markdown 报告构建
├── cache/                      # FinanceBench 评测缓存
├── reports/                    # 评测报告输出
├── runs/                      # 原始运行数据 JSONL
└── training/                 # 训练输出
    ├── hf_output/
    └── llamafactory_output/
```

---

### EvalSample 数据格式

每条样本为 JSONL 一行，包含以下结构:

```json
{
  "id": "kb_0001",
  "question": {
    "user_query": "苹果公司2023年净利润是多少？",
    "conversation_history": []
  },
  "retrieval": {
    "label_status": "labeled_chunk",
    "gold_chunk_ids": [10, 25],
    "gold_doc_ids": [1],
    "hard_negative_chunk_ids": [],
    "gold_evidence_section": "Income Statement"
  },
  "context": {
    "gold_context_blocks": ["原始文档块文本..."]
  },
  "answer": {
    "gold_answer": "$97 billion",
    "must_include": ["97", "billion"],
    "must_not_include": [],
    "faithfulness_requirements": []
  },
  "supervision": {
    "sft_messages_with_context": [],
    "sft_messages_no_context": [],
    "rejected_outputs": []
  },
  "evaluation": {
    "expected_behavior": "answer",
    "scoring_type": "answer",
    "error_type": null
  },
  "metadata": {
    "created_by": "human",
    "source": "financebench",
    "version": "1.0"
  }
}
```

#### label_status 状态

| 状态 | 检索评测 | 回答评测 |
|------|---------|---------|
| `unlabeled` | 跳过 | 是 |
| `labeled_doc` | 是 (doc 级) | 是 |
| `labeled_chunk` | 是 (chunk 级) | 是 |
| `unanswerable` | 跳过 | 是 (refuse 评测) |

---

### 评测指标

#### 检索层

| 指标 | 说明 |
|------|------|
| Hit@K | Top-K 是否命中任意 gold chunk |
| Recall@5 | Top-5 命中占总数比例 |
| MRR | 首个命中位置倒数均值 |

#### 回答层

| Label | 条件 |
|-------|------|
| `exact` | 归一化后与 gold_answer 完全一致 (≥85% token 重叠) |
| `partial` | `must_include` 满足 ≥50% 且无违规 |
| `wrong` | 既非 exact 也非 partial |
| `refuse_correct` | 期望拒答且答案含拒答语义 |
| `refuse_wrong` | 期望拒答但给出确定答案 |
| `clarify_correct` | 期望澄清且答案含澄清意图 |
| `clarify_wrong` | 期望澄清但未澄清 |

#### score_eval.py 多层评分

| 层级 | 指标 |
|------|------|
| Document | doc_hit_at_1/5, doc_mrr |
| Page (strict) | page_hit_at_1/5, page_mrr |
| Page (relaxed, window>0) | page_relaxed_hit_at_1/5 |
| Section | section_hit_at_1/5, section_mrr |
| Evidence text | evidence_text_hit_at_1/5, mrr |
| Evidence semantic | evidence_semantic_hit_at_1/5 (lexical OR embedding ≥ threshold) |
| Answer | numeric_match, normalized_exact_match, answer_label |
| Failure reason | doc_miss, section_miss, page_near_miss, semantic_drift... |

---

### 评测流程

#### 流程 1: 通用评测

```bash
python evals/scripts/run_eval.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --config evals/configs/baseline.yaml \
    --output evals/reports/run_baseline.json \
    --parallel 4
```

**EvalAdapter 双模式:**

- `mode: internal` → 直接调用 `app.qa.pipeline.answer_question()` 和 `app.retrieval.service.retrieve_chunks()`
- `mode: api` → HTTP 调用 `POST /ask` 和 `GET /retrieval`

#### 流程 2: FinanceBench 评测 (带缓存)

```bash
python evals/scripts/run_financebench_eval.py \
    --config evals/configs/baseline.yaml \
    --output evals/reports/fb_eval.json
```

- 缓存 key: `SHA256(top_k:doc_filter:question)[:16]`
- 自动从 query 关键词检测公司 → 过滤到对应 doc

#### 流程 3: 多层评分

```bash
python evals/scripts/score_eval.py \
    --run evals/reports/run_baseline.jsonl \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --output evals/reports/run_baseline_scored.json
```

#### 流程 4: 回归对比

```bash
python evals/scripts/analysis/compare_runs.py \
    --base evals/reports/run_baseline.json \
    --new evals/reports/run_exp.json \
    --output evals/reports/compare.md
```

---

### Gold 标注流程

```bash
# 1. 运行评测
python evals/scripts/run_eval.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --config evals/configs/baseline.yaml \
    --output evals/reports/run_debug.json

# 2. AI 推荐 gold chunk
python evals/scripts/gold/suggest_gold.py \
    --report evals/reports/run_debug.json \
    --only-need-labeling \
    --output evals/reports/suggestions.json

# 3. 人工审核候选
python evals/scripts/gold/review_candidates.py \
    --report evals/reports/run_debug.json \
    --only-need-labeling

# 4. 回填标注
python evals/scripts/gold/apply_gold.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --case kb_0001 \
    --gold-chunk 10 --gold-chunk 25 \
    --label-status labeled_chunk

# 5. 重新评测
python evals/scripts/run_eval.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --config evals/configs/baseline.yaml \
    --output evals/reports/run_after_labeling.json
```

---

### 评测配置

#### baseline.yaml
```yaml
mode: internal
top_k: 5
normalize_text: true
use_doc_level_fallback: true
```

#### retrieval_v4.yaml
```yaml
mode: internal
top_k: 5
retrieve_top_k: 20          # 两阶段: 召回 20 → LLM rerank → 返回 5
enable_query_enhance: true
use_multistage: true
enable_llm_rerank: true
llm_rerank_top_n: 8
llm_rerank_weight: 0.40
enable_answer_verifier: true
verifier_threshold: 0.5
enable_self_refine: false   # 默认关闭
max_refine_rounds: 1
```

---

## Model Serving (SFT 服务)

### Serving 目录结构

```
model_serving/
├── service/
│   ├── app.py            # FastAPI (Ollama 兼容)
│   └── model.py          # SFTModel 单例 (线程安全)
├── scripts/
│   ├── serve.py          # 入口 (uvicorn)
│   ├── eval_financebench_sft.py  # SFT 评测包装
│   └── check_model.py   # 健康检查
└── configs/
    └── serve_config.yaml  # 模型 + LoRA + 服务配置
```

### Serving API 接口

| 端点 | 兼容 | 说明 |
|------|------|------|
| `GET /` | - | 健康检查 + LoRA 状态 |
| `GET /api/tags` | Ollama | 列出可用模型 |
| `POST /api/chat` | Ollama | 聊天补全 |
| `POST /api/generate` | Ollama | 直接生成 |
| `POST /v1/chat/completions` | OpenAI | OpenAI 兼容 |
| `POST /model/switch?adapter_path=xxx` | - | **动态切换 LoRA** |

### LoRA 动态切换

```
POST /model/switch?adapter_path=/path/to/new/adapter
    │
    ├─→ SFTModel.reset()
    │     删除 model + tokenizer
    │     torch.cuda.empty_cache()
    │
    └─→ load_sft_model(new_adapter_path)
          PeftModel.from_pretrained(base_model, new_adapter)
```

### 启动 SFT 服务

```bash
cd /home/cz/agent/model_serving

# 配置 LoRA adapter
export SFT_LORA_PATH=/home/cz/agent/finance_sft/output/qwen25_finance_lora

# 启动 (端口 8001)
python scripts/serve.py
```

**serve_config.yaml 示例:**
```yaml
base_model:
  path: "/home/cz/python/knowledge_base/models/Qwen2.5-7B-Instruct"
  device: "cuda"
  torch_dtype: "bfloat16"

lora:
  enabled: true
  adapter_path: "/home/cz/agent/finance_sft/output/qwen25_finance_lora"

server:
  host: "0.0.0.0"
  port: 8001

inference:
  max_tokens: 512
  temperature: 0.1
  top_p: 0.9
```

---

## Finance SFT (训练)

### SFT 目录结构

```
finance_sft/
├── configs/
│   └── train.yaml          # LlamaFactory 训练配置
├── data/
│   ├── financebench_train.jsonl
│   └── financebench_train_sharegpt.jsonl
├── scripts/
│   └── eval_model.py      # 直接评测 (无需 serving)
├── output/
│   └── qwen25_finance_lora/  # 训练产出的 LoRA adapter
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       └── checkpoint-3/
└── models/                 # 基础模型 (符号链接)
```

### 训练流程

```bash
# 方式 A: 通过 evals/train_sft.py (推荐)
python evals/scripts/train_sft.py \
    --backend llamafactory \
    --data evals/reports/sft_data.jsonl \
    --output finance_sft/output/qwen25_finance_lora \
    --config evals/configs/sft_llamafactory.yaml

# 方式 B: LlamaFactory CLI
llamafactory-cli train finance_sft/configs/train.yaml
```

**LlamaFactory 训练配置 (train.yaml):**
```yaml
model_name_or_path: /path/to/Qwen2.5-7B-Instruct
stage: sft
finetuning_type: lora
dataset: financebench_train
template: qwen
cutoff_len: 4096
output_dir: output/qwen25_finance_lora
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1e-4
num_train_epochs: 3
bf16: true
lora:
  rank: 8
  alpha: 16
```

---

## 端到端流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        完整 RAG → 评测 → 训练 → Serving 流程            │
└─────────────────────────────────────────────────────────────────────┘

1. 导入 & 索引
   knowledge_base/
   python -m app.main import data/        # 导入文档
   python -m app.main index-all          # 建立索引

2. 评测 (internal 模式, 直接调 Python)
   evals/scripts/run_eval.py \
       --dataset evals/datasets/kb_eval_seed.jsonl \
       --config evals/configs/baseline.yaml \
       --output evals/reports/run_baseline.jsonl

3. 多层评分
   evals/scripts/score_eval.py \
       --run evals/reports/run_baseline.jsonl \
       --dataset evals/datasets/kb_eval_seed.jsonl \
       --output evals/reports/run_baseline_scored.jsonl

4. Gold 标注 (可选)
   evals/scripts/gold/suggest_gold.py --report evals/reports/run_baseline.jsonl ...
   evals/scripts/gold/apply_gold.py --dataset evals/datasets/kb_eval_seed.jsonl ...

5. 导出 SFT 训练数据
   evals/scripts/analysis/export_sft.py \
       --dataset evals/datasets/kb_eval_seed.jsonl \
       --from-evidence \
       --output evals/reports/sft_data.jsonl

6. SFT 训练
   evals/scripts/train_sft.py \
       --backend llamafactory \
       --data evals/reports/sft_data.jsonl \
       --output finance_sft/output/qwen25_finance_lora

7. 启动 SFT 服务
   cd model_serving
   SFT_LORA_PATH=finance_sft/output/qwen25_finance_lora \
   python scripts/serve.py

8. 用 SFT 模型评测
   USE_SFT_SERVICE=true \
   SFT_SERVICE_BASE_URL=http://localhost:8001 \
   python model_serving/scripts/eval_financebench_sft.py \
       --output evals/reports/run_sft.json

   或直接评测 (无需 serving):
   python finance_sft/scripts/eval_model.py
```

---

## 版本历史

| 版本 | 说明 |
|------|------|
| 1.0 | 基础 RAG (import → index → retrieve → ask) |
| 2.0 | Agent 自主推理 (LangGraph, 显式推理轨迹) |
| 3.0 | V3 多阶段检索 (section 级 embedding) |
| 4.0 | V4 答案验证 + 自修正 + 实体记忆 + Human-in-loop |
| 5.0 | Agent 学习能力：向量记忆(Qdrant) + 工具链记忆(MySQL)；成功问答自动存储、相似问题召回 |
| Evals 1.0 | 评测系统 (检索 + 回答评分, internal/API 双模式) |
| Evals 2.0 | Gold 标注流程 (AI 推荐 → 人工审核 → 回填) |
| Model Serving 1.0 | SFT 模型服务 (Ollama 兼容 + LoRA 动态切换) |
