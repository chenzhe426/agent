# Knowledge Base RAG System

本地知识库问答系统，支持混合检索、多轮对话、结构化输出、答案验证与自修正。

## 技术栈

| 组件 | 技术 |
|------|------|
| API 框架 | FastAPI |
| LLM | Ollama (qwen3:8b) |
| Embedding | Ollama (nomic-embed-text, 768 维) |
| 向量检索 | Qdrant |
| 关键词检索 | MySQL FULLTEXT + BM25 |
| 数据库 | MySQL |
| Agent | LangChain |

## 架构

```
knowledge_base/
├── app/
│   ├── api.py              # FastAPI 入口
│   ├── agent/              # LangChain Agent
│   ├── retrieval/          # 检索层（recall → rerank → diversity）
│   │   ├── recall.py       # 多路召回（向量/关键词/BM25/金融query扩展）
│   │   ├── rerank.py       # 重排序
│   │   ├── diversity.py    # 去重、邻接扩展、页内去重
│   │   ├── multistage.py   # 多阶段检索
│   │   └── query_understanding.py  # 查询增强/意图分类
│   ├── qa/                 # 问答层
│   │   ├── pipeline.py     # 主流程：改写→检索→生成→验证→修正
│   │   ├── prompts.py      # Prompt 模板
│   │   ├── session.py      # 会话管理
│   │   └── context.py      # 上下文组装
│   ├── services/           # 底层服务
│   │   ├── llm_service.py  # Ollama 调用
│   │   ├── verifier_service.py   # 答案验证（V4）
│   │   └── refine_service.py     # 答案自修正（V4）
│   ├── tools/              # LangChain Tools
│   ├── db/                  # MySQL 连接、schema、repository
│   └── ingestion/          # 文档导入 pipeline
├── evals/                  # 评测系统
└── data/                   # 示例文档
```

## 检索流程

```
用户 query
    ↓
查询增强（金融 query 专用扩展）
    ↓
多路召回 ──┬── 向量检索（Qdrant）
           ├── 关键词检索（MySQL FULLTEXT）
           ├── BM25 检索
           └── 金融 query 二次召回
    ↓
合并去重 + 邻接 chunk 扩展
    ↓
重排序（RERANK）
    ↓
返回 top_k chunks
```

## 问答流程

```
用户问题
    ↓
结合历史改写查询（指代消解）
    ↓
检索 chunks
    ↓
组装上下文
    ↓
生成答案（text / structured）
    ↓
[V4] 答案验证 → [V4] 自修正
    ↓
返回答案 + 来源 + 置信度
```

## 双模式

| 模式 | 接口 | 说明 |
|------|------|------|
| 直接 RAG | `POST /ask` | 检索 → 生成答案 |
| Agent 模式 | `POST /agent/ask` | LangChain Agent，可自主调用工具 |

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/ask` | POST | 直接 RAG 问答 |
| `/agent/ask` | POST | Agent 模式 |
| `/import/file` | POST | 导入文档 |
| `/import/folder` | POST | 批量导入 |
| `/index` | POST | 建立索引 |
| `/summary` | POST | 文档摘要 |
| `/chat/session` | POST | 创建会话 |
| `/chat/{session_id}` | GET | 会话历史 |
| `/demo` | GET | 演示页面 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
MYSQL_HOST=127.0.0.1
MYSQL_DATABASE=knowledge_base
QDRANT_URL=http://localhost:6333
```

### 3. 初始化

```bash
python -m app.main init-db
python -m app.main import data/
```

### 4. 启动

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

## 评测

详见 [evals/README.md](evals/README.md)

```bash
# 运行评测
python evals/scripts/run_eval.py \
    --dataset evals/datasets/kb_eval_seed.jsonl \
    --config evals/configs/baseline.yaml \
    --output evals/reports/run_baseline.json
```
