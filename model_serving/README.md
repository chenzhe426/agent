# Model Serving

SFT 模型的 HTTP 服务层，供 RAG 系统调用。

## 启动服务

```bash
cd model_serving
pip install fastapi uvicorn transformers peft torch

# 启动服务
python scripts/serve.py
```

服务默认运行在 `http://localhost:8001`，提供 Ollama 兼容的 API。

## API 接口

- `GET /` - 健康检查
- `GET /api/tags` - 列出可用模型
- `POST /api/chat` - Ollama 兼容聊天接口
- `POST /api/generate` - Ollama 兼容生成接口
- `POST /v1/chat/completions` - OpenAI 兼容接口
- `POST /model/switch?adapter_path=xxx` - 动态切换 LoRA 适配器

## RAG 层调用 SFT 模型

```bash
# 设置环境变量
export USE_SFT_SERVICE=true
export SFT_SERVICE_BASE_URL=http://localhost:8001
export SFT_SERVICE_MODEL=qwen25-finance-sft

# 运行评测
cd knowledge_base
USE_SFT_SERVICE=true python -m evals.scripts.run_eval \
    --dataset evals/data/financebench_v1_subset_3docs.jsonl \
    --config evals/configs/baseline.yaml \
    --output evals/reports/run_sft.json
```

## 配置

编辑 `configs/serve_config.yaml` 修改模型路径和 LoRA 适配器。

## 检查服务状态

```bash
python scripts/check_model.py
```
