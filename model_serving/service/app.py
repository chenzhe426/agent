"""
FastAPI 应用 - 提供 Ollama 兼容的 API 接口
供 RAG 层通过 HTTP 调用 SFT 模型
"""
import os
import time
from typing import Optional

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 加载配置
CONFIG_PATH = os.environ.get(
    "SERVE_CONFIG",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "serve_config.yaml")
)

def load_config() -> dict:
    """加载服务配置"""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

# 初始化模型（延迟加载）
_model = None

def get_model():
    """获取或初始化模型"""
    global _model
    if _model is None:
        from .model import load_sft_model

        base_cfg = config.get("base_model", {})
        lora_cfg = config.get("lora", {})
        infer_cfg = config.get("inference", {})
        server_cfg = config.get("server", {})

        device = base_cfg.get("device", "cuda")
        torch_dtype = base_cfg.get("torch_dtype", "bfloat16")

        lora_path = None
        if lora_cfg.get("enabled") and lora_cfg.get("adapter_path"):
            lora_path = lora_cfg["adapter_path"]

        print(f"[SFT Server] Initializing model...")
        print(f"  Base model: {base_cfg.get('path')}")
        print(f"  LoRA adapter: {lora_path}")
        print(f"  Device: {device}, dtype: {torch_dtype}")

        _model = load_sft_model(
            base_model_path=base_cfg.get("path"),
            lora_adapter_path=lora_path,
            device=device,
            torch_dtype=torch_dtype,
        )

    return _model


# ==================== API Models ====================

class Message(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    model: str = Field(default="qwen25-finance-sft", description="Model name")
    messages: list[Message] = Field(..., description="Chat messages")
    stream: bool = Field(default=False, description="Stream response")
    options: Optional[dict] = Field(default=None, description="Generation options")


class GenerateRequest(BaseModel):
    model: str = Field(default="qwen25-finance-sft", description="Model name")
    prompt: str = Field(..., description="Prompt text")
    stream: bool = Field(default=False, description="Stream response")
    options: Optional[dict] = Field(default=None, description="Generation options")


class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: Message
    done: bool = True


class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool = True


# ==================== FastAPI App ====================

app = FastAPI(
    title="SFT Model Serving API",
    description="Ollama-compatible API for SFT fine-tuned models",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "model": config.get("shared", {}).get("default_model", "qwen25-finance-sft"),
        "lora_enabled": config.get("lora", {}).get("enabled", True),
        "lora_adapter": config.get("lora", {}).get("adapter_path", ""),
    }


@app.get("/api/tags")
async def list_models():
    """List available models - compatible with Ollama API"""
    return {
        "models": [
            {
                "name": config.get("shared", {}).get("default_model", "qwen25-finance-sft"),
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "size": 0,
            }
        ]
    }


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Ollama-compatible chat API.
    RAG 层通过这个接口调用 SFT 模型生成答案。

    输入格式（ShareGPT风格）:
    {
        "model": "qwen25-finance-sft",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
    }
    """
    try:
        model = get_model()

        infer_cfg = config.get("inference", {})
        max_tokens = infer_cfg.get("max_tokens", 512)
        temperature = infer_cfg.get("temperature", 0.1)
        top_p = infer_cfg.get("top_p", 0.9)
        repetition_penalty = infer_cfg.get("repetition_penalty", 1.1)
        timeout = infer_cfg.get("timeout", 120)

        # 从 request options 覆盖默认参数
        if request.options:
            max_tokens = request.options.get("num_predict", max_tokens)
            temperature = request.options.get("temperature", temperature)
            top_p = request.options.get("top_p", top_p)

        # 转换为 dict 列表
        messages = [msg.model_dump() for msg in request.messages]

        start_time = time.time()
        response = model.chat(
            messages=messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        elapsed = time.time() - start_time

        print(f"[API /api/chat] Generated response in {elapsed:.2f}s, length={len(response['content'])}")

        return ChatResponse(
            model=request.model,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            message=Message(**response),
            done=True,
        )

    except Exception as e:
        print(f"[API /api/chat] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Ollama-compatible generate API.
    Direct prompt-based generation.
    """
    try:
        model = get_model()

        infer_cfg = config.get("inference", {})
        max_tokens = infer_cfg.get("max_tokens", 512)
        temperature = infer_cfg.get("temperature", 0.1)
        top_p = infer_cfg.get("top_p", 0.9)
        repetition_penalty = infer_cfg.get("repetition_penalty", 1.1)

        if request.options:
            max_tokens = request.options.get("num_predict", max_tokens)
            temperature = request.options.get("temperature", temperature)

        start_time = time.time()
        response = model.generate(
            prompt=request.prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        elapsed = time.time() - start_time

        print(f"[API /api/generate] Generated response in {elapsed:.2f}s")

        return GenerateResponse(
            model=request.model,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            response=response,
            done=True,
        )

    except Exception as e:
        print(f"[API /api/generate] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: dict) -> dict:
    """
    OpenAI-compatible chat completions API.
    支持直接传入 messages 列表。
    """
    try:
        model = get_model()

        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 512)
        temperature = request.get("temperature", 0.1)
        top_p = request.get("top_p", 0.9)

        response = model.chat(
            messages=messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "qwen25-finance-sft"),
            "choices": [
                {
                    "index": 0,
                    "message": response,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(response.get("content", "").split()),
                "total_tokens": 0,
            },
        }

    except Exception as e:
        print(f"[API /v1/chat/completions] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/switch")
async def switch_model(adapter_path: str) -> dict:
    """
    动态切换 LoRA 适配器（无需重启服务）
    """
    try:
        from .model import SFTModel

        print(f"[API /model/switch] Switching to adapter: {adapter_path}")
        SFTModel.reset()

        base_cfg = config.get("base_model", {})
        global _model
        from .model import load_sft_model
        _model = load_sft_model(
            base_model_path=base_cfg.get("path"),
            lora_adapter_path=adapter_path if adapter_path else None,
            device=base_cfg.get("device", "cuda"),
            torch_dtype=base_cfg.get("torch_dtype", "bfloat16"),
        )

        return {"status": "ok", "adapter_path": adapter_path}

    except Exception as e:
        print(f"[API /model/switch] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8001)
    workers = server_cfg.get("workers", 1)

    print(f"[SFT Server] Starting on {host}:{port}")
    uvicorn.run(
        "service.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
    )
