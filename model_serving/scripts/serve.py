#!/usr/bin/env python3
"""
启动 SFT 模型服务

Usage:
    python scripts/serve.py

Environment variables:
    SERVE_CONFIG: 配置文件路径 (default: configs/serve_config.yaml)
"""
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.app import app, get_model, config
import uvicorn


def main():
    print("=" * 60)
    print("SFT Model Serving")
    print("=" * 60)
    print(f"Config: {os.environ.get('SERVE_CONFIG', 'configs/serve_config.yaml')}")
    print(f"Base model: {config.get('base_model', {}).get('path', 'N/A')}")
    print(f"LoRA adapter: {config.get('lora', {}).get('adapter_path', 'None')}")
    print("=" * 60)

    # 预热模型（可选）
    warmup = os.environ.get("WARMUP_MODEL", "true").lower() == "true"
    if warmup:
        print("\nPre-warming model...")
        try:
            get_model()
            print("Model warmed up successfully")
        except Exception as e:
            print(f"Warning: Model warmup failed: {e}")
            print("Service will start but model will be loaded on first request")

    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8001)
    workers = server_cfg.get("workers", 1)

    print(f"\nStarting server on http://{host}:{port}")
    print(f"API docs: http://{host}:{port}/docs")
    print("=" * 60)

    uvicorn.run(
        "service.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
    )


if __name__ == "__main__":
    main()
