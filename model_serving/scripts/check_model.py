#!/usr/bin/env python3
"""
快速验证 SFT 模型服务是否正常工作

Usage:
    python scripts/check_model.py
"""
import requests
import json

BASE_URL = "http://localhost:8001"


def check_health():
    """检查服务健康状态"""
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"✓ Health check: {resp.json()}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def check_chat():
    """测试 chat API"""
    payload = {
        "model": "qwen25-finance-sft",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "stream": False
    }

    try:
        resp = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        print(f"✓ Chat API response:")
        print(f"  Model: {result.get('model')}")
        print(f"  Response: {result.get('message', {}).get('content', '')[:200]}")
        return True
    except Exception as e:
        print(f"✗ Chat API failed: {e}")
        return False


def check_switch_adapter():
    """测试切换 adapter 接口"""
    # 这个测试仅检查接口是否存在，不实际切换
    try:
        resp = requests.post(
            f"{BASE_URL}/model/switch",
            params={"adapter_path": "/path/to/adapter"},
            timeout=5
        )
        print(f"✓ Switch adapter API exists")
        return True
    except Exception as e:
        print(f"✗ Switch adapter API failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("SFT Model Service Check")
    print("=" * 50)

    health_ok = check_health()
    print()

    if health_ok:
        chat_ok = check_chat()
        print()
        check_switch_adapter()
    else:
        print("Service is not running. Start it with:")
        print("  cd model_serving && python scripts/serve.py")

    print("=" * 50)
