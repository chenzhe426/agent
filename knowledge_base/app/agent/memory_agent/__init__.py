"""
MemoryAgent - 专门负责存储和管理分层记忆

职责：
- 接收新消息，分析对话轮次
- 更新短期/中期/长期记忆
- 提供统一的记忆存储接口
"""

from app.agent.memory_agent.agent import MemoryAgent

__all__ = ["MemoryAgent"]
