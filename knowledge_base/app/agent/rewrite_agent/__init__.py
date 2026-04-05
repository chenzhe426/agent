"""
QueryRewriteAgent - 分层记忆的查询改写Agent（只负责读取）

架构：
- MemoryAgent: 专门负责存储（短期/中期/长期记忆）
- QueryRewriteAgent: 只负责读取记忆，用于查询改写

使用方式：
1. 新消息到达 → 调用 MemoryAgent.store_message() 存储记忆
2. 需要改写查询 → 调用 QueryRewriteAgent.rewrite() 进行改写
"""

from app.agent.rewrite_agent.agent import QueryRewriteAgent
from app.agent.rewrite_agent.schemas import RewriteInput, RewriteOutput

__all__ = ["QueryRewriteAgent", "RewriteInput", "RewriteOutput"]
