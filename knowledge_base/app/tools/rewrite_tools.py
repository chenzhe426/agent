"""
rewrite_tools.py - QueryRewriteAgent 工具封装

将 QueryRewriteAgent 注册为LangChain工具：
- kb_rewrite_query_v2: 基于分层记忆的查询改写
"""

from __future__ import annotations

import time
from typing import Any, Dict

from langchain.tools import tool

from app.agent.rewrite_agent.agent import query_rewrite_agent
from app.agent.rewrite_agent.schemas import RewriteInput
from app.tools.base import make_error, make_ok


TOOL_REWRITE_QUERY_V2 = "kb_rewrite_query_v2"


@tool
def kb_rewrite_query_v2(input_data: Dict[str, Any] | RewriteInput) -> Dict[str, Any]:
    """
    基于分层记忆的查询改写（新版）。

    使用MemoryAgent管理的三层记忆来改写用户问题：
    - 短期记忆：最近3轮对话的结构化摘要
    - 中期记忆：第4-10轮对话的关键信息（Redis, TTL=30min）
    - 长期记忆：>10轮对话的关键信息（MySQL, 跨session）

    适用于：
    - 用户问题包含代词或省略（"它"、"这个"、"怎么调用的"）
    - 需要结合多轮对话上下文理解的问题
    - 复杂的多轮对话问答场景

    输入参数:
        question: str - 用户原始问题
        session_id: str - 会话ID
        use_history: bool, optional - 是否使用历史记忆，默认True
    """
    start = time.perf_counter()
    try:
        payload = input_data if isinstance(input_data, RewriteInput) else RewriteInput(**input_data)
        result = query_rewrite_agent.rewrite(payload)

        duration_ms = int((time.perf_counter() - start) * 1000)
        if result.ok:
            return make_ok(TOOL_REWRITE_QUERY_V2, result.model_dump(), duration_ms)
        else:
            return make_error(TOOL_REWRITE_QUERY_V2, "REWRITE_FAILED", result.error or "改写失败", duration_ms)

    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_error(TOOL_REWRITE_QUERY_V2, "INTERNAL_ERROR", str(e), duration_ms)
