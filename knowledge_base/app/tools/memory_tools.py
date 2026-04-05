"""
memory_tools.py - MemoryAgent 工具封装

将 MemoryAgent 注册为LangChain工具：
- store_memory: 存储新消息并更新记忆
- get_memory_context: 获取记忆上下文
- clear_memory: 清除记忆
"""

from __future__ import annotations

import time
from typing import Any, Dict

from langchain.tools import tool

from app.agent.memory_agent.agent import memory_agent
from app.agent.memory_agent.schemas import StoreMessageInput, GetMemoryContextInput, ClearMemoryInput, MemoryLevel
from app.tools.base import make_error, make_ok


TOOL_STORE_MEMORY = "kb_store_memory"
TOOL_GET_MEMORY_CONTEXT = "kb_get_memory_context"
TOOL_CLEAR_MEMORY = "kb_clear_memory"


@tool
def kb_store_memory(input_data: Dict[str, Any] | StoreMessageInput) -> Dict[str, Any]:
    """
    存储新消息并更新分层记忆。

    当收到用户新消息或助手回复时，调用此工具来存储对话历史并更新记忆。

    输入参数:
        session_id: str - 会话ID
        role: str - 角色 ("user" 或 "assistant")
        message: str - 消息内容
        metadata: dict, optional - 额外元数据
    """
    start = time.perf_counter()
    try:
        payload = input_data if isinstance(input_data, StoreMessageInput) else StoreMessageInput(**input_data)
        result = memory_agent.store_message(payload)

        duration_ms = int((time.perf_counter() - start) * 1000)
        if result.ok:
            return make_ok(TOOL_STORE_MEMORY, result.model_dump(), duration_ms)
        else:
            return make_error(TOOL_STORE_MEMORY, "STORE_FAILED", result.error or "存储失败", duration_ms)

    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_error(TOOL_STORE_MEMORY, "INTERNAL_ERROR", str(e), duration_ms)


@tool
def kb_get_memory_context(input_data: Dict[str, Any] | GetMemoryContextInput) -> Dict[str, Any]:
    """
    获取分层记忆上下文。

    用于查询改写时获取历史对话记忆。

    输入参数:
        session_id: str - 会话ID
        question: str, optional - 关联的问题（用于长期记忆检索）
        include_levels: list[str], optional - 要获取的层级 ["short", "mid", "long"]
    """
    start = time.perf_counter()
    try:
        # 处理输入数据
        if isinstance(input_data, GetMemoryContextInput):
            payload = input_data
        else:
            # 转换 include_levels 字符串到枚举
            include_levels = input_data.get("include_levels", ["short", "mid", "long"])
            level_enum = []
            for lvl in include_levels:
                if lvl == "short":
                    level_enum.append(MemoryLevel.SHORT)
                elif lvl == "mid":
                    level_enum.append(MemoryLevel.MID)
                elif lvl == "long":
                    level_enum.append(MemoryLevel.LONG)

            payload = GetMemoryContextInput(
                session_id=input_data["session_id"],
                question=input_data.get("question"),
                include_levels=level_enum,
            )

        result = memory_agent.get_memory_context(payload)

        duration_ms = int((time.perf_counter() - start) * 1000)
        if result.ok:
            return make_ok(TOOL_GET_MEMORY_CONTEXT, result.model_dump(), duration_ms)
        else:
            return make_error(TOOL_GET_MEMORY_CONTEXT, "GET_CONTEXT_FAILED", result.error or "获取上下文失败", duration_ms)

    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_error(TOOL_GET_MEMORY_CONTEXT, "INTERNAL_ERROR", str(e), duration_ms)


@tool
def kb_clear_memory(input_data: Dict[str, Any] | ClearMemoryInput) -> Dict[str, Any]:
    """
    清除分层记忆。

    输入参数:
        session_id: str - 会话ID
        level: str, optional - 要清除的层级 ("short", "mid", "long")，None表示全部
    """
    start = time.perf_counter()
    try:
        if isinstance(input_data, ClearMemoryInput):
            payload = input_data
        else:
            level = input_data.get("level")
            if level:
                level_enum = MemoryLevel(level)
            else:
                level_enum = None

            payload = ClearMemoryInput(
                session_id=input_data["session_id"],
                level=level_enum,
            )

        result = memory_agent.clear_memory(payload)

        duration_ms = int((time.perf_counter() - start) * 1000)
        if result.ok:
            return make_ok(TOOL_CLEAR_MEMORY, result.model_dump(), duration_ms)
        else:
            return make_error(TOOL_CLEAR_MEMORY, "CLEAR_FAILED", str(result.error) if result.error else "清除失败", duration_ms)

    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_error(TOOL_CLEAR_MEMORY, "INTERNAL_ERROR", str(e), duration_ms)
