"""
mcp_adapter.py - LangChain 工具与 MCP 协议之间的适配层

职责：
1. 将 MCP 的 JSON 输入参数转换为工具函数的 input_data 格式
2. 将工具函数的 {"ok", "data", "error", "meta"} 输出转换为 MCP CallToolResult
3. 处理 Pydantic BaseModel 参数验证

MCP 协议使用 JSON-RPC 2.0，工具调用请求格式：
{
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": {...}
  }
}

工具返回格式：
CallToolResult(content=[TextContent(type="text", text="...")], isError=False)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from app.tools.base import ToolExecutionError


try:
    from mcp.types import CallToolResult, TextContent
except ImportError:
    # Fallback types if MCP is not installed
    class CallToolResult:
        def __init__(self, content: list, isError: bool = False):
            self.content = content
            self.isError = isError


    class TextContent:
        def __init__(self, type: str = "text", text: str = ""):
            self.type = type
            self.text = text


class MCPToolAdapter:
    """
    包装一个 kb_* 工具函数为 MCP 可调用的适配器。

    负责：
    1. 参数验证和转换
    2. 调用底层 kb_* 函数
    3. 将 ToolResult 格式转换为 MCP CallToolResult
    """

    def __init__(
        self,
        name: str,
        func: callable,
        input_schema: Type[BaseModel] | None = None,
        description: str = "",
    ):
        """
        Args:
            name: 工具名称
            func: 底层的 kb_* 函数
            input_schema: 输入参数的 Pydantic 模型类
            description: 工具描述
        """
        self.name = name
        self.func = func
        self.input_schema = input_schema
        self.description = description

    def parse_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 MCP JSON 参数转换为工具函数期望的格式。

        如果提供了 input_schema，则进行 Pydantic 验证和转换。
        否则直接返回原始字典。
        """
        if self.input_schema is None:
            return arguments

        try:
            # Pydantic 验证
            validated = self.input_schema(**arguments)
            # 转换为 dict 格式
            if hasattr(validated, "model_dump"):
                return validated.model_dump()
            return vars(validated)
        except Exception:
            # 如果验证失败，返回原始参数（工具函数会自行处理错误）
            return arguments

    def execute(self, arguments: Dict[str, Any]) -> CallToolResult:
        """
        执行工具并转换为 MCP 格式。

        Args:
            arguments: MCP 传来的参数字典

        Returns:
            CallToolResult: MCP 协议格式的返回结果
        """
        try:
            # 1. 解析和验证参数
            input_data = self.parse_arguments(arguments)

            # 2. 调用底层工具函数
            result = self.func(input_data)

            # 3. 转换为 MCP 格式
            return self._to_mcp_result(result)

        except ToolExecutionError as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {e.message}")],
                isError=True,
            )
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True,
            )

    def _to_mcp_result(self, result: Dict[str, Any]) -> CallToolResult:
        """
        将工具返回的 ToolResult 格式转换为 MCP CallToolResult。

        ToolResult 格式:
            {
                "ok": bool,
                "data": dict | None,
                "error": {"code": str, "message": str} | None,
                "meta": {"tool_name": str, "duration_ms": int}
            }

        CallToolResult 格式:
            CallToolResult(content=[TextContent(type="text", text="...")], isError=False)
        """
        if not isinstance(result, dict):
            result = {"ok": True, "data": result}

        if result.get("ok", True):
            data = result.get("data", {})
            # 将 data 格式化为可读文本
            if data is None:
                text = "OK"
            elif isinstance(data, dict):
                # 对于结构化数据，返回 JSON 格式化字符串
                text = json.dumps(data, ensure_ascii=False, indent=2)
            elif isinstance(data, list):
                text = json.dumps(data, ensure_ascii=False, indent=2)
            else:
                text = str(data)

            return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)
        else:
            error = result.get("error", {})
            if isinstance(error, dict):
                error_msg = f"[{error.get('code', 'ERROR')}] {error.get('message', 'Unknown error')}"
            else:
                error_msg = str(error)

            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {error_msg}")],
                isError=True,
            )


def create_mcp_adapter(
    name: str,
    func: callable,
    schema_class: Type[BaseModel] | None = None,
    description: str = "",
) -> MCPToolAdapter:
    """
    工厂函数：创建 MCP 工具适配器。

    Args:
        name: 工具名称
        func: 底层的 kb_* 函数
        schema_class: 输入参数的 Pydantic 模型类
        description: 工具描述

    Returns:
        MCPToolAdapter 实例
    """
    return MCPToolAdapter(
        name=name,
        func=func,
        input_schema=schema_class,
        description=description,
    )


def adapt_tool_result(result: Dict[str, Any]) -> CallToolResult:
    """
    快捷函数：将单个 ToolResult 字典直接转换为 MCP 格式。

    用于直接转换不通过适配器的场景。
    """
    adapter = MCPToolAdapter(name="", func=lambda x: x)
    return adapter._to_mcp_result(result)
