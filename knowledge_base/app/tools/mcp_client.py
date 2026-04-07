"""
mcp_client.py - Agent 使用的 MCP Client

Agent 通过此模块调用 MCP Server，实现：
1. 所有工具调用经过 MCP 协议层
2. 外部 MCP Client 可以看到/控制 Agent 的所有操作
3. governance_context 正确传递

设计原理：
- 使用 mcp 库的 ClientSession 通过 HTTP 与 MCP Server 通信
- MCP Server 需要以 streamable-http 模式运行
- 支持同步调用（内部处理 asyncio）
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional
from contextlib import asynccontextmanager

from loguru import logger


# MCP Client imports - 使用延迟导入
try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    MCP_AVAILABLE = True
except ImportError:
    logger.warning("mcp library not installed. MCP Client will not be available.")
    MCP_AVAILABLE = False
    ClientSession = None
    streamablehttp_client = None


class MCPClientManager:
    """
    MCP Client Session 管理器

    使用单例模式管理 ClientSession，确保整个 Agent 生命周期内
    复用同一个会话。

    使用方式：
    1. 启动 MCP Server: python -m app.tools.mcp_server streamable-http
    2. Agent 初始化时调用: await mcp_client_manager.connect()
    3. Agent 调用工具: await mcp_client_manager.call_tool(...)
    """

    _instance: Optional["MCPClientManager"] = None
    _session: Optional[ClientSession] = None
    _server_url: str = ""
    _loop: Optional[asyncio.AbstractEventLoop] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 默认从环境变量读取，或使用本地地址
        if not self._server_url:
            self._server_url = os.getenv(
                "MCP_SERVER_URL",
                "http://127.0.0.1:8000/mcp"
            )

    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._session is not None

    @property
    def server_url(self) -> str:
        """获取服务器 URL"""
        return self._server_url

    def set_server_url(self, url: str) -> None:
        """设置服务器 URL（在连接前调用）"""
        self._server_url = url
        logger.info(f"MCP Client server URL set to: {url}")

    async def connect(self, server_url: Optional[str] = None) -> None:
        """
        连接到 MCP Server

        Args:
            server_url: MCP Server 地址，默认使用环境变量 MCP_SERVER_URL
        """
        if not MCP_AVAILABLE:
            raise RuntimeError("mcp library is not installed")

        if server_url:
            self._server_url = server_url

        if self._session is not None:
            logger.debug("MCP Client already connected")
            return

        logger.info(f"Connecting to MCP Server at {self._server_url}")

        try:
            # 使用 streamablehttp_client 创建客户端
            transport = await streamablehttp_client(self._server_url)
            self._session = ClientSession(**transport)
            await self._session.initialize()
            logger.info("MCP Client connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MCP Server: {e}")
            self._session = None
            raise

    async def disconnect(self) -> None:
        """断开与 MCP Server 的连接"""
        if self._session is not None:
            await self._session.close()
            self._session = None
            logger.info("MCP Client disconnected")

    async def call_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        governance_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        通过 MCP 协议调用工具

        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            governance_context: 治理上下文（从中提取 session_id）

        Returns:
            工具执行结果 (ToolResult 格式)
        """
        if not MCP_AVAILABLE:
            raise RuntimeError("mcp library is not installed")

        if self._session is None:
            raise RuntimeError(
                "MCP Client not connected. Call connect() first or ensure "
                "MCP Server is running at http://127.0.0.1:8000/mcp"
            )

        # 从 governance_context 提取 session_id
        session_id = "default"
        if governance_context:
            session_id = governance_context.get("session_id", "default")

        try:
            # 通过 MCP 协议调用工具
            result = await self._session.call_tool(
                name=tool_name,
                arguments={
                    **tool_args,
                    "governance_session_id": session_id,
                }
            )

            # 转换为标准 ToolResult 格式
            return self._convert_mcp_result(result)

        except Exception as e:
            logger.error(f"MCP call_tool failed for {tool_name}: {e}")
            raise

    def _convert_mcp_result(self, mcp_result) -> dict[str, Any]:
        """
        将 MCP CallToolResult 转换为标准 ToolResult 格式

        Args:
            mcp_result: MCP 库的 CallToolResult 对象

        Returns:
            标准格式的 ToolResult 字典
        """
        from app.tools.mcp_adapter import adapt_tool_result

        # mcp_result 是 CallToolResult 类型
        # 需要转换为 dict 格式再通过 adapt_tool_result 转换
        if hasattr(mcp_result, 'isError') and hasattr(mcp_result, 'content'):
            is_error = mcp_result.isError
            content = mcp_result.content

            # 构建内部 ToolResult 格式
            internal_result = {
                "ok": not is_error,
                "data": {"content": content},
                "error": None if not is_error else {
                    "code": "MCP_ERROR",
                    "message": str(content)
                }
            }

            # 通过适配器转换
            adapted = adapt_tool_result(internal_result)

            # 提取内容
            if hasattr(adapted, 'content') and adapted.content:
                return {
                    "ok": not is_error,
                    "data": adapted.content[0].text if adapted.content else str(content),
                    "error": adapted.isError if hasattr(adapted, 'isError') else None
                }
            return internal_result

        # fallback: 假设已经是正确格式
        return mcp_result


# 全局实例
mcp_client_manager = MCPClientManager()


@asynccontextmanager
async def mcp_client_context(server_url: Optional[str] = None):
    """
    异步上下文管理器，用于 MCP Client 生命周期管理

    使用方式：
        async with mcp_client_context("http://127.0.0.1:8000/mcp"):
            result = await mcp_client_manager.call_tool("kb_search_knowledge_base", {...})
    """
    await mcp_client_manager.connect(server_url)
    try:
        yield mcp_client_manager
    finally:
        await mcp_client_manager.disconnect()


def call_mcp_tool_sync(
    tool_name: str,
    tool_args: dict[str, Any],
    governance_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    同步版本的 MCP 工具调用（内部使用 asyncio）

    这是 Agent 调用的主要入口，提供同步接口避免修改 Agent 代码结构。

    Args:
        tool_name: 工具名称
        tool_args: 工具参数
        governance_context: 治理上下文

    Returns:
        工具执行结果

    Note:
        如果 MCP Client 未连接，会自动连接。如果连接失败，会抛出异常。
    """
    if not MCP_AVAILABLE:
        raise RuntimeError("mcp library is not installed")

    try:
        # 获取或创建事件循环
        try:
            loop = asyncio.get_running_loop()
            has_running_loop = True
        except RuntimeError:
            has_running_loop = False

        if has_running_loop:
            # 如果有运行中的循环，创建任务
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    _sync_call_wrapper,
                    tool_name,
                    tool_args,
                    governance_context
                )
                return future.result()
        else:
            # 没有运行中的循环，可以直接使用 run_until_complete
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # 如果尚未连接，先连接
                if not mcp_client_manager.is_connected:
                    loop.run_until_complete(mcp_client_manager.connect())

                return loop.run_until_complete(
                    mcp_client_manager.call_tool(tool_name, tool_args, governance_context)
                )
            finally:
                # 不关闭循环，因为可能还需要复用
                pass

    except Exception as e:
        logger.error(f"call_mcp_tool_sync failed: {e}")
        raise


def _sync_call_wrapper(
    tool_name: str,
    tool_args: dict[str, Any],
    governance_context: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    同步调用包装器（在线程池中执行）

    用于在有运行中事件循环时，通过线程执行异步代码
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # 如果尚未连接，先连接
        if not mcp_client_manager.is_connected:
            loop.run_until_complete(mcp_client_manager.connect())

        return loop.run_until_complete(
            mcp_client_manager.call_tool(tool_name, tool_args, governance_context)
        )
    finally:
        loop.close()


# Agent 使用的便捷函数
def get_mcp_client() -> MCPClientManager:
    """获取全局 MCP Client Manager 实例"""
    return mcp_client_manager


def is_mcp_available() -> bool:
    """检查 MCP Client 是否可用（库已安装且已连接）"""
    return MCP_AVAILABLE and mcp_client_manager.is_connected


def call_tool_mcp_or_local(
    tool_name: str,
    args: dict[str, Any],
    agent: str,
    session_id: str,
    governance_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    尝试通过 MCP 调用工具，如果 MCP 不可用则回退到本地调用

    这是 Agent 调用工具的推荐入口，确保：
    1. 如果 MCP Server 可用，所有调用都经过 MCP 协议层
    2. 如果 MCP Server 不可用，回退到直接调用（保持兼容性）

    Args:
        tool_name: 工具名称
        args: 工具参数
        agent: 来源 agent (如 "qa", "document")
        session_id: 会话 ID
        governance_context: 治理上下文

    Returns:
        工具执行结果
    """
    if not MCP_AVAILABLE:
        logger.debug("MCP library not available, using local invoke_tool")
        from app.tools.tool_dispatcher import invoke_tool
        return invoke_tool(
            tool_name=tool_name,
            args=args,
            agent=agent,
            session_id=session_id,
            governance_context=governance_context,
        )

    # 构造 governance_context（包含 session_id）
    call_context = (governance_context.copy() if governance_context else {})
    call_context["session_id"] = session_id

    try:
        # 尝试通过 MCP 调用
        if not mcp_client_manager.is_connected:
            # 尝试连接（使用环境变量或默认地址）
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(mcp_client_manager.connect())
                loop.close()
            except Exception as connect_err:
                logger.warning(f"Failed to connect to MCP Server, falling back to local: {connect_err}")
                from app.tools.tool_dispatcher import invoke_tool
                return invoke_tool(
                    tool_name=tool_name,
                    args=args,
                    agent=agent,
                    session_id=session_id,
                    governance_context=governance_context,
                )

        result = call_mcp_tool_sync(tool_name, args, call_context)
        logger.debug(f"MCP call succeeded for {tool_name}")
        return result

    except Exception as e:
        logger.warning(f"MCP call failed for {tool_name}, falling back to local: {e}")
        from app.tools.tool_dispatcher import invoke_tool
        return invoke_tool(
            tool_name=tool_name,
            args=args,
            agent=agent,
            session_id=session_id,
            governance_context=governance_context,
        )
