"""
mcp_server.py - FastMCP Server 暴露知识库工具

使用 FastMCP 框架将现有的 kb_* 工具暴露为 MCP 协议工具。

设计原则：
1. 不修改现有 kb_* 工具函数
2. 复用现有的 Pydantic schemas
3. 通过 MCPToolAdapter 适配协议差异
4. 通过 Action Guard 进行治理检查
5. 支持文件系统 MCP 工具

启动方式：
    python -m app.tools.mcp_server [stdio|sse|http]
默认使用 stdio 传输（适合 MCP Client 如 Cline）

MCP Inspector 测试：
    npx @modelcontextprotocol/inspector python -m app.tools.mcp_server
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

# FastMCP import - 使用延迟导入以便在未安装时给出友好错误
try:
    from fastmcp import FastMCP
except ImportError:
    print("Error: fastmcp is not installed. Please run: pip install fastmcp>=2.0.0")
    sys.exit(1)

from pydantic import Field

# 导入底层工具函数（不含 @tool 装饰器）
from app.tools.kb_search_tools import kb_search_knowledge_base
from app.tools.kb_qa_tools import (
    kb_rewrite_query,
    kb_assemble_context,
    kb_generate_answer,
    kb_answer_question,
)
from app.tools.kb_index_tools import kb_index_document
from app.tools.kb_import_tools import kb_import_file, kb_import_folder
from app.tools.kb_history_tools import kb_create_chat_session, kb_get_chat_history
from app.tools.kb_summary_tools import kb_summarize_document
from app.tools.memory_tools import kb_store_memory, kb_get_memory_context, kb_clear_memory
from app.tools.rewrite_tools import kb_rewrite_query_v2

# 导入 Pydantic schemas
from app.tools.schemas import (
    KBSearchKnowledgeBaseInput,
    KBIndexDocumentInput,
    KBImportFileInput,
    KBImportFolderInput,
    KBCreateChatSessionInput,
    KBGetChatHistoryInput,
    KBSummarizeDocumentInput,
    KBAnswerQuestionInput,
    KBAssembleContextInput,
    KBGenerateAnswerInput,
    KBRewriteQueryInput,
)

# 导入 MemoryAgent schemas
from app.agent.memory_agent.schemas import (
    StoreMessageInput,
    GetMemoryContextInput,
    ClearMemoryInput,
    MemoryLevel,
)

# 导入适配器
from app.tools.mcp_adapter import MCPToolAdapter, adapt_tool_result

# 导入文件系统工具
from app.tools.filesystem_mcp import register_filesystem_tools

# 导入治理网关
from app.governance.gateway import governance_gateway
from app.governance.schemas import GovernanceDecision


# ==========================
# 创建 FastMCP 实例
# ==========================

mcp = FastMCP(
    "Knowledge Base Tools",
    include_stdlib=False,  # 不包含标准库工具
)


# ==========================
# 统一工具调用入口
# ==========================

def call_tool_via_dispatcher(
    tool_name: str,
    tool_args: dict[str, Any],
    session_id: str = "default",
    agent: str = "mcp",
) -> dict[str, Any]:
    """
    通过 ToolDispatcher 调用工具

    这是 MCP Server 调用工具的唯一入口
    所有工具调用都经过 Action Guard 检查
    """
    from app.tools.tool_dispatcher import invoke_tool

    return invoke_tool(
        tool_name=tool_name,
        args=tool_args,
        agent=agent,
        session_id=session_id,
    )


# ==========================
# 辅助函数：创建 MCP 工具包装
# ==========================

def make_mcp_tool(name: str, func: callable, schema_class: type | None = None):
    """创建 MCP 工具的装饰器工厂"""
    adapter = MCPToolAdapter(
        name=name,
        func=func,
        input_schema=schema_class,
        description=f"MCP wrapper for {name}",
    )
    return adapter.execute


# ==========================
# 注册知识搜索/问答工具
# ==========================


@mcp.tool(name="kb_search_knowledge_base")
def mcp_search_knowledge_base(
    query: str = Field(..., description="Search query for the knowledge base."),
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of hits to return."),
    include_full_text: bool = Field(True, description="Whether to include full chunk text."),
    text_max_length: int = Field(2000, ge=100, le=10000),
    session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """在知识库中搜索相关内容。适用于事实问答、文档检索、概念解释、对比分析。"""
    tool_args = {
        "query": query,
        "top_k": top_k,
        "include_full_text": include_full_text,
        "text_max_length": text_max_length,
    }
    result = call_tool_via_dispatcher("kb_search_knowledge_base", tool_args, session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


@mcp.tool(name="kb_answer_question")
def mcp_answer_question(
    question: str = Field(..., min_length=1, description="User question."),
    session_id: Optional[str] = Field(None, description="Chat session ID."),
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve."),
    response_mode: str = Field("text", description="Response format: 'text' or 'structured'."),
    highlight: bool = Field(True, description="Whether to highlight source spans."),
    use_chat_context: bool = Field(True, description="Whether to use chat history."),
    governance_session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """完整 RAG 问答：改写查询 → 检索 → 组装上下文 → 生成答案。"""
    tool_args = {
        "question": question,
        "session_id": session_id,
        "top_k": top_k,
        "response_mode": response_mode,
        "highlight": highlight,
        "use_chat_context": use_chat_context,
    }
    result = call_tool_via_dispatcher("kb_answer_question", tool_args, governance_session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


@mcp.tool(name="kb_rewrite_query")
def mcp_rewrite_query(
    question: str = Field(..., min_length=1, description="User question to rewrite."),
    session_id: Optional[str] = Field(None, description="Chat session ID for history context."),
    use_history: bool = Field(True, description="Whether to include chat history in rewriting."),
    governance_session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """将用户问题改写成适合检索的独立查询，结合对话历史理解代词。"""
    tool_args = {
        "question": question,
        "session_id": session_id,
        "use_history": use_history,
    }
    result = call_tool_via_dispatcher("kb_rewrite_query", tool_args, governance_session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


@mcp.tool(name="kb_assemble_context")
def mcp_assemble_context(
    hits: list = Field(..., description="Retrieved search hits to assemble into context."),
    max_chunks: int = Field(6, ge=1, le=20, description="Maximum number of chunks to include."),
    governance_session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """将搜索结果（hits）组装成可阅读的上下文文本。"""
    tool_args = {
        "hits": hits,
        "max_chunks": max_chunks,
    }
    result = call_tool_via_dispatcher("kb_assemble_context", tool_args, governance_session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


@mcp.tool(name="kb_generate_answer")
def mcp_generate_answer(
    question: str = Field(..., min_length=1, description="User question."),
    context: str = Field(..., description="Assembled context from knowledge base."),
    history_text: str = Field("", description="Formatted chat history for context."),
    response_mode: str = Field("text", description="Response format: 'text' or 'structured'."),
    governance_session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """根据组装好的上下文和问题生成答案。"""
    tool_args = {
        "question": question,
        "context": context,
        "history_text": history_text,
        "response_mode": response_mode,
    }
    result = call_tool_via_dispatcher("kb_generate_answer", tool_args, governance_session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


# ==========================
# 注册文档管理工具
# ==========================


@mcp.tool(name="kb_import_file")
def mcp_import_file(
    file_path: str = Field(..., description="Absolute or relative path to a local file."),
    governance_session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """导入单个文档文件（PDF、DOCX、TXT、MD）到知识库。"""
    tool_args = {"file_path": file_path}
    result = call_tool_via_dispatcher("kb_import_file", tool_args, governance_session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


@mcp.tool(name="kb_import_folder")
def mcp_import_folder(
    folder: str = Field(..., description="Absolute or relative path to a local folder."),
    governance_session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """批量导入文件夹中的所有文档到知识库。"""
    tool_args = {"folder": folder}
    result = call_tool_via_dispatcher("kb_import_folder", tool_args, governance_session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


@mcp.tool(name="kb_index_document")
def mcp_index_document(
    document_id: int = Field(..., ge=1, description="Document ID to index."),
    chunk_size: int = Field(800, ge=100, le=4000, description="Chunk size for splitting."),
    overlap: int = Field(120, ge=0, le=1000, description="Overlap between chunks."),
    governance_session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """根据 document_id 构建文档索引，将内容切分并写入向量数据库。"""
    tool_args = {
        "document_id": document_id,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
    result = call_tool_via_dispatcher("kb_index_document", tool_args, governance_session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


@mcp.tool(name="kb_summarize_document")
def mcp_summarize_document(
    document_id: int = Field(..., ge=1, description="Document ID to summarize."),
    governance_session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """根据 document_id 对指定文档做摘要。"""
    tool_args = {"document_id": document_id}
    result = call_tool_via_dispatcher("kb_summarize_document", tool_args, governance_session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


# ==========================
# 注册会话/历史工具
# ==========================


@mcp.tool(name="kb_create_chat_session")
def mcp_create_chat_session(
    session_id: Optional[str] = Field(None, description="Optional session ID (generated if not provided)."),
    title: Optional[str] = Field(None, description="Optional session title."),
    metadata: dict = Field(default_factory=dict, description="Optional metadata."),
    governance_session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """创建一个新的聊天会话。"""
    tool_args = {
        "session_id": session_id,
        "title": title,
        "metadata": metadata,
    }
    result = call_tool_via_dispatcher("kb_create_chat_session", tool_args, governance_session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


@mcp.tool(name="kb_get_chat_history")
def mcp_get_chat_history(
    session_id: str = Field(..., description="Session ID to get history for."),
    limit: int = Field(20, ge=1, le=100, description="Maximum number of messages to return."),
    governance_session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """根据 session_id 获取历史对话消息。"""
    tool_args = {
        "session_id": session_id,
        "limit": limit,
    }
    result = call_tool_via_dispatcher("kb_get_chat_history", tool_args, governance_session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


# ==========================
# 注册记忆工具
# ==========================


@mcp.tool(name="kb_store_memory")
def mcp_store_memory(
    session_id: str = Field(..., description="Session ID."),
    role: str = Field(..., description="Role: 'user' or 'assistant'."),
    message: str = Field(..., description="Message content."),
    metadata: Optional[dict] = Field(None, description="Optional metadata."),
    governance_session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """存储新消息并更新分层记忆。"""
    tool_args = {
        "session_id": session_id,
        "role": role,
        "message": message,
        "metadata": metadata or {},
    }
    result = call_tool_via_dispatcher("kb_store_memory", tool_args, governance_session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


@mcp.tool(name="kb_get_memory_context")
def mcp_get_memory_context(
    session_id: str = Field(..., description="Session ID."),
    question: Optional[str] = Field(None, description="Optional question for long-term memory retrieval."),
    include_levels: Optional[list[str]] = Field(None, description="Levels to retrieve: ['short', 'mid', 'long']."),
    governance_session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """获取分层记忆上下文（供查询改写用）。"""
    levels = [MemoryLevel(l) for l in (include_levels or ["short", "mid", "long"])]
    tool_args = {
        "session_id": session_id,
        "question": question,
        "include_levels": levels,
    }
    result = call_tool_via_dispatcher("kb_get_memory_context", tool_args, governance_session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


@mcp.tool(name="kb_clear_memory")
def mcp_clear_memory(
    session_id: str = Field(..., description="Session ID."),
    level: Optional[str] = Field(None, description="Level to clear: 'short', 'mid', 'long', or None for all."),
    governance_session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """清除分层记忆。"""
    tool_args = {
        "session_id": session_id,
        "level": level,
    }
    result = call_tool_via_dispatcher("kb_clear_memory", tool_args, governance_session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


@mcp.tool(name="kb_rewrite_query_v2")
def mcp_rewrite_query_v2(
    question: str = Field(..., min_length=1, description="User question to rewrite."),
    session_id: str = Field(..., description="Session ID."),
    use_history: bool = Field(True, description="Whether to use chat history."),
    governance_session_id: str = Field("default", description="Session ID for governance tracking."),
) -> str:
    """基于分层记忆的查询改写（新版）。使用MemoryAgent管理的三层记忆改写问题。"""
    tool_args = {
        "question": question,
        "session_id": session_id,
        "use_history": use_history,
    }
    result = call_tool_via_dispatcher("kb_rewrite_query_v2", tool_args, governance_session_id, "mcp")
    adapted = adapt_tool_result(result)
    return adapted.content[0].text if adapted.content else str(result)


# ==========================
# 注册文件系统工具
# ==========================

# 注册文件系统 MCP 工具
register_filesystem_tools(mcp)


# ==========================
# 启动函数
# ==========================

def run_server(transport: str = "stdio"):
    """
    启动 MCP Server

    Args:
        transport: 传输协议，可选 "stdio", "sse", "streamable-http"
    """
    print(f"Starting MCP Server with transport: {transport}", file=sys.stderr)
    mcp.run(transport=transport)


if __name__ == "__main__":
    import sys

    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    run_server(transport)
