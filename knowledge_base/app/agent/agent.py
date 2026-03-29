"""
Agent tool wrappers and TOOLS list for LangGraph agent.

The graph (graph.py) imports TOOLS from this module.
TOOLS is a list of langchain_core.tools.Tool objects wrapping the underlying
kb_* functions.
"""

from langchain_core.tools import Tool

from app.tools import (
    kb_answer_question,
    kb_assemble_context,
    kb_create_chat_session,
    kb_generate_answer,
    kb_get_chat_history,
    kb_import_file,
    kb_import_folder,
    kb_index_document,
    kb_rewrite_query,
    kb_search_knowledge_base,
    kb_summarize_document,
)


def _make_tool(name: str, fn, description: str) -> Tool:
    """Create a Tool object from a function."""
    return Tool(
        name=name,
        description=description,
        func=lambda input_data: fn(input_data) if isinstance(input_data, dict) else fn({"input_data": input_data}),
    )


# Build TOOLS as list of langchain.tools.Tool objects
TOOLS = [
    Tool(
        name="kb_search_knowledge_base",
        description="在知识库中搜索相关内容。适用于事实问答、文档检索、概念解释、对比分析。输入: query(str), top_k(int,默认5), include_full_text(bool,默认False)",
        func=lambda input_data: kb_search_knowledge_base(input_data),
    ),
    Tool(
        name="kb_summarize_document",
        description="根据 document_id 对指定文档做摘要。输入: document_id(int)",
        func=lambda input_data: kb_summarize_document(input_data),
    ),
    Tool(
        name="kb_get_chat_history",
        description="根据 session_id 获取历史对话消息。输入: session_id(str), limit(int,默认20)",
        func=lambda input_data: kb_get_chat_history(input_data),
    ),
    Tool(
        name="kb_import_file",
        description="导入单个文档文件（PDF、DOCX、TXT、MD）到知识库。输入: file_path(str)",
        func=lambda input_data: kb_import_file(input_data),
    ),
    Tool(
        name="kb_import_folder",
        description="批量导入文件夹中的所有文档到知识库。输入: folder(str)",
        func=lambda input_data: kb_import_folder(input_data),
    ),
    Tool(
        name="kb_index_document",
        description="根据 document_id 构建文档索引，将内容切分并写入向量数据库。输入: document_id(int), chunk_size(int,默认800), overlap(int,默认120)",
        func=lambda input_data: kb_index_document(input_data),
    ),
    Tool(
        name="kb_create_chat_session",
        description="创建一个新的聊天会话。输入: session_id(str,可选), title(str,可选), metadata(dict,可选)",
        func=lambda input_data: kb_create_chat_session(input_data),
    ),
    Tool(
        name="kb_rewrite_query",
        description="将用户问题改写成适合检索的独立查询，结合对话历史理解代词，省略和上下文。输入: question(str), session_id(str,可选), use_history(bool,默认True)",
        func=lambda input_data: kb_rewrite_query(input_data),
    ),
    Tool(
        name="kb_assemble_context",
        description="将搜索结果（hits）组装成可阅读的上下文文本。输入: hits(list), max_chunks(int,默认6)",
        func=lambda input_data: kb_assemble_context(input_data),
    ),
    Tool(
        name="kb_generate_answer",
        description="根据组装好的上下文和问题生成答案。输入: question(str), context(str), history_text(str,可选), response_mode(str,默认text)",
        func=lambda input_data: kb_generate_answer(input_data),
    ),
    Tool(
        name="kb_answer_question",
        description="完整 RAG 问答：改写查询 → 检索 → 组装上下文 → 生成答案。输入: question(str), session_id(str,可选), top_k(int,默认5), response_mode(str,默认text), highlight(bool,默认True), use_chat_context(bool,默认True)",
        func=lambda input_data: kb_answer_question(input_data),
    ),
]
