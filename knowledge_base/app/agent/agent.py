from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
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


# Direct passthrough wrappers (no re-application of @tool — tools in kb_qa_tools.py
# are already LangChain Tool objects after Phase 4 unification)


def search_knowledge_base(query: str, top_k: int = 5, include_full_text: bool = False) -> dict:
    """在知识库中搜索相关内容。适用于事实问答、文档检索、概念解释、对比分析。"""
    return kb_search_knowledge_base({"query": query, "top_k": top_k, "include_full_text": include_full_text})


def summarize_document(document_id: int) -> dict:
    """根据 document_id 对指定文档做摘要。"""
    return kb_summarize_document({"document_id": document_id})


def get_chat_history(session_id: str, limit: int = 20) -> dict:
    """根据 session_id 获取历史对话消息。"""
    return kb_get_chat_history({"session_id": session_id, "limit": limit})


def import_file(file_path: str) -> dict:
    """导入单个文档文件（PDF、DOCX、TXT、MD）到知识库。"""
    return kb_import_file({"file_path": file_path})


def import_folder(folder: str) -> dict:
    """批量导入文件夹中的所有文档到知识库。"""
    return kb_import_folder({"folder": folder})


def index_document(document_id: int, chunk_size: int = 800, overlap: int = 120) -> dict:
    """根据 document_id 构建文档索引，将内容切分并写入向量数据库。"""
    return kb_index_document({"document_id": document_id, "chunk_size": chunk_size, "overlap": overlap})


def create_chat_session(session_id: str = None, title: str = None, metadata: dict = None) -> dict:
    """创建一个新的聊天会话。"""
    return kb_create_chat_session({"session_id": session_id, "title": title, "metadata": metadata or {}})


def rewrite_query(question: str, session_id: str = None, use_history: bool = True) -> dict:
    """将用户问题改写成适合检索的独立查询，结合对话历史理解代词，省略和上下文。"""
    return kb_rewrite_query({"question": question, "session_id": session_id, "use_history": use_history})


def assemble_context(hits: list, max_chunks: int = 6) -> dict:
    """将搜索结果（hits）组装成可阅读的上下文文本。"""
    return kb_assemble_context({"hits": hits, "max_chunks": max_chunks})


def generate_answer(question: str, context: str, history_text: str = "", response_mode: str = "text") -> dict:
    """根据组装好的上下文和问题生成答案。"""
    return kb_generate_answer({"question": question, "context": context, "history_text": history_text, "response_mode": response_mode})


def answer_question(
    question: str,
    session_id: str = None,
    top_k: int = 5,
    response_mode: str = "text",
    highlight: bool = True,
    use_chat_context: bool = True,
) -> dict:
    """完整 RAG 问答：改写查询 → 检索 → 组装上下文 → 生成答案。"""
    return kb_answer_question({
        "question": question,
        "session_id": session_id,
        "top_k": top_k,
        "response_mode": response_mode,
        "highlight": highlight,
        "use_chat_context": use_chat_context,
    })


# Tools list for agent construction — kb_qa_tools functions are already @tool-decorated
TOOLS = [
    search_knowledge_base,
    summarize_document,
    get_chat_history,
    import_file,
    import_folder,
    index_document,
    create_chat_session,
    rewrite_query,
    assemble_context,
    generate_answer,
    answer_question,
]


def create_kb_agent():
    model = init_chat_model(
        model=f"ollama:{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    agent = create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=(
            "你是一个知识库问答助手。\n"
            "优先使用工具检索知识库后再回答。\n"
            "如果用户的问题涉及事实、概念解释、文档内容、总结、对比，出处定位，应先调用合适的工具。\n"
            "系统已经提供了当前对话历史，你必须结合历史理解用户的追问。\n"
            "如果当前问题像是在追问上一轮内容，要结合已有对话历史理解代词，省略和上下文。\n"
            "回答时尽量依据工具返回结果，不要编造。"
        ),
    )
    return agent
