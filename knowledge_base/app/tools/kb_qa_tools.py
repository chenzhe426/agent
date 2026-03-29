"""
KB QA tools: agent-facing tool functions with Chinese prompts.
All tools use LangChain's @tool decorator directly here.

Triple-layer cleanup (Phase 4):
  - @tool applied HERE, not re-applied in agent.py
  - Helpers deduplicated: imports from app/qa/config.py where not Chinese-specific
  - Tool functions re-use app/qa/ for retrieval/context/qa pipeline where applicable
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from langchain.tools import tool

from app.db import get_chat_messages
from app.services.common import normalize_whitespace, safe_get, to_float
from app.services.llm_service import chat_completion
from app.qa.context import assemble_context as _assemble_context
from app.qa.pipeline import rewrite_query_with_history as _rewrite_query_with_history
from app.retrieval.service import retrieve_chunks
from app.tools.base import ToolExecutionError, make_error, make_ok, require_field
from app.tools.schemas import (
    AnswerSource,
    KBAnswerQuestionInput,
    KBAnswerQuestionOutput,
    KBAssembleContextInput,
    KBAssembleContextOutput,
    KBRewriteQueryInput,
    KBRewriteQueryOutput,
    KBGenerateAnswerInput,
    KBGenerateAnswerOutput,
    SourceHighlightSpan,
    ToolResult,
)


TOOL_REWRITE_QUERY = "kb_rewrite_query"
TOOL_ASSEMBLE_CONTEXT = "kb_assemble_context"
TOOL_GENERATE_ANSWER = "kb_generate_answer"
TOOL_ANSWER_QUESTION = "kb_answer_question"

CHAT_HISTORY_LIMIT = 6
QA_MAX_CONTEXT_CHUNKS = 6
QA_STRUCTURED_ENABLE = True
QA_HIGHLIGHT_ENABLE = True


# Import shared helpers from app/qa (non Chinese-specific)
from app.qa.config import (
    _page_label,
    _truncate_text,
    _normalize_chunk_source as _qa_normalize_chunk_source,
    _build_highlight_spans as _qa_build_highlight_spans,
    _extract_query_terms as _qa_extract_query_terms,
)


def _normalize_chunk_source(chunk: Dict[str, Any]) -> AnswerSource:
    """Build an AnswerSource pydantic model from a chunk dict."""
    return AnswerSource(
        chunk_id=chunk.get("chunk_id"),
        document_id=chunk.get("document_id"),
        title=chunk.get("title", "") or "",
        section_title=chunk.get("section_title", "") or "",
        section_path=chunk.get("section_path", "") or "",
        page_start=chunk.get("page_start"),
        page_end=chunk.get("page_end"),
        quote=normalize_whitespace(chunk.get("chunk_text", "") or chunk.get("search_text", "") or ""),
        score=to_float(chunk.get("score")),
        highlight_spans=[],
    )


def _build_highlight_spans(text: str, terms: List[str]) -> List[Dict[str, Any]]:
    """Build highlight spans. Implementation shared with app/qa/config."""
    return _qa_build_highlight_spans(text, terms)


def _format_history_for_prompt(history: List[Dict[str, Any]], limit: int = CHAT_HISTORY_LIMIT) -> str:
    """Format chat history with Chinese labels (agent-facing)."""
    if not history:
        return ""
    selected = history[-limit:]
    lines: List[str] = []
    for msg in selected:
        role = msg.get("role", "user")
        message = normalize_whitespace(msg.get("message") or msg.get("content") or "")
        if not message:
            continue
        role_label = "用户" if role == "user" else "助手" if role == "assistant" else "系统"
        lines.append(f"{role_label}：{message}")
    return "\n".join(lines).strip()


def _build_answer_prompt(question: str, context: str, history_text: str = "") -> tuple[str, str]:
    """Chinese prompt for free-text answer generation (agent-facing)."""
    system_prompt = (
        "你是一个基于知识库回答问题的助手。\n"
        "你必须优先依据提供的上下文回答。\n"
        "如果上下文不足以支持结论，要明确说「知识库中没有足够信息支持该结论」。\n"
        "不要编造页码、来源或事实。\n"
        "回答要准确、简洁、中文输出。"
    )
    user_parts = []
    if history_text:
        user_parts.extend(["对话历史：", history_text, ""])
    user_parts.extend(["知识库上下文：", context or "（无可用上下文）", "", f"问题：{question}", "", "请直接给出答案。"])
    return system_prompt, "\n".join(user_parts)


def _build_structured_answer_prompt(question: str, context: str, history_text: str = "") -> tuple[str, str]:
    """Chinese prompt for structured answer generation (agent-facing)."""
    system_prompt = (
        "你是一个基于知识库回答问题的助手。\n"
        "你必须严格依据上下文作答，不要编造。\n"
        "输出必须是合法 JSON，且只输出 JSON，不要有额外解释。\n"
        '{"answer":"","summary":"","key_points":[],"confidence":0.0}\n'
        "其中 confidence 为 0 到 1 之间的小数。\n"
        "如果上下文不足，answer 和 summary 中要明确说明信息不足。"
    )
    user_parts = []
    if history_text:
        user_parts.extend(["对话历史：", history_text, ""])
    user_parts.extend(["知识库上下文：", context or "（无可用上下文）", "", f"问题：{question}", "", "请输出 JSON："])
    return system_prompt, "\n".join(user_parts)


def _safe_parse_structured_answer(raw_text: str) -> Dict[str, Any]:
    """Parse LLM structured output (agent-facing JSON format)."""
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {"answer": "", "summary": "", "key_points": [], "confidence": 0.0}
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, re.S)
    if fenced:
        raw_text = fenced.group(1).strip()
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return {
                "answer": normalize_whitespace(data.get("answer", "")),
                "summary": normalize_whitespace(data.get("summary", "")),
                "key_points": data.get("key_points") if isinstance(data.get("key_points"), list) else [],
                "confidence": float(data.get("confidence", 0.0) or 0.0),
            }
    except Exception:
        pass
    return {"answer": normalize_whitespace(raw_text), "summary": "", "key_points": [], "confidence": 0.0}


def _estimate_confidence(retrieved_chunks: List[Dict[str, Any]], answer_text: str) -> float:
    """Estimate confidence from retrieval scores."""
    if not retrieved_chunks:
        return 0.05
    top_score = max([to_float(c.get("score")) for c in retrieved_chunks] + [0.0])
    avg_top3 = sum(to_float(c.get("score")) for c in retrieved_chunks[:3]) / max(1, min(3, len(retrieved_chunks)))
    confidence = 0.55 * top_score + 0.45 * avg_top3
    if "没有足够信息" in (answer_text or ""):
        confidence *= 0.65
    return round(max(0.0, min(1.0, confidence)), 4)


def _build_sources(retrieved_chunks: List[Dict[str, Any]], highlight: bool, highlight_terms: List[str], limit: int = 5) -> List[AnswerSource]:
    """Build AnswerSource list from retrieved chunks."""
    sources: List[AnswerSource] = []
    for chunk in retrieved_chunks[:limit]:
        source = _normalize_chunk_source(chunk)
        quote = normalize_whitespace(source.quote or "")
        source.quote = _truncate_text(quote, max_len=260)
        if highlight and QA_HIGHLIGHT_ENABLE:
            source.highlight_spans = [
                SourceHighlightSpan(**span) for span in _build_highlight_spans(quote, highlight_terms)
            ]
        sources.append(source)
    return sources


# -------------------------
# Tool implementations (@tool applied directly here, not in agent.py)
# -------------------------


@tool
def kb_rewrite_query(input_data: KBAnswerQuestionInput | Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite user question using conversation history (agent tool)."""
    start = time.perf_counter()
    try:
        payload = input_data if isinstance(input_data, KBRewriteQueryInput) else KBRewriteQueryInput(**input_data)
        question = normalize_whitespace(payload.question)
        if not question:
            raise ToolExecutionError("EMPTY_QUESTION", "question cannot be empty")

        history: List[Dict[str, Any]] = []
        if payload.use_history and payload.session_id:
            rows = get_chat_messages(payload.session_id, limit=CHAT_HISTORY_LIMIT) or []
            history = [
                {"role": r.get("role", ""), "message": r.get("message") or r.get("content", "")}
                for r in rows
                if r.get("role") in {"user", "assistant"}
            ]

        rewritten_query = _rewrite_query_with_history(history, question) if history else question
        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_ok(TOOL_REWRITE_QUERY, KBRewriteQueryOutput(
            original_question=question,
            rewritten_query=rewritten_query,
            used_history=payload.use_history and bool(history),
        ).model_dump(), duration_ms)
    except ToolExecutionError as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_error(TOOL_REWRITE_QUERY, e.code, e.message, duration_ms)
    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_error(TOOL_REWRITE_QUERY, "INTERNAL_ERROR", str(e), duration_ms)


@tool
def kb_assemble_context(input_data: KBAssembleContextInput | Dict[str, Any]) -> Dict[str, Any]:
    """Assemble readable context from search hits (agent tool)."""
    start = time.perf_counter()
    try:
        payload = input_data if isinstance(input_data, KBAssembleContextInput) else KBAssembleContextInput(**input_data)

        raw_chunks: List[Dict[str, Any]] = []
        for hit in payload.hits:
            chunk: Dict[str, Any] = {
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "title": hit.title,
                "chunk_index": hit.chunk_index,
                "chunk_type": hit.chunk_type,
                "section_path": hit.section_path,
                "section_title": hit.section_title,
                "page_start": hit.page_start,
                "page_end": hit.page_end,
                "chunk_text": hit.text,
                "search_text": hit.text,
                "score": hit.score,
            }
            raw_chunks.append(chunk)

        context = _assemble_context(raw_chunks, max_chunks=payload.max_chunks)
        sources = _build_sources(raw_chunks, highlight=False, highlight_terms=[], limit=payload.max_chunks)

        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_ok(TOOL_ASSEMBLE_CONTEXT, KBAssembleContextOutput(
            context=context,
            chunk_count=len(raw_chunks),
            sources=sources,
        ).model_dump(), duration_ms)
    except ToolExecutionError as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_error(TOOL_ASSEMBLE_CONTEXT, e.code, e.message, duration_ms)
    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_error(TOOL_ASSEMBLE_CONTEXT, "INTERNAL_ERROR", str(e), duration_ms)


@tool
def kb_generate_answer(input_data: KBGenerateAnswerInput | Dict[str, Any]) -> Dict[str, Any]:
    """Generate answer from context (agent tool, Chinese prompts)."""
    start = time.perf_counter()
    try:
        payload = input_data if isinstance(input_data, KBGenerateAnswerInput) else KBGenerateAnswerInput(**input_data)
        question = normalize_whitespace(payload.question)
        if not question:
            raise ToolExecutionError("EMPTY_QUESTION", "question cannot be empty")

        context = payload.context or "（无可用上下文）"
        history_text = payload.history_text or ""

        answer_text = ""
        confidence = None
        key_points: List[str] = []
        summary = ""

        if payload.response_mode == "structured" and QA_STRUCTURED_ENABLE:
            system_prompt, user_prompt = _build_structured_answer_prompt(question, context, history_text)
            raw_output = chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)
            parsed = _safe_parse_structured_answer(raw_output)
            answer_text = parsed.get("answer", "") or ""
            confidence = parsed.get("confidence", 0.0)
            key_points = parsed.get("key_points", []) or []
            summary = parsed.get("summary", "") or ""
        else:
            system_prompt, user_prompt = _build_answer_prompt(question, context, history_text)
            answer_text = normalize_whitespace(chat_completion(system_prompt=system_prompt, user_prompt=user_prompt))

        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_ok(TOOL_GENERATE_ANSWER, KBGenerateAnswerOutput(
            answer=answer_text,
            confidence=confidence,
            key_points=key_points,
            summary=summary,
            sources=[],
        ).model_dump(), duration_ms)
    except ToolExecutionError as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_error(TOOL_GENERATE_ANSWER, e.code, e.message, duration_ms)
    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_error(TOOL_GENERATE_ANSWER, "INTERNAL_ERROR", str(e), duration_ms)


@tool
def kb_answer_question(input_data: KBAnswerQuestionInput | Dict[str, Any]) -> Dict[str, Any]:
    """Full RAG pipeline: rewrite → retrieve → assemble → answer (agent tool)."""
    start = time.perf_counter()
    try:
        payload = input_data if isinstance(input_data, KBAnswerQuestionInput) else KBAnswerQuestionInput(**input_data)
        question = normalize_whitespace(payload.question)
        if not question:
            raise ToolExecutionError("EMPTY_QUESTION", "question cannot be empty")

        history: List[Dict[str, Any]] = []
        if payload.use_chat_context and payload.session_id:
            rows = get_chat_messages(payload.session_id, limit=CHAT_HISTORY_LIMIT) or []
            history = [
                {"role": r.get("role", ""), "message": r.get("message") or r.get("content", "")}
                for r in rows
                if r.get("role") in {"user", "assistant"}
            ]

        rewritten_query = _rewrite_query_with_history(history, question) if history else question
        raw_chunks = retrieve_chunks(rewritten_query, top_k=payload.top_k) or []

        context = _assemble_context(raw_chunks, max_chunks=QA_MAX_CONTEXT_CHUNKS)
        history_text = _format_history_for_prompt(history) if payload.use_chat_context else ""

        answer_text = ""
        structured: Dict[str, Any] | None = None
        key_points: List[str] = []
        summary = ""

        if payload.response_mode == "structured" and QA_STRUCTURED_ENABLE:
            system_prompt, user_prompt = _build_structured_answer_prompt(question, context, history_text)
            raw_output = chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)
            parsed = _safe_parse_structured_answer(raw_output)
            answer_text = parsed.get("answer", "") or ""
            key_points = parsed.get("key_points", []) or []
            summary = parsed.get("summary", "") or ""
            confidence = parsed.get("confidence", 0.0)
            structured = {
                "answer": answer_text,
                "summary": summary,
                "key_points": key_points,
                "confidence": confidence,
                "sources": [],
            }
        else:
            system_prompt, user_prompt = _build_answer_prompt(question, context, history_text)
            answer_text = normalize_whitespace(chat_completion(system_prompt=system_prompt, user_prompt=user_prompt))
            confidence = _estimate_confidence(raw_chunks, answer_text)

        highlight_terms = _qa_extract_query_terms(question, rewritten_query, raw_chunks)
        sources = _build_sources(raw_chunks, highlight=payload.highlight, highlight_terms=highlight_terms, limit=min(5, payload.top_k))

        if structured is not None:
            structured["sources"] = [s.model_dump() for s in sources]

        retrieved_chunks = [
            {
                "chunk_id": c.get("chunk_id"),
                "document_id": c.get("document_id"),
                "title": c.get("title", "") or "",
                "chunk_index": c.get("chunk_index"),
                "chunk_type": c.get("chunk_type"),
                "section_path": c.get("section_path", "") or "",
                "section_title": c.get("section_title", "") or "",
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "score": c.get("score"),
                "text": c.get("chunk_text", "") or c.get("search_text", "") or "",
            }
            for c in raw_chunks
        ]

        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_ok(TOOL_ANSWER_QUESTION, KBAnswerQuestionOutput(
            question=question,
            rewritten_query=rewritten_query,
            answer=answer_text,
            confidence=confidence,
            structured=structured,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
            session_id=payload.session_id or "",
        ).model_dump(), duration_ms)
    except ToolExecutionError as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_error(TOOL_ANSWER_QUESTION, e.code, e.message, duration_ms)
    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return make_error(TOOL_ANSWER_QUESTION, "INTERNAL_ERROR", str(e), duration_ms)
