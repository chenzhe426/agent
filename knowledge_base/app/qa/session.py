"""
Session and document operations for QA pipeline.
"""
from __future__ import annotations

from typing import Any

from app.db import (
    create_chat_session,
    get_chat_messages,
    get_chat_session,
    get_document_by_id,
    update_chat_session,
)
from app.services.common import normalize_whitespace, safe_get
from app.services.llm_service import summarize_text

from .config import (
    CHAT_HISTORY_LIMIT,
    CHAT_SUMMARY_TRIGGER_TURNS,
    _truncate_text,
)
from .context import _format_history_for_prompt


def summarize_document(document_id: int) -> dict[str, Any]:
    """Summarize a document by ID."""
    row = get_document_by_id(document_id)
    if not row:
        raise ValueError("document not found")

    title = row.get("title", "") or ""
    content = normalize_whitespace(row.get("content") or row.get("raw_text") or "")

    if not content:
        summary = "文档内容为空，无法摘要。"
    else:
        summary = summarize_text(content)

    return {
        "document_id": document_id,
        "title": title,
        "summary": normalize_whitespace(summary),
    }


def get_chat_history(session_id: str, limit: int = 20) -> dict[str, Any]:
    """Get full chat history for a session."""
    session = get_chat_session(session_id)
    messages = get_chat_messages(session_id, limit=limit)

    normalized_messages: list[dict[str, Any]] = []
    for msg in messages:
        normalized_messages.append(
            {
                "id": msg.get("id"),
                "session_id": msg.get("session_id"),
                "role": msg.get("role"),
                "message": msg.get("message"),
                "rewritten_query": msg.get("rewritten_query"),
                "sources": msg.get("sources") or [],
                "metadata": msg.get("metadata") or {},
                "created_at": msg.get("created_at"),
            }
        )

    return {
        "session": {
            "session_id": safe_get(session, "session_id"),
            "title": safe_get(session, "title"),
            "summary_text": safe_get(session, "summary_text"),
            "metadata": safe_get(session, "metadata", {}) or {},
            "created_at": safe_get(session, "created_at"),
            "updated_at": safe_get(session, "updated_at"),
        }
        if session
        else None,
        "messages": normalized_messages,
    }


def _maybe_update_session_summary(session_id: str) -> None:
    """Conditionally generate a session summary after CHAT_SUMMARY_TRIGGER_TURNS messages."""
    session = get_chat_session(session_id)
    if not session:
        return

    history = get_chat_messages(session_id, limit=CHAT_SUMMARY_TRIGGER_TURNS)
    if len(history) < CHAT_SUMMARY_TRIGGER_TURNS:
        return

    summary_exists = normalize_whitespace(session.get("summary_text", "") or "")
    if summary_exists:
        return

    history_text = _format_history_for_prompt(history, limit=CHAT_SUMMARY_TRIGGER_TURNS)
    if not history_text:
        return

    try:
        summary = summarize_text(history_text)
        summary = normalize_whitespace(summary)
        if summary:
            update_chat_session(session_id=session_id, summary_text=summary)
    except Exception:
        pass
