from __future__ import annotations

import uuid
from typing import Any, Dict, List

from app.db import create_chat_session, get_chat_session
from app.qa.session import get_chat_history
from app.tools.base import ToolExecutionError, require_field, run_tool
from app.tools.schemas import (
    ChatMessage,
    KBCreateChatSessionInput,
    KBCreateChatSessionOutput,
    KBGetChatHistoryInput,
    KBGetChatHistoryOutput,
)

TOOL_CREATE_CHAT_SESSION = "kb_create_chat_session"
TOOL_GET_CHAT_HISTORY = "kb_get_chat_history"


def _to_iso(dt):
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    return dt.isoformat()

def kb_create_chat_session(input_data: KBCreateChatSessionInput | Dict[str, Any]) -> Dict[str, Any]:
    payload = (
        input_data
        if isinstance(input_data, KBCreateChatSessionInput)
        else KBCreateChatSessionInput(**input_data)
    )

    def _execute() -> Dict[str, Any]:
        
        resolved_session_id = payload.session_id or str(uuid.uuid4())
        session_id = create_chat_session(
            session_id=resolved_session_id,
            title=payload.title,
            metadata=payload.metadata,
        )
        

   
        session = get_chat_session(session_id)

        if not session:
            raise ToolExecutionError("SESSION_CREATE_FAILED", "failed to create session")

         

        normalized = KBCreateChatSessionOutput(
            session_id=require_field(session.get("session_id"), "SESSION_ID_MISSING", "session_id missing"),
            title=session.get("title"),
            summary_text=session.get("summary_text"),
            metadata=session.get("metadata") or session.get("metadata_json") or {},
            created_at=_to_iso(session.get("created_at")),
            updated_at=_to_iso(session.get("updated_at")),
        )
        return normalized.model_dump()

    return run_tool(TOOL_CREATE_CHAT_SESSION, _execute)


def kb_get_chat_history(input_data: KBGetChatHistoryInput | Dict[str, Any]) -> Dict[str, Any]:
    payload = (
        input_data
        if isinstance(input_data, KBGetChatHistoryInput)
        else KBGetChatHistoryInput(**input_data)
    )

    def _execute() -> Dict[str, Any]:
        result = get_chat_history(session_id=payload.session_id, limit=payload.limit)
        raw = result if isinstance(result, dict) else vars(result)

        messages_raw = raw.get("messages", []) or []
        messages: List[ChatMessage] = []

        for item in messages_raw:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif not isinstance(item, dict):
                item = vars(item)

            messages.append(
                ChatMessage(
                    role=item.get("role", ""),
                    content=item.get("content", ""),
                    created_at=item.get("created_at"),
                )
            )

        normalized = KBGetChatHistoryOutput(
            session_id=raw.get("session_id", payload.session_id),
            title=raw.get("title"),
            summary_text=raw.get("summary_text"),
            messages=messages,
        )
        return normalized.model_dump()

    return run_tool(TOOL_GET_CHAT_HISTORY, _execute)