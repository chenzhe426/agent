from __future__ import annotations
import uuid
import json
from typing import Any, Dict, Generator, List, Optional

from app.agent.agent import create_kb_agent

from app.db import create_chat_session, insert_chat_message, get_chat_messages


def _extract_final_text(result: Dict[str, Any]) -> str:
    messages = result.get("messages", []) or []
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(msg, dict):
            c = msg.get("content")
            if isinstance(c, str) and c.strip():
                return c
    return ""


def _serialize_message(msg: Any) -> Dict[str, Any]:
    if isinstance(msg, dict):
        return {
            "type": msg.get("type") or msg.get("role") or msg.__class__.__name__,
            "content": msg.get("content", ""),
        }

    tool_calls = getattr(msg, "tool_calls", None)
    name = msg.__class__.__name__
    content = getattr(msg, "content", "")

    payload = {
        "type": name,
        "content": content,
    }
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def _sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"



def _ensure_session(session_id: Optional[str]) -> str:
    if session_id:
        return session_id

    new_session_id = str(uuid.uuid4())

    create_chat_session(
        session_id=new_session_id,
        title="Agent Chat",
        metadata={"source": "agent_demo"},
    )

    return new_session_id

def _history_to_agent_messages(session_id: str, limit: int = 20) -> List[Dict[str, str]]:
    rows = get_chat_messages(session_id, limit) or []
    

    agent_messages: List[Dict[str, str]] = []
    for item in rows:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        elif not isinstance(item, dict):
            item = vars(item)

        role = item.get("role")
        content = item.get("content") or ""
        if role in {"user", "assistant"} and content:
            agent_messages.append({"role": role, "content": content})

    return agent_messages


def agent_ask(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    resolved_session_id = _ensure_session(session_id)
    history_messages = _history_to_agent_messages(resolved_session_id, limit=20)

    agent = create_kb_agent()
    input_messages = history_messages + [{"role": "user", "content": question}]

    result = agent.invoke({"messages": input_messages})
    final_answer = _extract_final_text(result)

    insert_chat_message(
        session_id=resolved_session_id,
        role="user",
        message=question,
        )
    insert_chat_message(
        session_id=resolved_session_id,
        role="assistant",
        message=final_answer,
        )
    

    messages = result.get("messages", []) or []
    return {
        "ok": True,
        "session_id": resolved_session_id,
        "question": question,
        "answer": final_answer,
        "messages": [_serialize_message(m) for m in messages],
    }


def agent_ask_stream(question: str, session_id: Optional[str] = None) -> Generator[str, None, None]:
    resolved_session_id = _ensure_session(session_id)
    history_messages = _history_to_agent_messages(resolved_session_id, limit=20)

    agent = create_kb_agent()
    input_messages = history_messages + [{"role": "user", "content": question}]

    yield _sse({"type": "start", "question": question, "session_id": resolved_session_id})

    final_result = agent.invoke({"messages": input_messages})
    final_answer = _extract_final_text(final_result)

    
    insert_chat_message(
        session_id=resolved_session_id,
        role="user",
        message=question,
        )
    insert_chat_message(
        session_id=resolved_session_id,
        role="assistant",
        message=final_answer,
        )

    messages = final_result.get("messages", []) or []

    for msg in messages:
        payload = _serialize_message(msg)
        msg_type = payload.get("type", "")

        if "HumanMessage" in msg_type:
            continue

        if payload.get("tool_calls"):
            yield _sse(
                {
                    "type": "tool_call",
                    "session_id": resolved_session_id,
                    "message": payload,
                }
            )
        elif "ToolMessage" in msg_type:
            yield _sse(
                {
                    "type": "tool_result",
                    "session_id": resolved_session_id,
                    "message": payload,
                }
            )
        else:
            yield _sse(
                {
                    "type": "message",
                    "session_id": resolved_session_id,
                    "message": payload,
                }
            )

    yield _sse(
        {
            "type": "final",
            "session_id": resolved_session_id,
            "answer": final_answer,
        }
    )
    yield _sse({"type": "done", "session_id": resolved_session_id})