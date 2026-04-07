from __future__ import annotations
import uuid
import json
from typing import Any, Dict, Generator, Optional

from app.db import create_chat_session, insert_chat_message


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


def agent_ask_multi(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Multi-agent question answering with automatic routing.

    Routes to appropriate specialist agents based on intent classification.

    Args:
        question: User question
        session_id: Session identifier

    Returns:
        Dict with answer and reasoning trace
    """
    from app.agent.multi_agent import run_multi_agent

    resolved_session_id = session_id or _ensure_session(session_id)

    result = run_multi_agent(
        question=question,
        session_id=resolved_session_id,
        max_steps=10,
    )

    # Save to chat history
    insert_chat_message(
        session_id=resolved_session_id,
        role="user",
        message=question,
    )
    insert_chat_message(
        session_id=resolved_session_id,
        role="assistant",
        message=result.get("answer", ""),
    )

    result["session_id"] = resolved_session_id
    result["question"] = question
    return result


def agent_ask_multi_stream(question: str, session_id: Optional[str] = None) -> Generator[str, None, None]:
    """
    Multi-agent streaming with automatic routing.

    Args:
        question: User question
        session_id: Session identifier

    Yields:
        SSE events
    """
    from app.agent.multi_agent import run_multi_agent_stream

    resolved_session_id = session_id or _ensure_session(session_id)

    yield _sse({"type": "start", "question": question, "session_id": resolved_session_id})

    try:
        events = run_multi_agent_stream(
            question=question,
            session_id=resolved_session_id,
        )

        for event in events:
            yield _sse(event)

    except Exception as e:
        yield _sse({"type": "error", "error": str(e)})

    yield _sse({"type": "done", "session_id": resolved_session_id})
