from __future__ import annotations
import uuid
import json
import time
from typing import Any, Dict, Generator, List, Optional

from app.db import create_chat_session, insert_chat_message, get_chat_messages
from app.agent.graph import run_agent, get_agent_graph, AgentState, TOOL_MAP, HIGH_RISK_TOOLS
from app.agent.confirmation import clear_pending_confirmation, get_pending_confirmation


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


def _execute_pending_confirmation(state: AgentState, confirmed: bool) -> AgentState:
    """Execute or skip a pending confirmation tool."""
    if not state.get("pending_confirmation"):
        return state

    tool_name = state["pending_confirmation"]["tool"]
    tool_args = state["pending_confirmation"]["args"]
    step = state.get("agent_step", 0)

    if not confirmed:
        # User rejected - skip the tool, continue reasoning
        from app.agent.graph import _reasoning_node
        reasoning_trace = list(state.get("reasoning_trace", []))
        if reasoning_trace:
            reasoning_trace[-1]["observation"] = f"用户拒绝执行 {tool_name}，跳过此操作"
        return {
            **state,
            "reasoning_trace": reasoning_trace,
            "pending_confirmation": None,
        }

    # Execute the tool
    from app.agent.graph import _execute_tool
    tool_result = _execute_tool(tool_name, tool_args)

    # Add tool result to messages
    from langchain_core.messages import ToolMessage
    messages = state.get("messages", [])
    tool_msg = ToolMessage(content=json.dumps(tool_result, ensure_ascii=False), tool_call_id="")
    new_messages = messages + [tool_msg]

    # Update trace
    reasoning_trace = list(state.get("reasoning_trace", []))
    if reasoning_trace:
        from app.agent.graph import _summarize_tool_result
        reasoning_trace[-1]["observation"] = _summarize_tool_result(tool_result)

    return {
        **state,
        "messages": new_messages,
        "current_tool_result": tool_result,
        "reasoning_trace": reasoning_trace,
        "pending_confirmation": None,
        "agent_step": step + 1,
    }


def _serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize agent result for API response."""
    return {
        "ok": True,
        "session_id": result.get("session_id"),
        "question": result.get("question"),
        "answer": result.get("final_answer", ""),
        "reasoning_trace": result.get("reasoning_trace", []),
        "tool_call_history": result.get("tool_call_history", []),
        "pending_confirmation": result.get("pending_confirmation"),
        "agent_step": result.get("agent_step", 0),
        "verification_result": result.get("verification_result"),
        "refine_result": result.get("refine_result"),
        "error": result.get("error"),
    }


# In-memory store for paused agent states (keyed by session_id)
# When a high-risk tool needs confirmation, we store the state here
# and let the client resume after confirmation
_agent_paused_states: Dict[str, AgentState] = {}


def handle_agent_confirmation(
    session_id: str,
    confirmed: bool,
    tool_name: Optional[str] = None,
    tool_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Handle user confirmation for a paused agent.

    When the agent needs confirmation for a high-risk tool, it stores its
    state and returns pending_confirmation. The client then calls this
    endpoint to confirm or reject. If confirmed, we execute the tool and
    continue the agent loop.
    """
    # Get the paused state
    paused_state = _agent_paused_states.get(session_id)
    if not paused_state:
        return {
            "ok": False,
            "session_id": session_id,
            "message": "No pending confirmation found for this session",
        }

    # Clear from paused states
    del _agent_paused_states[session_id]
    clear_pending_confirmation(session_id)

    # Execute or skip the tool
    from app.agent.graph import _execute_tool, _summarize_tool_result
    from langchain_core.messages import ToolMessage

    state = paused_state
    tool_to_execute = tool_name or state.get("current_tool")
    args_to_use = tool_args or state.get("current_tool_args", {})

    if not confirmed:
        # User rejected - skip the tool, update trace and continue reasoning
        reasoning_trace = list(state.get("reasoning_trace", []))
        if reasoning_trace:
            reasoning_trace[-1]["observation"] = f"用户拒绝执行 {tool_to_execute}，跳过此操作"

        # Continue agent loop without this tool
        state = {
            **state,
            "reasoning_trace": reasoning_trace,
            "pending_confirmation": None,
            "current_tool": None,
            "current_tool_args": None,
        }
    else:
        # Execute the tool
        tool_result = _execute_tool(tool_to_execute, args_to_use)

        # Add tool result to messages
        messages = state.get("messages", [])
        tool_msg = ToolMessage(content=json.dumps(tool_result, ensure_ascii=False), tool_call_id="")
        new_messages = messages + [tool_msg]

        # Update trace
        reasoning_trace = list(state.get("reasoning_trace", []))
        if reasoning_trace:
            reasoning_trace[-1]["observation"] = _summarize_tool_result(tool_result)

        # Continue agent loop
        state = {
            **state,
            "messages": new_messages,
            "current_tool_result": tool_result,
            "reasoning_trace": reasoning_trace,
            "pending_confirmation": None,
            "current_tool": None,
            "current_tool_args": None,
        }

    # Continue agent execution
    from app.agent.graph import get_agent_graph
    graph = get_agent_graph()
    result = graph.invoke(state)

    # Check if we need another confirmation
    if result.get("pending_confirmation"):
        # Store state and return confirmation needed
        _agent_paused_states[session_id] = result
        return _serialize_result(result)

    # Save to chat history
    insert_chat_message(
        session_id=session_id,
        role="user",
        message=state.get("question", ""),
    )
    insert_chat_message(
        session_id=session_id,
        role="assistant",
        message=result.get("final_answer", ""),
    )

    result["session_id"] = session_id
    result["question"] = state.get("question", "")
    return _serialize_result(result)


def agent_ask(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    resolved_session_id = _ensure_session(session_id)

    result = run_agent(
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
        message=result.get("final_answer", ""),
    )

    # Store to learning memory if verification passed (is_supported=True)
    try:
        from app.config import VECTOR_MEMORY_ENABLED, TOOL_CHAIN_MEMORY_ENABLED
        if VECTOR_MEMORY_ENABLED or TOOL_CHAIN_MEMORY_ENABLED:
            verification_result = result.get("verification_result")
            if verification_result and verification_result.get("is_supported") == True:
                from app.services.memory_service import memory_service

                # Extract retrieved_chunk_ids from evidence_chunks
                evidence_chunks = result.get("evidence_chunks") or []
                retrieved_chunk_ids = [
                    c.get("chunk_id") for c in evidence_chunks if c.get("chunk_id")
                ]

                memory_service.store_success(
                    question=question,
                    answer=result.get("final_answer", ""),
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    reasoning_trace=result.get("reasoning_trace", []),
                    session_id=resolved_session_id,
                )
    except Exception as e:
        print(f"[agent_ask] memory store failed: {e}")

    result["session_id"] = resolved_session_id
    result["question"] = question
    return _serialize_result(result)


def agent_ask_stream(question: str, session_id: Optional[str] = None) -> Generator[str, None, None]:
    resolved_session_id = _ensure_session(session_id)

    yield _sse({"type": "start", "question": question, "session_id": resolved_session_id})

    # Run graph step by step for streaming
    from app.agent.graph import get_agent_graph, AgentState
    graph = get_agent_graph()

    state: AgentState = {
        "messages": [],
        "session_id": resolved_session_id,
        "question": question,
        "reasoning_trace": [],
        "tool_call_history": [],
        "current_tool": None,
        "current_tool_args": None,
        "current_tool_result": None,
        "pending_confirmation": None,
        "agent_step": 0,
        "max_steps": 10,
        "final_answer": "",
        "verification_result": None,
        "refine_result": None,
        "error": None,
        "last_tool": None,
        "evidence_chunks": None,
    }

    # Run graph
    result = graph.invoke(state)

    # Stream reasoning trace
    for trace_entry in result.get("reasoning_trace", []):
        yield _sse({
            "type": "reasoning",
            "session_id": resolved_session_id,
            "step": trace_entry.get("step"),
            "thought": trace_entry.get("thought"),
            "action": trace_entry.get("action"),
            "action_input": trace_entry.get("action_input"),
            "observation": trace_entry.get("observation"),
        })

    # Stream tool calls
    for tool_entry in result.get("tool_call_history", []):
        yield _sse({
            "type": "tool_call",
            "session_id": resolved_session_id,
            "tool": tool_entry.get("tool"),
            "args": tool_entry.get("args"),
            "result": tool_entry.get("result"),
        })

    # Stream verification result
    if result.get("verification_result"):
        yield _sse({
            "type": "verification",
            "session_id": resolved_session_id,
            "verification_result": result["verification_result"],
        })

    # Stream refine result
    if result.get("refine_result"):
        yield _sse({
            "type": "refine",
            "session_id": resolved_session_id,
            "refine_result": result["refine_result"],
        })

    # Check for pending confirmation
    if result.get("pending_confirmation"):
        # Store the paused state so it can be resumed after confirmation
        _agent_paused_states[resolved_session_id] = result
        yield _sse({
            "type": "confirmation_required",
            "session_id": resolved_session_id,
            "tool": result["pending_confirmation"].get("tool"),
            "args": result["pending_confirmation"].get("args"),
            "reason": result["pending_confirmation"].get("reason"),
        })
        yield _sse({"type": "done", "session_id": resolved_session_id})
        return

    # Save to chat history
    insert_chat_message(
        session_id=resolved_session_id,
        role="user",
        message=question,
    )
    insert_chat_message(
        session_id=resolved_session_id,
        role="assistant",
        message=result.get("final_answer", ""),
    )

    # Store to learning memory if verification passed (is_supported=True)
    try:
        from app.config import VECTOR_MEMORY_ENABLED, TOOL_CHAIN_MEMORY_ENABLED
        if VECTOR_MEMORY_ENABLED or TOOL_CHAIN_MEMORY_ENABLED:
            verification_result = result.get("verification_result")
            if verification_result and verification_result.get("is_supported") == True:
                from app.services.memory_service import memory_service

                evidence_chunks = result.get("evidence_chunks") or []
                retrieved_chunk_ids = [
                    c.get("chunk_id") for c in evidence_chunks if c.get("chunk_id")
                ]

                memory_service.store_success(
                    question=question,
                    answer=result.get("final_answer", ""),
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    reasoning_trace=result.get("reasoning_trace", []),
                    session_id=resolved_session_id,
                )
    except Exception as e:
        print(f"[agent_ask_stream] memory store failed: {e}")

    yield _sse({
        "type": "final",
        "session_id": resolved_session_id,
        "answer": result.get("final_answer", ""),
    })
    yield _sse({"type": "done", "session_id": resolved_session_id})


# =============================================================================
# Multi-Agent Entry Points
# =============================================================================

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