"""
LangGraph StateGraph for KB Agent — Phase 1.

Nodes:
  reasoning_node   → decides next action (tool call or final answer) with explicit thought
  tool_execution_node → executes the selected tool
  final_answer_node  → assembles structured response with reasoning_trace

Edges:
  reasoning_node → tool_execution_node (if action is a tool)
               → final_answer_node (if action is "final")
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import Tool
from typing_extensions import TypedDict

import app.config as config
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.agent.agent import TOOLS


# =============================================================================
# Config
# =============================================================================

def _cfg(name: str, default: Any) -> Any:
    return getattr(config, name, default)


AGENT_ENABLE_VERIFIER = bool(_cfg("AGENT_ENABLE_VERIFIER", True))

# Memory / Learning config
VECTOR_MEMORY_ENABLED = bool(_cfg("VECTOR_MEMORY_ENABLED", True))
VECTOR_MEMORY_TOP_K = int(_cfg("VECTOR_MEMORY_TOP_K", 3))
TOOL_CHAIN_MEMORY_ENABLED = bool(_cfg("TOOL_CHAIN_MEMORY_ENABLED", True))
TOOL_CHAIN_MEMORY_MIN_SUCCESS = int(_cfg("TOOL_CHAIN_MEMORY_MIN_SUCCESS", 2))


# =============================================================================
# Tool name → function mapping (for graph node use)
# =============================================================================

def _build_tool_map() -> dict[str, Tool]:
    """Build {tool_name: Tool} map from the TOOLS list."""
    tool_map = {}
    for t in TOOLS:
        if isinstance(t, Tool):
            tool_map[t.name] = t
        else:
            # Fallback: assume it's a callable with a name attribute
            name = getattr(t, "name", None)
            if name:
                tool_map[name] = t
    return tool_map


TOOL_MAP = _build_tool_map()

HIGH_RISK_TOOLS = {"kb_import_file", "kb_index_document", "kb_import_folder"}

# Tool that triggers verification after execution
ANSWER_TOOL = "kb_answer_question"


# =============================================================================
# Agent State
# =============================================================================

class AgentState(TypedDict, total=False):
    messages: list[Any]
    session_id: str
    question: str
    reasoning_trace: list[dict[str, Any]]
    tool_call_history: list[dict[str, Any]]
    current_tool: str | None
    current_tool_args: dict[str, Any] | None
    current_tool_result: dict[str, Any] | None
    pending_confirmation: dict[str, Any] | None
    agent_step: int
    max_steps: int
    final_answer: str
    verification_result: dict[str, Any] | None
    refine_result: dict[str, Any] | None
    error: str | None
    # Additional fields for verification flow
    last_tool: str | None  # Track the last executed tool
    evidence_chunks: list[dict[str, Any]] | None  # Evidence chunks from last retrieval


# =============================================================================
# LLM for reasoning node
# =============================================================================

def _create_reasoning_llm() -> Runnable:
    from langchain.chat_models import init_chat_model
    return init_chat_model(
        model=f"ollama:{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


REASONING_LLM = _create_reasoning_llm()


SYSTEM_PROMPT = """你是一个知识库自主Agent。

当前会话信息：
- session_id: {session_id}
- 问题: {question}

{entity_memory}
{learning_memory}

你可以通过调用以下工具来回答问题：

{tool_descriptions}

推理规则：
1. 优先使用 kb_answer_question 工具进行完整问答
2. 如果用户问到具体文档内容，先用 kb_search_knowledge_base 搜索
3. 如果需要创建会话或获取历史，用 kb_get_chat_history / kb_create_chat_session
4. 高风险操作（import_file, index_document, import_folder）需要先确认

你的输出必须是合法的JSON对象，格式如下：
{{"thought": "你的推理过程", "action": "工具名或final", "action_input": {{"参数dict"}}, "observation": "工具执行结果或空"}}

如果 action 是 "final"，则 action_input 中需要包含 answer 字段。

示例：
{{"thought": "用户问的是苹果公司2023年净利润，这是一个财务问题，应该用完整问答工具", "action": "kb_answer_question", "action_input": {{"question": "苹果公司2023年净利润是多少？", "session_id": "{session_id}", "top_k": 5, "response_mode": "text"}}, "observation": ""}}
"""


def _build_tool_descriptions() -> str:
    lines = []
    for t in TOOLS:
        if isinstance(t, Tool):
            name = t.name
            desc = t.description or ""
            lines.append(f"- {name}: {desc}")
        else:
            name = getattr(t, "name", str(t))
            lines.append(f"- {name}")
    return "\n".join(lines)


TOOL_DESCRIPTIONS = _build_tool_descriptions()


def _reasoning_node(state: AgentState) -> AgentState:
    """LLM decides next action with explicit thought."""
    messages = state.get("messages", [])
    question = state.get("question", "")
    session_id = state.get("session_id", "")
    step = state.get("agent_step", 0)
    max_steps = state.get("max_steps", 10)

    # Check if we need confirmation first
    if state.get("pending_confirmation"):
        # Already waiting for confirmation, skip reasoning
        return state

    # Check max steps
    if step >= max_steps:
        final_answer = _extract_final_answer_from_messages(messages)
        return {
            **state,
            "final_answer": final_answer or "已达到最大步数限制。",
            "error": "max_steps_reached",
        }

    # Build tool description context
    history_text = _format_recent_history(messages)

    # Fetch entity memory context
    entity_memory = ""
    try:
        from app.db.repositories.entity_repository import get_entity_memory_context
        entity_memory = get_entity_memory_context(session_id)
    except Exception:
        pass

    if entity_memory:
        entity_memory = f"\n实体记忆：\n{entity_memory}\n"
    else:
        entity_memory = ""

    # Fetch learning memory context (vector memory + tool chain memory)
    learning_memory = ""
    if VECTOR_MEMORY_ENABLED or TOOL_CHAIN_MEMORY_ENABLED:
        try:
            from app.services.memory_service import memory_service
            memory_context = memory_service.retrieve_context(
                question,
                top_k=VECTOR_MEMORY_TOP_K,
            )
            parts = []
            if VECTOR_MEMORY_ENABLED and memory_context.get("vector_memory"):
                parts.append(f"相似案例记忆:\n{memory_context['vector_memory']}")
            if TOOL_CHAIN_MEMORY_ENABLED and memory_context.get("tool_chain_memory"):
                parts.append(memory_context['tool_chain_memory'])
            if parts:
                learning_memory = "\n学习记忆：\n" + "\n".join(parts) + "\n"
        except Exception:
            pass

    # Build prompt for structured output
    system_msg = SystemMessage(content=SYSTEM_PROMPT.format(
        session_id=session_id,
        question=question,
        tool_descriptions=TOOL_DESCRIPTIONS,
        entity_memory=entity_memory,
        learning_memory=learning_memory,
    ))

    user_msg = HumanMessage(content=f"问题：{question}\n\n对话历史：\n{history_text}\n\n请决定下一步操作（JSON格式）：")

    # Invoke LLM
    response = REASONING_LLM.invoke([system_msg, user_msg])
    raw_content = response.content if hasattr(response, "content") else str(response)

    # Parse structured output
    parsed = _parse_action_response(raw_content)

    thought = parsed.get("thought", "")
    action = parsed.get("action", "")
    action_input = parsed.get("action_input", {})
    observation = parsed.get("observation", "")

    # Append to reasoning trace
    trace_entry = {
        "step": step + 1,
        "thought": thought,
        "action": action,
        "action_input": action_input,
        "observation": observation,
    }
    reasoning_trace = list(state.get("reasoning_trace", []))
    reasoning_trace.append(trace_entry)

    # If action is "final", go to final answer
    if action == "final":
        answer = action_input.get("answer", observation or "")
        return {
            **state,
            "reasoning_trace": reasoning_trace,
            "final_answer": answer,
        }

    # High-risk tools need confirmation
    if action in HIGH_RISK_TOOLS:
        return {
            **state,
            "reasoning_trace": reasoning_trace,
            "current_tool": action,
            "current_tool_args": action_input,
            "pending_confirmation": {
                "tool": action,
                "args": action_input,
                "reason": thought,
            },
            "agent_step": step + 1,
        }

    # Set tool info for tool_execution_node to handle
    # Do NOT execute here - let the edge route to tool_execution_node
    return {
        **state,
        "reasoning_trace": reasoning_trace,
        "current_tool": action,
        "current_tool_args": action_input,
        "pending_confirmation": None,
        "agent_step": step + 1,
    }


def _tool_execution_node(state: AgentState) -> AgentState:
    """Execute a tool and return result."""
    tool_name = state.get("current_tool")
    tool_args = state.get("current_tool_args", {})
    session_id = state.get("session_id", "")

    if not tool_name or tool_name == "final":
        return _final_answer_node(state)

    tool_result = _execute_tool(tool_name, tool_args)

    # Track entities from question and tool result
    if session_id and tool_result.get("ok"):
        try:
            from app.db.repositories.entity_repository import track_entities, track_document_entity

            # Track entities from question
            question = state.get("question", "")
            if question:
                track_entities(session_id, question, {"source": "question"})

            # Track entities from tool result
            data = tool_result.get("data", {})
            result_text = ""

            if tool_name == ANSWER_TOOL:
                # For answer tool, track from answer text and document titles
                result_text = data.get("answer", "") or ""
                retrieved_chunks = data.get("retrieved_chunks", [])
                for chunk in retrieved_chunks:
                    title = chunk.get("title", "")
                    if title:
                        track_entities(session_id, title, {"source": "retrieved_chunk"})
                        # Also track document if we have document_id
                        doc_id = chunk.get("document_id")
                        if doc_id:
                            track_document_entity(session_id, doc_id, title, {"source": "retrieved_chunk"})
            else:
                # For other tools, track from result data
                result_text = json.dumps(data, ensure_ascii=False)

            if result_text:
                track_entities(session_id, result_text, {"source": tool_name})

        except Exception:
            # Entity tracking should not break the agent
            pass

    # Add tool result to messages
    messages = state.get("messages", [])
    tool_msg = ToolMessage(content=json.dumps(tool_result, ensure_ascii=False), tool_call_id="")
    new_messages = messages + [tool_msg]

    # Update trace observation
    reasoning_trace = list(state.get("reasoning_trace", []))
    if reasoning_trace:
        reasoning_trace[-1]["observation"] = _summarize_tool_result(tool_result)

    return {
        **state,
        "messages": new_messages,
        "current_tool_result": tool_result,
        "reasoning_trace": reasoning_trace,
        "last_tool": tool_name,
    }


def _final_answer_node(state: AgentState) -> AgentState:
    """Assemble final answer from state."""
    messages = state.get("messages", [])
    reasoning_trace = state.get("reasoning_trace", [])

    # Try to extract from trace first
    for entry in reversed(reasoning_trace):
        if entry.get("action") == "final":
            return {
                **state,
                "final_answer": entry.get("action_input", {}).get("answer", ""),
            }

    # Fall back to extracting from messages
    final_answer = _extract_final_answer_from_messages(messages)
    return {
        **state,
        "final_answer": final_answer or "无法生成答案。",
    }


def _verification_node(state: AgentState) -> AgentState:
    """
    Verify the answer from kb_answer_question tool.

    Calls the verifier service if enabled and routes to refine_node if verification fails.
    """
    if not AGENT_ENABLE_VERIFIER:
        # Verifier disabled, skip verification
        return {
            **state,
            "verification_result": {
                "is_supported": True,
                "support_level": "high",
                "method": "disabled",
            },
        }

    last_tool = state.get("last_tool")
    if last_tool != ANSWER_TOOL:
        # Only verify kb_answer_question results
        return state

    current_tool_result = state.get("current_tool_result")
    if not current_tool_result or not current_tool_result.get("ok"):
        return state

    question = state.get("question", "")
    data = current_tool_result.get("data", {})
    answer = data.get("answer", "") or ""
    retrieved_chunks = data.get("retrieved_chunks", [])

    if not answer or not retrieved_chunks:
        return state

    # Call verifier service
    try:
        from app.services.verifier_service import verify_answer
        verification_result = verify_answer(
            query=question,
            draft_answer=answer,
            evidence_chunks=retrieved_chunks,
            intent="unknown",
        )
    except Exception as e:
        verification_result = {
            "is_supported": True,
            "support_level": "high",
            "method": "error",
            "summary": f"Verification error: {str(e)}",
        }

    # Add to reasoning trace
    reasoning_trace = list(state.get("reasoning_trace", []))
    if reasoning_trace:
        reasoning_trace[-1]["observation"] = (
            f"验证结果: is_supported={verification_result.get('is_supported')}, "
            f"support_level={verification_result.get('support_level')}, "
            f"method={verification_result.get('method')}"
        )

    return {
        **state,
        "verification_result": verification_result,
        "reasoning_trace": reasoning_trace,
        "evidence_chunks": retrieved_chunks,
    }


def _refine_node(state: AgentState) -> AgentState:
    """
    Refine the answer based on verification feedback.

    Only triggers if verification indicates the answer is not fully supported.
    """
    verification_result = state.get("verification_result")

    if not verification_result:
        return state

    # Check if refinement is needed
    is_supported = verification_result.get("is_supported", True)
    if is_supported:
        # Answer is supported, no refinement needed
        return state

    question = state.get("question", "")
    current_tool_result = state.get("current_tool_result")
    data = current_tool_result.get("data", {}) if current_tool_result else {}
    draft_answer = data.get("answer", "") or ""
    evidence_chunks = state.get("evidence_chunks") or []

    if not draft_answer or not evidence_chunks:
        return state

    # Call refine service
    try:
        from app.services.refine_service import refine_answer
        refine_result = refine_answer(
            query=question,
            draft_answer=draft_answer,
            verification_result=verification_result,
            evidence_chunks=evidence_chunks,
        )
    except Exception as e:
        refine_result = {
            "refined_answer": draft_answer,
            "was_refined": False,
            "method": "error",
            "summary": f"Refine error: {str(e)}",
        }

    # Update the tool result with refined answer
    refined_answer = refine_result.get("refined_answer", draft_answer)
    updated_tool_result = {
        **current_tool_result,
        "data": {
            **data,
            "answer": refined_answer,
        },
    }

    # Add to reasoning trace
    reasoning_trace = list(state.get("reasoning_trace", []))
    if reasoning_trace:
        reasoning_trace[-1]["observation"] = (
            f"自我修正: was_refined={refine_result.get('was_refined')}, "
            f"trigger_reason={refine_result.get('trigger_reason')}"
        )

    return {
        **state,
        "refine_result": refine_result,
        "current_tool_result": updated_tool_result,
        "reasoning_trace": reasoning_trace,
    }


def _execute_tool(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name with arguments."""
    tool = TOOL_MAP.get(tool_name)
    if not tool:
        return {
            "ok": False,
            "error": {"code": "TOOL_NOT_FOUND", "message": f"Tool {tool_name} not found"},
            "meta": {"tool_name": tool_name, "duration_ms": 0},
        }

    start = time.perf_counter()
    try:
        # LangChain Tool.invoke takes a single dict arg
        if isinstance(tool, Tool):
            result = tool.invoke(tool_args)
        else:
            result = tool(tool_args)
        duration_ms = int((time.perf_counter() - start) * 1000)

        # Ensure result is a dict
        if not isinstance(result, dict):
            result = {"ok": True, "data": result}

        # Add meta if not present
        if "meta" not in result:
            result["meta"] = {"tool_name": tool_name, "duration_ms": duration_ms}
        return result
    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return {
            "ok": False,
            "error": {"code": "INTERNAL_ERROR", "message": str(e)},
            "meta": {"tool_name": tool_name, "duration_ms": duration_ms},
        }


def _parse_action_response(raw: str) -> dict[str, Any]:
    """Parse LLM JSON response for action decision."""
    raw = (raw or "").strip()

    # Try JSON object directly
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # Try to extract JSON from markdown
    import re
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find first { ... }
    brace = re.search(r"(\{.*\})", raw, re.S)
    if brace:
        try:
            return json.loads(brace.group(1))
        except json.JSONDecodeError:
            pass

    return {"thought": raw, "action": "final", "action_input": {"answer": raw}}


def _extract_final_answer_from_messages(messages: list[Any]) -> str:
    """Extract final answer text from messages."""
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            # Skip tool messages
            if hasattr(msg, "type") and "tool" in str(getattr(msg, "type", "")).lower():
                continue
            return content
    return ""


def _format_recent_history(messages: list[Any], limit: int = 6) -> str:
    """Format recent messages for prompt context."""
    relevant = []
    for msg in messages[-limit:]:
        role = getattr(msg, "type", "") or getattr(msg, "role", "")
        content = getattr(msg, "content", "") or ""
        if not content:
            continue
        if "tool" in role.lower():
            continue
        role_label = "用户" if "human" in role.lower() else "助手"
        relevant.append(f"{role_label}：{content[:200]}")
    return "\n".join(relevant) if relevant else "（无历史）"


def _summarize_tool_result(result: dict[str, Any]) -> str:
    """Create a brief observation string from tool result."""
    if not isinstance(result, dict):
        return str(result)[:200]

    ok = result.get("ok", True)
    if not ok:
        error = result.get("error", {})
        return f"工具执行失败: {error.get('message', 'unknown error')}"

    data = result.get("data", {})
    if isinstance(data, dict):
        # For search results
        hits = data.get("hits", [])
        if hits is not None:
            return f"检索到 {len(hits)} 个结果"
        # For answer results
        answer = data.get("answer", "")
        if answer:
            return f"答案生成成功: {str(answer)[:100]}"
        # For other results
        keys = list(data.keys())[:3]
        return f"执行成功，数据字段: {', '.join(keys)}"

    return "工具执行成功"


# =============================================================================
# Build LangGraph
# =============================================================================

def _build_graph() -> Any:
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("reasoning", _reasoning_node)
    workflow.add_node("tool_execution", _tool_execution_node)
    workflow.add_node("verification", _verification_node)
    workflow.add_node("refine", _refine_node)
    workflow.add_node("final_answer", _final_answer_node)

    # Set entry point
    workflow.set_entry_point("reasoning")

    # Edges from reasoning
    def reasoning_continue(state: AgentState) -> str:
        """Decide next node after reasoning."""
        if state.get("error") == "max_steps_reached":
            return "final_answer"
        if state.get("pending_confirmation"):
            # Wait for confirmation — halt and return
            return END
        current_tool = state.get("current_tool")
        if current_tool and current_tool != "final":
            return "tool_execution"
        return "final_answer"

    workflow.add_conditional_edges(
        "reasoning",
        reasoning_continue,
        {
            "tool_execution": "tool_execution",
            "final_answer": "final_answer",
        },
    )

    # Edges from tool_execution
    def tool_execution_continue(state: AgentState) -> str:
        """Decide next node after tool execution."""
        last_tool = state.get("last_tool")
        if last_tool == ANSWER_TOOL and AGENT_ENABLE_VERIFIER:
            # Route to verification for answer tool
            return "verification"
        return "reasoning"

    workflow.add_conditional_edges(
        "tool_execution",
        tool_execution_continue,
        {
            "verification": "verification",
            "reasoning": "reasoning",
        },
    )

    # Edges from verification
    def verification_continue(state: AgentState) -> str:
        """Decide next node after verification."""
        verification_result = state.get("verification_result")
        if not verification_result:
            # No verification result, skip to final answer
            return "final_answer"

        is_supported = verification_result.get("is_supported", True)
        if not is_supported:
            # Verification failed, try refinement
            return "refine"
        return "final_answer"

    workflow.add_conditional_edges(
        "verification",
        verification_continue,
        {
            "refine": "refine",
            "final_answer": "final_answer",
        },
    )

    # Edge from refine back to reasoning (to continue agent loop)
    workflow.add_edge("refine", "reasoning")
    workflow.add_edge("final_answer", END)

    return workflow.compile()


# Compile graph once at module load
_agent_graph = None


def get_agent_graph() -> Any:
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = _build_graph()
    return _agent_graph


def run_agent(
    question: str,
    session_id: str,
    max_steps: int = 10,
) -> dict[str, Any]:
    """Run the agent graph synchronously."""
    graph = get_agent_graph()

    initial_state: AgentState = {
        "messages": [],
        "session_id": session_id,
        "question": question,
        "reasoning_trace": [],
        "tool_call_history": [],
        "current_tool": None,
        "current_tool_args": None,
        "current_tool_result": None,
        "pending_confirmation": None,
        "agent_step": 0,
        "max_steps": max_steps,
        "final_answer": "",
        "verification_result": None,
        "refine_result": None,
        "error": None,
        "last_tool": None,
        "evidence_chunks": None,
    }

    result = graph.invoke(initial_state)

    return {
        "final_answer": result.get("final_answer", ""),
        "reasoning_trace": result.get("reasoning_trace", []),
        "tool_call_history": result.get("tool_call_history", []),
        "pending_confirmation": result.get("pending_confirmation"),
        "agent_step": result.get("agent_step", 0),
        "verification_result": result.get("verification_result"),
        "refine_result": result.get("refine_result"),
        "error": result.get("error"),
        "evidence_chunks": result.get("evidence_chunks"),
    }
