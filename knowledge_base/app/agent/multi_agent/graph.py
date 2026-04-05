"""
Multi-Agent StateGraph for Knowledge Base.

架构：
- 入口层：Rewrite (使用Memory改写问题)
- 编排层：Supervisor → QAAgent / DocumentAgent
- 共享状态层：Memory (记忆存储)
- 治理层：Governance Gateway（双阶段策略执行）

流程：
问题 → Rewrite → Context Guard → Supervisor → Action Guard → QAAgent/DocumentAgent → 聚合 → END

双阶段策略执行：
- 第一阶段（Context Guard）：Rewrite 输出进入 Supervisor 之前进行上下文治理
- 第二阶段（Action Guard）：Agent 发起工具调用之前进行动作治理
"""

from typing import Any, TypedDict
from langgraph.graph import END, StateGraph

from app.agent.multi_agent.messages import AgentResponse, TaskType, AgentRole
from app.agent.multi_agent.supervisor import supervisor_agent
from app.agent.multi_agent.qa_agent import qa_agent
from app.agent.multi_agent.document_agent import document_agent
from app.agent.multi_agent.memory_agent import memory_agent_wrapper
from app.agent.multi_agent.rewrite_agent import rewrite_agent

# 导入治理网关
from app.governance.gateway import governance_gateway
from app.governance.schemas import GovernanceDecision


# =============================================================================
# Multi-Agent State
# =============================================================================

class MultiAgentGraphState(TypedDict):
    """Multi-agent state using TypedDict for LangGraph compatibility."""
    session_id: str
    question: str
    messages: list[Any]
    task_type: Any
    supervisor_decision: dict[str, Any]
    agent_responses: dict[str, Any]
    final_answer: str
    reasoning_trace: list[dict[str, Any]]
    error: Any
    agent_step: int
    max_steps: int
    rewritten_question: str
    # 治理相关状态
    rewrite_output: dict[str, Any]  # Rewrite 结构化输出
    governance_context: dict[str, Any]  # 治理上下文
    context_guard_passed: bool  # Context Guard 是否通过


# =============================================================================
# Graph Nodes
# =============================================================================

def _rewrite_node(state: MultiAgentGraphState) -> MultiAgentGraphState:
    """入口层：Rewrite - 读取记忆并改写问题，然后经过 Context Guard"""
    question = state.get("question", "")
    session_id = state.get("session_id", "")

    # 先检查短期记忆是否有相同问题的缓存答案
    from app.agent.memory_agent.storage import short_term_storage
    short_mem = short_term_storage.get(session_id)
    if short_mem and short_mem.get("turns"):
        for turn in reversed(short_mem["turns"]):
            if turn.get("role") == "user" and turn.get("message") == question:
                # 找到相同问题，找下一个turn的assistant回答
                idx = short_mem["turns"].index(turn)
                if idx + 1 < len(short_mem["turns"]):
                    next_turn = short_mem["turns"][idx + 1]
                    if next_turn.get("role") == "assistant":
                        cached_answer = next_turn.get("message", "")
                        return {
                            **state,
                            "rewritten_question": question,
                            "final_answer": cached_answer,
                            "reasoning_trace": state.get("reasoning_trace", []) + [{
                                "step": 1,
                                "thought": "入口层：命中短期记忆缓存",
                                "action": "cache_hit",
                                "observation": "直接从记忆返回缓存答案",
                            }],
                            "agent_step": 1,
                            "agent_responses": {},
                            "rewrite_output": {"ok": True, "original_question": question, "rewritten_query": question},
                            "governance_context": {},
                            "context_guard_passed": True,
                        }

    # 获取记忆上下文
    memory_agent_wrapper.get_context(session_id, question)

    # 执行改写
    rewrite_response = rewrite_agent.execute(question, session_id)

    rewritten_question = question
    rewrite_output = {"ok": False, "original_question": question, "rewritten_query": question}

    if rewrite_response.success and rewrite_response.result:
        rewritten_question = rewrite_response.result.get("rewritten_query", question)
        rewrite_output = rewrite_response.result.copy()

    # ========== 第一阶段：Context Guard ==========
    # Context Guard 检查 Rewrite 输出，在进入 Supervisor 之前执行
    ctx_guard_result = governance_gateway.guard_context(
        rewrite_output=rewrite_output,
        session_id=session_id,
    )

    context_guard_passed = ctx_guard_result.decision == GovernanceDecision.ALLOW

    # 如果 Context Guard 拒绝，添加警告到 reasoning_trace
    guard_observation = f"Context Guard: {ctx_guard_result.decision.value}"
    if not context_guard_passed:
        guard_observation += f" - {ctx_guard_result.message}"
        if ctx_guard_result.risk_flags:
            guard_observation += f" (risk_flags: {ctx_guard_result.risk_flags})"

    reasoning_trace_entry = {
        "step": 1,
        "thought": "入口层：改写问题 + Context Guard",
        "action": "rewrite_with_context_guard",
        "observation": f"改写结果: {rewritten_question[:50]}... | {guard_observation}",
    }

    governance_context = {
        "risk_flags": ctx_guard_result.risk_flags,
        "source_tags": ctx_guard_result.source_tags,
        "risk_level": ctx_guard_result.risk_level.value if hasattr(ctx_guard_result.risk_level, 'value') else str(ctx_guard_result.risk_level),
        "context_guard_passed": context_guard_passed,
    }

    return {
        **state,
        "rewritten_question": rewritten_question,
        "rewrite_output": rewrite_output,
        "governance_context": governance_context,
        "context_guard_passed": context_guard_passed,
        "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_trace_entry],
        "agent_step": 1,
    }


def _supervisor_node(state: MultiAgentGraphState) -> MultiAgentGraphState:
    """Supervisor: 意图分类和路由"""
    question = state.get("question", "")
    session_id = state.get("session_id", "")
    context_guard_passed = state.get("context_guard_passed", True)

    # 如果 Context Guard 未通过，记录错误并继续（Supervisor 会降级处理）
    if not context_guard_passed:
        reasoning_trace_entry = {
            "step": state.get("agent_step", 0) + 1,
            "thought": "Supervisor：Context Guard 未通过，降级处理",
            "action": "supervisor_degraded",
            "observation": "由于上下文治理未通过，Supervisor 将进行保守路由",
        }
        routing = supervisor_agent.route_task(question, session_id)
        return {
            **state,
            "supervisor_decision": {**routing, "degraded": True},
            "task_type": routing["task_type"],
            "agent_step": state.get("agent_step", 0) + 1,
            "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_trace_entry],
        }

    routing = supervisor_agent.route_task(question, session_id)

    return {
        **state,
        "supervisor_decision": routing,
        "task_type": routing["task_type"],
        "agent_step": state.get("agent_step", 0) + 1,
    }


def _qa_node(state: MultiAgentGraphState) -> MultiAgentGraphState:
    """QAAgent: 回答问题"""
    question = state.get("question", "")
    session_id = state.get("session_id", "")
    rewritten_question = state.get("rewritten_question", "")
    max_steps = state.get("max_steps", 6)
    governance_context = state.get("governance_context", {})
    context_guard_passed = state.get("context_guard_passed", True)

    # 存储用户消息
    memory_agent_wrapper.store(session_id, "user", question)

    # 如果 Context Guard 未通过，返回降级响应
    if not context_guard_passed:
        degraded_answer = "[系统提示：由于上下文治理未通过，本次回答可能受限。请尝试重新提问或简化问题。]"
        memory_agent_wrapper.store(session_id, "assistant", degraded_answer)

        reasoning_trace_entry = {
            "step": state.get("agent_step", 0) + 1,
            "thought": "QAAgent：Context Guard 未通过，返回降级答案",
            "action": "degraded_answer",
            "observation": degraded_answer,
        }

        return {
            **state,
            "agent_responses": {AgentRole.QA.value: AgentResponse(
                agent=AgentRole.QA,
                task_type=TaskType.QA,
                result={"answer": degraded_answer, "degraded": True},
                reasoning_trace=[reasoning_trace_entry],
                success=True,
            )},
            "final_answer": degraded_answer,
            "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_trace_entry],
            "agent_step": state.get("agent_step", 0) + 1,
        }

    # 回答（传递 governance_context）
    response = qa_agent.execute(
        question=question,
        session_id=session_id,
        rewritten_question=rewritten_question,
        max_steps=max_steps,
        governance_context=governance_context,
    )

    # 存储助手回答
    if response.success and response.result:
        answer = response.result.get("answer", "")
        if answer:
            memory_agent_wrapper.store(session_id, "assistant", answer)

    return {
        **state,
        "agent_responses": {AgentRole.QA.value: response},
        "final_answer": response.result.get("answer", "") if response.success else "",
        "reasoning_trace": state.get("reasoning_trace", []) + response.reasoning_trace,
        "agent_step": state.get("agent_step", 0) + 1,
    }


def _document_node(state: MultiAgentGraphState) -> MultiAgentGraphState:
    """DocumentAgent: 文档管理/检索"""
    question = state.get("question", "")
    session_id = state.get("session_id", "")
    max_steps = state.get("max_steps", 5)
    governance_context = state.get("governance_context", {})
    context_guard_passed = state.get("context_guard_passed", True)

    # 存储用户消息
    memory_agent_wrapper.store(session_id, "user", question)

    # 如果 Context Guard 未通过，返回降级响应
    if not context_guard_passed:
        degraded_answer = "[系统提示：由于上下文治理未通过，文档操作受限。请尝试重新提问。]"
        memory_agent_wrapper.store(session_id, "assistant", degraded_answer)

        reasoning_trace_entry = {
            "step": state.get("agent_step", 0) + 1,
            "thought": "DocumentAgent：Context Guard 未通过，返回降级答案",
            "action": "degraded_answer",
            "observation": degraded_answer,
        }

        return {
            **state,
            "agent_responses": {AgentRole.DOCUMENT.value: AgentResponse(
                agent=AgentRole.DOCUMENT,
                task_type=TaskType.DOCUMENT,
                result={"message": degraded_answer, "degraded": True},
                reasoning_trace=[reasoning_trace_entry],
                success=True,
            )},
            "final_answer": degraded_answer,
            "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_trace_entry],
            "agent_step": state.get("agent_step", 0) + 1,
        }

    # 执行（传递 governance_context）
    response = document_agent.execute(
        question=question,
        session_id=session_id,
        max_steps=max_steps,
        governance_context=governance_context,
    )

    # 存储助手响应
    if response.success and response.result:
        result_msg = response.result.get("message", str(response.result))
        if result_msg:
            memory_agent_wrapper.store(session_id, "assistant", result_msg)

    return {
        **state,
        "agent_responses": {AgentRole.DOCUMENT.value: response},
        "final_answer": response.result.get("message", str(response.result)) if response.success else "",
        "reasoning_trace": state.get("reasoning_trace", []) + response.reasoning_trace,
        "agent_step": state.get("agent_step", 0) + 1,
    }


def _aggregation_node(state: MultiAgentGraphState) -> MultiAgentGraphState:
    """聚合节点"""
    agent_responses = state.get("agent_responses", {})
    question = state.get("question", "")

    responses_dict = {}
    for key, response in agent_responses.items():
        try:
            responses_dict[AgentRole(key)] = response
        except ValueError:
            pass

    aggregated = supervisor_agent.aggregate_results(responses_dict, question)
    final_response = supervisor_agent.build_final_answer(aggregated, question)

    return {
        **state,
        "final_answer": final_response.get("answer", state.get("final_answer", "")),
        "reasoning_trace": state.get("reasoning_trace", []) + [{
            "step": state.get("agent_step", 0) + 1,
            "thought": "聚合结果",
            "action": "aggregation",
            "observation": f"聚合了 {len(agent_responses)} 个Agent的响应",
        }],
    }


# =============================================================================
# Action Guard Integration
# =============================================================================

def _check_action_guard(
    tool_name: str,
    tool_args: dict[str, Any],
    agent: str,
    session_id: str,
    governance_context: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """
    检查 Action Guard

    Returns:
        (是否通过, 治理结果)
    """
    # 从治理上下文构建 rewrite_output
    rewrite_output = {
        "risk_flags": governance_context.get("risk_flags", []),
        "source_tags": governance_context.get("source_tags", []),
    }

    action_result = governance_gateway.guard_action(
        tool_name=tool_name,
        tool_args=tool_args,
        agent=agent,
        session_id=session_id,
        rewrite_output=rewrite_output,
    )

    passed = action_result.decision in [
        GovernanceDecision.ALLOW,
        GovernanceDecision.GRADE,  # 降级也允许执行（只是返回简化结果）
    ]

    return passed, action_result.model_dump()


# =============================================================================
# Routing Logic
# =============================================================================

def _route_after_supervisor(state: MultiAgentGraphState) -> str:
    """Supervisor后的路由"""
    routing = state.get("supervisor_decision", {})
    task_type = routing.get("task_type")

    if task_type == TaskType.DOCUMENT:
        return "document"

    return "qa"


# =============================================================================
# Build Graph
# =============================================================================

def _build_graph() -> Any:
    """构建多Agent状态图"""
    workflow = StateGraph(MultiAgentGraphState)

    # 添加节点
    workflow.add_node("rewrite", _rewrite_node)
    workflow.add_node("supervisor", _supervisor_node)
    workflow.add_node("qa", _qa_node)
    workflow.add_node("document", _document_node)
    workflow.add_node("aggregation", _aggregation_node)

    # 设置入口
    workflow.set_entry_point("rewrite")

    # 入口改写 → 条件路由（缓存命中则结束，否则到Supervisor）
    def _route_after_rewrite(state: MultiAgentGraphState) -> str:
        """Rewrite后的路由 - 缓存命中则直接结束"""
        if state.get("final_answer") and state.get("agent_responses") == {}:
            return "__end__"
        return "supervisor"

    workflow.add_conditional_edges(
        "rewrite",
        _route_after_rewrite,
        {
            "__end__": END,
            "supervisor": "supervisor",
        },
    )

    # Supervisor路由
    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "qa": "qa",
            "document": "document",
        },
    )

    # 汇聚到aggregation
    workflow.add_edge("qa", "aggregation")
    workflow.add_edge("document", "aggregation")

    # Aggregation结束
    workflow.add_edge("aggregation", END)

    return workflow.compile()


# 编译一次
_multi_agent_graph = None


def get_multi_agent_graph() -> Any:
    global _multi_agent_graph
    if _multi_agent_graph is None:
        _multi_agent_graph = _build_graph()
    return _multi_agent_graph


# =============================================================================
# Public API
# =============================================================================

def run_multi_agent(
    question: str,
    session_id: str = "",
    max_steps: int = 10,
) -> dict[str, Any]:
    """运行多Agent系统"""
    graph = get_multi_agent_graph()

    initial_state = MultiAgentGraphState(
        session_id=session_id,
        question=question,
        messages=[],
        task_type=None,
        supervisor_decision={},
        agent_responses={},
        final_answer="",
        reasoning_trace=[],
        error=None,
        agent_step=0,
        max_steps=max_steps,
        rewritten_question="",
        # 治理相关状态初始化
        rewrite_output={"ok": False, "original_question": question, "rewritten_query": question},
        governance_context={},
        context_guard_passed=True,
    )

    result = graph.invoke(initial_state)

    return {
        "question": result.get("question", question),
        "rewritten_question": result.get("rewritten_question", question),
        "answer": result.get("final_answer", ""),
        "reasoning_trace": result.get("reasoning_trace", []),
        "agent_responses": {
            k: v.model_dump() if hasattr(v, 'model_dump') else v
            for k, v in result.get("agent_responses", {}).items()
        },
        "supervisor_decision": result.get("supervisor_decision", {}),
        "task_type": result.get("task_type"),
        "error": result.get("error"),
        # 治理相关返回
        "governance_context": result.get("governance_context", {}),
        "context_guard_passed": result.get("context_guard_passed", True),
        "rewrite_output": result.get("rewrite_output", {}),
    }


def run_multi_agent_stream(
    question: str,
    session_id: str = "",
) -> list[dict[str, Any]]:
    """流式运行"""
    events = []

    # Step 1: Rewrite
    rewrite_response = rewrite_agent.execute(question, session_id)
    rewritten_question = question
    if rewrite_response.success and rewrite_response.result:
        rewritten_question = rewrite_response.result.get("rewritten_query", question)

    events.append({
        "type": "rewrite",
        "rewritten_question": rewritten_question,
        "used_memory_levels": rewrite_response.result.get("used_memory_levels", []) if rewrite_response.success else [],
    })

    # Step 2: Supervisor
    routing = supervisor_agent.route_task(question, session_id)
    events.append({
        "type": "supervisor",
        "task_type": routing["task_type"].value,
        "target_agents": [ag.value for ag in routing["target_agents"]],
    })

    # Step 3: 执行
    if routing["task_type"] == TaskType.DOCUMENT:
        response = document_agent.execute(question, session_id)
        events.append({
            "type": "document_result",
            "result": response.result,
            "success": response.success,
        })
    else:
        # QA
        memory_agent_wrapper.store(session_id, "user", question)
        qa_response = qa_agent.execute(
            question=question,
            session_id=session_id,
            rewritten_question=rewritten_question,
        )

        # 存储助手回答
        if qa_response.success and qa_response.result:
            answer = qa_response.result.get("answer", "")
            if answer:
                memory_agent_wrapper.store(session_id, "assistant", answer)

        events.append({
            "type": "qa_result",
            "result": qa_response.result,
            "success": qa_response.success,
        })

        events.append({
            "type": "final",
            "answer": qa_response.result.get("answer", "") if qa_response.success else "",
        })

    return events
