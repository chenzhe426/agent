"""
Multi-Agent StateGraph for Knowledge Base.

Architecture:
- Supervisor: Intent classification and routing
- Specialist Agents: QA, Search, Import
- Parallel execution for complex tasks
"""

from typing import Any, Optional
from langgraph.graph import END, StateGraph

from app.agent.multi_agent.messages import AgentMessage, AgentResponse, TaskType, AgentRole, MultiAgentState
from app.agent.multi_agent.supervisor import supervisor_agent
from app.agent.multi_agent.qa_agent import qa_agent
from app.agent.multi_agent.search_agent import search_agent
from app.agent.multi_agent.import_agent import import_agent


# =============================================================================
# Multi-Agent State
# =============================================================================

class MultiAgentGraphState(MultiAgentState):
    """Extended state for multi-agent graph."""

    max_steps: int = 10


# =============================================================================
# Graph Nodes
# =============================================================================

def _supervisor_node(state: MultiAgentGraphState) -> MultiAgentGraphState:
    """
    Supervisor node: classify intent and route to specialists.
    """
    question = state.get("question", "")
    session_id = state.get("session_id", "")

    # Get routing decision from supervisor
    routing = supervisor_agent.route_task(question, session_id)

    # Update state with supervisor decision
    new_state = {
        **state,
        "supervisor_decision": routing,
        "task_type": routing["task_type"],
        "agent_step": state.get("agent_step", 0) + 1,
    }

    return new_state


def _qa_node(state: MultiAgentGraphState) -> MultiAgentGraphState:
    """
    QA Specialist node: handle question answering.
    """
    question = state.get("question", "")
    session_id = state.get("session_id", "")
    max_steps = state.get("max_steps", 6)

    # Execute QA agent
    response = qa_agent.execute(question, session_id, max_steps)

    # Store response
    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses[AgentRole.QA.value] = response

    return {
        **state,
        "agent_responses": agent_responses,
        "reasoning_trace": state.get("reasoning_trace", []) + response.reasoning_trace,
        "agent_step": state.get("agent_step", 0) + 1,
    }


def _search_node(state: MultiAgentGraphState) -> MultiAgentGraphState:
    """
    Search Specialist node: handle knowledge base search.
    """
    question = state.get("question", "")
    session_id = state.get("session_id", "")
    max_steps = state.get("max_steps", 4)

    # Execute Search agent
    response = search_agent.execute(question, session_id, top_k=5, max_steps=max_steps)

    # Store response
    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses[AgentRole.SEARCH.value] = response

    return {
        **state,
        "agent_responses": agent_responses,
        "reasoning_trace": state.get("reasoning_trace", []) + response.reasoning_trace,
        "agent_step": state.get("agent_step", 0) + 1,
    }


def _import_node(state: MultiAgentGraphState) -> MultiAgentGraphState:
    """
    Import Specialist node: handle document management.
    """
    question = state.get("question", "")
    session_id = state.get("session_id", "")
    max_steps = state.get("max_steps", 5)

    # Execute Import agent
    response = import_agent.execute(question, session_id, max_steps)

    # Store response
    agent_responses = dict(state.get("agent_responses", {}))
    agent_responses[AgentRole.IMPORT.value] = response

    return {
        **state,
        "agent_responses": agent_responses,
        "reasoning_trace": state.get("reasoning_trace", []) + response.reasoning_trace,
        "agent_step": state.get("agent_step", 0) + 1,
    }


def _aggregation_node(state: MultiAgentGraphState) -> MultiAgentGraphState:
    """
    Aggregation node: combine results from multiple agents.
    """
    agent_responses = state.get("agent_responses", {})
    question = state.get("question", "")

    # Convert string keys back to AgentRole
    responses_dict = {}
    for key, response in agent_responses.items():
        if isinstance(key, str):
            try:
                responses_dict[AgentRole(key)] = response
            except ValueError:
                responses_dict[AgentRole.QA] = response
        else:
            responses_dict[key] = response

    # Aggregate results
    aggregated = supervisor_agent.aggregate_results(responses_dict, question)
    final_response = supervisor_agent.build_final_answer(aggregated, question)

    return {
        **state,
        "final_answer": final_response.get("answer", ""),
        "reasoning_trace": state.get("reasoning_trace", []) + [{
            "step": state.get("agent_step", 0) + 1,
            "thought": "聚合多Agent结果",
            "action": "aggregation",
            "observation": f"聚合了 {len(agent_responses)} 个Agent的响应",
        }],
    }


# =============================================================================
# Routing Logic
# =============================================================================

def _route_after_supervisor(state: MultiAgentGraphState) -> list[str]:
    """
    Route to appropriate specialist agents after supervisor decision.
    """
    routing = state.get("supervisor_decision", {})
    target_agents = routing.get("target_agents", [])
    task_type = routing.get("task_type")

    # Build routing targets
    targets = []
    for agent_role in target_agents:
        if agent_role == AgentRole.QA:
            targets.append("qa")
        elif agent_role == AgentRole.SEARCH:
            targets.append("search")
        elif agent_role == AgentRole.IMPORT:
            targets.append("import")

    # For multi-task, go to parallel execution
    if task_type == TaskType.MULTI and len(targets) > 1:
        return targets

    # Single task - go directly to that agent
    if targets:
        return targets

    # Default to QA
    return ["qa"]


def _should_aggregate(state: MultiAgentGraphState) -> bool:
    """
    Check if we should go to aggregation node.
    """
    routing = state.get("supervisor_decision", {})
    target_agents = routing.get("target_agents", [])

    # If multiple agents were called, aggregate
    return len(target_agents) > 1


# =============================================================================
# Build Graph
# =============================================================================

def _build_graph() -> Any:
    """Build the multi-agent state graph."""
    from langgraph.graph import END

    workflow = StateGraph(MultiAgentGraphState)

    # Add nodes
    workflow.add_node("supervisor", _supervisor_node)
    workflow.add_node("qa", _qa_node)
    workflow.add_node("search", _search_node)
    workflow.add_node("import", _import_node)
    workflow.add_node("aggregation", _aggregation_node)

    # Set entry point
    workflow.set_entry_point("supervisor")

    # Supervisor routing
    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "qa": "qa",
            "search": "search",
            "import": "import",
        },
    )

    # Specialist nodes route to aggregation or back to supervisor for multi-task
    def specialist_route(state: MultiAgentGraphState) -> str:
        routing = state.get("supervisor_decision", {})
        target_agents = routing.get("target_agents", [])

        if len(target_agents) > 1:
            # Multi-task: check if all agents have responded
            agent_responses = state.get("agent_responses", {})
            responded = set()
            for key in agent_responses.keys():
                if isinstance(key, str):
                    responded.add(key)
                else:
                    responded.add(key.value)

            # If all targeted agents have responded, aggregate
            targeted = {ag.value for ag in target_agents}
            if targeted.issubset(responded):
                return "aggregation"

        # Single task or waiting for more agents
        return END

    workflow.add_conditional_edges(
        "qa",
        specialist_route,
        {"aggregation": "aggregation", END: END},
    )

    workflow.add_conditional_edges(
        "search",
        specialist_route,
        {"aggregation": "aggregation", END: END},
    )

    workflow.add_conditional_edges(
        "import",
        specialist_route,
        {"aggregation": "aggregation", END: END},
    )

    # Aggregation always ends
    workflow.add_edge("aggregation", END)

    return workflow.compile()


# Compile graph once at module load
_multi_agent_graph = None


def get_multi_agent_graph() -> Any:
    """Get the compiled multi-agent graph."""
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
    """
    Run the multi-agent system synchronously.

    Args:
        question: User question
        session_id: Session identifier
        max_steps: Maximum total steps

    Returns:
        Dict with final answer and reasoning trace
    """
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
    )

    result = graph.invoke(initial_state)

    return {
        "question": result.get("question", question),
        "answer": result.get("final_answer", ""),
        "reasoning_trace": result.get("reasoning_trace", []),
        "agent_responses": {
            k: v.model_dump() if hasattr(v, 'model_dump') else v
            for k, v in result.get("agent_responses", {}).items()
        },
        "supervisor_decision": result.get("supervisor_decision", {}),
        "task_type": result.get("task_type"),
        "error": result.get("error"),
    }


def run_multi_agent_stream(
    question: str,
    session_id: str = "",
) -> list[dict[str, Any]]:
    """
    Run the multi-agent system and return streaming events.

    Args:
        question: User question
        session_id: Session identifier

    Returns:
        List of event dicts for SSE streaming
    """
    events = []

    # Step 1: Supervisor decision
    routing = supervisor_agent.route_task(question, session_id)
    events.append({
        "type": "supervisor",
        "task_type": routing["task_type"].value,
        "target_agents": [ag.value for ag in routing["target_agents"]],
        "classification": routing["classification"],
    })

    # Step 2: Execute target agents
    target_agents = routing["target_agents"]
    agent_responses = {}

    for agent_role in target_agents:
        if agent_role == AgentRole.QA:
            response = qa_agent.execute(question, session_id)
        elif agent_role == AgentRole.SEARCH:
            response = search_agent.execute(question, session_id)
        elif agent_role == AgentRole.IMPORT:
            response = import_agent.execute(question, session_id)
        else:
            continue

        agent_responses[agent_role] = response
        events.append({
            "type": f"{agent_role.value}_result",
            "agent": agent_role.value,
            "success": response.success,
            "result": response.result,
            "reasoning_trace": response.reasoning_trace,
        })

    # Step 3: Aggregate results
    aggregated = supervisor_agent.aggregate_results(agent_responses, question)
    final_response = supervisor_agent.build_final_answer(aggregated, question)

    events.append({
        "type": "final",
        "answer": final_response.get("answer", ""),
        "success": final_response.get("success", True),
    })

    return events
