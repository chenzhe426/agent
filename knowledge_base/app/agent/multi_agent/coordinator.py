"""
Multi-Agent Coordinator for task distribution and result aggregation.
"""

from typing import Any
from app.agent.multi_agent.messages import AgentMessage, AgentResponse, TaskType, AgentRole


class MultiAgentCoordinator:
    """
    Coordinates task distribution to specialist agents and result aggregation.

    Responsibilities:
    - Route tasks to appropriate specialist agents
    - Collect results from agents
    - Aggregate multi-agent responses
    - Handle failures and retries
    """

    def __init__(self):
        self._agent_registry: dict[AgentRole, Any] = {}

    def register_agent(self, role: AgentRole, agent: Any) -> None:
        """Register a specialist agent."""
        self._agent_registry[role] = agent

    def get_agent(self, role: AgentRole) -> Any:
        """Get a registered agent by role."""
        return self._agent_registry.get(role)

    def create_task_message(
        self,
        sender: AgentRole,
        receiver: AgentRole,
        task_type: TaskType,
        payload: dict[str, Any],
        session_id: str = "",
        trace: list[dict[str, Any]] | None = None,
    ) -> AgentMessage:
        """Create a message for inter-agent communication."""
        return AgentMessage(
            sender=sender,
            receiver=receiver,
            task_type=task_type,
            payload=payload,
            session_id=session_id,
            trace=trace or [],
        )

    def aggregate_responses(
        self,
        responses: dict[AgentRole, AgentResponse],
        question: str,
    ) -> dict[str, Any]:
        """
        Aggregate responses from multiple specialist agents.

        Args:
            responses: Dict mapping agent roles to their responses
            question: The original user question

        Returns:
            Aggregated result with reasoning trace
        """
        aggregated = {
            "success": True,
            "primary_response": None,
            "all_responses": {},
            "reasoning_trace": [],
            "error": None,
        }

        # Collect all responses
        for role, response in responses.items():
            aggregated["all_responses"][role.value] = response.result

            if response.success:
                aggregated["reasoning_trace"].extend(response.reasoning_trace)
            else:
                if aggregated["error"]:
                    aggregated["error"] += f"; {response.error}"
                else:
                    aggregated["error"] = response.error

            # Use the first successful response as primary
            if response.success and aggregated["primary_response"] is None:
                aggregated["primary_response"] = response.result

        # If no successful response, use the first one
        if aggregated["primary_response"] is None and responses:
            first_response = next(iter(responses.values()))
            aggregated["primary_response"] = first_response.result
            aggregated["success"] = False

        return aggregated

    def should_use_multiple_agents(self, task_type: TaskType, question: str) -> list[TaskType]:
        """
        Determine if multiple agents should be used for a task.

        Complex questions may benefit from multiple perspectives.
        """
        complex_keywords = [
            "比较", "对比", "分析", "总结",
            "compare", "analyze", "summarize",
            "both", "and", "also",
        ]

        # Check if question suggests complex multi-step reasoning
        question_lower = question.lower()
        if any(kw in question_lower for kw in complex_keywords):
            return [TaskType.QA, TaskType.SEARCH]

        return [task_type]

    def build_final_response(
        self,
        aggregated: dict[str, Any],
        question: str,
    ) -> dict[str, Any]:
        """
        Build the final response from aggregated agent results.

        Args:
            aggregated: Aggregated results from aggregate_responses
            question: Original user question

        Returns:
            Final response dict
        """
        primary = aggregated.get("primary_response") or {}

        # Extract answer based on task type
        answer = ""
        if isinstance(primary, dict):
            answer = primary.get("answer", "") or primary.get("final_answer", "")
        elif isinstance(primary, str):
            answer = primary

        return {
            "question": question,
            "answer": answer,
            "reasoning_trace": aggregated.get("reasoning_trace", []),
            "agent_responses": aggregated.get("all_responses", {}),
            "success": aggregated.get("success", True),
            "error": aggregated.get("error"),
        }


# Global coordinator instance
coordinator = MultiAgentCoordinator()
