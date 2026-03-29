"""
Multi-Agent system for Knowledge Base.

Architecture: Supervisor + Specialist Agents
- Supervisor: Intent classification and task routing
- QA Agent: Complex Q&A with verification
- Search Agent: Precise retrieval and search
- Import Agent: Document management
"""

from app.agent.multi_agent.messages import AgentMessage, AgentResponse, TaskType
from app.agent.multi_agent.coordinator import MultiAgentCoordinator
from app.agent.multi_agent.supervisor import SupervisorAgent
from app.agent.multi_agent.qa_agent import QAAgent
from app.agent.multi_agent.search_agent import SearchAgent
from app.agent.multi_agent.import_agent import ImportAgent
from app.agent.multi_agent.graph import run_multi_agent, run_multi_agent_stream

__all__ = [
    "AgentMessage",
    "AgentResponse",
    "TaskType",
    "MultiAgentCoordinator",
    "SupervisorAgent",
    "QAAgent",
    "SearchAgent",
    "ImportAgent",
    "run_multi_agent",
    "run_multi_agent_stream",
]
