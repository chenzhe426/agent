"""
Multi-Agent system for Knowledge Base.

架构：
- 入口层：Rewrite (读取Memory改写问题)
- 编排层：Supervisor → QAAgent / DocumentAgent
- 共享状态层：Memory (记忆存储)
"""

from app.agent.multi_agent.messages import AgentMessage, AgentResponse, TaskType, AgentRole
from app.agent.multi_agent.coordinator import MultiAgentCoordinator
from app.agent.multi_agent.supervisor import SupervisorAgent
from app.agent.multi_agent.graph import run_multi_agent, run_multi_agent_stream

__all__ = [
    "AgentMessage",
    "AgentResponse",
    "TaskType",
    "AgentRole",
    "MultiAgentCoordinator",
    "SupervisorAgent",
    "run_multi_agent",
    "run_multi_agent_stream",
]
