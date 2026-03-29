"""
Agent communication protocol and message types.
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Task types for agent routing."""
    QA = "qa"                    # Question answering
    SEARCH = "search"            # Knowledge base search
    IMPORT = "import"            # Document import/index
    SUMMARY = "summary"          # Document summarization
    CHAT = "chat"                # General chat
    MULTI = "multi"             # Multi-task (requires multiple agents)


class AgentRole(str, Enum):
    """Agent roles in the system."""
    SUPERVISOR = "supervisor"
    QA = "qa"
    SEARCH = "search"
    IMPORT = "import"
    COORDINATOR = "coordinator"


class AgentMessage(BaseModel):
    """Message passed between agents."""

    sender: AgentRole = Field(..., description="Sender agent role")
    receiver: AgentRole = Field(..., description="Receiver agent role")
    task_type: TaskType = Field(..., description="Type of task")
    payload: dict[str, Any] = Field(default_factory=dict, description="Task payload")
    trace: list[dict[str, Any]] = Field(default_factory=list, description="Agent reasoning trace")
    session_id: str = Field(default="", description="Session identifier")
    step: int = Field(default=0, description="Current step number")

    class Config:
        use_enum_values = True


class AgentResponse(BaseModel):
    """Response from an agent after processing."""

    agent: AgentRole = Field(..., description="Agent that generated this response")
    task_type: TaskType = Field(..., description="Type of task processed")
    result: dict[str, Any] = Field(default_factory=dict, description="Task result data")
    reasoning_trace: list[dict[str, Any]] = Field(default_factory=list, description="Agent reasoning steps")
    success: bool = Field(default=True, description="Whether task succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        use_enum_values = True


class MultiAgentState(BaseModel):
    """Shared state for multi-agent coordination."""

    session_id: str = ""
    question: str = ""
    messages: list[Any] = Field(default_factory=list)
    task_type: Optional[TaskType] = None
    supervisor_decision: dict[str, Any] = Field(default_factory=dict)
    agent_responses: dict[str, AgentResponse] = Field(default_factory=dict)
    final_answer: str = ""
    reasoning_trace: list[dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    agent_step: int = 0

    class Config:
        arbitrary_types_allowed = True
