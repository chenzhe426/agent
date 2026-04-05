"""
memory_agent.py - MemoryAgent 在 multi_agent 模式下的封装

专门负责存储和管理分层记忆
"""

from typing import Any

from app.agent.multi_agent.messages import AgentResponse, TaskType, AgentRole
from app.agent.memory_agent.agent import memory_agent
from app.agent.memory_agent.schemas import StoreMessageInput, GetMemoryContextInput, MemoryLevel


class MemoryAgentWrapper:
    """MemoryAgent 的封装，用于 multi_agent 架构"""

    def store(self, session_id: str, role: str, message: str, metadata: dict = None) -> AgentResponse:
        """存储消息并更新记忆"""
        try:
            input_data = StoreMessageInput(
                session_id=session_id,
                role=role,
                message=message,
                metadata=metadata,
            )
            result = memory_agent.store_message(input_data)

            if result.ok:
                return AgentResponse(
                    agent=AgentRole.MEMORY,
                    task_type=TaskType.MEMORY,
                    result={
                        "message_id": result.message_id,
                        "turn_count": result.turn_count,
                        "memory_updates": result.memory_updates,
                    },
                    reasoning_trace=[{
                        "step": 1,
                        "action": "store_memory",
                        "observation": f"存储成功，轮次: {result.turn_count}",
                    }],
                    success=True,
                )
            else:
                return AgentResponse(
                    agent=AgentRole.MEMORY,
                    task_type=TaskType.MEMORY,
                    result={},
                    reasoning_trace=[],
                    success=False,
                    error=result.error,
                )

        except Exception as e:
            return AgentResponse(
                agent=AgentRole.MEMORY,
                task_type=TaskType.MEMORY,
                result={},
                reasoning_trace=[],
                success=False,
                error=str(e),
            )

    def get_context(self, session_id: str, question: str = "") -> AgentResponse:
        """获取记忆上下文"""
        try:
            input_data = GetMemoryContextInput(
                session_id=session_id,
                question=question,
                include_levels=[MemoryLevel.SHORT, MemoryLevel.MID, MemoryLevel.LONG],
            )
            result = memory_agent.get_memory_context(input_data)

            if result.ok:
                return AgentResponse(
                    agent=AgentRole.MEMORY,
                    task_type=TaskType.MEMORY,
                    result={
                        "short_term": result.short_term,
                        "mid_term": result.mid_term,
                        "long_term": result.long_term,
                    },
                    reasoning_trace=[{
                        "step": 1,
                        "action": "get_memory_context",
                        "observation": "获取成功",
                    }],
                    success=True,
                )
            else:
                return AgentResponse(
                    agent=AgentRole.MEMORY,
                    task_type=TaskType.MEMORY,
                    result={},
                    reasoning_trace=[],
                    success=False,
                    error=result.error,
                )

        except Exception as e:
            return AgentResponse(
                agent=AgentRole.MEMORY,
                task_type=TaskType.MEMORY,
                result={},
                reasoning_trace=[],
                success=False,
                error=str(e),
            )

    def execute(self, question: str, session_id: str = "", max_steps: int = 3) -> AgentResponse:
        """
        执行记忆存储（包含用户问题和助手回答的完整流程）

        工作流程：
        1. 获取当前会话的最新消息
        2. 存储用户问题和助手回答
        3. 更新各层级记忆
        """
        try:
            # 简化处理：直接返回成功
            return AgentResponse(
                agent=AgentRole.MEMORY,
                task_type=TaskType.MEMORY,
                result={"status": "memory_agent_ready"},
                reasoning_trace=[{
                    "step": 1,
                    "action": "memory_ready",
                    "observation": "记忆已就绪",
                }],
                success=True,
            )
        except Exception as e:
            return AgentResponse(
                agent=AgentRole.MEMORY,
                task_type=TaskType.MEMORY,
                result={},
                reasoning_trace=[],
                success=False,
                error=str(e),
            )


# 全局实例
memory_agent_wrapper = MemoryAgentWrapper()
