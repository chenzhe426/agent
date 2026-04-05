"""
rewrite_agent.py - QueryRewriteAgent 在 multi_agent 模式下的封装

专门负责读取记忆并进行查询改写
"""

from typing import Any

from app.agent.multi_agent.messages import AgentResponse, TaskType, AgentRole
from app.agent.rewrite_agent.agent import query_rewrite_agent
from app.agent.rewrite_agent.schemas import RewriteInput


class RewriteAgentWrapper:
    """QueryRewriteAgent 的封装，用于 multi_agent 架构"""

    def execute(self, question: str, session_id: str = "", max_steps: int = 3) -> AgentResponse:
        """
        执行查询改写。

        Args:
            question: 用户问题
            session_id: 会话ID
            max_steps: 最大步数（此类操作通常1步完成）

        Returns:
            AgentResponse with rewritten query
        """
        try:
            # 调用 QueryRewriteAgent
            input_data = RewriteInput(
                question=question,
                session_id=session_id,
                use_history=True,
            )
            result = query_rewrite_agent.rewrite(input_data)

            if result.ok:
                return AgentResponse(
                    agent=AgentRole.REWRITE,
                    task_type=TaskType.REWRITE,
                    result={
                        "original_question": result.original_question,
                        "rewritten_query": result.rewritten_query,
                        "used_memory_levels": result.used_memory_levels,
                        "confidence": result.confidence,
                        # 结构化扩展字段
                        "intent": result.intent,
                        "normalized_query": result.normalized_query,
                        "entities": result.entities,
                        "doc_scope": result.doc_scope,
                        "memory_refs": result.memory_refs,
                        "source_tags": result.source_tags,
                        "provenance": result.provenance,
                    },
                    reasoning_trace=[{
                        "step": 1,
                        "action": "rewrite_query",
                        "observation": f"改写成功: {result.rewritten_query[:50]}... (intent={result.intent})",
                    }],
                    success=True,
                )
            else:
                return AgentResponse(
                    agent=AgentRole.REWRITE,
                    task_type=TaskType.REWRITE,
                    result={},
                    reasoning_trace=[],
                    success=False,
                    error=result.error,
                )

        except Exception as e:
            return AgentResponse(
                agent=AgentRole.REWRITE,
                task_type=TaskType.REWRITE,
                result={},
                reasoning_trace=[{
                    "step": 1,
                    "action": "rewrite_query",
                    "observation": f"改写失败: {str(e)}",
                }],
                success=False,
                error=str(e),
            )


# 全局实例
rewrite_agent = RewriteAgentWrapper()
