"""
tool_dispatcher.py - 统一工具调用分发器

作为所有工具调用的唯一入口：
- Agent（QAAgent、DocumentAgent）通过此分发器调用工具
- MCP Server 通过此分发器调用工具
- 所有调用都经过 Action Guard 检查

设计原则：
1. 所有工具调用必须经过此分发器
2. Action Guard 在此处统一执行
3. 不直接暴露 TOOL_MAP 给 Agent
"""

from __future__ import annotations

import time
from typing import Any, Optional

from app.governance.gateway import governance_gateway
from app.governance.schemas import GovernanceDecision


class ToolDispatcher:
    """
    统一工具调用分发器

    所有工具调用必须通过此类，禁止直接调用 TOOL_MAP
    """

    def __init__(self):
        # 延迟导入避免循环依赖
        self._tool_map = None

    @property
    def tool_map(self) -> dict[str, Any]:
        """延迟加载 TOOL_MAP"""
        if self._tool_map is None:
            from app.agent.agent import TOOL_MAP
            self._tool_map = TOOL_MAP
        return self._tool_map

    def invoke(
        self,
        tool_name: str,
        args: dict[str, Any],
        agent: str,
        session_id: str,
        tenant_id: str = "default",
        governance_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        调用工具的统一入口

        Args:
            tool_name: 工具名称
            args: 工具参数
            agent: 来源 Agent（qa/document/memory/rewrite/mcp）
            session_id: 会话ID
            tenant_id: 租户ID
            governance_context: 治理上下文（从 Rewrite 传递过来的）

        Returns:
            工具执行结果
        """
        start_time = time.perf_counter()

        # 1. 获取工具
        tool = self.tool_map.get(tool_name)
        if not tool:
            return {
                "ok": False,
                "error": {"code": "TOOL_NOT_FOUND", "message": f"Tool {tool_name} not found"},
                "meta": {"tool_name": tool_name, "duration_ms": 0},
            }

        # 2. Action Guard 检查
        # 构建 rewrite_output 用于传递风险标记
        rewrite_output = {}
        if governance_context:
            rewrite_output = {
                "risk_flags": governance_context.get("risk_flags", []),
                "source_tags": governance_context.get("source_tags", []),
            }

        guard_result = governance_gateway.guard_action(
            tool_name=tool_name,
            tool_args=args,
            agent=agent,
            session_id=session_id,
            tenant_id=tenant_id,
            rewrite_output=rewrite_output,
        )

        # 3. 如果 Action Guard 拒绝，直接返回拒绝结果
        if guard_result.decision == GovernanceDecision.DENY:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return {
                "ok": False,
                "error": {
                    "code": "ACTION_GUARD_DENIED",
                    "message": f"Action denied: {guard_result.message}",
                    "risk_flags": guard_result.risk_flags,
                    "governance_decision": guard_result.decision.value,
                },
                "meta": {
                    "tool_name": tool_name,
                    "duration_ms": duration_ms,
                    "governance_check": True,
                    "governance_decision": guard_result.decision.value,
                    "risk_level": guard_result.risk_level.value if hasattr(guard_result.risk_level, 'value') else str(guard_result.risk_level),
                },
            }

        # 4. 如果需要确认，记录警告但继续执行
        requires_confirmation = guard_result.decision == GovernanceDecision.CONFIRM

        # 5. 执行工具
        try:
            result = tool.invoke(args)
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            if not isinstance(result, dict):
                result = {"ok": True, "data": result}

            # 添加治理元数据
            if "meta" not in result:
                result["meta"] = {}
            result["meta"].update({
                "tool_name": tool_name,
                "duration_ms": duration_ms,
                "governance_check": True,
                "governance_decision": guard_result.decision.value,
                "risk_level": guard_result.risk_level.value if hasattr(guard_result.risk_level, 'value') else str(guard_result.risk_level),
                "requires_confirmation": requires_confirmation,
            })

            return result

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return {
                "ok": False,
                "error": {"code": "TOOL_ERROR", "message": str(e)},
                "meta": {
                    "tool_name": tool_name,
                    "duration_ms": duration_ms,
                    "governance_check": True,
                    "governance_decision": guard_result.decision.value,
                },
            }

    def check_guard(
        self,
        tool_name: str,
        args: dict[str, Any],
        agent: str,
        session_id: str,
        tenant_id: str = "default",
        governance_context: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        仅检查 Action Guard，不执行工具

        Returns:
            (是否通过, 治理结果)
        """
        rewrite_output = {}
        if governance_context:
            rewrite_output = {
                "risk_flags": governance_context.get("risk_flags", []),
                "source_tags": governance_context.get("source_tags", []),
            }

        result = governance_gateway.guard_action(
            tool_name=tool_name,
            tool_args=args,
            agent=agent,
            session_id=session_id,
            tenant_id=tenant_id,
            rewrite_output=rewrite_output,
        )

        passed = result.decision in [GovernanceDecision.ALLOW, GovernanceDecision.GRADE]
        return passed, result.model_dump()


# 全局实例
tool_dispatcher = ToolDispatcher()


def invoke_tool(
    tool_name: str,
    args: dict[str, Any],
    agent: str,
    session_id: str,
    tenant_id: str = "default",
    governance_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    便捷函数：调用工具（经过 Action Guard）

    推荐所有工具调用都使用此函数
    """
    return tool_dispatcher.invoke(
        tool_name=tool_name,
        args=args,
        agent=agent,
        session_id=session_id,
        tenant_id=tenant_id,
        governance_context=governance_context,
    )


def check_tool_guard(
    tool_name: str,
    args: dict[str, Any],
    agent: str,
    session_id: str,
    tenant_id: str = "default",
    governance_context: Optional[dict[str, Any]] = None,
) -> tuple[bool, dict[str, Any]]:
    """
    便捷函数：检查工具调用是否通过治理（不执行）

    用于预先检查高风险操作
    """
    return tool_dispatcher.check_guard(
        tool_name=tool_name,
        args=args,
        agent=agent,
        session_id=session_id,
        tenant_id=tenant_id,
        governance_context=governance_context,
    )
