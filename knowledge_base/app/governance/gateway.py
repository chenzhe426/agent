"""
governance/gateway.py - 统一治理网关（Unified Governance Gateway）

整合 Context Guard 和 Action Guard，提供统一的治理入口。

职责：
1. 接收治理请求并路由到相应的 Guard
2. 管理治理状态和审计日志
3. 提供高风险操作的人工确认接口
4. 聚合双阶段治理结果
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from app.governance.schemas import (
    GovernanceDecision,
    GovernanceResult,
    RiskLevel,
    RewriteStructuredOutput,
    ActionContext,
    AuditRecord,
)
from app.governance.context_guard import ContextGuard, context_guard
from app.governance.action_guard import ActionGuard, action_guard


class GovernanceGateway:
    """
    统一治理网关

    提供双阶段策略执行的统一入口：
    - 第一阶段：Context Guard（上下文治理）
    - 第二阶段：Action Guard（动作治理）

    使用方式：
        gateway = GovernanceGateway()

        # 第一阶段：Rewrite 输出治理
        ctx_result = gateway.guard_context(rewrite_output, session_id)

        # 第二阶段：工具调用治理
        action_result = gateway.guard_action(tool_name, tool_args, agent, session_id)
    """

    def __init__(
        self,
        context_guard: Optional[ContextGuard] = None,
        action_guard: Optional[ActionGuard] = None,
    ):
        """
        初始化治理网关

        Args:
            context_guard: 上下文治理器（可选，默认使用全局实例）
            action_guard: 动作治理器（可选，默认使用全局实例）
        """
        self._context_guard = context_guard or context_guard
        self._action_guard = action_guard or action_guard

        # 全局审计记录
        self._audit_records: list[AuditRecord] = []

        # 治理统计
        self._stats = {
            "total_context_guards": 0,
            "total_action_guards": 0,
            "denied_context": 0,
            "denied_action": 0,
            "confirmed_action": 0,
        }

    def guard_context(
        self,
        rewrite_output: dict[str, Any],
        session_id: str,
        tenant_id: str = "default",
    ) -> GovernanceResult:
        """
        第一阶段：上下文治理

        在 Rewrite 输出进入 Supervisor 之前执行

        Args:
            rewrite_output: Rewrite 节点的输出
            session_id: 会话ID
            tenant_id: 租户ID

        Returns:
            GovernanceResult: 治理决策结果
        """
        self._stats["total_context_guards"] += 1

        result = self._context_guard.guard(rewrite_output, session_id, tenant_id)

        if result.decision == GovernanceDecision.DENY:
            self._stats["denied_context"] += 1

        return result

    def guard_action(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        agent: str,
        session_id: str,
        tenant_id: str = "default",
        rewrite_output: Optional[dict[str, Any]] = None,
    ) -> GovernanceResult:
        """
        第二阶段：动作治理

        在 Agent 发起 MCP 调用之前执行

        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            agent: 来源 Agent
            session_id: 会话ID
            tenant_id: 租户ID
            rewrite_output: Rewrite 结构化输出（可选，用于传递上下文）

        Returns:
            GovernanceResult: 治理决策结果
        """
        self._stats["total_action_guards"] += 1

        result = self._action_guard.guard_tool_call(
            tool_name=tool_name,
            tool_args=tool_args,
            agent=agent,
            session_id=session_id,
            tenant_id=tenant_id,
            rewrite_output=rewrite_output,
        )

        if result.decision == GovernanceDecision.DENY:
            self._stats["denied_action"] += 1
        elif result.decision == GovernanceDecision.CONFIRM:
            self._stats["confirmed_action"] += 1

        return result

    def guard_context_structured(
        self,
        rewrite_output: RewriteStructuredOutput,
        session_id: str,
        tenant_id: str = "default",
    ) -> GovernanceResult:
        """
        便捷方法：使用结构化输入进行上下文治理

        Args:
            rewrite_output: Rewrite 结构化输出对象
            session_id: 会话ID
            tenant_id: 租户ID

        Returns:
            GovernanceResult: 治理决策结果
        """
        return self.guard_context(
            rewrite_output=rewrite_output.model_dump(),
            session_id=session_id,
            tenant_id=tenant_id,
        )

    def guard_action_structured(
        self,
        action_context: ActionContext,
    ) -> GovernanceResult:
        """
        便捷方法：使用结构化输入进行动作治理

        Args:
            action_context: 动作上下文

        Returns:
            GovernanceResult: 治理决策结果
        """
        return self._action_guard.guard(action_context)

    def confirm_pending_action(
        self,
        confirmation_id: str,
        approved: bool,
    ) -> bool:
        """
        确认或拒绝待定的高风险操作

        Args:
            confirmation_id: 确认ID
            approved: True=批准，False=拒绝

        Returns:
            是否成功
        """
        return self._action_guard.confirm_action(confirmation_id, approved)

    def get_pending_confirmation(
        self,
        confirmation_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        获取待确认操作

        Args:
            confirmation_id: 确认ID

        Returns:
            待确认操作详情
        """
        return self._action_guard.get_pending_confirmation(confirmation_id)

    def get_governance_context(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """
        获取会话的治理上下文

        Args:
            session_id: 会话ID

        Returns:
            治理上下文信息
        """
        ctx_records = self._context_guard.get_audit_records(session_id)
        action_records = self._action_guard.get_audit_records(session_id)

        return {
            "session_id": session_id,
            "context_guards": [r.model_dump() for r in ctx_records],
            "action_guards": [r.model_dump() for r in action_records],
            "stats": self._stats.copy(),
        }

    def get_stats(self) -> dict[str, Any]:
        """获取治理统计信息"""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """重置统计信息"""
        for key in self._stats:
            self._stats[key] = 0


# 全局实例
governance_gateway = GovernanceGateway()
