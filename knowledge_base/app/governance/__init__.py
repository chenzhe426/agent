"""
Governance Module - 统一治理网关

提供双阶段策略执行能力：
- Context Guard（上下文治理）：Rewrite 输出进入 Supervisor 之前
- Action Guard（动作治理）：Agent 发起 MCP 调用之前

模块结构：
- schemas.py: 治理相关数据模型
- policies.py: 策略定义和风险分级
- context_guard.py: 上下文治理
- action_guard.py: 动作治理
- gateway.py: 统一治理网关
"""

from app.governance.gateway import GovernanceGateway, governance_gateway
from app.governance.context_guard import ContextGuard, context_guard
from app.governance.action_guard import ActionGuard, action_guard
from app.governance.schemas import (
    GovernanceDecision,
    RiskLevel,
    GovernanceResult,
    AuditRecord,
)

__all__ = [
    "GovernanceGateway",
    "governance_gateway",
    "ContextGuard",
    "context_guard",
    "ActionGuard",
    "action_guard",
    "GovernanceDecision",
    "RiskLevel",
    "GovernanceResult",
    "AuditRecord",
]
