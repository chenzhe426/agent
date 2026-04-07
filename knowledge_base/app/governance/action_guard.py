"""
governance/action_guard.py - 动作治理（Action Guard）

第二阶段治理：Agent 发起 MCP 调用之前

职责：
1. 操作类型识别
2. 权限判断
3. 参数 schema 校验
4. 风险等级判定
5. 审计日志记录（输出到外部存储）
6. 高风险动作钩子（人工确认）
7. 状态写入和资源访问治理
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.governance.schemas import (
    GovernanceDecision,
    GovernanceResult,
    RiskLevel,
    OperationType,
    ActionContext,
    AuditRecord,
    RewriteStructuredOutput,
)
from app.governance.policies import (
    HIGH_RISK_TOOLS,
    MEDIUM_RISK_TOOLS,
    TOOL_OPERATION_MAP,
    TOOL_RESOURCE_MAP,
    evaluate_tool_risk,
    check_injection_risk,
    check_sensitive_action,
)
from loguru import logger


class AuditLogger:
    """
    审计日志输出器

    将审计记录输出到文件（JSONL 格式），支持：
    1. 本地文件存储
    2. 可配置日志目录
    3. 自动按日期分割
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or os.getenv("AUDIT_LOG_DIR", "logs/audit")
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """确保日志目录存在"""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def _get_log_path(self) -> str:
        """获取当日志文件路径"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"audit_{date_str}.jsonl")

    def log(self, record: AuditRecord) -> None:
        """
        输出审计日志到文件

        Args:
            record: 审计记录
        """
        try:
            log_path = self._get_log_path()

            # 将 record 转换为字典
            record_dict = {
                "timestamp": record.timestamp,
                "session_id": record.session_id,
                "tenant_id": record.tenant_id,
                "operation": record.operation.value if hasattr(record.operation, 'value') else str(record.operation),
                "operation_detail": record.operation_detail,
                "decision": record.decision.value if hasattr(record.decision, 'value') else str(record.decision),
                "risk_level": record.risk_level.value if hasattr(record.risk_level, 'value') else str(record.risk_level),
                "user_input": record.user_input,
                "rewritten_query": record.rewritten_query,
                "agent": record.agent,
                "tool_name": record.tool_name,
                "tool_args": self._sanitize_args(record.tool_args),
                "success": record.success,
                "error_message": record.error_message,
                "risk_flags": record.risk_flags,
            }

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")

            logger.debug(f"Audit log written: {record.operation_detail} by {record.agent}")

        except Exception as e:
            # 日志写入失败不影响主流程，只记录错误
            logger.error(f"Failed to write audit log: {e}")

    def _sanitize_args(self, args: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """
        脱敏工具参数中的敏感信息

        Args:
            args: 原始工具参数

        Returns:
            脱敏后的参数
        """
        if not args:
            return args

        # 敏感字段列表
        sensitive_keys = {"password", "token", "api_key", "secret", "credential", "auth"}

        sanitized = {}
        for key, value in args.items():
            if any(s in key.lower() for s in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value

        return sanitized


# 全局审计日志器实例
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """获取全局审计日志器实例"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


class ActionGuard:
    """
    动作治理器

    在 Agent 发起 MCP 调用之前执行治理

    治理流程：
    1. 解析操作上下文
    2. 识别操作类型
    3. 评估风险等级
    4. 校验参数 schema
    5. 执行权限检查
    6. 决策：允许/拒绝/确认/降级
    7. 记录审计日志
    8. 触发高风险钩子
    """

    def __init__(self):
        self._audit_records: list[AuditRecord] = []
        # 待确认的操作存储（用于人工确认机制）
        self._pending_confirmations: dict[str, dict[str, Any]] = {}

    def guard(
        self,
        action_context: ActionContext,
    ) -> GovernanceResult:
        """
        执行动作治理

        Args:
            action_context: 动作上下文

        Returns:
            GovernanceResult: 治理决策结果
        """
        start_time = time.perf_counter()
        operation_detail = f"action_guard:{action_context.operation.value}:{action_context.agent}"

        try:
            risk_flags = []

            # 1. 检查参数中的注入风险
            if action_context.tool_args:
                args_str = str(action_context.tool_args)
                risk_flags.extend(check_injection_risk(args_str))
                risk_flags.extend(check_sensitive_action(args_str))

            # 2. 继承 Rewrite 阶段的风险标记
            if action_context.risk_flags:
                risk_flags.extend(action_context.risk_flags)

            # 3. 评估风险等级
            tool_name = action_context.tool_name or ""
            risk_level = evaluate_tool_risk(tool_name, action_context.tool_args)

            # 如果已有高风险标记，提升风险等级
            if risk_flags and risk_level == RiskLevel.LOW:
                if any("injection" in f for f in risk_flags):
                    risk_level = RiskLevel.CRITICAL
                elif any("sensitive" in f for f in risk_flags):
                    risk_level = RiskLevel.HIGH

            # 4. 根据风险等级决策
            if risk_level == RiskLevel.LOW:
                decision = GovernanceDecision.ALLOW
                message = f"Action {action_context.operation.value} allowed"

            elif risk_level == RiskLevel.MEDIUM:
                # 中风险：记录并通过
                decision = GovernanceDecision.ALLOW
                message = f"Action {action_context.operation.value} allowed with logging"

            elif risk_level == RiskLevel.HIGH:
                # 高风险：需要确认
                decision = GovernanceDecision.CONFIRM
                message = f"High risk action requires confirmation"
                confirmation_id = str(uuid.uuid4())
                self._store_pending_confirmation(
                    confirmation_id=confirmation_id,
                    action_context=action_context,
                    risk_level=risk_level,
                )

            else:  # CRITICAL
                # 极高风险：拒绝
                decision = GovernanceDecision.DENY
                message = "Critical risk action denied"
                confirmation_id = None

            # 5. 构建返回结果
            result = GovernanceResult(
                decision=decision,
                risk_level=risk_level,
                message=message,
                risk_flags=risk_flags,
                source_tags=action_context.source_tags,
                requires_confirmation=(decision == GovernanceDecision.CONFIRM),
                confirmation_id=str(uuid.uuid4()) if decision == GovernanceDecision.CONFIRM else None,
                metadata={
                    "operation": action_context.operation.value,
                    "operation_detail": action_context.operation_detail,
                    "agent": action_context.agent,
                    "tool_name": tool_name,
                    "resource_type": action_context.resource_type,
                    "duration_ms": int((time.perf_counter() - start_time) * 1000),
                },
            )

            # 6. 记录审计日志
            self._record_audit(
                action_context=action_context,
                decision=decision,
                risk_level=risk_level,
                risk_flags=risk_flags,
                success=(decision != GovernanceDecision.DENY),
                operation_detail=operation_detail,
            )

            return result

        except Exception as e:
            # 出错作为中风险处理
            return GovernanceResult(
                decision=GovernanceDecision.DENY,
                risk_level=RiskLevel.MEDIUM,
                message=f"Action guard error: {str(e)}",
                risk_flags=["action_guard_error"],
                source_tags=action_context.source_tags,
                requires_confirmation=False,
                confirmation_id=None,
                metadata={"error": str(e)},
            )

    def guard_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        agent: str,
        session_id: str,
        tenant_id: str = "default",
        rewrite_output: Optional[dict[str, Any]] = None,
    ) -> GovernanceResult:
        """
        便捷方法：治理工具调用

        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            agent: 来源 Agent
            session_id: 会话ID
            tenant_id: 租户ID
            rewrite_output: Rewrite 结构化输出（可选）

        Returns:
            GovernanceResult: 治理决策结果
        """
        # 获取操作类型
        operation = TOOL_OPERATION_MAP.get(tool_name, OperationType.EXECUTE_TOOL)

        # 获取资源类型
        resource_type = TOOL_RESOURCE_MAP.get(tool_name, "unknown")

        # 构建动作上下文
        action_context = ActionContext(
            session_id=session_id,
            tenant_id=tenant_id,
            operation=operation,
            operation_detail=f"{operation.value}:{tool_name}",
            agent=agent,
            tool_name=tool_name,
            tool_args=tool_args,
            resource_type=resource_type,
            resource_id=self._extract_resource_id(tool_name, tool_args),
            risk_flags=rewrite_output.get("risk_flags", []) if rewrite_output else [],
            source_tags=rewrite_output.get("source_tags", []) if rewrite_output else [],
        )

        return self.guard(action_context)

    def _extract_resource_id(self, tool_name: str, tool_args: dict[str, Any]) -> Optional[str]:
        """从工具参数中提取资源ID"""
        if "document_id" in tool_args:
            return f"doc:{tool_args['document_id']}"
        if "session_id" in tool_args:
            return f"session:{tool_args['session_id']}"
        if "file_path" in tool_args:
            return f"file:{tool_args['file_path']}"
        return None

    def _store_pending_confirmation(
        self,
        confirmation_id: str,
        action_context: ActionContext,
        risk_level: RiskLevel,
    ) -> None:
        """存储待确认操作"""
        self._pending_confirmations[confirmation_id] = {
            "action_context": action_context,
            "risk_level": risk_level,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def get_pending_confirmation(self, confirmation_id: str) -> Optional[dict[str, Any]]:
        """获取待确认状态"""
        return self._pending_confirmations.get(confirmation_id)

    def confirm_action(self, confirmation_id: str, approved: bool) -> bool:
        """
        确认或拒绝操作

        Args:
            confirmation_id: 确认ID
            approved: True=批准，False=拒绝

        Returns:
            是否成功
        """
        if confirmation_id not in self._pending_confirmations:
            return False

        pending = self._pending_confirmations[confirmation_id]

        if approved:
            # 批准后记录审计
            action_context = pending["action_context"]
            self._record_audit(
                action_context=action_context,
                decision=GovernanceDecision.ALLOW,
                risk_level=pending["risk_level"],
                risk_flags=["confirmed_by_user"],
                success=True,
                operation_detail=f"action_confirmed:{confirmation_id}",
            )

        # 移除待确认记录
        del self._pending_confirmations[confirmation_id]
        return True

    def _record_audit(
        self,
        action_context: ActionContext,
        decision: GovernanceDecision,
        risk_level: RiskLevel,
        risk_flags: list[str],
        success: bool,
        operation_detail: str,
        error_message: Optional[str] = None,
    ) -> None:
        """记录审计日志（内存 + 外部文件）"""
        record = AuditRecord(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            session_id=action_context.session_id,
            tenant_id=action_context.tenant_id,
            operation=action_context.operation,
            operation_detail=operation_detail,
            decision=decision,
            risk_level=risk_level,
            user_input="",
            rewritten_query="",
            agent=action_context.agent,
            tool_name=action_context.tool_name,
            tool_args=action_context.tool_args,
            success=success,
            error_message=error_message,
            risk_flags=risk_flags,
        )

        # 1. 保存到内存列表（用于程序查询）
        self._audit_records.append(record)

        # 2. 输出到外部文件（用于持久化审计）
        try:
            audit_logger = get_audit_logger()
            audit_logger.log(record)
        except Exception as e:
            logger.warning(f"Failed to write external audit log: {e}")

    def get_audit_records(self, session_id: Optional[str] = None) -> list[AuditRecord]:
        """获取审计记录"""
        if session_id:
            return [r for r in self._audit_records if r.session_id == session_id]
        return self._audit_records

    def clear_audit_records(self, session_id: Optional[str] = None) -> None:
        """清除审计记录"""
        if session_id:
            self._audit_records = [r for r in self._audit_records if r.session_id != session_id]
        else:
            self._audit_records = []


# 全局实例
action_guard = ActionGuard()
