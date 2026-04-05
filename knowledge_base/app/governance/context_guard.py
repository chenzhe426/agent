"""
governance/context_guard.py - 上下文治理（Context Guard）

第一阶段治理：Rewrite 输出进入 Supervisor 之前

职责：
1. Layer 1: 基础模式检测（XSS/HTML注入）
2. Layer 2: Schema 验证（必填字段、置信度、长度）
3. Layer 3: 语义风险评估（指令覆盖、角色提升、数据泄露等）
4. Layer 4: 来源追溯与信任治理
5. Layer 5: 分级决策系统（ALLOW/GRADE/DENY）

核心逻辑：
判断 Rewrite 输出有没有把不可信内容包装成可信任务意图
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
    AuditRecord,
    SourceTag,
)
from app.governance.policies import (
    # Layer 1
    check_injection_risk,
    check_sensitive_action,
    # Layer 2
    validate_rewrite_output,
    # Layer 3
    check_semantic_risk,
    flatten_semantic_risks,
    evaluate_semantic_risk_level,
    # Layer 4
    evaluate_provenance_trust,
    evaluate_intent_consistency,
    # Layer 5
    evaluate_context_risk,
    get_degradation_config,
)


class ContextGuard:
    """
    上下文治理器

    在 Rewrite 输出进入 Supervisor 之前执行多层次治理

    治理流程（5层）：
    1. Layer 1 - 基础模式检测：XSS/HTML/代码注入
    2. Layer 2 - Schema验证：必填字段、置信度、长度
    3. Layer 3 - 语义风险评估：指令覆盖、角色提升、数据泄露等
    4. Layer 4 - 来源追溯：验证source_tags、provenance、intent一致性
    5. Layer 5 - 分级决策：ALLOW/GRADE/DEGRADED/DENY
    """

    def __init__(self):
        self._audit_records: list[AuditRecord] = []

    def guard(
        self,
        rewrite_output: dict[str, Any],
        session_id: str,
        tenant_id: str = "default",
    ) -> GovernanceResult:
        """
        执行上下文治理（多层次）

        Args:
            rewrite_output: Rewrite 节点的输出（字典格式）
            session_id: 会话ID
            tenant_id: 租户ID

        Returns:
            GovernanceResult: 治理决策结果
        """
        start_time = time.perf_counter()
        operation_detail = f"context_guard:rewrite_output:{session_id}"

        try:
            # ========== Layer 2: Schema 验证 ==========
            is_valid, validation_errors = validate_rewrite_output(rewrite_output)
            if not is_valid:
                return self._make_deny_result(
                    message=f"Schema validation failed: {', '.join(validation_errors)}",
                    risk_level=RiskLevel.MEDIUM,
                    risk_flags=[f"schema_error:{e}" for e in validation_errors],
                    session_id=session_id,
                    tenant_id=tenant_id,
                    operation_detail=operation_detail,
                )

            # 转换为结构化输出
            structured = RewriteStructuredOutput(**rewrite_output)
            structured.session_id = session_id
            structured.tenant_id = tenant_id

            # ========== 收集所有风险标记 ==========
            all_risk_flags: list[str] = []
            semantic_risks: dict[str, list[str]] = {}

            # ========== Layer 1: 基础注入检测 ==========
            layer1_flags = self._layer1_pattern_detection(structured)
            all_risk_flags.extend(layer1_flags)

            # ========== Layer 3: 语义风险检测 ==========
            layer3_risks = self._layer3_semantic_analysis(structured)
            semantic_risks.update(layer3_risks)
            all_risk_flags.extend(flatten_semantic_risks(layer3_risks))

            # ========== Layer 4: 来源追溯与信任治理 ==========
            layer4_warnings = self._layer4_provenance_trust(structured)
            all_risk_flags.extend(layer4_warnings)

            # ========== 检查会话一致性 ==========
            if not self._check_session_consistency(structured, session_id):
                all_risk_flags.append("session_inconsistency")

            # ========== Layer 5: 综合风险评估 ==========
            risk_level = self._layer5_grading(
                rewrite_output={"ok": structured.ok, "risk_flags": all_risk_flags},
                semantic_risks=semantic_risks,
                intent_consistency=self._check_intent_consistency(structured),
                provenance_trust=self._check_provenance_trust(structured),
            )

            # ========== 决策生成 ==========
            decision, message, metadata = self._make_decision(
                risk_level=risk_level,
                semantic_risks=semantic_risks,
                all_risk_flags=all_risk_flags,
                structured=structured,
            )

            # ========== 构建返回结果 ==========
            result = GovernanceResult(
                decision=decision,
                risk_level=risk_level,
                message=message,
                risk_flags=list(set(all_risk_flags)),
                source_tags=list(structured.source_tags or []),
                requires_confirmation=(decision == GovernanceDecision.CONFIRM),
                confirmation_id=str(uuid.uuid4()) if decision == GovernanceDecision.CONFIRM else None,
                metadata=metadata,
            )

            # ========== 记录审计日志 ==========
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_audit(
                session_id=session_id,
                tenant_id=tenant_id,
                operation_detail=operation_detail,
                decision=result.decision,
                risk_level=result.risk_level,
                user_input=structured.original_question,
                rewritten_query=structured.rewritten_query,
                risk_flags=all_risk_flags,
                success=(result.decision != GovernanceDecision.DENY),
            )

            return result

        except Exception as e:
            # 解析失败，作为中风险处理
            return self._make_deny_result(
                message=f"Failed to process rewrite output: {str(e)}",
                risk_level=RiskLevel.MEDIUM,
                risk_flags=["parse_error"],
                session_id=session_id,
                tenant_id=tenant_id,
                operation_detail=operation_detail,
            )

    def _layer1_pattern_detection(self, structured: RewriteStructuredOutput) -> list[str]:
        """
        Layer 1: 基础模式检测

        检测 XSS/HTML/代码注入模式
        """
        risk_flags = []

        # 检查原始问题
        risk_flags.extend(check_injection_risk(structured.original_question))
        risk_flags.extend(check_sensitive_action(structured.original_question))

        # 检查改写后的问题
        risk_flags.extend(check_injection_risk(structured.rewritten_query))
        risk_flags.extend(check_sensitive_action(structured.rewritten_query))

        # 检查标准化查询
        if structured.normalized_query:
            risk_flags.extend(check_injection_risk(structured.normalized_query))

        # 检查实体列表
        for entity in structured.entities:
            risk_flags.extend(check_injection_risk(entity))

        return list(set(risk_flags))

    def _layer3_semantic_analysis(self, structured: RewriteStructuredOutput) -> dict[str, list[str]]:
        """
        Layer 3: 语义风险检测

        检测 9 大类语义风险：
        - instruction_override: 指令覆盖
        - role_escalation: 角色提升
        - data_exfiltration: 数据泄露
        - unauthorized_persistence: 未授权持久化
        - routing_manipulation: 路由操纵
        - cross_session_pollution: 跨会话污染
        - tool_action_priming: 工具/动作 priming
        - intent_hijacking: 意图劫持
        - context_pollution: 上下文污染
        """
        all_texts = [
            structured.original_question,
            structured.rewritten_query,
        ]
        if structured.normalized_query:
            all_texts.append(structured.normalized_query)
        all_texts.extend(structured.entities)

        combined_text = " ".join(all_texts)
        return check_semantic_risk(combined_text)

    def _layer4_provenance_trust(self, structured: RewriteStructuredOutput) -> list[str]:
        """
        Layer 4: 来源追溯与信任治理

        验证 source_tags 和 provenance 的可信度
        """
        warnings = []

        # 评估来源信任
        trust_score, trust_warnings = evaluate_provenance_trust(
            source_tags=structured.source_tags or [],
            provenance=structured.provenance or "rewrite",
        )
        warnings.extend(trust_warnings)

        # 低信任度警告
        if trust_score < 0.5:
            warnings.append(f"low_trust_score:{trust_score:.2f}")

        # 检查意图一致性
        is_consistent, consistency_msg = evaluate_intent_consistency(
            original_question=structured.original_question,
            rewritten_query=structured.rewritten_query,
            intent=structured.intent,
        )
        if not is_consistent:
            warnings.append(f"intent_inconsistency:{consistency_msg}")

        return warnings

    def _check_session_consistency(self, structured: RewriteStructuredOutput, session_id: str) -> bool:
        """检查会话一致性"""
        if structured.session_id and structured.session_id != session_id:
            return False
        return True

    def _check_intent_consistency(self, structured: RewriteStructuredOutput) -> bool:
        """检查意图一致性"""
        is_consistent, _ = evaluate_intent_consistency(
            original_question=structured.original_question,
            rewritten_query=structured.rewritten_query,
            intent=structured.intent,
        )
        return is_consistent

    def _check_provenance_trust(self, structured: RewriteStructuredOutput) -> float:
        """检查来源信任分数"""
        trust_score, _ = evaluate_provenance_trust(
            source_tags=structured.source_tags or [],
            provenance=structured.provenance or "rewrite",
        )
        return trust_score

    def _layer5_grading(
        self,
        rewrite_output: dict[str, Any],
        semantic_risks: dict[str, list[str]],
        intent_consistency: bool,
        provenance_trust: float,
    ) -> RiskLevel:
        """
        Layer 5: 分级决策系统

        综合所有层次的检测结果，确定最终风险等级
        """
        risk_flags = rewrite_output.get("risk_flags", [])

        # 1. 基础注入检测（Layer 1）- 最高优先级
        injection_flags = [f for f in risk_flags if "injection" in f.lower()]
        if injection_flags:
            return RiskLevel.CRITICAL

        # 2. 指令覆盖风险（Layer 3）- 直接高风险
        if "instruction_override" in semantic_risks:
            return RiskLevel.HIGH

        # 3. 数据泄露风险（Layer 3）- 直接高风险
        if "data_exfiltration" in semantic_risks:
            return RiskLevel.HIGH

        # 4. 路由操纵风险（Layer 3）- 直接高风险
        if "routing_manipulation" in semantic_risks:
            return RiskLevel.HIGH

        # 5. 意图不一致 - 中风险
        if not intent_consistency:
            return RiskLevel.MEDIUM

        # 6. 低信任来源 - 中风险
        if provenance_trust < 0.4:
            return RiskLevel.MEDIUM

        # 7. 其他语义风险 - 中风险
        if semantic_risks:
            return RiskLevel.MEDIUM

        # 8. 其他风险标记 - 中风险
        if risk_flags:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def _make_decision(
        self,
        risk_level: RiskLevel,
        semantic_risks: dict[str, list[str]],
        all_risk_flags: list[str],
        structured: RewriteStructuredOutput,
    ) -> tuple[GovernanceDecision, str, dict[str, Any]]:
        """
        生成治理决策

        Returns:
            (决策, 消息, 元数据)
        """
        # 获取降级配置
        config = get_degradation_config(risk_level)
        decision = config["decision"]

        # 构建消息
        if risk_level == RiskLevel.LOW:
            message = "Context passed governance check"
        elif risk_level == RiskLevel.MEDIUM:
            if semantic_risks:
                risk_types = list(semantic_risks.keys())
                message = f"Context passed with semantic risk flags: {', '.join(risk_types)}"
            else:
                message = "Context passed with risk flags"
        elif risk_level == RiskLevel.HIGH:
            risk_types = list(semantic_risks.keys()) if semantic_risks else all_risk_flags
            message = f"High risk detected, requires review: {', '.join(risk_types[:3])}"
        else:  # CRITICAL
            message = "Critical risk detected, context rejected"

        # 构建元数据
        metadata = {
            "confidence": structured.confidence,
            "intent": structured.intent,
            "entities": structured.entities,
            "can_use_memory": config["can_use_memory"],
            "can_use_external": config["can_use_external"],
            "semantic_risks": list(semantic_risks.keys()),
        }

        return decision, message, metadata

    def _make_deny_result(
        self,
        message: str,
        risk_level: RiskLevel,
        risk_flags: list[str],
        session_id: str,
        tenant_id: str,
        operation_detail: str,
    ) -> GovernanceResult:
        """构建拒绝结果"""
        self._record_audit(
            session_id=session_id,
            tenant_id=tenant_id,
            operation_detail=operation_detail,
            decision=GovernanceDecision.DENY,
            risk_level=risk_level,
            user_input="",
            rewritten_query="",
            risk_flags=risk_flags,
            success=False,
            error_message=message,
        )

        return GovernanceResult(
            decision=GovernanceDecision.DENY,
            risk_level=risk_level,
            message=message,
            risk_flags=risk_flags,
            source_tags=[],
            requires_confirmation=False,
            confirmation_id=None,
            metadata={},
        )

    def _record_audit(
        self,
        session_id: str,
        tenant_id: str,
        operation_detail: str,
        decision: GovernanceDecision,
        risk_level: RiskLevel,
        user_input: str,
        rewritten_query: str,
        risk_flags: list[str],
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """记录审计日志"""
        from app.governance.schemas import OperationType

        record = AuditRecord(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            session_id=session_id,
            tenant_id=tenant_id,
            operation=OperationType.QUERY,
            operation_detail=operation_detail,
            decision=decision,
            risk_level=risk_level,
            user_input=user_input,
            rewritten_query=rewritten_query,
            agent="context_guard",
            success=success,
            error_message=error_message,
            risk_flags=risk_flags,
        )
        self._audit_records.append(record)

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
context_guard = ContextGuard()
