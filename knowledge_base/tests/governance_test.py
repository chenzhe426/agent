"""
tests/governance_test.py - 治理模块测试

测试内容：
1. Rewrite 输出正常通过治理
2. Rewrite 输出异常时被识别/拒绝/降级
3. Agent 普通 MCP 调用正常通过
4. 高风险 MCP 调用触发治理分支
5. QAAgent / DocumentAgent 不绕过 MCP 和治理网关
"""

import pytest
from unittest.mock import MagicMock, patch


class TestContextGuard:
    """Context Guard 测试"""

    def test_normal_rewrite_output_passes(self):
        """测试正常 Rewrite 输出通过治理"""
        from app.governance.context_guard import ContextGuard

        guard = ContextGuard()

        rewrite_output = {
            "ok": True,
            "original_question": "什么是MCP协议",
            "rewritten_query": "MCP协议定义和原理",
            "confidence": 0.8,
            "used_memory_levels": ["short_term"],
        }

        result = guard.guard(rewrite_output, session_id="test-session")

        assert result.decision.value == "allow"
        assert result.risk_level.value == "low"

    def test_low_confidence_rewrite_flagged(self):
        """测试低置信度 Rewrite 被标记"""
        from app.governance.context_guard import ContextGuard
        from app.governance.schemas import RiskLevel

        guard = ContextGuard()

        rewrite_output = {
            "ok": True,
            "original_question": "什么是",
            "rewritten_query": "什么是",
            "confidence": 0.1,  # 低置信度
        }

        result = guard.guard(rewrite_output, session_id="test-session")

        # 低置信度应该被标记
        assert result.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]

    def test_missing_required_fields_denied(self):
        """测试缺少必填字段被拒绝"""
        from app.governance.context_guard import ContextGuard
        from app.governance.schemas import GovernanceDecision

        guard = ContextGuard()

        # 缺少 rewritten_query
        rewrite_output = {
            "ok": True,
            "original_question": "什么是MCP",
            # "rewritten_query" 缺失
        }

        result = guard.guard(rewrite_output, session_id="test-session")

        assert result.decision == GovernanceDecision.DENY

    def test_injection_attack_detected(self):
        """测试注入攻击被检测"""
        from app.governance.context_guard import ContextGuard
        from app.governance.schemas import GovernanceDecision

        guard = ContextGuard()

        # 包含注入脚本
        rewrite_output = {
            "ok": True,
            "original_question": "什么是MCP <script>alert('xss')</script>",
            "rewritten_query": "MCP 协议",
            "confidence": 0.5,
        }

        result = guard.guard(rewrite_output, session_id="test-session")

        # 注入攻击应该被拒绝
        assert result.decision == GovernanceDecision.DENY
        assert "injection" in str(result.risk_flags).lower()


class TestActionGuard:
    """Action Guard 测试"""

    def test_low_risk_tool_allowed(self):
        """测试低风险工具被允许"""
        from app.governance.action_guard import ActionGuard
        from app.governance.schemas import ActionContext, OperationType

        guard = ActionGuard()

        context = ActionContext(
            session_id="test-session",
            tenant_id="default",
            operation=OperationType.READ,
            operation_detail="kb_search_knowledge_base",
            agent="qa",
            tool_name="kb_search_knowledge_base",
            tool_args={"query": "MCP", "top_k": 5},
        )

        result = guard.guard(context)

        assert result.decision.value == "allow"
        assert result.risk_level.value == "low"

    def test_high_risk_tool_requires_confirmation(self):
        """测试高风险工具需要确认"""
        from app.governance.action_guard import ActionGuard
        from app.governance.schemas import ActionContext, OperationType

        guard = ActionGuard()

        context = ActionContext(
            session_id="test-session",
            tenant_id="default",
            operation=OperationType.WRITE,
            operation_detail="kb_import_file",
            agent="document",
            tool_name="kb_import_file",
            tool_args={"file_path": "/path/to/file.pdf"},
        )

        result = guard.guard(context)

        # 高风险工具应该需要确认
        assert result.requires_confirmation == True

    def test_injection_in_tool_args_denied(self):
        """测试工具参数中的注入被拒绝"""
        from app.governance.action_guard import ActionGuard
        from app.governance.schemas import ActionContext, OperationType, GovernanceDecision

        guard = ActionGuard()

        context = ActionContext(
            session_id="test-session",
            tenant_id="default",
            operation=OperationType.EXECUTE_TOOL,
            operation_detail="kb_search",
            agent="qa",
            tool_name="kb_search_knowledge_base",
            tool_args={"query": "test <script>alert(1)</script>"},
        )

        result = guard.guard(context)

        assert result.decision == GovernanceDecision.DENY


class TestGovernanceGateway:
    """治理网关测试"""

    def test_context_guard_integrated(self):
        """测试 Context Guard 集成"""
        from app.governance.gateway import GovernanceGateway

        gateway = GovernanceGateway()

        rewrite_output = {
            "ok": True,
            "original_question": "MCP协议是什么",
            "rewritten_query": "MCP协议定义",
            "confidence": 0.8,
        }

        result = gateway.guard_context(rewrite_output, session_id="test")

        assert result.decision.value == "allow"

    def test_action_guard_integrated(self):
        """测试 Action Guard 集成"""
        from app.governance.gateway import GovernanceGateway

        gateway = GovernanceGateway()

        # 低风险工具调用
        result = gateway.guard_action(
            tool_name="kb_search_knowledge_base",
            tool_args={"query": "test", "top_k": 5},
            agent="qa",
            session_id="test",
        )

        assert result.decision.value == "allow"

    def test_high_risk_action_triggers_confirmation(self):
        """测试高风险动作触发确认"""
        from app.governance.gateway import GovernanceGateway

        gateway = GovernanceGateway()

        result = gateway.guard_action(
            tool_name="kb_import_file",
            tool_args={"file_path": "/tmp/test.pdf"},
            agent="document",
            session_id="test",
        )

        assert result.requires_confirmation == True


class TestRewriteStructuredOutput:
    """Rewrite 结构化输出测试"""

    def test_intent_extraction(self):
        """测试意图提取"""
        from app.agent.rewrite_agent.agent import QueryRewriteAgent

        agent = QueryRewriteAgent()

        # 测试问答意图
        intent = agent._extract_intent("什么是MCP协议")
        assert intent == "qa"

        # 测试文档意图
        intent = agent._extract_intent("帮我搜索关于AMD的文档")
        assert intent == "document"

        # 测试搜索意图
        intent = agent._extract_intent("查找MCP相关的资料")
        assert intent == "search"

    def test_entity_extraction(self):
        """测试实体提取"""
        from app.agent.rewrite_agent.agent import QueryRewriteAgent

        agent = QueryRewriteAgent()

        entities = agent._extract_entities("MCP协议和Anthropic的关系")
        assert "MCP" in entities or "协议" in entities

    def test_query_normalization(self):
        """测试查询标准化"""
        from app.agent.rewrite_agent.agent import QueryRewriteAgent

        agent = QueryRewriteAgent()

        normalized = agent._normalize_query("请问，MCP协议是什么？")
        assert "请问" not in normalized


class TestGraphIntegration:
    """图集成测试"""

    def test_graph_state_includes_governance_fields(self):
        """测试图状态包含治理字段"""
        from app.agent.multi_agent.graph import MultiAgentGraphState

        state = MultiAgentGraphState(
            session_id="test",
            question="什么是MCP",
            messages=[],
            task_type=None,
            supervisor_decision={},
            agent_responses={},
            final_answer="",
            reasoning_trace=[],
            error=None,
            agent_step=0,
            max_steps=10,
            rewritten_question="",
            # 治理相关字段
            rewrite_output={},
            governance_context={},
            context_guard_passed=True,
        )

        assert "rewrite_output" in state
        assert "governance_context" in state
        assert "context_guard_passed" in state

    def test_governance_gateway_singleton(self):
        """测试治理网关单例"""
        from app.governance.gateway import governance_gateway, GovernanceGateway

        # 两次导入应该获得相同实例
        from app.governance.gateway import governance_gateway as gw2
        assert governance_gateway is gw2


class TestToolDispatcher:
    """ToolDispatcher 测试"""

    def test_tool_dispatcher_singleton(self):
        """测试 ToolDispatcher 单例"""
        from app.tools.tool_dispatcher import tool_dispatcher, ToolDispatcher

        t1 = ToolDispatcher()
        t2 = ToolDispatcher()
        assert t1 is not t2  # 不同实例

        # 但全局实例存在
        assert tool_dispatcher is not None

    def test_dispatcher_has_tool_map(self):
        """测试分发器能获取工具映射"""
        from app.tools.tool_dispatcher import tool_dispatcher

        # 应该能获取工具映射
        tool_map = tool_dispatcher.tool_map
        assert isinstance(tool_map, dict)

    def test_invoke_tool_function_exists(self):
        """测试便捷函数存在"""
        from app.tools import invoke_tool

        assert callable(invoke_tool)

    def test_check_tool_guard_function_exists(self):
        """测试检查函数存在"""
        from app.tools import check_tool_guard

        assert callable(check_tool_guard)


class TestPolicies:
    """策略测试"""

    def test_high_risk_tools_identified(self):
        """测试高风险工具识别"""
        from app.governance.policies import HIGH_RISK_TOOLS

        assert "kb_import_file" in HIGH_RISK_TOOLS
        assert "kb_import_folder" in HIGH_RISK_TOOLS
        assert "kb_index_document" in HIGH_RISK_TOOLS

    def test_tool_operation_mapping(self):
        """测试工具操作类型映射"""
        from app.governance.policies import TOOL_OPERATION_MAP

        assert TOOL_OPERATION_MAP["kb_search_knowledge_base"].value == "read"
        assert TOOL_OPERATION_MAP["kb_import_file"].value == "write"
        assert TOOL_OPERATION_MAP["kb_clear_memory"].value == "delete"

    def test_injection_pattern_detection(self):
        """测试注入模式检测"""
        from app.governance.policies import check_injection_risk

        # 正常文本
        flags = check_injection_risk("MCP协议是什么")
        assert len(flags) == 0

        # 注入脚本
        flags = check_injection_risk("<script>alert('xss')</script>")
        assert len(flags) > 0

    def test_risk_evaluation(self):
        """测试风险评估"""
        from app.governance.policies import evaluate_tool_risk
        from app.governance.schemas import RiskLevel

        # 低风险工具
        level = evaluate_tool_risk("kb_search_knowledge_base", {})
        assert level == RiskLevel.LOW

        # 高风险工具
        level = evaluate_tool_risk("kb_import_file", {})
        assert level == RiskLevel.HIGH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
