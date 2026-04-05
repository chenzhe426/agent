# Governance 快速参考

## 风险等级

| 等级 | 值 | 说明 | 决策 |
|-----|-----|------|------|
| LOW | `"low"` | 低风险，正常执行 | ALLOW |
| MEDIUM | `"medium"` | 中风险，需记录 | ALLOW + flags |
| HIGH | `"high"` | 高风险，需确认 | GRADE |
| CRITICAL | `"critical"` | 极高风险，拦截 | DENY |

## 治理决策

| 决策 | 值 | 说明 |
|-----|-----|------|
| ALLOW | `"allow"` | 允许通过 |
| GRADE | `"grade"` | 降级处理（禁用外部来源） |
| DENY | `"deny"` | 直接拒绝 |

## 来源信任等级

| 来源 | 信任分数 |
|-----|---------|
| USER_INPUT | 1.0 |
| VECTOR_SEARCH | 0.9 |
| CACHE | 0.7 |
| MEMORY_SHORT | 0.5 |
| MEMORY_MID | 0.4 |
| MEMORY_LONG | 0.3 |
| EXTERNAL_API | 0.2 |

## 语义风险类型

| 类型 | 风险等级 |
|-----|---------|
| instruction_override | HIGH |
| data_exfiltration | HIGH |
| routing_manipulation | HIGH |
| role_escalation | MEDIUM |
| cross_session_pollution | MEDIUM |
| context_pollution | MEDIUM |
| unauthorized_persistence | MEDIUM |
| tool_action_priming | MEDIUM |
| intent_hijacking | MEDIUM |

## 降级配置

| 风险等级 | 允许使用Memory | 允许使用External |
|---------|---------------|-----------------|
| LOW | ✓ | ✓ |
| MEDIUM | ✓ | ✗ |
| HIGH | ✗ | ✗ |
| CRITICAL | ✗ | ✗ |

## 调用示例

```python
from app.governance.context_guard import ContextGuard
from app.governance.schemas import GovernanceDecision, RiskLevel

guard = ContextGuard()

result = guard.guard(
    rewrite_output={
        "ok": True,
        "original_question": "什么是MCP",
        "rewritten_query": "MCP协议定义",
        "confidence": 0.8,
    },
    session_id="session-123",
)

# 检查结果
if result.decision == GovernanceDecision.ALLOW:
    print("通过治理")
elif result.decision == GovernanceDecision.GRADE:
    print(f"降级处理: {result.message}")
    print(f"可使用Memory: {result.metadata.get('can_use_memory')}")
    print(f"可使用External: {result.metadata.get('can_use_external')}")
elif result.decision == GovernanceDecision.DENY:
    print(f"拒绝: {result.message}")
    print(f"风险标记: {result.risk_flags}")
```
