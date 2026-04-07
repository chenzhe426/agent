# Action Guard 动作治理技术文档

## 概述

Action Guard 是知识库 Agent 系统的第二阶段治理组件，位于 Agent 发起 MCP 调用之前的治理防线。其核心职责是**判断 Agent 发起的工具调用是否安全、是否具有相应权限、参数是否符合预期**。

## 架构位置

```
用户输入 → Rewrite → Context Guard → [Action Guard] → MCP 工具调用
                                    ↑
                              第二阶段治理
```

## 核心职责

1. **操作类型识别** - 识别当前操作的类型（读/写/删除/执行工具等）
2. **权限判断** - 判断 Agent 是否有权执行该操作
3. **参数 Schema 校验** - 验证工具参数是否符合预期格式
4. **风险等级判定** - 评估操作的风险等级
5. **审计日志记录** - 记录所有操作的审计日志
6. **高风险动作钩子** - 对高风险操作触发人工确认机制
7. **状态写入治理** - 对写入操作进行特殊治理

## 治理流程

```
ActionContext 输入
       ↓
┌──────────────────┐
│ 1. 注入风险检测   │ ← 检查 tool_args 中的 XSS/HTML/代码注入
└────────┬─────────┘
       ↓
┌──────────────────┐
│ 2. 敏感操作检测   │ ← 检查危险操作模式
└────────┬─────────┘
       ↓
┌──────────────────┐
│ 3. 风险等级评估   │ ← 根据工具类型和参数判定风险等级
└────────┬─────────┘
       ↓
┌──────────────────┐
│ 4. 决策判定       │
│  LOW → ALLOW     │
│  MEDIUM → ALLOW  │
│  HIGH → CONFIRM  │
│  CRITICAL → DENY │
└────────┬─────────┘
       ↓
┌──────────────────┐
│ 5. 审计日志记录   │ ← 记录完整操作审计信息
└────────┬─────────┘
       ↓
    GovernanceResult
```

## 风险等级与决策

| 风险等级 | 决策 | 说明 |
|---------|------|------|
| `LOW` | ALLOW | 低风险操作，直接放行 |
| `MEDIUM` | ALLOW | 中风险操作，放行但记录日志 |
| `HIGH` | CONFIRM | 高风险操作，需要人工确认 |
| `CRITICAL` | DENY | 极高风险操作，直接拒绝 |

## 核心组件

### ActionGuard 类

```python
class ActionGuard:
    """动作治理器"""

    def guard(self, action_context: ActionContext) -> GovernanceResult:
        """执行动作治理"""

    def guard_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        agent: str,
        session_id: str,
        tenant_id: str = "default",
        rewrite_output: Optional[dict[str, Any]] = None,
    ) -> GovernanceResult:
        """便捷方法：治理工具调用"""

    def confirm_action(self, confirmation_id: str, approved: bool) -> bool:
        """确认或拒绝待定操作"""
```

### ActionContext 数据模型

```python
class ActionContext(BaseModel):
    session_id: str           # 会话ID
    tenant_id: str            # 租户ID
    operation: OperationType # 操作类型
    operation_detail: str     # 操作详情

    agent: str                # 来源Agent
    tool_name: Optional[str]  # 工具名称
    tool_args: dict[str, Any] # 工具参数

    resource_type: Optional[str]  # 资源类型
    resource_id: Optional[str]     # 资源ID

    risk_flags: list[str]     # 继承的风险标记
    source_tags: list[str]     # 继承的来源标记
```

### GovernanceResult 返回结构

```python
class GovernanceResult(BaseModel):
    decision: GovernanceDecision       # 治理决策
    risk_level: RiskLevel               # 风险等级
    message: str                         # 决策消息
    risk_flags: list[str]                # 识别的风险标记
    source_tags: list[str]               # 来源标记
    requires_confirmation: bool          # 是否需要人工确认
    confirmation_id: Optional[str]       # 确认ID
    metadata: dict[str, Any]             # 额外元数据
```

## 工具风险分级

### HIGH_RISK_TOOLS (高风险工具)

需要人工确认才能执行：

| 工具名称 | 说明 |
|---------|------|
| `kb_import_file` | 文件导入 |
| `kb_import_folder` | 文件夹批量导入 |
| `kb_index_document` | 文档索引构建 |
| `kb_clear_memory` | 清除记忆 |

### MEDIUM_RISK_TOOLS (中风险工具)

正常执行，但需要记录审计日志：

| 工具名称 | 说明 |
|---------|------|
| `kb_store_memory` | 存储记忆 |
| `kb_delete_document` | 删除文档 |
| `kb_update_document` | 更新文档 |

## 工具到操作类型映射

```python
TOOL_OPERATION_MAP = {
    "kb_search_knowledge_base": OperationType.READ,
    "kb_summarize_document": OperationType.READ,
    "kb_get_chat_history": OperationType.READ_MEMORY,
    "kb_import_file": OperationType.WRITE,
    "kb_import_folder": OperationType.WRITE,
    "kb_index_document": OperationType.WRITE,
    "kb_create_chat_session": OperationType.WRITE,
    "kb_rewrite_query": OperationType.READ,
    "kb_assemble_context": OperationType.READ,
    "kb_generate_answer": OperationType.READ,
    "kb_answer_question": OperationType.QUERY,
    "kb_store_memory": OperationType.WRITE_MEMORY,
    "kb_get_memory_context": OperationType.READ_MEMORY,
    "kb_clear_memory": OperationType.DELETE,
}
```

## 风险检测模式

### 注入风险检测 (Layer 1)

```python
INJECTION_PATTERNS = [
    r"<script[^>]*>.*?</script>",  # Script 注入
    r"javascript:",                   # JavaScript 协议
    r"on\w+\s*=",                     # 事件处理器注入
    r"\$\{.*\}",                      # 模板注入
    r"\{\{.*\}\}",                    # 双大括号注入
    r"<!--.*?-->",                    # HTML 注释注入
    r"<iframe[^>]*>",                 # Iframe 注入
    r"<object[^>]*>",                # Object 注入
    r"<embed[^>]*>",                 # Embed 注入
    r"expression\s*\(",               # CSS 表达式注入
    r"url\s*\(",                      # CSS URL 注入
]
```

### 敏感操作检测

```python
SENSITIVE_ACTION_PATTERNS = [
    r"删[除除].*数据",
    r"删除.*文档",
    r"清空.*记忆",
    r"drop\s+table",
    r"delete\s+from",
    r"truncate",
]
```

## 人工确认机制

### 工作流程

```
HIGH 风险操作触发
       ↓
生成 confirmation_id
       ↓
存储到 _pending_confirmations
       ↓
返回 requires_confirmation=True
       ↓
外部系统展示确认界面
       ↓
用户确认/拒绝
       ↓
调用 confirm_action(confirmation_id, approved)
       ↓
记录审计日志 / 移除待确认记录
```

### 代码示例

```python
# 发起高风险操作治理
result = action_guard.guard_tool_call(
    tool_name="kb_clear_memory",
    tool_args={"session_id": "xxx"},
    agent="document_agent",
    session_id="session-123",
)

if result.requires_confirmation:
    # 展示确认界面给用户
    confirmation_id = result.confirmation_id
    # 等待用户确认...

# 用户确认后
action_guard.confirm_action(confirmation_id, approved=True)
```

## 审计日志

### AuditRecord 结构

```python
class AuditRecord(BaseModel):
    timestamp: str              # 时间戳
    session_id: str             # 会话ID
    tenant_id: str              # 租户ID

    operation: OperationType     # 操作类型
    operation_detail: str       # 操作详情
    decision: GovernanceDecision # 治理决策
    risk_level: RiskLevel       # 风险等级

    agent: str                  # 执行的Agent
    tool_name: Optional[str]    # 工具名称
    tool_args: Optional[dict]   # 工具参数

    success: bool               # 是否成功
    error_message: Optional[str] # 错误信息
    risk_flags: list[str]       # 风险标记
```

### 获取审计记录

```python
# 获取所有审计记录
records = action_guard.get_audit_records()

# 获取特定会话的审计记录
records = action_guard.get_audit_records(session_id="session-123")

# 清除审计记录
action_guard.clear_audit_records()  # 清除所有
action_guard.clear_audit_records(session_id="session-123")  # 清除特定会话
```

## 全局实例

系统提供全局单例实例：

```python
from app.governance.action_guard import action_guard

result = action_guard.guard_tool_call(...)
```

## 错误处理

当 Action Guard 内部发生异常时：

- 决策设为 `DENY`
- 风险等级设为 `MEDIUM`
- 添加 `action_guard_error` 风险标记
- 记录错误信息到 metadata

```python
except Exception as e:
    return GovernanceResult(
        decision=GovernanceDecision.DENY,
        risk_level=RiskLevel.MEDIUM,
        message=f"Action guard error: {str(e)}",
        risk_flags=["action_guard_error"],
        ...
    )
```

## 与 Context Guard 的协作

```
Context Guard (第一阶段)
      ↓ 传递 risk_flags 和 source_tags
Action Guard (第二阶段)
      ↓
1. 继承 Context Guard 发现的风险标记
2. 额外检测工具参数的注入风险
3. 根据工具类型判定风险等级
4. 综合决策
```

Context Guard 发现的风险标记会传递给 Action Guard，Action Guard 在此基础上进行额外检测和综合决策。
