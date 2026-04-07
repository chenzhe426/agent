# 安全防护系统文档

## 概述

本系统采用**多层防护架构**，从输入到输出全程治理：

```
用户输入 → Context Guard → LangGraph → Action Guard → MCP Server → 工具 → 数据库
              ↓              ↓           ↓            ↓           ↓
          输入治理      节点状态    动作治理     工具审计     限流
                      Checkpoint    频率限制    审计日志
```

---

## 1. 输入治理 (Context Guard)

**文件**: `app/governance/context_guard.py`

**职责**:
- 检测恶意提示词注入
- 识别风险标记（injection, sensitive）
- 标记来源标签

**流程**:
```
用户输入 → Rewrite → Context Guard → 决策
                        ↓
           ALLOW: 正常继续
           DENY: 拒绝请求
```

**风险标记类型**:
- `injection`: 提示词注入尝试
- `sensitive`: 敏感内容识别
- `malicious_url`: 恶意 URL
- `prompt_leakage`: Prompt 泄露尝试

---

## 2. 节点状态记录 (LangGraph Checkpoint)

**文件**: `app/agent/multi_agent/graph.py`

**职责**:
- 记录每个节点的输入/输出状态
- 支持对话续接
- 用于问题追溯

**State 结构**:
```python
class MultiAgentGraphState(TypedDict):
    session_id: str
    question: str
    rewritten_question: str
    governance_context: dict[str, Any]
    context_guard_passed: bool
    agent_responses: dict[str, Any]
    current_node: str
    # ... 其他字段
```

**Checkpoint 存储**:
- 使用 `MemorySaver` 内存存储
- 可配置为持久化存储（Redis, PostgreSQL 等）

---

## 3. 动作治理 (Action Guard)

**文件**: `app/governance/action_guard.py`

**职责**:
- 工具调用前的安全检查
- 风险等级评估
- 权限判断
- **审计日志记录**

**流程**:
```
Agent 调用工具 → Action Guard → 决策
                          ↓
           ALLOW: 执行
           CONFIRM: 需要确认
           DENY: 拒绝
```

**风险等级**:
| 等级 | 处理 |
|------|------|
| LOW | 直接放行 |
| MEDIUM | 记录日志后放行 |
| HIGH | 需要人工确认 |
| CRITICAL | 直接拒绝 |

**高风险工具** (`app/governance/policies.py`):
- `kb_import_file` - 文件导入
- `kb_import_folder` - 文件夹导入
- `kb_index_document` - 文档索引
- `kb_clear_memory` - 清除记忆
- `kb_clear_agent` - 清除 Agent

---

## 4. MCP 统一入口

**文件**: `app/tools/mcp_server.py`

**设计原则**:
- 所有工具调用必须经过 MCP Server
- Agent 内部调用也通过 MCP Client
- **无例外路径，所有调用都经过 Action Guard**

**工具列表**:
| 工具 | 用途 | 风险等级 |
|------|------|---------|
| `kb_search_knowledge_base` | 知识库搜索 | LOW |
| `kb_answer_question` | RAG 问答 | LOW |
| `kb_assemble_context` | 上下文组装 | LOW |
| `kb_generate_answer` | 答案生成 | LOW |
| `kb_rewrite_query` | 查询改写 | LOW |
| `kb_import_file` | 文件导入 | HIGH |
| `kb_import_folder` | 文件夹导入 | HIGH |
| `kb_index_document` | 文档索引 | HIGH |
| `kb_summarize_document` | 文档摘要 | MEDIUM |
| `kb_create_chat_session` | 创建会话 | LOW |
| `kb_get_chat_history` | 获取历史 | MEDIUM |
| `kb_store_memory` | 存储记忆 | LOW |
| `kb_get_memory_context` | 获取记忆 | MEDIUM |
| `kb_clear_memory` | 清除记忆 | CRITICAL |
| `kb_rewrite_query_v2` | V2 查询改写 | LOW |

---

## 5. 频率限制 (Rate Limiter)

**文件**: `app/tools/tool_dispatcher.py`

**职责**:
- 防止恶意提示词注入导致的自我 DDoS
- 防止单用户恶意请求

**两级限流**:

### Session 级别
| 配置 | 默认值 | 说明 |
|------|--------|------|
| `RATE_LIMIT_SESSION_MAX` | 60 | 每分钟最大调用数 |
| `RATE_LIMIT_SESSION_WINDOW` | 60 | 时间窗口（秒） |

### Agent 级别
| 配置 | 默认值 | 说明 |
|------|--------|------|
| `RATE_LIMIT_AGENT_MAX` | 120 | 每分钟最大调用数 |
| `RATE_LIMIT_AGENT_WINDOW` | 60 | 时间窗口（秒） |

**返回格式**:
```json
{
  "ok": false,
  "error": {
    "code": "RATE_LIMIT_SESSION",
    "message": "Session rate limit exceeded. Retry after 45s",
    "retry_after": 45
  }
}
```

---

## 6. 审计日志

**文件**: `app/governance/action_guard.py` → `AuditLogger`

**输出位置**: `logs/audit/audit_YYYY-MM-DD.jsonl`

**日志格式**:
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "session_id": "abc123",
  "tenant_id": "default",
  "operation": "EXECUTE_TOOL",
  "operation_detail": "EXECUTE_TOOL:kb_search_knowledge_base",
  "decision": "ALLOW",
  "risk_level": "LOW",
  "agent": "qa",
  "tool_name": "kb_search_knowledge_base",
  "tool_args": {"query": "AMD产品", "top_k": 5},
  "success": true,
  "risk_flags": []
}
```

**敏感信息脱敏**:
- `password` → `***REDACTED***`
- `token` → `***REDACTED***`
- `api_key` → `***REDACTED***`
- `secret` → `***REDACTED***`
- `credential` → `***REDACTED***`

---

## 7. 架构调用链

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户请求                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Context Guard                                │
│                 (提示词注入检测)                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌───────────────────┐
                    │  LangGraph State   │
                    │  (Checkpoint 记录)  │
                    └───────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    QAAgent / DocumentAgent                       │
│                   (通过 MCP Client 调用)                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Client                                   │
│                  (streamable-http)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Server                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ToolDispatcher                                  │
│                 (频率限制检查)                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Session 限流: 60次/分钟                                  │   │
│  │ Agent 限流: 120次/分钟                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Action Guard                                 │
│                 (工具调用治理)                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 风险评估 → 决策: ALLOW/CONFIRM/DENY                      │   │
│  │ 审计日志 → logs/audit/                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      工具执行                                     │
│             (kb_search, kb_import, etc.)                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       数据库                                      │
│         (MySQL / Qdrant / Redis - 参数化查询)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RATE_LIMIT_SESSION_MAX` | 60 | Session 每分钟最大调用 |
| `RATE_LIMIT_SESSION_WINDOW` | 60 | Session 时间窗口（秒） |
| `RATE_LIMIT_AGENT_MAX` | 120 | Agent 每分钟最大调用 |
| `RATE_LIMIT_AGENT_WINDOW` | 60 | Agent 时间窗口（秒） |
| `AUDIT_LOG_DIR` | logs/audit | 审计日志目录 |

---

## 9. 安全特性总结

| 层级 | 防护措施 | 状态 |
|------|---------|------|
| 输入 | Context Guard (提示词注入检测) | ✅ 已实现 |
| 输入 | Pydantic Schema 验证 | ✅ 已实现 |
| 节点 | LangGraph Checkpoint (状态记录) | ✅ 已实现 |
| 调用 | MCP 统一入口 (无例外路径) | ✅ 已实现 |
| 调用 | Action Guard (动作治理) | ✅ 已实现 |
| 调用 | 频率限制 (Session + Agent) | ✅ 已实现 |
| 工具 | 参数化 SQL 查询 | ✅ 已实现 |
| 输出 | 审计日志 (JSONL) | ✅ 已实现 |
| 输出 | 敏感信息脱敏 | ✅ 已实现 |

---

## 10. 待加强项

### API 层防护 (建议实现)

| 防护项 | 说明 |
|--------|------|
| CORS 限制 | 当前 `allow_origins=["*"]` 建议限制 |
| API 认证 | 添加 API Key 认证 |
| API 限流 | 限制 API 层面的请求频率 |
| 异常脱敏 | 避免内部信息泄露 |
| 文件路径验证 | 防止路径遍历攻击 |
| Session 归属校验 | 防止跨用户访问 |

### 数据库层防护 (可选)

| 防护项 | 说明 |
|--------|------|
| Qdrant 认证 | 启用 API Key |
| Redis 认证 | 启用密码 |
| 数据库防火墙 | 限制访问来源 |

---

## 11. 故障排查

### 限流触发
```
症状: {"ok": false, "error": {"code": "RATE_LIMIT_SESSION"}}
解决: 等待 retry_after 秒后重试，或调整环境变量
```

### Action Guard 拒绝
```
症状: {"ok": false, "error": {"code": "ACTION_GUARD_DENIED"}}
解决: 检查 risk_flags 和 risk_level，联系管理员
```

### 审计日志查询
```bash
# 查看今日审计日志
tail -f logs/audit/audit_$(date +%Y-%m-%d).jsonl

# 查询特定 session 的日志
grep "session_id" logs/audit/audit_2024-01-01.jsonl

# 查询被拒绝的操作
grep "DENY" logs/audit/audit_2024-01-01.jsonl
```
