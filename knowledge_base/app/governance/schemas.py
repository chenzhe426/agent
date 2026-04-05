"""
governance/schemas.py - 治理相关数据模型

包含：
- 风险等级定义
- 治理决策结果
- 审计记录
- 来源标记
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """风险等级枚举"""
    LOW = "low"           # 低风险，正常执行
    MEDIUM = "medium"     # 中风险，需要记录
    HIGH = "high"         # 高风险，需要确认
    CRITICAL = "critical" # 极高风险，拦截


class GovernanceDecision(str, Enum):
    """治理决策枚举"""
    ALLOW = "allow"       # 允许通过
    DENY = "deny"         # 直接拒绝
    CONFIRM = "confirm"   # 需要人工确认
    GRADE = "grade"       # 降级处理（如只返回摘要）
    REJECT = "reject"     # 拒绝并返回错误


class SourceTag(str, Enum):
    """来源标记枚举"""
    USER_INPUT = "user_input"           # 用户直接输入
    MEMORY_SHORT = "memory_short"       # 短期记忆
    MEMORY_MID = "memory_mid"           # 中期记忆
    MEMORY_LONG = "memory_long"         # 长期记忆
    VECTOR_SEARCH = "vector_search"     # 向量检索
    EXTERNAL_API = "external_api"       # 外部API
    CACHE = "cache"                    # 缓存


class OperationType(str, Enum):
    """操作类型枚举"""
    READ = "read"               # 读取资源
    WRITE = "write"             # 写入资源
    DELETE = "delete"           # 删除资源
    EXECUTE_TOOL = "execute_tool"  # 执行工具
    READ_MEMORY = "read_memory"   # 读取记忆
    WRITE_MEMORY = "write_memory" # 写入记忆
    QUERY = "query"             # 查询操作


class RewriteStructuredOutput(BaseModel):
    """
    Rewrite 节点的结构化输出

    用于在 Context Guard 阶段进行校验和传递
    """
    # 基础字段
    ok: bool = Field(..., description="改写是否成功")
    original_question: str = Field(..., description="原始问题")
    rewritten_query: str = Field(..., description="改写后的问题")

    # 结构化扩展字段
    intent: str = Field(default="unknown", description="用户意图")
    normalized_query: str = Field(default="", description="标准化查询")
    entities: list[str] = Field(default_factory=list, description="识别的实体列表")
    doc_scope: str = Field(default="", description="文档范围限定")
    memory_refs: list[str] = Field(default_factory=list, description="引用的记忆层级")

    # 治理相关字段
    risk_flags: list[str] = Field(default_factory=list, description="风险标记列表")
    source_tags: list[str] = Field(default_factory=list, description="来源标记")
    provenance: str = Field(default="rewrite", description="来源标识")

    # 元数据
    session_id: str = Field(default="", description="会话ID")
    tenant_id: str = Field(default="default", description="租户ID")
    confidence: float = Field(default=0.0, description="置信度")
    error: Optional[str] = Field(None, description="错误信息")

    # 标记字段
    used_memory_levels: list[str] = Field(default_factory=list, description="使用的记忆层级")


class GovernanceResult(BaseModel):
    """
    治理结果

    返回给调用方的治理决策
    """
    decision: GovernanceDecision = Field(..., description="治理决策")
    risk_level: RiskLevel = Field(..., description="风险等级")
    message: str = Field(default="", description="决策消息")
    risk_flags: list[str] = Field(default_factory=list, description="识别的风险标记")
    source_tags: list[str] = Field(default_factory=list, description="来源标记")
    requires_confirmation: bool = Field(default=False, description="是否需要人工确认")
    confirmation_id: Optional[str] = Field(None, description="确认ID，用于获取确认状态")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class AuditRecord(BaseModel):
    """
    审计记录

    用于日志记录和追溯
    """
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="时间戳")
    session_id: str = Field(..., description="会话ID")
    tenant_id: str = Field(default="default", description="租户ID")

    # 操作信息
    operation: OperationType = Field(..., description="操作类型")
    operation_detail: str = Field(..., description="操作详情")

    # 治理决策
    decision: GovernanceDecision = Field(..., description="治理决策")
    risk_level: RiskLevel = Field(..., description="风险等级")

    # 上下文信息
    user_input: str = Field(default="", description="用户输入")
    rewritten_query: str = Field(default="", description="改写后查询")
    agent: str = Field(default="", description="执行的Agent")
    tool_name: Optional[str] = Field(None, description="工具名称")
    tool_args: Optional[dict[str, Any]] = Field(None, description="工具参数")

    # 结果
    success: bool = Field(default=True, description="是否成功")
    error_message: Optional[str] = Field(None, description="错误信息")
    risk_flags: list[str] = Field(default_factory=list, description="风险标记")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class ActionContext(BaseModel):
    """
    动作治理上下文

    在 Action Guard 阶段使用
    """
    session_id: str = Field(..., description="会话ID")
    tenant_id: str = Field(default="default", description="租户ID")
    operation: OperationType = Field(..., description="操作类型")
    operation_detail: str = Field(..., description="操作详情")

    # Agent 信息
    agent: str = Field(default="", description="来源Agent")
    tool_name: Optional[str] = Field(None, description="工具名称")
    tool_args: dict[str, Any] = Field(default_factory=dict, description="工具参数")

    # Rewrite 输出（用于上下文传递）
    rewrite_output: Optional[RewriteStructuredOutput] = Field(None, description="Rewrite结构化输出")

    # 资源信息
    resource_type: Optional[str] = Field(None, description="资源类型")
    resource_id: Optional[str] = Field(None, description="资源ID")

    # 风险标记传递
    risk_flags: list[str] = Field(default_factory=list, description="继承的风险标记")
    source_tags: list[str] = Field(default_factory=list, description="继承的来源标记")
