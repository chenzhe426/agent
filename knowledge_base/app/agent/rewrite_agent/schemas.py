"""
rewrite_agent/schemas.py - QueryRewriteAgent 的数据结构和输入输出Schema
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class RewriteInput(BaseModel):
    """查询改写输入"""
    question: str = Field(..., description="用户原始问题")
    session_id: str = Field(..., description="会话ID")
    use_history: bool = Field(default=True, description="是否使用对话历史")


class RewriteOutput(BaseModel):
    """查询改写输出（结构化）"""
    ok: bool = Field(..., description="是否成功")
    original_question: str = Field(..., description="原始问题")
    rewritten_query: str = Field(..., description="改写后的问题")
    used_memory_levels: list[str] = Field(default_factory=list, description="使用了哪些记忆层级")
    confidence: float = Field(default=0.0, description="改写置信度")
    error: Optional[str] = Field(None, description="错误信息")

    # ========== 结构化扩展字段（用于治理） ==========
    # 意图识别
    intent: str = Field(default="unknown", description="用户意图：qa/document/search/chat")
    # 标准化查询（去噪声、标准化格式）
    normalized_query: str = Field(default="", description="标准化后的查询")
    # 识别的实体列表
    entities: list[str] = Field(default_factory=list, description="问题中识别的实体")
    # 文档范围限定
    doc_scope: str = Field(default="", description="文档范围限定，如特定文档或领域")
    # 引用的记忆层级
    memory_refs: list[str] = Field(default_factory=list, description="引用的记忆来源")
    # 风险标记（由治理网关填充）
    risk_flags: list[str] = Field(default_factory=list, description="风险标记列表")
    # 来源标记
    source_tags: list[str] = Field(default_factory=list, description="来源标记")
    # 来源标识
    provenance: str = Field(default="rewrite", description="来源标识")


class MemoryContext(BaseModel):
    """记忆上下文（用于构建prompt）"""
    short_term: Optional[dict[str, Any]] = Field(None, description="短期记忆")
    mid_term: Optional[dict[str, Any]] = Field(None, description="中期记忆")
    long_term: Optional[list[dict[str, Any]]] = Field(None, description="长期记忆列表")
