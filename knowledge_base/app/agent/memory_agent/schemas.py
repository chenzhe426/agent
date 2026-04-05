"""
MemoryAgent 的数据结构和输入输出Schema
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class MemoryLevel(str, Enum):
    """记忆层级"""
    SHORT = "short"   # 短期记忆
    MID = "mid"       # 中期记忆
    LONG = "long"     # 长期记忆


class StoreMessageInput(BaseModel):
    """存储新消息的输入"""
    session_id: str = Field(..., description="会话ID")
    role: str = Field(..., description="角色: user/assistant")
    message: str = Field(..., description="消息内容")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="额外元数据")


class StoreMessageOutput(BaseModel):
    """存储消息的输出"""
    ok: bool = Field(..., description="是否成功")
    memory_updates: dict[str, Any] = Field(default_factory=dict, description="各层级记忆的更新情况")
    message_id: Optional[int] = Field(None, description="消息ID")
    turn_count: int = Field(..., description="当前会话总轮次")
    error: Optional[str] = Field(None, description="错误信息")


class GetMemoryContextInput(BaseModel):
    """获取记忆上下文的输入"""
    session_id: str = Field(..., description="会话ID")
    question: Optional[str] = Field(None, description="关联的问题（用于长期记忆检索）")
    include_levels: list[MemoryLevel] = Field(
        default_factory=lambda: [MemoryLevel.SHORT, MemoryLevel.MID, MemoryLevel.LONG],
        description="需要获取的记忆层级"
    )


class GetMemoryContextOutput(BaseModel):
    """获取记忆上下文的输出"""
    ok: bool
    short_term: Optional[dict[str, Any]] = Field(None, description="短期记忆")
    mid_term: Optional[dict[str, Any]] = Field(None, description="中期记忆")
    long_term: Optional[dict[str, Any]] = Field(None, description="长期记忆")
    error: Optional[str] = None


class ClearMemoryInput(BaseModel):
    """清除记忆的输入"""
    session_id: str = Field(..., description="会话ID")
    level: Optional[MemoryLevel] = Field(None, description="要清除的层级，None表示全部")


class ClearMemoryOutput(BaseModel):
    """清除记忆的输出"""
    ok: bool
    cleared_levels: list[MemoryLevel] = Field(default_factory=list)


class ShortTermSummary(BaseModel):
    """短期记忆结构化摘要"""
    entities: list[str] = Field(default_factory=list, description="关键实体")
    topics: list[str] = Field(default_factory=list, description="话题")
    qa_pairs: list[dict[str, str]] = Field(default_factory=list, description="问答对")
    coreferences: list[str] = Field(default_factory=list, description="指代关系")


class MidTermSummary(BaseModel):
    """中期记忆摘要"""
    summary: str = Field(..., description="关键信息摘要")
    key_entities: list[str] = Field(default_factory=list)
    key_topics: list[str] = Field(default_factory=list)
    turn_range: tuple[int, int] = Field(..., description="涉及的轮次范围")


class LongTermSummary(BaseModel):
    """长期记忆摘要"""
    summary: str = Field(..., description="关键信息摘要")
    key_entities: list[str] = Field(default_factory=list)
    key_topics: list[str] = Field(default_factory=list)
    session_id: str = Field(..., description="会话ID")
    created_at: datetime = Field(default_factory=datetime.now)
