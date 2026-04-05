"""
memory_agent/agent.py - MemoryAgent 核心逻辑

专门负责存储和管理分层记忆：
- 接收新消息
- 分析对话轮次
- 更新对应层级的记忆
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from app.agent.memory_agent.schemas import (
    StoreMessageInput,
    StoreMessageOutput,
    GetMemoryContextInput,
    GetMemoryContextOutput,
    ClearMemoryInput,
    ClearMemoryOutput,
    MemoryLevel,
)
from app.agent.memory_agent.storage import short_term_storage, mid_term_storage, long_term_storage
from app.db.repositories.chat_repository import insert_chat_message, get_chat_messages
from app.services.llm_service import chat_completion


# =============================================================================
# 配置
# =============================================================================

# 轮次阈值
MID_TERM_START_TURN = 3   # 超过3轮开始中期记忆
LONG_TERM_START_TURN = 10  # 超过10轮开始长期记忆

# 摘要生成相关
SHORT_TERM_SUMMARY_PROMPT = """你是一个对话摘要助手。请分析以下对话，生成结构化摘要。

对话：
{conversations}

请提取并返回JSON格式的摘要：
{{
    "entities": ["实体1", "实体2", ...],  // 关键实体（公司、人名、数字等）
    "topics": ["话题1", "话题2", ...],    // 讨论的话题
    "qa_pairs": [{{"q": "问题", "a": "答案"}}, ...],  // 重要问答对
    "coreferences": ["它", "他", ...]     // 指代关系
}}

只返回JSON，不要有其他解释。"""

MID_TERM_SUMMARY_PROMPT = """你是一个对话摘要助手。请分析以下对话，生成关键信息摘要。

对话：
{conversations}

请生成一段简洁的摘要，包含：
1. 对话的核心内容
2. 涉及的关键实体
3. 讨论的主要话题

返回JSON格式：
{{
    "summary": "摘要内容...",
    "key_entities": ["实体1", "实体2"],
    "key_topics": ["话题1", "话题2"]
}}

只返回JSON，不要有其他解释。"""


# =============================================================================
# 核心Agent类
# =============================================================================

class MemoryAgent:
    """
    MemoryAgent 负责存储和管理分层记忆。

    工作流程：
    1. 收到新消息
    2. 追加到chat_messages表
    3. 分析当前会话轮次
    4. 根据轮次更新对应层级的记忆
    """

    def store_message(self, input_data: StoreMessageInput) -> StoreMessageOutput:
        """
        存储新消息并更新记忆。

        Args:
            input_data: 包含session_id, role, message

        Returns:
            StoreMessageOutput: 更新结果
        """
        try:
            # 1. 插入消息到数据库
            message_id = insert_chat_message(
                session_id=input_data.session_id,
                role=input_data.role,
                message=input_data.message,
                metadata=input_data.metadata,
            )

            # 2. 获取当前会话的所有消息来计算轮次
            all_messages = get_chat_messages(input_data.session_id, limit=1000)
            turn_count = len([m for m in all_messages if m.get("role") in ("user", "assistant")])

            # 3. 更新各层级记忆
            memory_updates = {}

            # 短期记忆：追加轮次
            recent_turns = short_term_storage.append_turn(
                input_data.session_id,
                input_data.role,
                input_data.message,
            )

            # 当用户是问问题，助手回答后，更新摘要
            if input_data.role == "assistant" and recent_turns:
                # 找到对应的用户消息
                user_turns = [t for t in recent_turns if t.get("role") == "user"]
                assistant_turns = [t for t in recent_turns if t.get("role") == "assistant"]

                if user_turns and assistant_turns:
                    # 构建对话文本用于摘要
                    conv_text = self._build_conversation_text(recent_turns)
                    summary = self._generate_short_term_summary(conv_text)
                    short_term_storage.update_summary(input_data.session_id, summary)
                    memory_updates["short_term"] = {"updated": True, "turn_count": len(recent_turns)}

            # 中期记忆：turn > 3 时生成
            if turn_count > MID_TERM_START_TURN:
                mid_update = self._update_mid_term(input_data.session_id, all_messages)
                if mid_update:
                    memory_updates["mid_term"] = mid_update

            # 长期记忆：turn > 10 时生成
            if turn_count > LONG_TERM_START_TURN:
                long_update = self._update_long_term(input_data.session_id, all_messages)
                if long_update:
                    memory_updates["long_term"] = long_update

            return StoreMessageOutput(
                ok=True,
                memory_updates=memory_updates,
                message_id=message_id,
                turn_count=turn_count,
            )

        except Exception as e:
            return StoreMessageOutput(
                ok=False,
                memory_updates={},
                message_id=None,
                turn_count=0,
                error=str(e),
            )

    def get_memory_context(self, input_data: GetMemoryContextInput) -> GetMemoryContextOutput:
        """
        获取记忆上下文（供QueryRewriteAgent使用）。

        Args:
            input_data: 包含session_id, question, include_levels

        Returns:
            GetMemoryContextOutput: 各层级记忆内容
        """
        result = {}

        try:
            # 短期记忆
            if MemoryLevel.SHORT in input_data.include_levels:
                short = short_term_storage.get(input_data.session_id)
                if short:
                    result["short_term"] = {
                        "turns": short.get("turns", []),
                        "summary": short.get("summary", {}),
                    }

            # 中期记忆
            if MemoryLevel.MID in input_data.include_levels:
                mid = mid_term_storage.get(input_data.session_id)
                if mid:
                    result["mid_term"] = mid

            # 长期记忆
            if MemoryLevel.LONG in input_data.include_levels:
                if input_data.question:
                    # 基于问题关键词搜索
                    keywords = self._extract_keywords(input_data.question)
                    long_results = []
                    for kw in keywords[:3]:  # 取前3个关键词
                        found = long_term_storage.search(kw, limit=3)
                        long_results.extend(found)
                    # 去重
                    seen = set()
                    unique_results = []
                    for r in long_results:
                        sid = r.get("session_id")
                        if sid not in seen and sid != input_data.session_id:
                            seen.add(sid)
                            unique_results.append(r)
                    if unique_results:
                        result["long_term"] = unique_results[:3]
                else:
                    # 直接获取当前session的长期记忆
                    long = long_term_storage.get(input_data.session_id)
                    if long:
                        result["long_term"] = [long]

            return GetMemoryContextOutput(ok=True, **result)

        except Exception as e:
            return GetMemoryContextOutput(ok=False, error=str(e))

    def clear_memory(self, input_data: ClearMemoryInput) -> ClearMemoryOutput:
        """
        清除记忆。

        Args:
            input_data: 包含session_id, level

        Returns:
            ClearMemoryOutput: 清除结果
        """
        cleared = []

        try:
            if input_data.level is None or input_data.level == MemoryLevel.SHORT:
                short_term_storage.clear(input_data.session_id)
                cleared.append(MemoryLevel.SHORT)

            if input_data.level is None or input_data.level == MemoryLevel.MID:
                mid_term_storage.delete(input_data.session_id)
                cleared.append(MemoryLevel.MID)

            if input_data.level is None or input_data.level == MemoryLevel.LONG:
                long_term_storage.delete(input_data.session_id)
                cleared.append(MemoryLevel.LONG)

            return ClearMemoryOutput(ok=True, cleared_levels=cleared)

        except Exception as e:
            return ClearMemoryOutput(ok=False, cleared_levels=cleared, error=str(e))

    # =============================================================================
    # 内部方法
    # =============================================================================

    def _build_conversation_text(self, turns: list[dict[str, Any]]) -> str:
        """构建对话文本用于摘要生成"""
        lines = []
        for turn in turns:
            role = "用户" if turn.get("role") == "user" else "助手"
            msg = turn.get("message", "")
            lines.append(f"{role}：{msg}")
        return "\n".join(lines)

    def _generate_short_term_summary(self, conv_text: str) -> dict[str, Any]:
        """生成短期记忆结构化摘要"""
        try:
            prompt = SHORT_TERM_SUMMARY_PROMPT.format(conversations=conv_text)
            raw = chat_completion(
                system_prompt="你是一个对话摘要助手。",
                user_prompt=prompt,
            )
            # 解析JSON
            return json.loads(raw)
        except Exception as e:
            print(f"[_generate_short_term_summary] error: {e}")
            return {"entities": [], "topics": [], "qa_pairs": [], "coreferences": []}

    def _generate_mid_term_summary(self, conv_text: str) -> dict[str, Any]:
        """生成中期记忆摘要"""
        try:
            prompt = MID_TERM_SUMMARY_PROMPT.format(conversations=conv_text)
            raw = chat_completion(
                system_prompt="你是一个对话摘要助手。",
                user_prompt=prompt,
            )
            return json.loads(raw)
        except Exception as e:
            print(f"[_generate_mid_term_summary] error: {e}")
            return {"summary": "", "key_entities": [], "key_topics": []}

    def _update_mid_term(self, session_id: str, all_messages: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """更新中期记忆"""
        # 只保留第4-10轮（跳过前3轮）
        relevant = all_messages[3:10] if len(all_messages) > 3 else []
        if not relevant:
            return None

        conv_text = self._build_conversation_text_from_messages(relevant)
        summary_data = self._generate_mid_term_summary(conv_text)

        success = mid_term_storage.store(
            session_id=session_id,
            turn_range=(4, min(10, len(all_messages))),
            summary=summary_data.get("summary", ""),
            key_entities=summary_data.get("key_entities", []),
            key_topics=summary_data.get("key_topics", []),
        )

        return {"updated": success, "turn_range": (4, min(10, len(all_messages)))}

    def _update_long_term(self, session_id: str, all_messages: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """更新长期记忆"""
        # 取第11轮及以后
        relevant = all_messages[10:] if len(all_messages) > 10 else []
        if not relevant:
            return None

        conv_text = self._build_conversation_text_from_messages(relevant)
        summary_data = self._generate_mid_term_summary(conv_text)  # 复用中期摘要prompt

        # 生成question_hint用于搜索
        question_hint = self._extract_question_hint(all_messages)

        success = long_term_storage.store(
            session_id=session_id,
            summary=summary_data.get("summary", ""),
            key_entities=summary_data.get("key_entities", []),
            key_topics=summary_data.get("key_topics", []),
            question_hint=question_hint,
        )

        return {"updated": success}

    def _build_conversation_text_from_messages(self, messages: list[dict[str, Any]]) -> str:
        """从消息列表构建对话文本"""
        lines = []
        for msg in messages:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("message") or msg.get("content", "")
            lines.append(f"{role}：{content}")
        return "\n".join(lines)

    def _extract_keywords(self, text: str) -> list[str]:
        """简单提取关键词（实际可用更复杂的算法）"""
        import re
        # 提取连续的中文/英文词组
        words = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', text)
        # 过滤停用词
        stopwords = {"什么", "怎么", "如何", "为什么", "这个", "那个", "一个", "的", "是", "了"}
        return [w for w in words if w not in stopwords]

    def _extract_question_hint(self, messages: list[dict[str, Any]]) -> str:
        """从对话中提取问题提示"""
        # 取用户第一轮问题作为hint
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("message") or msg.get("content", "")
                return content[:200]  # 限制长度
        return ""


# 全局实例
memory_agent = MemoryAgent()
