"""
rewrite_agent/agent.py - QueryRewriteAgent 核心逻辑

专门负责读取记忆并进行查询改写：
- 从三层记忆读取上下文
- 结合上下文改写用户问题
- 输出改写后的完整查询
"""

from __future__ import annotations

import json
from typing import Any, Optional

from app.agent.memory_agent.storage import short_term_storage, mid_term_storage, long_term_storage
from app.agent.memory_agent.schemas import MemoryLevel
from app.agent.memory_agent.agent import memory_agent
from app.agent.rewrite_agent.schemas import RewriteInput, RewriteOutput, MemoryContext
from app.agent.rewrite_agent.prompts import (
    REWRITE_SYSTEM_PROMPT,
    REWRITE_NO_CONTEXT_PROMPT,
    SHORT_TERM_PROMPT,
    MID_TERM_PROMPT,
    LONG_TERM_PROMPT,
)
from app.services.llm_service import chat_completion


class QueryRewriteAgent:
    """
    QueryRewriteAgent 专门负责读取记忆并进行查询改写。

    它不负责存储记忆，存储由MemoryAgent负责。

    工作流程：
    1. 接收 question + session_id
    2. 从三层记忆读取上下文
    3. 构建带上下文的prompt
    4. 调用LLM进行改写
    5. 输出结构化改写结果（包含 intent、entities、doc_scope 等）
    """

    # 意图关键词映射
    INTENT_KEYWORDS = {
        "qa": ["什么", "怎么", "如何", "为什么", "是不是", "是不是", "请问", "回答", "解释", "定义", "概念"],
        "document": ["文档", "文件", "报告", "文章", "上传", "导入", "索引", "查找", "搜索", "检索"],
        "search": ["查找", "搜索", "检索", "找", "获取", "拿到"],
        "chat": ["你好", "嗨", "在吗", "聊", "说"],
    }

    def rewrite(self, input_data: RewriteInput) -> RewriteOutput:
        """
        执行查询改写。

        Args:
            input_data: 包含 question, session_id, use_history

        Returns:
            RewriteOutput: 结构化改写结果
        """
        try:
            if not input_data.use_history:
                return self._rewrite_without_context(input_data.question)

            # 1. 获取三层记忆上下文
            context = self._get_memory_context(input_data.session_id, input_data.question)

            # 2. 构建上下文prompt
            context_prompt = self._build_context_prompt(context, input_data.question)

            # 3. 调用LLM改写
            result = self._do_rewrite(input_data.question, context_prompt)

            # 4. 确定使用了哪些记忆层级
            used_levels = []
            if context.short_term:
                used_levels.append("short_term")
            if context.mid_term:
                used_levels.append("mid_term")
            if context.long_term:
                used_levels.append("long_term")

            # 5. 提取结构化信息
            intent = self._extract_intent(input_data.question)
            entities = self._extract_entities(input_data.question)
            normalized_query = self._normalize_query(input_data.question)
            doc_scope = self._extract_doc_scope(input_data.question)
            memory_refs = self._build_memory_refs(context)
            source_tags = self._build_source_tags(used_levels)

            return RewriteOutput(
                ok=True,
                original_question=input_data.question,
                rewritten_query=result.get("rewritten_query", input_data.question),
                used_memory_levels=used_levels,
                confidence=result.get("confidence", 0.5),
                # 结构化扩展字段
                intent=intent,
                normalized_query=normalized_query,
                entities=entities,
                doc_scope=doc_scope,
                memory_refs=memory_refs,
                source_tags=source_tags,
                provenance="rewrite",
            )

        except Exception as e:
            return RewriteOutput(
                ok=False,
                original_question=input_data.question,
                rewritten_query=input_data.question,
                used_memory_levels=[],
                confidence=0.0,
                error=str(e),
            )

    def _get_memory_context(self, session_id: str, question: str) -> MemoryContext:
        """获取三层记忆上下文"""
        context = {}

        # 短期记忆：直接读取
        short = short_term_storage.get(session_id)
        if short:
            context["short_term"] = {
                "turns": short.get("turns", []),
                "summary": short.get("summary", {}),
            }

        # 中期记忆：直接读取
        mid = mid_term_storage.get(session_id)
        if mid:
            context["mid_term"] = mid

        # 长期记忆：基于问题关键词搜索
        keywords = self._extract_keywords(question)
        long_results = []
        for kw in keywords[:3]:
            found = long_term_storage.search(kw, limit=2)
            long_results.extend(found)
        # 去重，排除当前session
        seen = set()
        unique = []
        for r in long_results:
            sid = r.get("session_id")
            if sid not in seen and sid != session_id:
                seen.add(sid)
                unique.append(r)
        if unique:
            context["long_term"] = unique[:3]

        return MemoryContext(**context)

    def _build_context_prompt(self, context: MemoryContext, question: str) -> str:
        """构建带上下文的prompt"""
        parts = []

        # 短期记忆
        if context.short_term:
            turns = context.short_term.get("turns", [])
            if turns:
                turn_lines = []
                for t in turns:
                    role = "用户" if t.get("role") == "user" else "助手"
                    turn_lines.append(f"{role}：{t.get('message', '')}")
                short_content = "\n".join(turn_lines)
                parts.append(SHORT_TERM_PROMPT.format(short_term_content=short_content))

        # 中期记忆
        if context.mid_term:
            mid_content = context.mid_term.get("summary", "")
            if mid_content:
                parts.append(MID_TERM_PROMPT.format(mid_term_content=mid_content))

        # 长期记忆
        if context.long_term:
            long_parts = []
            for lt in context.long_term:
                summary = lt.get("summary", "")
                if summary:
                    long_parts.append(f"- {summary}")
            if long_parts:
                long_content = "\n".join(long_parts)
                parts.append(LONG_TERM_PROMPT.format(long_term_content=long_content))

        return "\n".join(parts)

    def _do_rewrite(self, question: str, context_prompt: str) -> dict[str, Any]:
        """执行改写"""
        if not context_prompt:
            # 无上下文时的简化改写
            prompt = REWRITE_NO_CONTEXT_PROMPT.format(question=question)
        else:
            # 带上下文的改写
            user_prompt = f"{context_prompt}\n\n【当前问题】\n{question}\n\n请根据以上上下文改写问题："

        raw = chat_completion(
            system_prompt=REWRITE_SYSTEM_PROMPT,
            user_prompt=user_prompt if context_prompt else prompt,
        )

        # 解析JSON响应
        return self._parse_json_response(raw)

    def _rewrite_without_context(self, question: str) -> RewriteOutput:
        """无历史上下文时的改写"""
        try:
            prompt = REWRITE_NO_CONTEXT_PROMPT.format(question=question)
            raw = chat_completion(
                system_prompt="你是一个查询改写助手。",
                user_prompt=prompt,
            )
            result = self._parse_json_response(raw)

            # 提取结构化信息
            intent = self._extract_intent(question)
            entities = self._extract_entities(question)
            normalized_query = self._normalize_query(question)
            doc_scope = self._extract_doc_scope(question)

            return RewriteOutput(
                ok=True,
                original_question=question,
                rewritten_query=result.get("rewritten_query", question),
                used_memory_levels=[],
                confidence=result.get("confidence", 0.5),
                # 结构化扩展字段
                intent=intent,
                normalized_query=normalized_query,
                entities=entities,
                doc_scope=doc_scope,
                memory_refs=[],
                source_tags=["user_input"],
                provenance="rewrite",
            )
        except Exception as e:
            return RewriteOutput(
                ok=False,
                original_question=question,
                rewritten_query=question,
                used_memory_levels=[],
                confidence=0.0,
                error=str(e),
            )

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        """解析JSON响应"""
        raw = (raw or "").strip()

        # 直接解析
        if raw.startswith("{"):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

        # 尝试从markdown代码块中提取
        import re
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试找第一个 { ... }
        brace = re.search(r"(\{.*\})", raw, re.S)
        if brace:
            try:
                return json.loads(brace.group(1))
            except json.JSONDecodeError:
                pass

        return {"rewritten_query": raw, "confidence": 0.0}

    def _extract_keywords(self, text: str) -> list[str]:
        """简单提取关键词"""
        import re
        words = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', text)
        stopwords = {"什么", "怎么", "如何", "为什么", "这个", "那个", "一个", "的", "是", "了"}
        return [w for w in words if w not in stopwords]

    def _extract_intent(self, question: str) -> str:
        """
        从问题中提取用户意图

        Returns:
            intent: qa/document/search/chat/unknown
        """
        question_lower = question.lower()

        # 文档操作意图
        doc_keywords = ["文档", "文件", "报告", "文章", "上传", "导入", "索引", "上传文件", "导入文档"]
        for kw in doc_keywords:
            if kw in question_lower:
                return "document"

        # 搜索意图
        search_keywords = ["查找", "搜索", "检索", "找", "获取", "拿到", "有没有"]
        for kw in search_keywords:
            if kw in question_lower:
                return "search"

        # 问答意图
        qa_keywords = ["什么", "怎么", "如何", "为什么", "是不是", "请问", "回答", "解释", "定义", "概念", "原因", "办法"]
        for kw in qa_keywords:
            if kw in question_lower:
                return "qa"

        # 闲聊意图
        chat_keywords = ["你好", "嗨", "在吗", "聊", "说", "帮", "给我"]
        for kw in chat_keywords:
            if kw in question_lower:
                return "chat"

        return "unknown"

    def _extract_entities(self, question: str) -> list[str]:
        """
        从问题中提取实体

        Returns:
            entities: 实体列表
        """
        import re

        # 提取中文实体（2个字以上）
        chinese_entities = re.findall(r'[\u4e00-\u9fff]{2,}', question)

        # 提取英文实体
        english_entities = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', question)

        # 过滤停用词
        stopwords = {
            "什么", "怎么", "如何", "为什么", "这个", "那个", "一个", "的", "是", "了",
            "请问", "帮我", "给我", "我想", "我要", "你能", "可以", "有没有", "是不是",
        }

        all_entities = chinese_entities + english_entities
        filtered = [e for e in all_entities if e not in stopwords and len(e) > 1]

        # 去重
        seen = set()
        unique = []
        for e in filtered:
            if e not in seen:
                seen.add(e)
                unique.append(e)

        return unique[:10]  # 最多返回10个实体

    def _normalize_query(self, question: str) -> str:
        """
        标准化查询

        - 去噪声词
        - 标准化格式
        """
        import re

        # 去除多余空格
        normalized = re.sub(r'\s+', ' ', question).strip()

        # 去除常见前缀
        prefixes = ["请问", "我想问", "我想知道", "帮我查", "帮我找", "请帮我", "你能告诉我"]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()

        return normalized

    def _extract_doc_scope(self, question: str) -> str:
        """
        从问题中提取文档范围限定

        Returns:
            doc_scope: 文档范围描述
        """
        question_lower = question.lower()

        # 检查特定文档标识
        doc_indicators = [
            ("这个文档", "current_doc"),
            ("该文档", "current_doc"),
            ("上面", "current_doc"),
            ("之前", "previous_doc"),
            ("那篇", "previous_doc"),
            ("所有", "all_docs"),
            ("全部", "all_docs"),
        ]

        for indicator, scope in doc_indicators:
            if indicator in question_lower:
                return scope

        return "all_docs"  # 默认搜索所有文档

    def _build_memory_refs(self, context: MemoryContext) -> list[str]:
        """
        构建记忆引用列表

        Returns:
            memory_refs: 引用的记忆来源
        """
        refs = []

        if context.short_term:
            refs.append("memory:short_term")
        if context.mid_term:
            refs.append("memory:mid_term")
        if context.long_term:
            refs.append("memory:long_term")

        return refs

    def _build_source_tags(self, used_levels: list[str]) -> list[str]:
        """
        构建来源标签

        Returns:
            source_tags: 来源标签列表
        """
        tags = []

        level_to_tag = {
            "short_term": "memory_short",
            "mid_term": "memory_mid",
            "long_term": "memory_long",
        }

        for level in used_levels:
            tag = level_to_tag.get(level)
            if tag:
                tags.append(tag)

        return tags


# 全局实例
query_rewrite_agent = QueryRewriteAgent()
