"""
Supervisor Agent for intent classification and task routing.
"""

import json
import re
from typing import Any

from app.agent.multi_agent.messages import TaskType, AgentRole, AgentResponse
from app.agent.multi_agent.coordinator import coordinator


# Supervisor prompt for intent classification
SUPERVISOR_SYSTEM_PROMPT = """你是一个知识库系统的Supervisor Agent。

你的职责是分析用户问题，决定最合适的Agent进行响应。

任务类型：
- qa: 需要完整问答、验证、答案生成的问题
- search: 只需要搜索检索知识库的问题
- import: 文档导入、索引、摘要等文档管理任务
- summary: 文档摘要
- chat: 一般性对话
- multi: 需要多个Agent协作的复杂问题

分析规则：
1. 如果问题涉及"导入"、"索引"、"文档"，且需要执行操作 → import
2. 如果问题只要求"搜索"、"查找"、"检索"且不需要生成答案 → search
3. 如果问题要求"比较"、"分析"、"总结"、"回答" → qa
4. 如果问题要求"摘要"文档 → summary
5. 复杂问题（包含多个子问题） → multi

输出必须是合法的JSON对象：
{"task_type": "任务类型", "reason": "分析理由", "confidence": 0.95}

如果 task_type 是 "multi"，需要额外说明需要哪些Agent协作：
{"task_type": "multi", "reason": "...", "confidence": 0.8, "sub_agents": ["qa", "search"]}
"""


class SupervisorAgent:
    """
    Supervisor Agent responsible for:
    - Intent classification
    - Task routing
    - Delegation to specialist agents
    """

    def __init__(self, llm=None):
        self._llm = llm
        self._coordinator = coordinator

    def _get_llm(self):
        """Lazy load LLM."""
        if self._llm is None:
            from app.agent.llm import get_chat_llm
            self._llm = get_chat_llm()
        return self._llm

    def classify_intent(self, question: str, session_id: str = "") -> dict[str, Any]:
        """
        Classify user intent and determine task type.

        Args:
            question: User question
            session_id: Session identifier

        Returns:
            Dict with task_type, reason, confidence
        """
        # Try LLM-based classification first
        try:
            llm = self._get_llm()
            from langchain_core.messages import HumanMessage, SystemMessage

            response = llm.invoke([
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(content=f"问题：{question}"),
            ])

            raw_content = response.content if hasattr(response, "content") else str(response)
            parsed = self._parse_json_response(raw_content)

            if parsed and "task_type" in parsed:
                return {
                    "task_type": TaskType(parsed["task_type"]),
                    "reason": parsed.get("reason", ""),
                    "confidence": parsed.get("confidence", 0.5),
                    "sub_agents": parsed.get("sub_agents", []),
                }
        except Exception as e:
            print(f"[Supervisor] LLM classification failed: {e}")

        # Fallback to rule-based classification
        return self._rule_based_classification(question)

    def _rule_based_classification(self, question: str) -> dict[str, Any]:
        """
        Rule-based intent classification as fallback.
        """
        question_lower = question.lower()

        # Import keywords
        import_keywords = ["导入", "索引", "index", "import", "上传", "添加文档"]
        if any(kw in question_lower for kw in import_keywords):
            return {
                "task_type": TaskType.IMPORT,
                "reason": "检测到文档管理关键词",
                "confidence": 0.9,
                "sub_agents": [],
            }

        # Search keywords
        search_keywords = ["搜索", "查找", "检索", "search", "find", "在哪"]
        only_search = any(kw in question_lower for kw in search_keywords)
        has_answer_indicators = any(kw in question_lower for kw in ["多少", "是什么", "如何", "为什么", "哪个"])
        if only_search and not has_answer_indicators:
            return {
                "task_type": TaskType.SEARCH,
                "reason": "检测到仅检索关键词",
                "confidence": 0.8,
                "sub_agents": [],
            }

        # Multi-task indicators
        multi_keywords = ["比较", "对比", "both", "and", "also", "以及"]
        if any(kw in question_lower for kw in multi_keywords):
            return {
                "task_type": TaskType.MULTI,
                "reason": "检测到复杂多步问题",
                "confidence": 0.7,
                "sub_agents": ["qa", "search"],
            }

        # QA as default
        return {
            "task_type": TaskType.QA,
            "reason": "默认分类为问答任务",
            "confidence": 0.6,
            "sub_agents": [],
        }

    def _parse_json_response(self, raw: str) -> dict[str, Any] | None:
        """Parse JSON from LLM response."""
        raw = (raw or "").strip()

        if raw.startswith("{"):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

        # Try markdown code blocks
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find first { ... }
        brace = re.search(r"(\{.*\})", raw, re.S)
        if brace:
            try:
                return json.loads(brace.group(1))
            except json.JSONDecodeError:
                pass

        return None

    def route_task(
        self,
        question: str,
        session_id: str = "",
    ) -> dict[str, Any]:
        """
        Route task to appropriate specialist agent(s).

        Args:
            question: User question
            session_id: Session identifier

        Returns:
            Dict with routing decision
        """
        # Step 1: Classify intent
        classification = self.classify_intent(question, session_id)
        task_type = classification["task_type"]

        # Step 2: Determine target agents
        if task_type == TaskType.MULTI:
            target_agents = [AgentRole(ag) for ag in classification.get("sub_agents", ["qa"])]
        else:
            agent_mapping = {
                TaskType.QA: AgentRole.QA,
                TaskType.SEARCH: AgentRole.SEARCH,
                TaskType.IMPORT: AgentRole.IMPORT,
                TaskType.SUMMARY: AgentRole.IMPORT,  # Summary also handled by import agent
                TaskType.CHAT: AgentRole.QA,
            }
            target_agents = [agent_mapping.get(task_type, AgentRole.QA)]

        return {
            "task_type": task_type,
            "target_agents": target_agents,
            "classification": classification,
            "question": question,
            "session_id": session_id,
        }

    def aggregate_results(
        self,
        agent_responses: dict[AgentRole, AgentResponse],
        question: str,
    ) -> dict[str, Any]:
        """
        Aggregate results from multiple specialist agents.

        Args:
            agent_responses: Dict of agent role -> response
            question: Original user question

        Returns:
            Aggregated final response
        """
        return self._coordinator.aggregate_responses(agent_responses, question)

    def build_final_answer(
        self,
        aggregated: dict[str, Any],
        question: str,
    ) -> dict[str, Any]:
        """
        Build final answer from aggregated results.

        Args:
            aggregated: Aggregated results
            question: Original question

        Returns:
            Final response dict
        """
        return self._coordinator.build_final_response(aggregated, question)


# Global supervisor instance
supervisor_agent = SupervisorAgent()
