"""
Supervisor Agent for intent classification and task routing.

职责：
- 意图分类
- 路由到 QAAgent 或 DocumentAgent
"""

import json
import re
from typing import Any

from app.agent.multi_agent.messages import TaskType, AgentRole, AgentResponse


SUPERVISOR_SYSTEM_PROMPT = """你是一个知识库系统的Supervisor Agent。

你的职责是分析用户问题，决定需要哪个Agent处理。

任务类型：
- qa: 需要完整问答 → 路由到QA Agent
- document: 文档管理/检索 → 路由到Document Agent

分析规则：
1. 如果用户要求"给我一篇XXX"、"查找XXX文档"、"检索XXX" → document
2. 如果问题涉及"导入"、"索引"、"上传"、"添加文档" → document
3. 其他问题默认 → qa

输出必须是合法的JSON对象：
{"task_type": "任务类型", "reason": "分析理由", "confidence": 0.95}
"""


class SupervisorAgent:
    """Supervisor Agent 负责意图分类和任务路由"""

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            from app.agent.llm import get_chat_llm
            self._llm = get_chat_llm()
        return self._llm

    def classify_intent(self, question: str, session_id: str = "") -> dict[str, Any]:
        """分类用户意图"""
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
                task_type = parsed["task_type"]
                if task_type not in ["qa", "document"]:
                    task_type = "qa"
                return {
                    "task_type": TaskType(task_type),
                    "reason": parsed.get("reason", ""),
                    "confidence": parsed.get("confidence", 0.5),
                }
        except Exception as e:
            print(f"[Supervisor] LLM classification failed: {e}")

        # Fallback: 规则分类
        return self._rule_based_classification(question)

    def _rule_based_classification(self, question: str) -> dict[str, Any]:
        """规则分类 fallback"""
        question_lower = question.lower()

        # 文档检索关键词
        doc_keywords = [
            "给我", "查找", "检索", "找到", "搜索",
            "文档", "研报", "报告", "文章", "文件",
            "导入", "索引", "index", "import", "上传",
            "添加文档", "上传文件"
        ]
        if any(kw in question_lower for kw in doc_keywords):
            return {
                "task_type": TaskType.DOCUMENT,
                "reason": "检测到文档管理/检索关键词",
                "confidence": 0.9,
            }

        return {
            "task_type": TaskType.QA,
            "reason": "默认分类为问答",
            "confidence": 0.6,
        }

    def route_task(self, question: str, session_id: str = "") -> dict[str, Any]:
        """路由任务"""
        classification = self.classify_intent(question, session_id)
        task_type = classification["task_type"]

        # 确定目标Agent
        if task_type == TaskType.DOCUMENT:
            target_agents = [AgentRole.DOCUMENT]
        else:
            target_agents = [AgentRole.QA]

        return {
            "task_type": task_type,
            "target_agents": target_agents,
            "classification": classification,
            "question": question,
            "session_id": session_id,
        }

    def aggregate_results(self, agent_responses: dict[AgentRole, AgentResponse], question: str) -> dict[str, Any]:
        """聚合多Agent结果"""
        aggregated = {
            "success": True,
            "primary_response": None,
            "all_responses": {},
            "reasoning_trace": [],
            "error": None,
        }

        for role, response in agent_responses.items():
            aggregated["all_responses"][role.value] = response.result

            if response.success:
                aggregated["reasoning_trace"].extend(response.reasoning_trace)
            else:
                if aggregated["error"]:
                    aggregated["error"] += f"; {response.error}"
                else:
                    aggregated["error"] = response.error

            if response.success and aggregated["primary_response"] is None:
                aggregated["primary_response"] = response.result

        if aggregated["primary_response"] is None and agent_responses:
            first = next(iter(agent_responses.values()))
            aggregated["primary_response"] = first.result
            aggregated["success"] = first.success

        return aggregated

    def build_final_answer(self, aggregated: dict[str, Any], question: str) -> dict[str, Any]:
        """构建最终响应"""
        primary = aggregated.get("primary_response") or {}

        answer = ""
        if isinstance(primary, dict):
            answer = primary.get("answer", "") or primary.get("message", str(primary))
        elif isinstance(primary, str):
            answer = primary

        return {
            "question": question,
            "answer": answer,
            "reasoning_trace": aggregated.get("reasoning_trace", []),
            "agent_responses": aggregated.get("all_responses", {}),
            "success": aggregated.get("success", True),
            "error": aggregated.get("error"),
        }

    def _parse_json_response(self, raw: str) -> dict[str, Any] | None:
        """解析JSON响应"""
        raw = (raw or "").strip()

        if raw.startswith("{"):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        brace = re.search(r"(\{.*\})", raw, re.S)
        if brace:
            try:
                return json.loads(brace.group(1))
            except json.JSONDecodeError:
                pass

        return None


supervisor_agent = SupervisorAgent()
