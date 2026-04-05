"""
QA Specialist Agent - 只接收改写后的问题进行回答

在新的多Agent架构中：
- QueryRewriteAgent 负责改写问题
- MemoryAgent 负责存储记忆
- QA Agent 只负责用改写后的问题回答
"""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import Tool

from app.agent.multi_agent.messages import AgentResponse, TaskType, AgentRole


# QA Agent专用工具（只保留问答相关，移除改写）
QA_TOOLS = [
    "kb_answer_question",
    "kb_assemble_context",
    "kb_generate_answer",
    "kb_get_chat_history",
    "kb_create_chat_session",
]


QA_SYSTEM_PROMPT = """你是一个专业的问答Agent，专注于知识库问答任务。

注意：你只接收已经改写好的问题，不需要再进行改写。

你的职责：
1. 使用RAG工具检索相关知识
2. 生成准确、有依据的答案
3. 对答案进行验证确保质量

可用工具：
- kb_answer_question: 完整RAG问答流程
- kb_assemble_context: 组装检索结果为上下文
- kb_generate_answer: 基于上下文生成答案
- kb_get_chat_history: 获取对话历史
- kb_create_chat_session: 创建会话

工作流程：
1. 直接调用 kb_answer_question 获取答案
2. 组装最终答案

输出格式（JSON）：
{"thought": "推理过程", "action": "工具名或final", "action_input": {"参数"}, "observation": "结果"}
"""


class QAAgent:
    """
    QA Specialist Agent - 只负责回答问题

    不负责改写（由QueryRewriteAgent负责）
    不负责存储记忆（由MemoryAgent负责）
    """

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            from app.agent.llm import get_chat_llm
            self._llm = get_chat_llm()
        return self._llm

    def _get_qa_tools(self) -> list[Tool]:
        from app.agent.agent import TOOL_MAP
        return [TOOL_MAP[name] for name in QA_TOOLS if name in TOOL_MAP]

    def _build_tool_map(self) -> dict[str, Tool]:
        from app.agent.agent import TOOL_MAP
        return {name: TOOL_MAP[name] for name in QA_TOOLS if name in TOOL_MAP}

    def execute(self, question: str, session_id: str = "", rewritten_question: str = None, max_steps: int = 6, governance_context: dict[str, Any] = None) -> AgentResponse:
        """
        执行QA任务。

        Args:
            question: 原始问题
            session_id: 会话ID
            rewritten_question: 改写后的问题（由QueryRewriteAgent提供）
            max_steps: 最大步数
            governance_context: 治理上下文（可选）

        Returns:
            AgentResponse with answer
        """
        self._governance_context = governance_context or {}
        self._session_id = session_id
        tool_map = self._build_tool_map()
        reasoning_trace = []
        messages = []

        # 使用改写后的问题，如果没有则用原问题
        query = rewritten_question if rewritten_question else question

        for step in range(max_steps):
            action_response = self._reasoning_step(
                query,
                session_id,
                messages,
                reasoning_trace,
            )

            thought = action_response.get("thought", "")
            action = action_response.get("action", "")
            action_input = action_response.get("action_input", {})
            observation = action_response.get("observation", "")

            trace_entry = {
                "step": step + 1,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation,
            }
            reasoning_trace.append(trace_entry)

            # If final, return answer
            if action == "final":
                answer = action_input.get("answer", observation or "")
                return AgentResponse(
                    agent=AgentRole.QA,
                    task_type=TaskType.QA,
                    result={"answer": answer, "question": question, "rewritten_query": query},
                    reasoning_trace=reasoning_trace,
                    success=True,
                )

            # Execute tool
            if action in tool_map:
                tool_result = self._execute_tool(tool_map[action], action_input)
                tool_msg = ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    tool_call_id="",
                )
                messages.append(tool_msg)

                reasoning_trace[-1]["observation"] = self._summarize_tool_result(tool_result)

                # kb_answer_question 返回答案
                if action == "kb_answer_question":
                    data = tool_result.get("data", {})
                    if data.get("answer"):
                        return AgentResponse(
                            agent=AgentRole.QA,
                            task_type=TaskType.QA,
                            result={
                                "answer": data["answer"],
                                "question": question,
                                "rewritten_query": query,
                                "sources": data.get("sources", []),
                                "confidence": data.get("confidence"),
                                "retrieved_chunks": data.get("retrieved_chunks", []),
                            },
                            reasoning_trace=reasoning_trace,
                            success=True,
                        )
            else:
                break

        # Max steps reached
        return AgentResponse(
            agent=AgentRole.QA,
            task_type=TaskType.QA,
            result={"answer": reasoning_trace[-1].get("observation", "") if reasoning_trace else ""},
            reasoning_trace=reasoning_trace,
            success=False,
            error="max_steps_reached",
        )

    def _reasoning_step(self, question: str, session_id: str, messages: list, reasoning_trace: list) -> dict[str, Any]:
        """执行一步推理"""
        llm = self._get_llm()
        tools = self._get_qa_tools()

        tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])

        trace_text = "\n".join([
            f"Step {e['step']}: {e.get('action', 'unknown')} -> {e.get('observation', '')}"
            for e in reasoning_trace[-3:]
        ])

        prompt = f"""问题（已改写）：{question}

可用工具：
{tool_descriptions}

之前的推理：
{trace_text or "无"}

请决定下一步操作（JSON格式）：
{{"thought": "推理过程", "action": "工具名或final", "action_input": {{"参数"}}, "observation": ""}}
"""

        try:
            response = llm.invoke([
                SystemMessage(content=QA_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            raw_content = response.content if hasattr(response, "content") else str(response)
            return self._parse_action_response(raw_content)
        except Exception as e:
            return {"thought": f"LLM调用失败: {e}", "action": "final", "action_input": {"answer": str(e)}}

    def _execute_tool(self, tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
        """执行工具（通过 ToolDispatcher 统一入口）"""
        # 使用 ToolDispatcher 作为统一入口，所有调用都经过 Action Guard
        from app.tools.tool_dispatcher import invoke_tool

        governance_context = getattr(self, '_governance_context', {})
        session_id = getattr(self, '_session_id', '')

        return invoke_tool(
            tool_name=tool.name,
            args=args,
            agent="qa",
            session_id=session_id,
            governance_context=governance_context,
        )

    def _parse_action_response(self, raw: str) -> dict[str, Any]:
        """解析JSON响应"""
        import re
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

        return {"thought": raw, "action": "final", "action_input": {"answer": raw}}

    def _summarize_tool_result(self, result: dict[str, Any]) -> str:
        """总结工具结果"""
        if not isinstance(result, dict):
            return str(result)[:200]

        if not result.get("ok", True):
            error = result.get("error", {})
            return f"工具执行失败: {error.get('message', 'unknown')}"

        data = result.get("data", {})
        if isinstance(data, dict):
            if data.get("answer"):
                return f"答案生成: {str(data['answer'])[:100]}"
            hits = data.get("hits", [])
            if hits is not None:
                return f"检索到 {len(hits)} 个结果"

        return "执行成功"


# 全局实例
qa_agent = QAAgent()
