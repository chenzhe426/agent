"""
QA Specialist Agent for complex question answering with verification.
"""

import json
import time
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import Tool
from typing_extensions import TypedDict

from app.agent.multi_agent.messages import AgentResponse, TaskType, AgentRole
from app.agent.agent import TOOLS


# QA Agent专用工具（只暴露问答相关工具）
QA_TOOLS = [
    "kb_answer_question",
    "kb_rewrite_query",
    "kb_assemble_context",
    "kb_generate_answer",
    "kb_get_chat_history",
    "kb_create_chat_session",
]


QA_SYSTEM_PROMPT = """你是一个专业的问答Agent，专注于知识库问答任务。

你的职责：
1. 理解用户问题，结合对话历史
2. 使用RAG工具检索相关知识
3. 生成准确、有依据的答案
4. 对答案进行验证确保质量

可用工具：
- kb_answer_question: 完整RAG问答流程
- kb_rewrite_query: 改写问题以更好匹配检索
- kb_assemble_context: 组装检索结果为上下文
- kb_generate_answer: 基于上下文生成答案
- kb_get_chat_history: 获取对话历史
- kb_create_chat_session: 创建会话

工作流程：
1. 分析问题，判断是否需要改写
2. 调用 kb_answer_question 获取答案
3. 如果需要验证，进行多步检索确认
4. 组装最终答案

输出格式（JSON）：
{"thought": "推理过程", "action": "工具名或final", "action_input": {"参数"}, "observation": "结果"}
"""


class QAAgentState(TypedDict, total=False):
    """QA Agent state."""
    messages: list[Any]
    session_id: str
    question: str
    reasoning_trace: list[dict[str, Any]]
    current_tool: str | None
    current_tool_args: dict[str, Any] | None
    current_tool_result: dict[str, Any] | None
    agent_step: int
    max_steps: int
    final_answer: str
    error: str | None


class QAAgent:
    """
    QA Specialist Agent for complex question answering.

    Handles:
    - RAG question answering
    - Answer verification
    - Multi-step reasoning
    """

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        """Lazy load LLM."""
        if self._llm is None:
            from app.agent.llm import get_chat_llm
            self._llm = get_chat_llm()
        return self._llm

    def _get_qa_tools(self) -> list[Tool]:
        """Get QA-specific tools."""
        from app.agent.agent import TOOL_MAP
        return [TOOL_MAP[name] for name in QA_TOOLS if name in TOOL_MAP]

    def _build_tool_map(self) -> dict[str, Tool]:
        """Build tool map for QA tools."""
        from app.agent.agent import TOOL_MAP
        return {name: TOOL_MAP[name] for name in QA_TOOLS if name in TOOL_MAP}

    def execute(self, question: str, session_id: str = "", max_steps: int = 6) -> AgentResponse:
        """
        Execute QA task.

        Args:
            question: User question
            session_id: Session identifier
            max_steps: Maximum reasoning steps

        Returns:
            AgentResponse with answer
        """
        tool_map = self._build_tool_map()
        reasoning_trace = []
        messages = []

        current_question = question
        current_session = session_id

        for step in range(max_steps):
            # LLM decides next action
            action_response = self._reasoning_step(
                current_question,
                current_session,
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
                    result={"answer": answer, "question": question},
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

                # Update observation in trace
                reasoning_trace[-1]["observation"] = self._summarize_tool_result(tool_result)

                # Update state for next iteration
                if action == "kb_rewrite_query":
                    rewritten = tool_result.get("data", {}).get("rewritten_query", "")
                    if rewritten:
                        current_question = rewritten
                elif action == "kb_answer_question":
                    data = tool_result.get("data", {})
                    if data.get("answer"):
                        return AgentResponse(
                            agent=AgentRole.QA,
                            task_type=TaskType.QA,
                            result={
                                "answer": data["answer"],
                                "question": question,
                                "sources": data.get("sources", []),
                                "confidence": data.get("confidence"),
                                "retrieved_chunks": data.get("retrieved_chunks", []),
                            },
                            reasoning_trace=reasoning_trace,
                            success=True,
                        )
            else:
                # Unknown action, return what we have
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

    def _reasoning_step(
        self,
        question: str,
        session_id: str,
        messages: list,
        reasoning_trace: list,
    ) -> dict[str, Any]:
        """Execute one reasoning step."""
        llm = self._get_llm()
        tools = self._get_qa_tools()

        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}"
            for t in tools
        ])

        history_text = self._format_history(messages)
        trace_text = "\n".join([
            f"Step {e['step']}: {e.get('action', 'unknown')} -> {e.get('observation', '')}"
            for e in reasoning_trace[-3:]  # Last 3 steps
        ])

        prompt = f"""问题：{question}

对话历史：
{history_text}

之前的推理步骤：
{trace_text}

可用工具：
{tool_descriptions}

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
        """Execute a tool."""
        start = time.perf_counter()
        try:
            result = tool.invoke(args)
            duration_ms = int((time.perf_counter() - start) * 1000)

            if not isinstance(result, dict):
                result = {"ok": True, "data": result}

            if "meta" not in result:
                result["meta"] = {"tool_name": tool.name, "duration_ms": duration_ms}

            return result
        except Exception as e:
            duration_ms = int((time.perf_counter() - start) * 1000)
            return {
                "ok": False,
                "error": {"code": "TOOL_ERROR", "message": str(e)},
                "meta": {"tool_name": tool.name, "duration_ms": duration_ms},
            }

    def _parse_action_response(self, raw: str) -> dict[str, Any]:
        """Parse JSON action from LLM response."""
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

    def _format_history(self, messages: list) -> str:
        """Format conversation history."""
        relevant = []
        for msg in messages[-6:]:
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            content = getattr(msg, "content", "") or ""
            if not content:
                continue
            if "tool" in role.lower():
                continue
            role_label = "用户" if "human" in role.lower() else "助手"
            relevant.append(f"{role_label}：{content[:200]}")
        return "\n".join(relevant) if relevant else "（无历史）"

    def _summarize_tool_result(self, result: dict[str, Any]) -> str:
        """Summarize tool result."""
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


# Global QA agent instance
qa_agent = QAAgent()
