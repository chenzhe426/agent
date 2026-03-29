"""
Import Specialist Agent for document management tasks.
"""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from typing_extensions import TypedDict

from app.agent.multi_agent.messages import AgentResponse, TaskType, AgentRole


# Import Agent专用工具
IMPORT_TOOLS = [
    "kb_import_file",
    "kb_import_folder",
    "kb_index_document",
    "kb_summarize_document",
    "kb_create_chat_session",
]


IMPORT_SYSTEM_PROMPT = """你是一个专业的文档管理Agent，专注于文档导入、索引和摘要任务。

你的职责：
1. 处理文档导入请求
2. 构建文档索引
3. 生成文档摘要
4. 管理会话

可用工具：
- kb_import_file: 导入单个文档文件（PDF、DOCX、TXT、MD）
- kb_import_folder: 批量导入文件夹中的所有文档
- kb_index_document: 为指定文档构建索引
- kb_summarize_document: 生成文档摘要
- kb_create_chat_session: 创建新的聊天会话

工作流程：
1. 分析用户请求类型（导入/索引/摘要）
2. 调用相应工具
3. 返回操作结果

输出格式（JSON）：
{"thought": "推理过程", "action": "工具名或final", "action_input": {"参数"}, "observation": "结果"}
"""


class ImportAgentState(TypedDict, total=False):
    """Import Agent state."""
    messages: list[Any]
    session_id: str
    task_description: str
    reasoning_trace: list[dict[str, Any]]
    current_tool: str | None
    current_tool_args: dict[str, Any] | None
    agent_step: int
    max_steps: int
    final_result: dict[str, Any]
    error: str | None


class ImportAgent:
    """
    Import Specialist Agent for document management.

    Handles:
    - Document import (single/file or batch/folder)
    - Document indexing
    - Document summarization
    - Session management
    """

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        """Lazy load LLM."""
        if self._llm is None:
            from app.agent.llm import get_chat_llm
            self._llm = get_chat_llm()
        return self._llm

    def _get_import_tools(self) -> list[Tool]:
        """Get import-specific tools."""
        from app.agent.agent import TOOL_MAP
        return [TOOL_MAP[name] for name in IMPORT_TOOLS if name in TOOL_MAP]

    def _build_tool_map(self) -> dict[str, Tool]:
        """Build tool map for import tools."""
        from app.agent.agent import TOOL_MAP
        return {name: TOOL_MAP[name] for name in IMPORT_TOOLS if name in TOOL_MAP}

    def execute(
        self,
        task: str,
        session_id: str = "",
        max_steps: int = 5,
    ) -> AgentResponse:
        """
        Execute import/document management task.

        Args:
            task: Task description (can be natural language)
            session_id: Session identifier
            max_steps: Maximum reasoning steps

        Returns:
            AgentResponse with task result
        """
        tool_map = self._build_tool_map()
        reasoning_trace = []

        # First, determine the appropriate tool from task description
        tool_decision = self._decide_tool(task)

        if tool_decision["action"] == "final":
            return AgentResponse(
                agent=AgentRole.IMPORT,
                task_type=TaskType.IMPORT,
                result=tool_decision.get("result", {}),
                reasoning_trace=reasoning_trace,
                success=False,
                error=tool_decision.get("error", "无法确定操作"),
            )

        # Execute the tool directly (import tasks are typically single-step)
        action = tool_decision["action"]
        action_input = tool_decision["action_input"]

        trace_entry = {
            "step": 1,
            "thought": tool_decision.get("thought", ""),
            "action": action,
            "action_input": action_input,
            "observation": "",
        }
        reasoning_trace.append(trace_entry)

        if action in tool_map:
            tool_result = self._execute_tool(tool_map[action], action_input)
            reasoning_trace[-1]["observation"] = self._summarize_tool_result(tool_result)

            if tool_result.get("ok"):
                return AgentResponse(
                    agent=AgentRole.IMPORT,
                    task_type=TaskType.IMPORT,
                    result={
                        "success": True,
                        "action": action,
                        "result": tool_result.get("data", tool_result),
                        "message": f"{action} 执行成功",
                    },
                    reasoning_trace=reasoning_trace,
                    success=True,
                )
            else:
                return AgentResponse(
                    agent=AgentRole.IMPORT,
                    task_type=TaskType.IMPORT,
                    result={},
                    reasoning_trace=reasoning_trace,
                    success=False,
                    error=tool_result.get("error", {}).get("message", "执行失败"),
                )

        return AgentResponse(
            agent=AgentRole.IMPORT,
            task_type=TaskType.IMPORT,
            result={},
            reasoning_trace=reasoning_trace,
            success=False,
            error="未知操作",
        )

    def _decide_tool(self, task: str) -> dict[str, Any]:
        """
        Decide which tool to use based on task description.

        Uses LLM to understand the intent.
        """
        llm = self._get_llm()
        tools = self._get_import_tools()

        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}"
            for t in tools
        ])

        prompt = f"""任务描述：{task}

可用工具：
{tool_descriptions}

分析任务并决定使用哪个工具。如果任务不明确或不需要工具操作，返回：
{{"action": "final", "reason": "原因", "result": {{"message": "说明"}}}}

否则返回：
{{"action": "工具名", "thought": "推理过程", "action_input": {{"参数"}}}}
"""

        try:
            response = llm.invoke([
                SystemMessage(content=IMPORT_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            raw_content = response.content if hasattr(response, "content") else str(response)
            return self._parse_action_response(raw_content)
        except Exception as e:
            return {"action": "final", "error": str(e)}

    def summarize_document(self, document_id: int) -> AgentResponse:
        """
        Summarize a specific document.

        Args:
            document_id: Document ID to summarize

        Returns:
            AgentResponse with summary
        """
        from app.agent.agent import TOOL_MAP
        summarize_tool = TOOL_MAP.get("kb_summarize_document")

        if not summarize_tool:
            return AgentResponse(
                agent=AgentRole.IMPORT,
                task_type=TaskType.SUMMARY,
                result={},
                success=False,
                error="kb_summarize_document tool not found",
            )

        tool_result = self._execute_tool(summarize_tool, {"document_id": document_id})

        if tool_result.get("ok"):
            return AgentResponse(
                agent=AgentRole.IMPORT,
                task_type=TaskType.SUMMARY,
                result={
                    "success": True,
                    "summary": tool_result.get("data", {}).get("summary", ""),
                    "document_id": document_id,
                },
                reasoning_trace=[{
                    "step": 1,
                    "action": "kb_summarize_document",
                    "observation": "摘要生成成功",
                }],
                success=True,
            )
        else:
            return AgentResponse(
                agent=AgentRole.IMPORT,
                task_type=TaskType.SUMMARY,
                result={},
                success=False,
                error=tool_result.get("error", {}).get("message", "摘要生成失败"),
            )

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

        return {"action": "final", "error": "无法解析响应"}

    def _summarize_tool_result(self, result: dict[str, Any]) -> str:
        """Summarize tool result."""
        if not isinstance(result, dict):
            return str(result)[:200]

        if not result.get("ok", True):
            error = result.get("error", {})
            return f"操作失败: {error.get('message', 'unknown')}"

        data = result.get("data", {})
        if isinstance(data, dict):
            keys = list(data.keys())
            if "document_id" in data:
                return f"文档 ID {data['document_id']} 操作成功"
            if "summary" in data:
                return f"摘要生成成功: {str(data['summary'])[:100]}"
            if "imported_count" in data:
                return f"导入成功: {data.get('imported_count', 0)} 个文档"
            return f"执行成功: {', '.join(keys[:3])}"

        return "执行成功"


# Global import agent instance
import_agent = ImportAgent()
