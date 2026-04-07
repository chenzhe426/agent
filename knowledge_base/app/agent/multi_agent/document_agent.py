"""
DocumentAgent - 文档管理Agent

职责：
- 文档检索：根据用户描述查找文档（如"AMD 2023年金融研报"）
- 文档导入：导入新的文档到知识库
- 文档摘要：生成文档摘要
- 文档索引：构建文档索引
"""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from typing_extensions import TypedDict

from app.agent.multi_agent.messages import AgentResponse, TaskType, AgentRole


# Document Agent专用工具
DOCUMENT_TOOLS = [
    "kb_import_file",
    "kb_import_folder",
    "kb_index_document",
    "kb_summarize_document",
    "kb_create_chat_session",
]


DOCUMENT_SYSTEM_PROMPT = """你是一个专业的文档管理Agent，专注于文档检索、导入、索引和摘要任务。

你的职责：
1. 根据用户描述检索已有文档（如"AMD 2023年金融研报"）
2. 处理文档导入请求
3. 构建文档索引
4. 生成文档摘要

可用工具：
- kb_import_file: 导入单个文档文件（PDF、DOCX、TXT、MD）
- kb_import_folder: 批量导入文件夹中的所有文档
- kb_index_document: 为指定文档构建索引
- kb_summarize_document: 生成文档摘要
- kb_create_chat_session: 创建新的聊天会话

工作流程：
1. 分析用户请求类型（检索/导入/索引/摘要）
2. 调用相应工具
3. 返回结果

输出格式（JSON）：
{"thought": "推理过程", "action": "工具名或final", "action_input": {"参数"}, "observation": "结果"}
"""


class DocumentAgentState(TypedDict, total=False):
    """Document Agent state."""
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


class DocumentAgent:
    """
    Document Agent - 文档管理

    职责：
    - 文档检索：根据描述查找文档
    - 文档导入：导入新文档
    - 文档摘要：生成摘要
    - 文档索引：构建索引
    """

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            from app.agent.llm import get_chat_llm
            self._llm = get_chat_llm()
        return self._llm

    def _get_document_tools(self) -> list[Tool]:
        from app.agent.agent import TOOL_MAP
        return [TOOL_MAP[name] for name in DOCUMENT_TOOLS if name in TOOL_MAP]

    def _build_tool_map(self) -> dict[str, Tool]:
        from app.agent.agent import TOOL_MAP
        return {name: TOOL_MAP[name] for name in DOCUMENT_TOOLS if name in TOOL_MAP}

    def execute(self, task: str, session_id: str = "", max_steps: int = 5, governance_context: dict[str, Any] = None) -> AgentResponse:
        """
        执行文档管理任务。

        Args:
            task: 任务描述（如"给我AMD 2023年的金融研报"）
            session_id: 会话ID
            max_steps: 最大步数
            governance_context: 治理上下文（可选）

        Returns:
            AgentResponse with task result
        """
        self._governance_context = governance_context or {}
        self._session_id = session_id
        tool_map = self._build_tool_map()
        reasoning_trace = []

        # 分析任务，决定使用哪个工具
        tool_decision = self._decide_tool(task)

        if tool_decision["action"] == "final":
            return AgentResponse(
                agent=AgentRole.DOCUMENT,
                task_type=TaskType.DOCUMENT,
                result=tool_decision.get("result", {}),
                reasoning_trace=reasoning_trace,
                success=False,
                error=tool_decision.get("error", "无法确定操作"),
            )

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
                    agent=AgentRole.DOCUMENT,
                    task_type=TaskType.DOCUMENT,
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
                    agent=AgentRole.DOCUMENT,
                    task_type=TaskType.DOCUMENT,
                    result={},
                    reasoning_trace=reasoning_trace,
                    success=False,
                    error=tool_result.get("error", {}).get("message", "执行失败"),
                )

        return AgentResponse(
            agent=AgentRole.DOCUMENT,
            task_type=TaskType.DOCUMENT,
            result={},
            reasoning_trace=reasoning_trace,
            success=False,
            error="未知操作",
        )

    def _decide_tool(self, task: str) -> dict[str, Any]:
        """根据任务描述决定使用哪个工具"""
        llm = self._get_llm()
        tools = self._get_document_tools()

        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}"
            for t in tools
        ])

        prompt = f"""任务描述：{task}

可用工具：
{tool_descriptions}

分析任务并决定使用哪个工具：
- 如果用户要求检索/查找文档，先搜索知识库找到相关文档
- 如果用户要求导入新文档，使用 kb_import_file 或 kb_import_folder
- 如果用户要求对文档建索引，使用 kb_index_document
- 如果用户要求摘要文档，使用 kb_summarize_document

如果任务不明确或不需要工具操作，返回：
{{"action": "final", "reason": "原因", "result": {{"message": "说明"}}}}

否则返回：
{{"action": "工具名", "thought": "推理过程", "action_input": {{"参数"}}}}
"""

        try:
            response = llm.invoke([
                SystemMessage(content=DOCUMENT_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            raw_content = response.content if hasattr(response, "content") else str(response)
            return self._parse_action_response(raw_content)
        except Exception as e:
            return {"action": "final", "error": str(e)}

    def summarize_document(self, document_id: int) -> AgentResponse:
        """对指定文档生成摘要"""
        from app.agent.agent import TOOL_MAP
        summarize_tool = TOOL_MAP.get("kb_summarize_document")

        if not summarize_tool:
            return AgentResponse(
                agent=AgentRole.DOCUMENT,
                task_type=TaskType.SUMMARY,
                result={},
                success=False,
                error="kb_summarize_document tool not found",
            )

        tool_result = self._execute_tool(summarize_tool, {"document_id": document_id})

        if tool_result.get("ok"):
            return AgentResponse(
                agent=AgentRole.DOCUMENT,
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
                agent=AgentRole.DOCUMENT,
                task_type=TaskType.SUMMARY,
                result={},
                success=False,
                error=tool_result.get("error", {}).get("message", "摘要生成失败"),
            )

    def _execute_tool(self, tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
        """执行工具（优先通过 MCP Client，失败则回退到直接调用）"""
        governance_context = getattr(self, '_governance_context', {})
        session_id = getattr(self, '_session_id', '')

        # 优先通过 MCP 调用（所有工具调用都经过 MCP 协议层）
        from app.tools.mcp_client import call_tool_mcp_or_local

        return call_tool_mcp_or_local(
            tool_name=tool.name,
            args=args,
            agent="document",
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

        return {"action": "final", "error": "无法解析响应"}

    def _summarize_tool_result(self, result: dict[str, Any]) -> str:
        """总结工具结果"""
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


# 全局实例
document_agent = DocumentAgent()
