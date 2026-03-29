"""
Search Specialist Agent for precise knowledge base retrieval.
"""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from typing_extensions import TypedDict

from app.agent.multi_agent.messages import AgentResponse, TaskType, AgentRole


# Search Agent专用工具
SEARCH_TOOLS = [
    "kb_search_knowledge_base",
    "kb_rewrite_query",
    "kb_assemble_context",
    "kb_get_chat_history",
]


SEARCH_SYSTEM_PROMPT = """你是一个专业的检索Agent，专注于从知识库中精确检索相关信息。

你的职责：
1. 理解用户查询意图
2. 优化查询以获得更好的检索结果
3. 从知识库中检索相关文档块
4. 组装检索结果为可阅读的上下文

可用工具：
- kb_search_knowledge_base: 在知识库中搜索相关内容
- kb_rewrite_query: 改写查询以提高检索效果
- kb_assemble_context: 将检索结果组装为上下文
- kb_get_chat_history: 获取对话历史

工作流程：
1. 分析查询，决定是否需要改写
2. 执行检索
3. 如有必要，改写后重新检索
4. 组装最终上下文

输出格式（JSON）：
{"thought": "推理过程", "action": "工具名或final", "action_input": {"参数"}, "observation": "结果"}
"""


class SearchAgentState(TypedDict, total=False):
    """Search Agent state."""
    messages: list[Any]
    session_id: str
    question: str
    reasoning_trace: list[dict[str, Any]]
    current_tool: str | None
    current_tool_args: dict[str, Any] | None
    agent_step: int
    max_steps: int
    final_context: str
    error: str | None


class SearchAgent:
    """
    Search Specialist Agent for precise retrieval.

    Handles:
    - Query understanding and rewrite
    - Knowledge base search
    - Result assembly
    """

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        """Lazy load LLM."""
        if self._llm is None:
            from app.agent.llm import get_chat_llm
            self._llm = get_chat_llm()
        return self._llm

    def _get_search_tools(self) -> list[Tool]:
        """Get search-specific tools."""
        from app.agent.agent import TOOL_MAP
        return [TOOL_MAP[name] for name in SEARCH_TOOLS if name in TOOL_MAP]

    def _build_tool_map(self) -> dict[str, Tool]:
        """Build tool map for search tools."""
        from app.agent.agent import TOOL_MAP
        return {name: TOOL_MAP[name] for name in SEARCH_TOOLS if name in TOOL_MAP}

    def execute(
        self,
        question: str,
        session_id: str = "",
        top_k: int = 5,
        max_steps: int = 4,
    ) -> AgentResponse:
        """
        Execute search task.

        Args:
            question: User query
            session_id: Session identifier
            top_k: Number of results to retrieve
            max_steps: Maximum reasoning steps

        Returns:
            AgentResponse with search results
        """
        tool_map = self._build_tool_map()
        reasoning_trace = []
        messages = []

        current_query = question

        for step in range(max_steps):
            # LLM decides next action
            action_response = self._reasoning_step(
                current_query,
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

            # If final, assemble and return
            if action == "final":
                context = observation or self._assemble_from_messages(messages)
                return AgentResponse(
                    agent=AgentRole.SEARCH,
                    task_type=TaskType.SEARCH,
                    result={
                        "context": context,
                        "query": question,
                        "hits": self._extract_hits(messages),
                    },
                    reasoning_trace=reasoning_trace,
                    success=True,
                )

            # Execute tool
            if action in tool_map:
                tool_result = self._execute_tool(tool_map[action], action_input)
                tool_msg = json.dumps(tool_result, ensure_ascii=False)
                messages.append(tool_msg)

                # Update observation
                reasoning_trace[-1]["observation"] = self._summarize_tool_result(tool_result)

                # If search, update current_query from rewritten query if needed
                if action == "kb_rewrite_query":
                    rewritten = tool_result.get("data", {}).get("rewritten_query", "")
                    if rewritten:
                        current_query = rewritten
                elif action == "kb_search_knowledge_base":
                    # Search completed, try to assemble
                    hits = tool_result.get("data", {}).get("hits", [])
                    if hits:
                        assemble_result = self._assemble_context(hits, max_chunks=top_k)
                        return AgentResponse(
                            agent=AgentRole.SEARCH,
                            task_type=TaskType.SEARCH,
                            result={
                                "context": assemble_result,
                                "query": question,
                                "hits": hits,
                            },
                            reasoning_trace=reasoning_trace,
                            success=True,
                        )
            else:
                break

        # Max steps reached
        return AgentResponse(
            agent=AgentRole.SEARCH,
            task_type=TaskType.SEARCH,
            result={
                "context": self._assemble_from_messages(messages),
                "query": question,
                "hits": self._extract_hits(messages),
            },
            reasoning_trace=reasoning_trace,
            success=True,
        )

    def _reasoning_step(
        self,
        query: str,
        session_id: str,
        messages: list,
        reasoning_trace: list,
    ) -> dict[str, Any]:
        """Execute one reasoning step."""
        llm = self._get_llm()
        tools = self._get_search_tools()

        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}"
            for t in tools
        ])

        trace_text = "\n".join([
            f"Step {e['step']}: {e.get('action', 'unknown')} -> {e.get('observation', '')}"
            for e in reasoning_trace[-3:]
        ])

        prompt = f"""查询：{query}

之前的推理步骤：
{trace_text}

可用工具：
{tool_descriptions}

请决定下一步操作（JSON格式）：
{{"thought": "推理过程", "action": "工具名或final", "action_input": {{"参数"}}, "observation": ""}}
"""

        try:
            response = llm.invoke([
                SystemMessage(content=SEARCH_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            raw_content = response.content if hasattr(response, "content") else str(response)
            return self._parse_action_response(raw_content)
        except Exception as e:
            return {"thought": f"LLM调用失败: {e}", "action": "final", "action_input": {}, "observation": str(e)}

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

    def _assemble_context(self, hits: list, max_chunks: int = 5) -> str:
        """Assemble context from hits using the tool."""
        from app.agent.agent import TOOL_MAP
        assemble_tool = TOOL_MAP.get("kb_assemble_context")
        if assemble_tool:
            result = self._execute_tool(assemble_tool, {"hits": hits, "max_chunks": max_chunks})
            if result.get("ok"):
                return result.get("data", {}).get("context", "")
        return self._format_hits(hits)

    def _format_hits(self, hits: list) -> str:
        """Format hits as context string."""
        if not hits:
            return "（未找到相关内容）"

        lines = []
        for i, hit in enumerate(hits[:5], 1):
            title = hit.get("title", "无标题")
            text = hit.get("text", hit.get("chunk_text", ""))
            score = hit.get("score", 0)
            lines.append(f"[{i}] {title} (相关度: {score:.2f})\n{text[:200]}...")
        return "\n\n".join(lines)

    def _assemble_from_messages(self, messages: list) -> str:
        """Extract context from messages."""
        context_parts = []
        for msg in messages:
            try:
                data = json.loads(msg) if isinstance(msg, str) else msg
                if data.get("ok"):
                    d = data.get("data", {})
                    if d.get("context"):
                        context_parts.append(d["context"])
                    elif d.get("hits"):
                        context_parts.append(self._format_hits(d["hits"]))
            except (json.JSONDecodeError, TypeError):
                pass
        return "\n\n".join(context_parts) if context_parts else "（未找到相关内容）"

    def _extract_hits(self, messages: list) -> list:
        """Extract hits from messages."""
        for msg in reversed(messages):
            try:
                data = json.loads(msg) if isinstance(msg, str) else msg
                if data.get("ok"):
                    hits = data.get("data", {}).get("hits", [])
                    if hits:
                        return hits
            except (json.JSONDecodeError, TypeError):
                pass
        return []

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

        return {"thought": raw, "action": "final", "action_input": {}, "observation": raw}

    def _summarize_tool_result(self, result: dict[str, Any]) -> str:
        """Summarize tool result."""
        if not isinstance(result, dict):
            return str(result)[:200]

        if not result.get("ok", True):
            error = result.get("error", {})
            return f"检索失败: {error.get('message', 'unknown')}"

        data = result.get("data", {})
        hits = data.get("hits", [])
        if hits is not None:
            return f"检索到 {len(hits)} 个结果"

        return "执行成功"


# Global search agent instance
search_agent = SearchAgent()
