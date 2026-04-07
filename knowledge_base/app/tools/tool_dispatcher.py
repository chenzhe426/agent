"""
tool_dispatcher.py - 统一工具调用分发器

作为所有工具调用的唯一入口：
- Agent（QAAgent、DocumentAgent）通过此分发器调用工具
- MCP Server 通过此分发器调用工具
- 所有调用都经过 Action Guard 检查

设计原则：
1. 所有工具调用必须经过此分发器
2. Action Guard 在此处统一执行
3. 集成频率限制防止 DDoS
4. 不直接暴露 TOOL_MAP 给 Agent
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from threading import Lock
from typing import Any, Optional

from app.governance.gateway import governance_gateway
from app.governance.schemas import GovernanceDecision
from loguru import logger


class RateLimiter:
    """
    基于滑动窗口的频率限制器

    用于防止恶意提示词注入导致的自我 DDoS 攻击
    （例如：让 Agent 在循环中疯狂调用工具）

    配置：
    - SESSION_RATE_LIMIT: 每 session 每窗口的最多调用次数
    - AGENT_RATE_LIMIT: 每 agent 每窗口的最多调用次数
    """

    def __init__(
        self,
        max_calls: int = 60,
        window_seconds: int = 60,
    ):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._calls: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def _cleanup(self, key: str, now: float) -> None:
        """清理过期的调用记录"""
        self._calls[key] = [
            t for t in self._calls[key]
            if now - t < self.window_seconds
        ]

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        检查是否允许调用

        Args:
            key: 限流键（如 session_id 或 agent）

        Returns:
            (是否允许, 剩余调用次数)
        """
        now = time.time()

        with self._lock:
            self._cleanup(key, now)

            remaining = self.max_calls - len(self._calls[key])
            if remaining <= 0:
                return False, 0

            self._calls[key].append(now)
            return True, remaining - 1

    def get_retry_after(self, key: str) -> int:
        """获取需要等待的秒数"""
        now = time.time()

        with self._lock:
            if key not in self._calls or not self._calls[key]:
                return 0

            oldest = min(self._calls[key])
            elapsed = now - oldest
            if elapsed >= self.window_seconds:
                return 0
            return int(self.window_seconds - elapsed)

    def get_usage(self, key: str) -> dict[str, Any]:
        """获取当前使用情况"""
        now = time.time()

        with self._lock:
            self._cleanup(key, now)
            used = len(self._calls[key])
            return {
                "used": used,
                "remaining": max(0, self.max_calls - used),
                "limit": self.max_calls,
                "window_seconds": self.window_seconds,
            }


# 从环境变量读取配置（支持运行时调整）
def _get_rate_limit_config() -> tuple[int, int, int, int]:
    """获取频率限制配置"""
    session_max = int(os.getenv("RATE_LIMIT_SESSION_MAX", "60"))
    session_window = int(os.getenv("RATE_LIMIT_SESSION_WINDOW", "60"))
    agent_max = int(os.getenv("RATE_LIMIT_AGENT_MAX", "120"))
    agent_window = int(os.getenv("RATE_LIMIT_AGENT_WINDOW", "60"))
    return session_max, session_window, agent_max, agent_window


# 全局限流器实例
_session_limiter: Optional[RateLimiter] = None
_agent_limiter: Optional[RateLimiter] = None


def get_session_limiter() -> RateLimiter:
    """获取 session 级别限流器"""
    global _session_limiter
    if _session_limiter is None:
        max_calls, window, _, _ = _get_rate_limit_config()
        _session_limiter = RateLimiter(max_calls=max_calls, window_seconds=window)
    return _session_limiter


def get_agent_limiter() -> RateLimiter:
    """获取 agent 级别限流器"""
    global _agent_limiter
    if _agent_limiter is None:
        _, _, max_calls, window = _get_rate_limit_config()
        _agent_limiter = RateLimiter(max_calls=max_calls, window_seconds=window)
    return _agent_limiter


class ToolDispatcher:
    """
    统一工具调用分发器

    所有工具调用必须通过此类，禁止直接调用 TOOL_MAP
    """

    def __init__(self):
        # 延迟导入避免循环依赖
        self._tool_map = None

    @property
    def tool_map(self) -> dict[str, Any]:
        """延迟加载 TOOL_MAP"""
        if self._tool_map is None:
            from app.agent.agent import TOOL_MAP
            self._tool_map = TOOL_MAP
        return self._tool_map

    def invoke(
        self,
        tool_name: str,
        args: dict[str, Any],
        agent: str,
        session_id: str,
        tenant_id: str = "default",
        governance_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        调用工具的统一入口

        Args:
            tool_name: 工具名称
            args: 工具参数
            agent: 来源 Agent（qa/document/memory/rewrite/mcp）
            session_id: 会话ID
            tenant_id: 租户ID
            governance_context: 治理上下文（从 Rewrite 传递过来的）

        Returns:
            工具执行结果
        """
        start_time = time.perf_counter()

        # 1. 获取工具
        tool = self.tool_map.get(tool_name)
        if not tool:
            return {
                "ok": False,
                "error": {"code": "TOOL_NOT_FOUND", "message": f"Tool {tool_name} not found"},
                "meta": {"tool_name": tool_name, "duration_ms": 0},
            }

        # 2. 频率限制检查（防止 DDoS / 恶意提示词注入导致的自我攻击）
        rate_limit_key = f"{agent}:{session_id}"

        # 2.1 Session 级别限流
        session_limiter = get_session_limiter()
        session_allowed, session_remaining = session_limiter.is_allowed(session_id)
        if not session_allowed:
            retry_after = session_limiter.get_retry_after(session_id)
            logger.warning(f"Session rate limit exceeded: session_id={session_id}")
            return {
                "ok": False,
                "error": {
                    "code": "RATE_LIMIT_SESSION",
                    "message": f"Session rate limit exceeded. Retry after {retry_after}s",
                    "retry_after": retry_after,
                },
                "meta": {
                    "tool_name": tool_name,
                    "duration_ms": int((time.perf_counter() - start_time) * 1000),
                    "rate_limit": {
                        "session_remaining": 0,
                        "session_limit": session_limiter.max_calls,
                        "session_window": session_limiter.window_seconds,
                    },
                },
            }

        # 2.2 Agent 级别限流
        agent_limiter = get_agent_limiter()
        agent_allowed, agent_remaining = agent_limiter.is_allowed(agent)
        if not agent_allowed:
            retry_after = agent_limiter.get_retry_after(agent)
            logger.warning(f"Agent rate limit exceeded: agent={agent}")
            return {
                "ok": False,
                "error": {
                    "code": "RATE_LIMIT_AGENT",
                    "message": f"Agent rate limit exceeded. Retry after {retry_after}s",
                    "retry_after": retry_after,
                },
                "meta": {
                    "tool_name": tool_name,
                    "duration_ms": int((time.perf_counter() - start_time) * 1000),
                    "rate_limit": {
                        "agent_remaining": 0,
                        "agent_limit": agent_limiter.max_calls,
                        "agent_window": agent_limiter.window_seconds,
                    },
                },
            }

        # 3. Action Guard 检查
        # 构建 rewrite_output 用于传递风险标记
        rewrite_output = {}
        if governance_context:
            rewrite_output = {
                "risk_flags": governance_context.get("risk_flags", []),
                "source_tags": governance_context.get("source_tags", []),
            }

        guard_result = governance_gateway.guard_action(
            tool_name=tool_name,
            tool_args=args,
            agent=agent,
            session_id=session_id,
            tenant_id=tenant_id,
            rewrite_output=rewrite_output,
        )

        # 3. 如果 Action Guard 拒绝，直接返回拒绝结果
        if guard_result.decision == GovernanceDecision.DENY:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return {
                "ok": False,
                "error": {
                    "code": "ACTION_GUARD_DENIED",
                    "message": f"Action denied: {guard_result.message}",
                    "risk_flags": guard_result.risk_flags,
                    "governance_decision": guard_result.decision.value,
                },
                "meta": {
                    "tool_name": tool_name,
                    "duration_ms": duration_ms,
                    "governance_check": True,
                    "governance_decision": guard_result.decision.value,
                    "risk_level": guard_result.risk_level.value if hasattr(guard_result.risk_level, 'value') else str(guard_result.risk_level),
                },
            }

        # 4. 如果需要确认，记录警告但继续执行
        requires_confirmation = guard_result.decision == GovernanceDecision.CONFIRM

        # 5. 执行工具
        try:
            result = tool.invoke(args)
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            if not isinstance(result, dict):
                result = {"ok": True, "data": result}

            # 添加治理元数据
            if "meta" not in result:
                result["meta"] = {}
            result["meta"].update({
                "tool_name": tool_name,
                "duration_ms": duration_ms,
                "governance_check": True,
                "governance_decision": guard_result.decision.value,
                "risk_level": guard_result.risk_level.value if hasattr(guard_result.risk_level, 'value') else str(guard_result.risk_level),
                "requires_confirmation": requires_confirmation,
            })

            return result

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return {
                "ok": False,
                "error": {"code": "TOOL_ERROR", "message": str(e)},
                "meta": {
                    "tool_name": tool_name,
                    "duration_ms": duration_ms,
                    "governance_check": True,
                    "governance_decision": guard_result.decision.value,
                },
            }

    def check_guard(
        self,
        tool_name: str,
        args: dict[str, Any],
        agent: str,
        session_id: str,
        tenant_id: str = "default",
        governance_context: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        仅检查 Action Guard，不执行工具

        Returns:
            (是否通过, 治理结果)
        """
        rewrite_output = {}
        if governance_context:
            rewrite_output = {
                "risk_flags": governance_context.get("risk_flags", []),
                "source_tags": governance_context.get("source_tags", []),
            }

        result = governance_gateway.guard_action(
            tool_name=tool_name,
            tool_args=args,
            agent=agent,
            session_id=session_id,
            tenant_id=tenant_id,
            rewrite_output=rewrite_output,
        )

        passed = result.decision in [GovernanceDecision.ALLOW, GovernanceDecision.GRADE]
        return passed, result.model_dump()


# 全局实例
tool_dispatcher = ToolDispatcher()


def invoke_tool(
    tool_name: str,
    args: dict[str, Any],
    agent: str,
    session_id: str,
    tenant_id: str = "default",
    governance_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    便捷函数：调用工具（经过 Action Guard）

    推荐所有工具调用都使用此函数
    """
    return tool_dispatcher.invoke(
        tool_name=tool_name,
        args=args,
        agent=agent,
        session_id=session_id,
        tenant_id=tenant_id,
        governance_context=governance_context,
    )


def check_tool_guard(
    tool_name: str,
    args: dict[str, Any],
    agent: str,
    session_id: str,
    tenant_id: str = "default",
    governance_context: Optional[dict[str, Any]] = None,
) -> tuple[bool, dict[str, Any]]:
    """
    便捷函数：检查工具调用是否通过治理（不执行）

    用于预先检查高风险操作
    """
    return tool_dispatcher.check_guard(
        tool_name=tool_name,
        args=args,
        agent=agent,
        session_id=session_id,
        tenant_id=tenant_id,
        governance_context=governance_context,
    )
