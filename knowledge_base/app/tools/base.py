from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from app.tools.schemas import ToolError, ToolMeta, ToolResult


class ToolExecutionError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def make_ok(tool_name: str, data: Dict[str, Any], duration_ms: Optional[int] = None) -> Dict[str, Any]:
    result = ToolResult(
        ok=True,
        data=data,
        error=None,
        meta=ToolMeta(tool_name=tool_name, duration_ms=duration_ms),
    )
    return result.model_dump()


def make_error(
    tool_name: str,
    code: str,
    message: str,
    duration_ms: Optional[int] = None,
) -> Dict[str, Any]:
    result = ToolResult(
        ok=False,
        data=None,
        error=ToolError(code=code, message=message),
        meta=ToolMeta(tool_name=tool_name, duration_ms=duration_ms),
    )
    return result.model_dump()


def run_tool(
    tool_name: str,
    fn: Callable[[], Dict[str, Any]],
    max_retries: int = 3,
    retry_delay: float = 0.5,
    backoff_factor: float = 2.0,
) -> Dict[str, Any]:
    """
    Execute a tool with retry logic and exponential backoff.

    Args:
        tool_name: Name of the tool for logging/meta purposes
        fn: The tool function to execute
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 0.5)
        backoff_factor: Multiplier for delay on each retry (default: 2.0)
    """
    last_error: Exception | None = None
    duration_ms = 0

    for attempt in range(max_retries + 1):
        start = time.perf_counter()
        try:
            data = fn()
            duration_ms = int((time.perf_counter() - start) * 1000)
            return make_ok(tool_name=tool_name, data=data, duration_ms=duration_ms)
        except ToolExecutionError as e:
            duration_ms = int((time.perf_counter() - start) * 1000)
            last_error = e
            if attempt < max_retries:
                time.sleep(retry_delay * (backoff_factor ** attempt))
                continue
            return make_error(
                tool_name=tool_name,
                code=e.code,
                message=e.message,
                duration_ms=duration_ms,
            )
        except FileNotFoundError as e:
            duration_ms = int((time.perf_counter() - start) * 1000)
            last_error = e
            if attempt < max_retries:
                time.sleep(retry_delay * (backoff_factor ** attempt))
                continue
            return make_error(
                tool_name=tool_name,
                code="FILE_NOT_FOUND",
                message=str(e),
                duration_ms=duration_ms,
            )
        except ValueError as e:
            duration_ms = int((time.perf_counter() - start) * 1000)
            last_error = e
            if attempt < max_retries:
                time.sleep(retry_delay * (backoff_factor ** attempt))
                continue
            return make_error(
                tool_name=tool_name,
                code="INVALID_INPUT",
                message=str(e),
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = int((time.perf_counter() - start) * 1000)
            last_error = e
            if attempt < max_retries:
                time.sleep(retry_delay * (backoff_factor ** attempt))
                continue
            return make_error(
                tool_name=tool_name,
                code="INTERNAL_ERROR",
                message=str(e),
                duration_ms=duration_ms,
            )

    # Should not reach here, but just in case
    return make_error(
        tool_name=tool_name,
        code="INTERNAL_ERROR",
        message=str(last_error) if last_error else "Unknown error after retries",
        duration_ms=duration_ms,
    )


def require_field(value: Any, code: str, message: str) -> Any:
    if value is None:
        raise ToolExecutionError(code=code, message=message)
    return value