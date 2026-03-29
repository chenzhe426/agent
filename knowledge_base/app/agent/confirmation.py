"""
confirmation.py — Human-in-loop confirmation models and utilities.

Provides structured confirmation request/response types and the ask_confirmation
function used by the agent graph when high-risk operations require user approval.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ConfirmationRequest(BaseModel):
    """Request model for agent confirmation."""

    session_id: str = Field(..., description="Chat session ID")
    confirmed: bool = Field(..., description="Whether the user confirmed the action")
    tool_name: Optional[str] = Field(default=None, description="Tool name being confirmed")
    tool_args: Optional[dict[str, Any]] = Field(default=None, description="Tool arguments")


class ConfirmationResponse(BaseModel):
    """Response model for confirmation result."""

    ok: bool = Field(..., description="Whether the confirmation was processed successfully")
    session_id: str = Field(..., description="Chat session ID")
    tool_name: Optional[str] = Field(default=None, description="Tool name that was confirmed/rejected")
    message: str = Field(default="", description="Status message")


# In-memory store for pending confirmations (keyed by session_id)
# In production, this would be stored in Redis or the database
_pending_confirmations: dict[str, dict[str, Any]] = {}


def ask_confirmation(
    session_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """
    Register a pending confirmation request for a session.

    Returns the confirmation context that will be sent via SSE to the client.
    The client must then call POST /agent/confirm to approve or reject.
    """
    confirmation_context = {
        "session_id": session_id,
        "tool": tool_name,
        "args": tool_args,
        "reason": reason,
    }
    _pending_confirmations[session_id] = confirmation_context
    return confirmation_context


def get_pending_confirmation(session_id: str) -> dict[str, Any] | None:
    """Get pending confirmation for a session, if any."""
    return _pending_confirmations.get(session_id)


def clear_pending_confirmation(session_id: str) -> None:
    """Clear the pending confirmation for a session after it's been handled."""
    _pending_confirmations.pop(session_id, None)


def is_confirmation_pending(session_id: str) -> bool:
    """Check if there's a pending confirmation for the session."""
    return session_id in _pending_confirmations
