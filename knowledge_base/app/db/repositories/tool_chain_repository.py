"""
tool_chain_repository.py — Agent tool chain memory for tracking successful action sequences.

Stores successful tool call sequences (from reasoning_trace) in MySQL.
Enables the agent to replay successful chains for similar questions.
"""

from __future__ import annotations

import hashlib
from typing import Any

from app.db.connection import get_cursor
from app.db.utils import safe_json_dumps, safe_json_loads


def _hash_question(question: str) -> str:
    """Create a normalized hash of the question for lookup."""
    normalized = question.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _build_tool_sequence_from_trace(reasoning_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract tool sequence from reasoning trace.

    Each entry in reasoning_trace has: action, action_input, step, thought, observation.
    We skip 'final' actions.
    """
    sequence = []
    for entry in reasoning_trace:
        action = entry.get("action", "")
        if action and action != "final" and action not in {"", None}:
            sequence.append({
                "tool": action,
                "args": entry.get("action_input", {}),
            })
    return sequence


def upsert_tool_chain(
    question: str,
    reasoning_trace: list[dict[str, Any]],
    session_id: str,
) -> bool:
    """
    Insert or update a tool chain for a question.

    If an entry for the same question_hash exists, increment success_count and update.
    Otherwise, insert a new row.
    """
    question_hash = _hash_question(question)
    tool_sequence = _build_tool_sequence_from_trace(reasoning_trace)

    if not tool_sequence:
        return False

    try:
        with get_cursor(commit=True) as (_, cursor):
            # Try to update existing
            cursor.execute(
                """
                UPDATE agent_tool_chains
                SET success_count = success_count + 1,
                    last_used = CURRENT_TIMESTAMP,
                    tool_sequence = %s,
                    question_text = %s,
                    session_id = %s
                WHERE question_hash = %s
                """,
                (safe_json_dumps(tool_sequence), question, session_id, question_hash),
            )

            if cursor.rowcount == 0:
                # Insert new
                cursor.execute(
                    """
                    INSERT INTO agent_tool_chains
                    (question_hash, question_text, tool_sequence, session_id, success_count, last_used)
                    VALUES (%s, %s, %s, %s, 1, CURRENT_TIMESTAMP)
                    """,
                    (question_hash, question, safe_json_dumps(tool_sequence), session_id),
                )
        return True
    except Exception as e:
        print(f"[tool_chain_repository] upsert_tool_chain failed: {e}")
        return False


def get_similar_tool_chain(
    question: str,
    min_success_count: int = 2,
) -> dict[str, Any] | None:
    """
    Look up a similar tool chain by question hash.

    Returns the tool chain with highest success_count for the given question hash,
    if success_count >= min_success_count.
    """
    question_hash = _hash_question(question)

    try:
        with get_cursor() as (_, cursor):
            cursor.execute(
                """
                SELECT id, question_hash, question_text, tool_sequence,
                       success_count, last_used, created_at
                FROM agent_tool_chains
                WHERE question_hash = %s AND success_count >= %s
                ORDER BY success_count DESC, last_used DESC
                LIMIT 1
                """,
                (question_hash, min_success_count),
            )
            row = cursor.fetchone()
            if not row:
                return None

            # row is a dict-like object (PyMySQL DictCursor)
            return {
                "id": row["id"],
                "question_hash": row["question_hash"],
                "question_text": row["question_text"],
                "tool_sequence": safe_json_loads(row["tool_sequence"], fallback=[]),
                "success_count": row["success_count"],
                "last_used": row["last_used"],
                "created_at": row["created_at"],
            }
    except Exception as e:
        print(f"[tool_chain_repository] get_similar_tool_chain failed: {e}")
        return None


def get_memory_context(question: str, min_success_count: int = 2) -> str:
    """
    Get a formatted context string for tool chain injection.

    Returns a string like:
    "类似问题的成功工具链: kb_answer_question -> kb_search_knowledge_base (成功率: 3次)"
    """
    chain = get_similar_tool_chain(question, min_success_count)
    if not chain:
        return ""

    tool_sequence = chain.get("tool_sequence", [])
    if not tool_sequence:
        return ""

    tools = " -> ".join(step.get("tool", "") for step in tool_sequence if step.get("tool"))
    if not tools:
        return ""

    success_count = chain.get("success_count", 1)
    return f"类似问题的成功工具链: {tools} (成功率: {success_count}次)"
