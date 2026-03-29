"""
Context assembly: build readable context text from retrieved chunks.
"""
from __future__ import annotations

import re
from typing import Any

from app.services.common import normalize_whitespace

from .config import (
    CHAT_HISTORY_LIMIT,
    QA_MAX_CONTEXT_CHUNKS,
    _page_label,
    _truncate_text,
)


def _clean_table_text(text: str) -> str:
    """
    Clean up table-formatted text: remove col_0= prefixes and simplify layout.
    For paired financial data (two fiscal years), add a header to clarify column order.
    """
    text = normalize_whitespace(text)

    has_col_prefix = bool(re.search(r'^col_0=', text, re.MULTILINE))
    cleaned = re.sub(r'(?m)^col_0=', '', text)

    if has_col_prefix:
        # Detect if this has paired values (two numeric columns)
        lines = cleaned.split('\n')
        numeric_line_count = sum(
            1 for line in lines
            if re.match(r'^\s*\$?[\d,]+\.?\d*\s*$', line.strip())
        )
        if numeric_line_count >= 4:
            # Add header to clarify two-column format
            return (
                "[NOTE: Financial table with two fiscal year columns. "
                "When data appears as:\n"
                "  Label\n"
                "  $value1\n"
                "  $value2\n"
                "value1 = most recent fiscal year, value2 = prior fiscal year. "
                "Use the FIRST value (most recent year) for current-year analysis.]\n\n"
                + cleaned
            )

    return cleaned


def assemble_context(chunks: list[dict[str, Any]], max_chunks: int = QA_MAX_CONTEXT_CHUNKS) -> str:
    """Build a readable context string from retrieved chunks."""
    if not chunks:
        return ""

    context_parts: list[str] = []

    for idx, chunk in enumerate(chunks[:max_chunks], start=1):
        title = chunk.get("title", "") or ""
        section_title = chunk.get("section_title", "") or ""
        section_path = chunk.get("section_path", "") or ""
        page_lbl = _page_label(chunk.get("page_start"), chunk.get("page_end"))
        raw_text = chunk.get("search_text") or chunk.get("chunk_text") or ""

        # Clean up table formatting for readability
        chunk_text = _clean_table_text(raw_text)

        part = [
            f"[Source {idx}]",
            f"Document: {title or '-'}",
            f"Section: {section_title or section_path or '-'}",
            f"Page: {page_lbl}",
            "Content:",
            chunk_text,
        ]
        context_parts.append("\n".join(part))

    return "\n\n".join(context_parts)


def _format_history_for_prompt(history: list[dict[str, Any]], limit: int = CHAT_HISTORY_LIMIT) -> str:
    """Format chat history as a string for prompt injection."""
    if not history:
        return ""

    selected = history[-limit:]
    lines: list[str] = []

    for msg in selected:
        role = msg.get("role", "user")
        message = normalize_whitespace(msg.get("message", ""))
        if not message:
            continue

        role_label = "User" if role == "user" else "Assistant" if role == "assistant" else "System"
        lines.append(f"{role_label}: {message}")

    return "\n".join(lines).strip()
