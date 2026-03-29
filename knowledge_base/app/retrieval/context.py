"""
Contextual text assembly for embeddings and display.
"""
from __future__ import annotations

from typing import Any


def build_contextual_header(
    doc_title: str,
    page_start: int | None,
    section_title: str,
    section_path: str,
    chunk_type: str | None = None,
) -> str:
    """
    Build a contextual header string for a chunk.

    Format:
        "{doc_title} | Page {page} | Section: {section_path} > {section_title} | {chunk_type}"

    This header is prepended to the chunk text when constructing
    the contextual embedding, helping disambiguate sections that
    discuss similar topics in different parts of the document.

    Example:
        "AMD_2022_10K | Page 55 | Section: Item 8 > Consolidated Balance Sheets | table"
    """
    parts = []

    if doc_title:
        parts.append(str(doc_title))
    if page_start is not None:
        parts.append(f"Page {page_start}")
    if section_path or section_title:
        section_str = " > ".join(filter(None, [section_path, section_title]))
        if section_str:
            parts.append(f"Section: {section_str}")
    if chunk_type:
        parts.append(f"Type: {chunk_type}")

    return " | ".join(parts)


def build_contextual_text(cand: dict[str, Any]) -> str:
    """
    Build the full contextual text for a chunk candidate.
    Used when generating contextual embeddings for retrieval.
    """
    header = build_contextual_header(
        doc_title=cand.get("title", ""),
        page_start=cand.get("page_start"),
        section_title=cand.get("section_title", ""),
        section_path=cand.get("section_path", ""),
        chunk_type=cand.get("chunk_type"),
    )
    chunk_text = cand.get("search_text", "") or cand.get("chunk_text", "")
    return f"{header}\n{chunk_text}"
