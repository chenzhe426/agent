"""
QA module configuration constants and utility helpers.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

import app.config as config
from app.services.common import normalize_whitespace, to_float


def _cfg(name: str, default: Any):
    return getattr(config, name, default)


# --- Session / chat ---
CHAT_HISTORY_LIMIT = int(_cfg("CHAT_HISTORY_LIMIT", 6))
CHAT_SUMMARY_TRIGGER_TURNS = int(_cfg("CHAT_SUMMARY_TRIGGER_TURNS", 12))
QA_MAX_CONTEXT_CHUNKS = int(_cfg("QA_MAX_CONTEXT_CHUNKS", 6))
QA_STRUCTURED_ENABLE = bool(_cfg("QA_STRUCTURED_ENABLE", True))
QA_HIGHLIGHT_ENABLE = bool(_cfg("QA_HIGHLIGHT_ENABLE", True))

# --- V4 pipeline ---
V4_ENABLE_ANSWER_VERIFIER = bool(_cfg("V4_ENABLE_ANSWER_VERIFIER", True))
V4_ENABLE_SELF_REFINE = bool(_cfg("V4_ENABLE_SELF_REFINE", False))
V4_MAX_REFINE_ROUNDS = int(_cfg("V4_MAX_REFINE_ROUNDS", 1))
V4_ANSWER_USE_STRUCTURED_OUTPUT = bool(_cfg("V4_ANSWER_USE_STRUCTURED_OUTPUT", True))
V4_NUMERIC_FIRST_FOR_NUMERIC_QUERIES = bool(_cfg("V4_NUMERIC_FIRST_FOR_NUMERIC_QUERIES", True))


def _new_session_id() -> str:
    return uuid.uuid4().hex


def _truncate_text(text: str, max_len: int = 220) -> str:
    text = normalize_whitespace(text or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _page_label(page_start: Any, page_end: Any) -> str:
    if page_start is None and page_end is None:
        return "-"
    if page_start is not None and page_end is not None:
        if page_start == page_end:
            return str(page_start)
        return f"{page_start}-{page_end}"
    return str(page_start if page_start is not None else page_end)


def _normalize_chunk_source(chunk: dict[str, Any]) -> dict[str, Any]:
    """Normalize chunk into the AnswerSource output structure."""
    return {
        "chunk_id": chunk.get("chunk_id"),
        "document_id": chunk.get("document_id"),
        "title": chunk.get("title", "") or "",
        "section_title": chunk.get("section_title", "") or "",
        "section_path": chunk.get("section_path", "") or "",
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "quote": chunk.get("chunk_text", "") or "",
        "score": to_float(chunk.get("score")),
        "highlight_spans": [],
    }


def _build_highlight_spans(text: str, terms: list[str]) -> list[dict[str, Any]]:
    """Build highlight spans by finding occurrences of query terms in text."""
    text = text or ""
    if not text or not terms:
        return []

    spans: list[tuple[int, int]] = []

    for term in terms:
        term = normalize_whitespace(term)
        if not term:
            continue

        try:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
        except Exception:
            continue

        for match in pattern.finditer(text):
            start, end = match.span()
            if start == end:
                continue
            spans.append((start, end))

    if not spans:
        return []

    spans.sort(key=lambda x: (x[0], x[1]))
    merged: list[list[int]] = []

    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return [
        {"start": start, "end": end, "text": text[start:end]}
        for start, end in merged
    ]


def _extract_query_terms(
    question: str,
    rewritten_query: str | None,
    retrieved_chunks: list[dict[str, Any]],
) -> list[str]:
    """Extract unique query terms from question and retrieved chunks."""
    terms = set()

    for raw in [question, rewritten_query or ""]:
        raw = normalize_whitespace(raw)
        if not raw:
            continue

        for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_./:-]{1,}", raw):
            terms.add(token)

    for chunk in retrieved_chunks:
        for term, count in (chunk.get("term_hits") or {}).items():
            if count:
                terms.add(term)

    return sorted(terms, key=lambda x: (-len(x), x))


def _normalize_retrieved_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Normalize a retrieved chunk into the ChunkResult output structure."""
    return {
        "chunk_id": chunk.get("chunk_id"),
        "document_id": chunk.get("document_id"),
        "chunk_index": chunk.get("chunk_index"),
        "score": to_float(chunk.get("score")),
        "rerank_score": to_float(chunk.get("rerank_score")),
        "embedding_score": to_float(chunk.get("embedding_score")),
        "keyword_score": to_float(chunk.get("keyword_score")),
        "bm25_score": to_float(chunk.get("bm25_score")),
        "title_match_score": to_float(chunk.get("title_match_score")),
        "section_match_score": to_float(chunk.get("section_match_score")),
        "coverage_score": to_float(chunk.get("coverage_score")),
        "matched_term_count": chunk.get("matched_term_count"),
        "title": chunk.get("title", "") or "",
        "section_title": chunk.get("section_title", "") or "",
        "section_path": chunk.get("section_path", "") or "",
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "chunk_type": chunk.get("chunk_type"),
        "chunk_text": chunk.get("chunk_text", "") or "",
        "term_hits": chunk.get("term_hits") or {},
        "term_hit_detail": chunk.get("term_hit_detail") or {},
        "is_neighbor": bool(chunk.get("is_neighbor", False)),
        "_retrieval_query": chunk.get("_retrieval_query"),
    }
