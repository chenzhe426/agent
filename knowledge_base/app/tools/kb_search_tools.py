from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.retrieval.service import retrieve_chunks
from app.tools.base import run_tool
from app.tools.schemas import (
    KBSearchKnowledgeBaseInput,
    KBSearchKnowledgeBaseOutput,
    SearchHit,
)

TOOL_SEARCH_KNOWLEDGE_BASE = "kb_search_knowledge_base"


def _to_dict(item: Any) -> Dict[str, Any]:
    if item is None:
        return {}
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return item
    try:
        return vars(item)
    except TypeError:
        return {"value": item}


def _pick_text(raw: Dict[str, Any]) -> str:
    return (
        raw.get("text")
        or raw.get("content")
        or raw.get("chunk_text")
        or raw.get("body")
        or raw.get("preview")
        or ""
    )


def _pick_score(raw: Dict[str, Any]) -> Optional[float]:
    for key in ("score", "final_score", "rerank_score", "similarity", "distance"):
        value = raw.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _pick_chunk_id(raw: Dict[str, Any]) -> Optional[int]:
    value = raw.get("chunk_id") or raw.get("id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pick_document_id(raw: Dict[str, Any]) -> Optional[int]:
    value = raw.get("document_id") or raw.get("doc_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_hit(item: Any) -> SearchHit:
    raw = _to_dict(item)

    metadata = {
        "document_id": _pick_document_id(raw),
        "chunk_id": _pick_chunk_id(raw),
        "title": raw.get("title") or raw.get("document_title"),
        "chunk_index": raw.get("chunk_index"),
        "chunk_type": raw.get("chunk_type"),
        "section_path": raw.get("section_path"),
        "section_title": raw.get("section_title"),
        "page_start": raw.get("page_start"),
        "page_end": raw.get("page_end"),
        "token_count": raw.get("token_count"),
        "block_start_index": raw.get("block_start_index"),
        "block_end_index": raw.get("block_end_index"),
        "source_type": raw.get("source_type"),
        "file_type": raw.get("file_type"),
    }

    metadata = {k: v for k, v in metadata.items() if v is not None}

    return SearchHit(
        chunk_id=_pick_chunk_id(raw),
        document_id=_pick_document_id(raw),
        title=raw.get("title") or raw.get("document_title"),
        chunk_index=raw.get("chunk_index"),
        chunk_type=raw.get("chunk_type"),
        section_path=raw.get("section_path"),
        section_title=raw.get("section_title"),
        page_start=raw.get("page_start"),
        page_end=raw.get("page_end"),
        score=_pick_score(raw),
        text=_pick_text(raw),
        preview=raw.get("preview") or _pick_text(raw)[:300],
        metadata=metadata,
    )


def kb_search_knowledge_base(
    input_data: KBSearchKnowledgeBaseInput | Dict[str, Any],
) -> Dict[str, Any]:
    payload = (
        input_data
        if isinstance(input_data, KBSearchKnowledgeBaseInput)
        else KBSearchKnowledgeBaseInput(**input_data)
    )

    def _execute() -> Dict[str, Any]:
        # 兼容 retrieve_chunks 的不同返回风格：
        # 1) 直接返回 list
        # 2) 返回 {"items": [...]} / {"chunks": [...]} / {"results": [...]}
        result = retrieve_chunks(
            query=payload.query,
            top_k=payload.top_k,
        )

        if isinstance(result, dict):
            raw_items = (
                result.get("items")
                or result.get("chunks")
                or result.get("results")
                or result.get("hits")
                or []
            )
        else:
            raw_items = result or []

        hits: List[SearchHit] = [_normalize_hit(item) for item in raw_items]

        if not payload.include_full_text:
            for hit in hits:
                if hit.text:
                    hit.text = hit.text[: payload.text_max_length]

        normalized = KBSearchKnowledgeBaseOutput(
            query=payload.query,
            top_k=payload.top_k,
            count=len(hits),
            hits=hits,
        )
        return normalized.model_dump()

    return run_tool(TOOL_SEARCH_KNOWLEDGE_BASE, _execute)