"""
Recall: vector search, keyword search, lexical search, candidate merging.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from app.db import (
    get_chunks_by_ids,
    get_neighbor_chunks,
    search_chunks_boolean,
    search_chunks_fulltext,
)
from app.retrieval._common import normalize_embedding, normalize_whitespace, safe_get, to_float

from .config import (
    HYBRID_BOOLEAN_FETCH_K,
    HYBRID_LEXICAL_FETCH_K,
    _NUMERIC_FINANCE_PATTERNS,
    _NUMERIC_FACT_PATTER,
    _LIST_FACT_PATTER,
    _DESCRIPTIVE_PATTER,
    KEYWORD_EXACT_MATCH_WEIGHT,
    KEYWORD_SUBSTRING_MATCH_WEIGHT,
    RETRIEVAL_WEIGHT_BM25,
    RETRIEVAL_WEIGHT_EMBEDDING,
    RETRIEVAL_WEIGHT_KEYWORD,
    RETRIEVAL_WEIGHT_SECTION,
    RETRIEVAL_WEIGHT_TITLE,
    SECTION_MATCH_WEIGHT,
    TITLE_MATCH_WEIGHT,
)


def _safe_section_path(value: Any) -> str:
    if isinstance(value, list):
        return " / ".join(str(x).strip() for x in value if str(x).strip())
    if isinstance(value, dict):
        return normalize_whitespace(" ".join(str(v) for v in value.values()))
    return normalize_whitespace(str(value or ""))


def _row_to_candidate(row: Any) -> dict[str, Any]:
    metadata = safe_get(row, "metadata") or {}
    raw_section_path = safe_get(row, "section_path", "")

    lexical_text = normalize_whitespace(
        safe_get(row, "lexical_text", "")
        or safe_get(row, "search_text", "")
        or safe_get(row, "chunk_text", "")
    )
    search_text = normalize_whitespace(
        safe_get(row, "search_text", "")
        or safe_get(row, "chunk_text", "")
    )

    return {
        "chunk_id": safe_get(row, "id"),
        "document_id": safe_get(row, "document_id"),
        "chunk_index": safe_get(row, "chunk_index"),
        "chunk_text": normalize_whitespace(safe_get(row, "chunk_text", "")),
        "search_text": search_text,
        "lexical_text": lexical_text,
        "embedding": normalize_embedding(safe_get(row, "embedding")),
        "title": normalize_whitespace(
            safe_get(row, "title")
            or metadata.get("title", "")
        ),
        "section_path": _safe_section_path(raw_section_path),
        "section_title": normalize_whitespace(
            safe_get(row, "section_title")
            or metadata.get("section_title", "")
        ),
        "page_start": safe_get(row, "page_start"),
        "page_end": safe_get(row, "page_end"),
        "block_start_index": safe_get(row, "block_start_index"),
        "block_end_index": safe_get(row, "block_end_index"),
        "chunk_type": safe_get(row, "chunk_type"),
        "metadata": metadata,
        "embedding_score": 0.0,
        "keyword_score": 0.0,
        "bm25_score": 0.0,
        "title_match_score": 0.0,
        "section_match_score": 0.0,
        "coverage_score": 0.0,
        "matched_term_count": 0,
        "lexical_db_score": to_float(safe_get(row, "lexical_score")),
        "final_score": 0.0,
        "term_hits": {},
        "term_hit_detail": {},
        "is_neighbor": False,
    }


def _collect_row_texts(cand: dict[str, Any]) -> dict[str, str]:
    title = normalize_whitespace(cand.get("title", ""))
    section_title = normalize_whitespace(cand.get("section_title", ""))
    section_path = normalize_whitespace(cand.get("section_path", ""))
    search_text = normalize_whitespace(cand.get("search_text", ""))
    lexical_text = normalize_whitespace(cand.get("lexical_text", ""))

    return {
        "title": title,
        "section_title": section_title,
        "section_path": section_path,
        "section_text": normalize_whitespace(" ".join([section_path, section_title])),
        "search_text": search_text,
        "lexical_text": lexical_text,
    }


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = (sum(x * x for x in a)) ** 0.5
    nb = (sum(y * y for y in b)) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _term_occurrence_detail(term: str, text: str) -> dict[str, Any]:
    from .query_understanding import _contains_cjk
    if not term or not text:
        return {"score": 0.0, "exact_count": 0, "substring_count": 0, "whole_word": False, "prefix_hit": False}

    normalized_text = text or ""
    lower_text = normalized_text.lower()
    lower_term = term.lower()

    exact_count = normalized_text.count(term)
    substring_count = lower_text.count(lower_term)

    whole_word = False
    prefix_hit = False

    if _contains_cjk(term):
        whole_word = exact_count > 0 or substring_count > 0
    else:
        import re
        word_pat = re.compile(rf"(?<![a-z0-9_]){re.escape(lower_term)}(?![a-z0-9_])")
        whole_word = bool(word_pat.search(lower_text))
        prefix_hit = bool(re.search(rf"(^|[\s:/._-]){re.escape(lower_term)}", lower_text))

    return {
        "score": float(whole_word) * KEYWORD_EXACT_MATCH_WEIGHT
        + float(prefix_hit) * KEYWORD_SUBSTRING_MATCH_WEIGHT * 0.7,
        "exact_count": exact_count,
        "substring_count": substring_count,
        "whole_word": whole_word,
        "prefix_hit": prefix_hit,
    }


def _normalize_scores(items: list[dict[str, Any]], field: str) -> None:
    max_score = max((to_float(item.get(field)) for item in items), default=0.0)
    if max_score <= 0:
        return
    for item in items:
        item[field] = to_float(item.get(field)) / max_score


def _hydrate_candidates(raw_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not raw_hits:
        return []

    by_id: dict[int, dict[str, Any]] = {}
    ordered_ids: list[int] = []

    for item in raw_hits:
        chunk_id = safe_get(item, "id")
        if chunk_id is None:
            chunk_id = safe_get(item, "chunk_id")
        if chunk_id is None:
            continue
        chunk_id = int(chunk_id)
        if chunk_id not in by_id:
            by_id[chunk_id] = dict(item)
            ordered_ids.append(chunk_id)
        else:
            existing = by_id[chunk_id]
            existing["lexical_score"] = max(
                to_float(existing.get("lexical_score")),
                to_float(item.get("lexical_score")),
            )

    hydrated_rows = get_chunks_by_ids(ordered_ids) or []
    row_map = {int(row["id"]): row for row in hydrated_rows if safe_get(row, "id") is not None}

    results: list[dict[str, Any]] = []
    for chunk_id in ordered_ids:
        row = row_map.get(chunk_id)
        if not row:
            continue
        cand = _row_to_candidate(row)
        extra = by_id.get(chunk_id, {})
        cand["lexical_db_score"] = max(
            to_float(cand.get("lexical_db_score")),
            to_float(extra.get("lexical_score")),
        )
        cand["bm25_score"] = to_float(cand.get("lexical_db_score"))
        results.append(cand)

    return results


def _compute_keyword_components(query_info: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    """Compute keyword match scores for a candidate chunk."""
    texts = _collect_row_texts(cand)
    title_text = texts["title"].lower()
    section_text = texts["section_text"].lower()
    search_text = texts["search_text"].lower()

    terms = query_info.get("important_terms") or query_info.get("unique_terms") or []

    title_hits = 0
    section_hits = 0
    term_hits: dict[str, int] = {}
    term_hit_detail: dict[str, Any] = {}

    for term in terms:
        detail = _term_occurrence_detail(term, texts["search_text"])
        if detail.get("whole_word"):
            term_hits[term] = term_hits.get(term, 0) + 1
            term_hit_detail[term] = detail

        for text_to_check, weight in [
            (title_text, TITLE_MATCH_WEIGHT),
            (section_text, SECTION_MATCH_WEIGHT),
        ]:
            detail = _term_occurrence_detail(term, text_to_check)
            if detail.get("score", 0) >= KEYWORD_EXACT_MATCH_WEIGHT * 0.5:
                if term in title_text:
                    title_hits += 1
                if term in section_text:
                    section_hits += 1

    keyword_score = sum(
        min(1.0, term_hits.get(t, 0) / 2) * 1.0 for t in terms
    ) / max(1, len(terms))

    title_match_score = float(title_hits >= 2) if len(terms) >= 2 else float(any(t in title_text for t in terms))
    section_match_score = float(section_hits >= 2) if len(terms) >= 2 else float(any(t in section_text for t in terms))

    matched_count = len(term_hits)
    coverage_score = matched_count / max(1, len(terms))

    return {
        "keyword_score": keyword_score,
        "title_match_score": title_match_score,
        "section_match_score": section_match_score,
        "coverage_score": coverage_score,
        "matched_term_count": matched_count,
        "term_hits": term_hits,
        "term_hit_detail": term_hit_detail,
    }


def _vector_recall_from_qdrant(
    query_info: dict[str, Any],
    top_k: int,
    doc_filter: int | None = None,
) -> list[dict[str, Any]]:
    from app.services.vector_store import vector_store
    query_text = query_info.get("normalized_query", "")
    if not query_text or top_k <= 0:
        return []

    search_filters = {"document_id": doc_filter} if doc_filter is not None else None
    hits = vector_store.search(query_text, top_k=top_k, filters=search_filters)
    if not hits:
        return []

    chunk_ids: list[int] = []
    for item in hits:
        chunk_id = item.get("chunk_id")
        if chunk_id is None:
            continue
        try:
            chunk_ids.append(int(chunk_id))
        except (TypeError, ValueError):
            continue

    if not chunk_ids:
        return []

    hydrated_rows = get_chunks_by_ids(chunk_ids) or []
    row_map = {int(row["id"]): row for row in hydrated_rows if safe_get(row, "id") is not None}

    results: list[dict[str, Any]] = []
    for hit in hits:
        chunk_id = hit.get("chunk_id")
        if chunk_id is None:
            continue
        try:
            chunk_id = int(chunk_id)
        except (TypeError, ValueError):
            continue

        row = row_map.get(chunk_id)
        if not row:
            continue

        cand = _row_to_candidate(row)
        cand["embedding_score"] = to_float(hit.get("embedding_score"))
        cand["final_score"] = cand["embedding_score"]

        if hit.get("document_id") is not None:
            cand["document_id"] = hit.get("document_id")
        if hit.get("chunk_index") is not None:
            cand["chunk_index"] = hit.get("chunk_index")

        results.append(cand)

    _normalize_scores(results, "embedding_score")
    for item in results:
        item["final_score"] = to_float(item.get("embedding_score"))

    results.sort(key=lambda x: x.get("embedding_score", 0.0), reverse=True)
    return results[:top_k]


def _keyword_recall_from_candidates(
    query_info: dict[str, Any],
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for cand in candidates:
        item = dict(cand)
        item.update(_compute_keyword_components(query_info, item))
        lexical_score = (
            0.60 * to_float(item.get("keyword_score"))
            + 0.23 * to_float(item.get("title_match_score"))
            + 0.17 * to_float(item.get("section_match_score"))
        )
        item["final_score"] = lexical_score
        has_any_lexical = (
            to_float(item.get("keyword_score")) > 0
            or to_float(item.get("coverage_score")) > 0
            or to_float(item.get("lexical_db_score")) > 0
        )
        if has_any_lexical:
            results.append(item)

    results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return results[:top_k]


def _lexical_recall_from_db(
    query_info: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    query_text = query_info.get("normalized_query", "")
    if not query_text:
        return []

    natural_hits = search_chunks_fulltext(
        query_text,
        limit=max(top_k, HYBRID_LEXICAL_FETCH_K),
    ) or []

    boolean_hits = search_chunks_boolean(
        query_text,
        limit=max(top_k, HYBRID_BOOLEAN_FETCH_K),
        require_all_terms=False,
    ) or []

    merged_raw: dict[int, dict[str, Any]] = {}
    for row in [*natural_hits, *boolean_hits]:
        chunk_id = safe_get(row, "id")
        if chunk_id is None:
            continue
        chunk_id = int(chunk_id)
        if chunk_id not in merged_raw:
            merged_raw[chunk_id] = dict(row)
        else:
            merged_raw[chunk_id]["lexical_score"] = max(
                to_float(merged_raw[chunk_id].get("lexical_score")),
                to_float(row.get("lexical_score")),
            )

    hydrated = _hydrate_candidates(list(merged_raw.values()))
    _normalize_scores(hydrated, "lexical_db_score")

    for cand in hydrated:
        cand["bm25_score"] = to_float(cand.get("lexical_db_score"))

    hydrated.sort(key=lambda x: x.get("bm25_score", 0.0), reverse=True)
    return hydrated[:top_k]


def _merge_recall_candidates(*candidate_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[Any, dict[str, Any]] = {}

    def merge_one(item: dict[str, Any]) -> None:
        chunk_id = item.get("chunk_id")
        if chunk_id not in merged:
            merged[chunk_id] = dict(item)
            return

        current = merged[chunk_id]

        for field in [
            "embedding_score",
            "keyword_score",
            "bm25_score",
            "lexical_db_score",
            "title_match_score",
            "section_match_score",
            "coverage_score",
            "matched_term_count",
            "final_score",
        ]:
            current[field] = max(to_float(current.get(field)), to_float(item.get(field)))

        if item.get("term_hits"):
            existing_hits = Counter(current.get("term_hits") or {})
            existing_hits.update(item.get("term_hits") or {})
            current["term_hits"] = dict(existing_hits)

        if item.get("term_hit_detail"):
            details = dict(current.get("term_hit_detail") or {})
            details.update(item.get("term_hit_detail") or {})
            current["term_hit_detail"] = details

    for group in candidate_groups:
        for item in group:
            merge_one(item)

    return list(merged.values())


def _secondary_financial_recall(
    query_info: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Perform a secondary lexical recall specifically for financial statement content.
    Only activated when _detect_financial_query returns True.
    """
    from app.db import search_chunks_boolean, search_chunks_fulltext
    from app.services.common import safe_get
    from app.retrieval.query_understanding import (
        _detect_financial_query,
        _build_expanded_query,
    )

    if not _detect_financial_query(query_info):
        return []

    expanded_query = _build_expanded_query(query_info)
    if not expanded_query.strip():
        return []

    financial_hits = search_chunks_fulltext(
        expanded_query,
        limit=max(top_k, 60),
    ) or []

    boolean_hits = search_chunks_boolean(
        expanded_query,
        limit=max(top_k, 40),
        require_all_terms=False,
    ) or []

    # For ratio queries (quick ratio, current ratio, etc.), also specifically
    # search for chunks containing "total current liabilities" as these are
    # critical for ratio calculations but may not match query terms
    query_lower = query_info.get("normalized_query", "").lower()
    ratio_hits: list[dict[str, Any]] = []
    if any(ratio in query_lower for ratio in ["quick ratio", "current ratio", "liquidity"]):
        ratio_hits = search_chunks_fulltext(
            "total current liabilities",
            limit=20,
        ) or []

    seen_ids: set[int] = set()
    merged: list[dict[str, Any]] = []
    for row in [*financial_hits, *boolean_hits, *ratio_hits]:
        chunk_id = safe_get(row, "id")
        if chunk_id is None:
            continue
        chunk_id = int(chunk_id)
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            merged.append(dict(row))

    candidates = _hydrate_candidates(merged)
    _normalize_scores(candidates, "lexical_db_score")
    for cand in candidates:
        cand["bm25_score"] = to_float(cand.get("lexical_db_score"))

    candidates.sort(key=lambda x: x.get("bm25_score", 0.0), reverse=True)

    # For ratio queries, ensure we return enough candidates to include the ratio_hits
    # which may have low lexical scores but are critical for ratio calculations
    query_lower = query_info.get("normalized_query", "").lower()
    is_ratio_query = any(ratio in query_lower for ratio in ["quick ratio", "current ratio", "liquidity"])
    if is_ratio_query and ratio_hits:
        # Return up to 2x to ensure ratio_hits chunks are included
        return candidates[:top_k * 2]

    return candidates[:top_k]
