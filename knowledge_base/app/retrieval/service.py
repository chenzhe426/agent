"""
Retrieval service: public API (retrieve_chunks) and two-stage pipeline.
"""
from __future__ import annotations

import time
from typing import Any

from .config import DEFAULT_TOP_K


def retrieve_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    enhanced_query: str | None = None,
    use_multistage: bool = False,
    doc_filter: int | None = None,
) -> dict[str, Any]:
    """
    Two-stage retrieval: fetch large candidate pool → rerank → return top_k.
    """
    # Local imports to avoid circular dependency chain:
    # app/retrieval/__init__.py → query_understanding → app.services.common →
    # app.services.__init__ → qa_service → retrieval_service → service → multistage
    from .config import (
        HYBRID_HYDRATE_TOP_K,
        HYBRID_KEYWORD_TOP_K,
        HYBRID_VECTOR_TOP_K,
        RETRIEVAL_USE_MULTISTAGE,
    )
    from app.services.common import to_float
    from .diversity import (
        _cap_page_duplicates,
        _deduplicate_candidates,
        _expand_neighbor_chunks,
    )
    from .query_understanding import _normalize_query
    from .recall import (
        _keyword_recall_from_candidates,
        _lexical_recall_from_db,
        _merge_recall_candidates,
        _secondary_financial_recall,
        _vector_recall_from_qdrant,
    )
    from .rerank import _rerank_hybrid_candidates

    if use_multistage or RETRIEVAL_USE_MULTISTAGE:
        from .multistage import retrieve_chunks_multistage
        result = retrieve_chunks_multistage(
            query=query,
            top_k=top_k,
            enhanced_query=enhanced_query,
        )
        # multistage returns list, wrap it
        return {"chunks": result, "latency_ms": 0.0, "stage_timings": {"multistage": 0.0}}

    retrieval_query = enhanced_query if enhanced_query else query

    t0 = time.perf_counter()
    t_normalize = t0
    query_info = _normalize_query(retrieval_query)
    t_normalize_end = time.perf_counter()
    if not query_info.get("normalized_query"):
        return {"chunks": [], "latency_ms": 0.0, "stage_timings": {}}

    candidate_top_k = max(top_k, 20)

    # Parallelize lexical and vector recall (both I/O bound)
    from concurrent.futures import ThreadPoolExecutor
    t_recall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        lexical_future = pool.submit(
            _lexical_recall_from_db,
            query_info,
            max(candidate_top_k, HYBRID_HYDRATE_TOP_K),
        )
        vector_future = pool.submit(
            _vector_recall_from_qdrant,
            query_info,
            max(candidate_top_k, HYBRID_VECTOR_TOP_K),
            doc_filter,
        )
        lexical_hits = lexical_future.result()
        vector_hits = vector_future.result()
    t_recall_end = time.perf_counter()

    if not lexical_hits and not vector_hits:
        return {
            "chunks": [],
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "stage_timings": {
                "normalize_query": round((t_normalize_end - t_normalize) * 1000, 2),
                "parallel_recall": round((t_recall_end - t_recall_start) * 1000, 2),
            },
        }

    seed_candidates = _merge_recall_candidates(lexical_hits, vector_hits)

    t_keyword = time.perf_counter()
    keyword_hits = _keyword_recall_from_candidates(
        query_info=query_info,
        candidates=seed_candidates,
        top_k=max(candidate_top_k, HYBRID_KEYWORD_TOP_K),
    )
    t_keyword_end = time.perf_counter()

    merged = _merge_recall_candidates(seed_candidates, keyword_hits)

    t_secondary = time.perf_counter()
    secondary_hits = _secondary_financial_recall(
        query_info=query_info,
        top_k=max(candidate_top_k, 60),
    )
    t_secondary_end = time.perf_counter()
    if secondary_hits:
        merged = _merge_recall_candidates(merged, secondary_hits)

    t_rerank = time.perf_counter()
    reranked = _rerank_hybrid_candidates(merged, query_info=query_info)
    t_rerank_end = time.perf_counter()

    # Post-filter: restrict to specific document (e.g., company filter from query)
    # Apply BEFORE deduplication so that deduplication respects per-document limits.
    if doc_filter is not None:
        reranked = [c for c in reranked if c.get("document_id") == doc_filter]

    primary = _deduplicate_candidates(reranked, top_k=candidate_top_k)

    t_diversity = time.perf_counter()
    expanded = _expand_neighbor_chunks(
        primary,
        target_limit=max(candidate_top_k, len(primary) + 2),
    )
    expanded = _cap_page_duplicates(expanded, top_k=candidate_top_k)
    t_diversity_end = time.perf_counter()

    expanded.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    final_hits = expanded[:top_k]

    results: list[dict[str, Any]] = []
    for item in final_hits:
        results.append(
            {
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "chunk_index": item.get("chunk_index"),
                "score": round(to_float(item.get("final_score")), 6),
                "rerank_score": round(to_float(item.get("final_score")), 6),
                "embedding_score": round(to_float(item.get("embedding_score")), 6),
                "keyword_score": round(to_float(item.get("keyword_score")), 6),
                "bm25_score": round(to_float(item.get("bm25_score")), 6),
                "title_match_score": round(to_float(item.get("title_match_score")), 6),
                "section_match_score": round(to_float(item.get("section_match_score")), 6),
                "coverage_score": round(to_float(item.get("coverage_score")), 6),
                "matched_term_count": int(to_float(item.get("matched_term_count"))),
                "title": item.get("title", ""),
                "section_path": item.get("section_path", ""),
                "section_title": item.get("section_title", ""),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "chunk_type": item.get("chunk_type"),
                "chunk_text": item.get("chunk_text", ""),
                "search_text": item.get("search_text", ""),
                "lexical_text": item.get("lexical_text", ""),
                "term_hits": item.get("term_hits", {}),
                "term_hit_detail": item.get("term_hit_detail", {}),
                "is_neighbor": bool(item.get("is_neighbor", False)),
                "_retrieval_query": retrieval_query if retrieval_query != query else None,
                "_numeric_boost": round(to_float(item.get("_numeric_boost", 0.0)), 6),
                "_table_boost": round(to_float(item.get("_table_boost", 0.0)), 6),
                "_query_aware_boost": round(to_float(item.get("_query_aware_boost", 0.0)), 6),
                "_anti_noise_penalty": round(to_float(item.get("_anti_noise_penalty", 0.0)), 6),
                "_page_cluster_bonus": round(to_float(item.get("_page_cluster_bonus", 0.0)), 6),
                "_query_intent": item.get("_query_intent", "unknown"),
            }
        )

    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "chunks": results,
        "latency_ms": round(total_ms, 2),
        "stage_timings": {
            "normalize_query": round((t_normalize_end - t_normalize) * 1000, 2),
            "parallel_recall": round((t_recall_end - t_recall_start) * 1000, 2),
            "keyword_recall": round((t_keyword_end - t_keyword) * 1000, 2),
            "secondary_recall": round((t_secondary_end - t_secondary) * 1000, 2),
            "rerank": round((t_rerank_end - t_rerank) * 1000, 2),
            "diversity_dedup": round((t_diversity_end - t_diversity) * 1000, 2),
        },
    }
