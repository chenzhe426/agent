"""
V3 Multi-Stage Retrieval: section grouping, section-level retrieval, chunk retrieval within sections.
"""
from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any

from app.services.common import to_float

from .config import (
    RETRIEVAL_CANDIDATE_SECTIONS,
    RETRIEVAL_CHUNKS_PER_SECTION,
    RETRIEVAL_HYBRID_W_DENSE,
    RETRIEVAL_HYBRID_W_LEXICAL,
    RETRIEVAL_PAGE_CLUSTER_ALPHA,
    RETRIEVAL_SECTION_PAGE_WINDOW,
    RETRIEVAL_SECTION_RELEVANCE_BOOST,
    V4_ENABLE_LLM_RERANK,
)
from .recall import (
    _hydrate_candidates,
    _keyword_recall_from_candidates,
    _lexical_recall_from_db,
    _merge_recall_candidates,
    _normalize_scores,
    _vector_recall_from_qdrant,
    cosine_similarity,
)
from .rerank import _rerank_hybrid_candidates
from .diversity import (
    _cap_page_duplicates,
    _deduplicate_candidates,
    _expand_neighbor_chunks,
)
from .signals import _smooth_page_scores
from app.services.reranker_service import rerank_with_llm
from app.services.llm_service import get_embedding


# Bounded LRU Cache for section embeddings
_SECTION_EMBEDDING_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_SECTION_EMBEDDING_CACHE_LOCK = Lock()
_SECTION_EMBEDDING_CACHE_MAX_SIZE = 512


def _group_chunks_into_sections(
    candidates: list[dict[str, Any]],
    page_window: int | None = None,
) -> list[dict[str, Any]]:
    """
    Group a list of chunk candidates into sections.

    A section is defined as:
      - Same document_id
      - Consecutive/nearby pages (within page_window)
      - Same section_path (optional grouping key)

    Each section is returned as a dict with:
      - section_id: unique string
      - doc_id: int
      - page_start, page_end: int
      - section_title: str
      - section_path: str
      - combined_text: str  (concatenated chunk texts)
      - chunk_ids: list[int]
      - chunk_count: int
      - avg_score: float  (average final_score across chunks)

    Returns sections sorted by avg_score descending.
    """
    from collections import Counter

    if page_window is None:
        page_window = RETRIEVAL_SECTION_PAGE_WINDOW

    if not candidates:
        return []

    # Group by doc_id
    by_doc: dict[int, list[dict[str, Any]]] = {}
    for cand in candidates:
        did = cand.get("document_id")
        if did is not None:
            try:
                did_int = int(did)
                if did_int not in by_doc:
                    by_doc[did_int] = []
                by_doc[did_int].append(cand)
            except (TypeError, ValueError):
                pass

    all_sections: list[dict[str, Any]] = []

    for doc_id, doc_cands in by_doc.items():
        # Sort by page_start
        doc_cands.sort(key=lambda x: to_float(x.get("page_start")) or 0)

        # Group into sections by page proximity
        current_group: list[dict[str, Any]] = []
        group_start_page = None

        def flush_group(group: list[dict[str, Any]], start_page: int | None) -> None:
            if not group:
                return
            titles = Counter(c.get("section_title", "") for c in group if c.get("section_title"))
            top_title = titles.most_common(1)[0][0] if titles else ""

            paths = Counter(c.get("section_path", "") for c in group if c.get("section_path"))
            top_path = paths.most_common(1)[0][0] if paths else ""

            pages = [c.get("page_start") for c in group if c.get("page_start") is not None]
            chunk_ids = [c.get("chunk_id") for c in group if c.get("chunk_id") is not None]
            scores = [to_float(c.get("final_score", 0.0)) for c in group]

            combined_text = "\n".join(
                c.get("search_text", "") or c.get("chunk_text", "") or ""
                for c in group
            )

            section = {
                "section_id": f"doc_{doc_id}_page_{start_page if start_page is not None else 0}",
                "doc_id": doc_id,
                "page_start": min(pages) if pages else 0,
                "page_end": max(pages) if pages else 0,
                "section_title": top_title,
                "section_path": top_path,
                "title": group[0].get("title", "") if group else "",
                "combined_text": combined_text,
                "chunk_ids": chunk_ids,
                "chunk_count": len(group),
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "gold_chunks": group,
            }
            all_sections.append(section)

        for cand in doc_cands:
            p = cand.get("page_start")
            if p is None:
                continue
            try:
                page = int(p)
            except (TypeError, ValueError):
                continue

            if current_group and group_start_page is not None:
                if page - group_start_page <= page_window:
                    current_group.append(cand)
                else:
                    flush_group(current_group, group_start_page)
                    current_group = [cand]
                    group_start_page = page
            else:
                current_group.append(cand)
                group_start_page = page

        flush_group(current_group, group_start_page)

    all_sections.sort(key=lambda x: x.get("avg_score", 0.0), reverse=True)
    return all_sections


def _score_sections_by_lexical(
    sections: list[dict[str, Any]],
    query_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Score sections by lexical match (keyword/BM25 overlap). Updates section['lexical_score'] in place."""
    terms = query_info.get("important_terms") or query_info.get("unique_terms") or []
    for section in sections:
        text = section.get("combined_text", "").lower()
        section_title = section.get("section_title", "").lower()
        section_path = section.get("section_path", "").lower()
        combined = (section_title + " " + section_path + " " + text).lower()

        hits = 0
        for term in terms:
            term_lower = term.lower()
            hits += combined.count(term_lower)

        coverage = hits / max(1, len(terms))
        lexical_score = min(1.0, coverage + hits * 0.02)
        section["lexical_score"] = lexical_score

    return sections


def _hybrid_normalize(sections: list[dict[str, Any]], score_key: str) -> None:
    """Min-max normalize a score field across sections."""
    vals = [to_float(s.get(score_key, 0.0)) for s in sections]
    max_val = max(vals) if vals else 1.0
    if max_val <= 0:
        max_val = 1.0
    for s in sections:
        s[score_key] = to_float(s.get(score_key, 0.0)) / max_val


def _get_section_embedding(section: dict[str, Any]) -> list[float]:
    """
    Get or compute embedding for a section's combined text.
    Uses contextual header text for the section.
    Uses a bounded LRU cache to avoid unbounded memory growth.
    """
    section_id = section.get("section_id", "")
    with _SECTION_EMBEDDING_CACHE_LOCK:
        if section_id in _SECTION_EMBEDDING_CACHE:
            _SECTION_EMBEDDING_CACHE.move_to_end(section_id)
            return _SECTION_EMBEDDING_CACHE[section_id]

    from .context import build_contextual_header

    header = build_contextual_header(
        doc_title=section.get("title", ""),
        page_start=section.get("page_start"),
        section_title=section.get("section_title", ""),
        section_path=section.get("section_path", ""),
    )
    text = section.get("combined_text", "")[:1500]
    full_text = f"{header}\n{text}"

    emb = get_embedding(full_text)

    with _SECTION_EMBEDDING_CACHE_LOCK:
        if section_id in _SECTION_EMBEDDING_CACHE:
            _SECTION_EMBEDDING_CACHE.move_to_end(section_id)
        else:
            if len(_SECTION_EMBEDDING_CACHE) >= _SECTION_EMBEDDING_CACHE_MAX_SIZE:
                _SECTION_EMBEDDING_CACHE.popitem(last=False)
            _SECTION_EMBEDDING_CACHE[section_id] = emb

    return emb


def _retrieve_sections_from_candidates(
    candidates: list[dict[str, Any]],
    query_info: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Given a set of chunk candidates (from initial retrieval),
    group them into sections and score each section by:
      1. Average embedding similarity (from initial chunk scores)
      2. Lexical keyword overlap
      3. Section-title relevance

    Returns top-k sections sorted by combined score.
    """
    if not candidates:
        return []

    sections = _group_chunks_into_sections(candidates, page_window=RETRIEVAL_SECTION_PAGE_WINDOW)

    if not sections:
        return []

    query_text = query_info.get("normalized_query", "")
    query_emb = get_embedding(query_text) if query_text else []
    query_emb_arr = query_emb if isinstance(query_emb, list) and query_emb else None

    section_emb_scores: dict[str, float] = {}
    for section in sections:
        section_emb = _get_section_embedding(section)
        score = 0.0
        if section_emb and query_emb_arr and len(section_emb) == len(query_emb_arr):
            score = cosine_similarity(section_emb, query_emb_arr)
        section_emb_scores[section["section_id"]] = score
        section["embedding_score"] = score

    if not any(section_emb_scores.values()):
        section_chunk_scores: dict[str, float] = {}
        for cand in candidates:
            section_key = f"doc_{cand.get('document_id')}_page_{cand.get('page_start')}"
            score = to_float(cand.get("final_score", 0.0)) or to_float(cand.get("embedding_score", 0.0))
            if section_key not in section_chunk_scores or score > section_chunk_scores[section_key]:
                section_chunk_scores[section_key] = score
        for section in sections:
            section["embedding_score"] = section_chunk_scores.get(section["section_id"], 0.0)

    _score_sections_by_lexical(sections, query_info)

    _hybrid_normalize(sections, "embedding_score")
    _hybrid_normalize(sections, "lexical_score")

    for section in sections:
        title_path = (
            section.get("section_title", "").lower() + " " +
            section.get("section_path", "").lower()
        )
        title_hits = sum(
            1 for kw in [
                "balance sheet", "income statement", "cash flow",
                "management discussion", "overview", "business",
                "risk factor", "notes to", "financial summary",
            ]
            if kw in title_path
        )
        section["title_boost"] = min(0.1, title_hits * 0.03)

    for section in sections:
        section["section_score"] = (
            RETRIEVAL_HYBRID_W_DENSE * section.get("embedding_score", 0.0)
            + RETRIEVAL_HYBRID_W_LEXICAL * section.get("lexical_score", 0.0)
            + section.get("title_boost", 0.0)
        )

    sections.sort(key=lambda x: x.get("section_score", 0.0), reverse=True)
    return sections[:top_k]


def _retrieve_chunks_within_sections(
    sections: list[dict[str, Any]],
    query_info: dict[str, Any],
    chunks_per_section: int = 5,
) -> list[dict[str, Any]]:
    """
    For each top section, retrieve the best chunks from within that section.
    Uses the original chunk candidates preserved in section['gold_chunks'].
    """
    selected_chunks: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for section in sections:
        section_chunks = section.get("gold_chunks", [])
        if not section_chunks:
            continue

        sorted_chunks = sorted(
            section_chunks,
            key=lambda x: to_float(x.get("final_score", 0.0)),
            reverse=True,
        )

        section_selected = 0
        for chunk in sorted_chunks:
            cid = chunk.get("chunk_id")
            if cid is None or cid in seen_ids:
                continue

            chunk["section_score"] = section.get("section_score", 0.0)
            chunk["section_page_start"] = section.get("page_start")
            chunk["section_page_end"] = section.get("page_end")
            chunk["section_title"] = section.get("section_title", "")
            chunk["_from_section_retrieval"] = True

            selected_chunks.append(chunk)
            seen_ids.add(cid)
            section_selected += 1

            if section_selected >= chunks_per_section:
                break

    return selected_chunks


def _rerank_with_section_relevance(
    candidates: list[dict[str, Any]],
    query_info: dict[str, Any],
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Re-score chunks by combining their existing scores with section relevance.
    Sections that are highly ranked contribute a bonus to their constituent chunks.
    """
    if not sections:
        return candidates

    section_scores: dict[str, float] = {
        s["section_id"]: s.get("section_score", 0.0)
        for s in sections
    }

    for cand in candidates:
        doc_id = cand.get("document_id")
        page = cand.get("page_start")
        if doc_id is not None and page is not None:
            section_key = f"doc_{doc_id}_page_{page}"
            sec_score = section_scores.get(section_key, 0.0)
            if sec_score > 0:
                section_bonus = RETRIEVAL_SECTION_RELEVANCE_BOOST * sec_score
                cand["final_score"] = to_float(cand.get("final_score", 0.0)) + section_bonus
                cand["_section_relevance_bonus"] = round(section_bonus, 6)

    candidates.sort(key=lambda x: to_float(x.get("final_score", 0.0)), reverse=True)
    return candidates


def retrieve_chunks_multistage(
    query: str,
    top_k: int = 5,
    enhanced_query: str | None = None,
) -> list[dict[str, Any]]:
    """
    V3 Multi-stage retrieval pipeline:

      1. Query understanding (classify + rewrite)
      2. Initial candidate retrieval (vector + lexical + keyword)
      3. Section-level retrieval & scoring
      4. Chunk retrieval within top sections
      5. Multi-signal rerank with section relevance
      6. Deduplication + page clustering

    Returns the same chunk dict format as retrieve_chunks(), with additional
    v3-specific fields (section_score, _from_section_retrieval, etc.).
    """
    from app.retrieval.config import (
        DEFAULT_TOP_K,
        HYBRID_HYDRATE_TOP_K,
        HYBRID_KEYWORD_TOP_K,
        HYBRID_VECTOR_TOP_K,
    )
    from app.retrieval.recall import _secondary_financial_recall
    from app.retrieval.query_understanding import _normalize_query, rewrite_query

    rewrite_result = rewrite_query(query)
    retrieval_query = enhanced_query if enhanced_query else rewrite_result["rewritten_query"]
    intent = rewrite_result["intent"]
    query_info = _normalize_query(retrieval_query)
    if not query_info.get("normalized_query"):
        return []

    candidate_top_k = max(top_k, 20)

    lexical_hits = _lexical_recall_from_db(
        query_info=query_info,
        top_k=max(candidate_top_k, HYBRID_HYDRATE_TOP_K),
    )
    vector_hits = _vector_recall_from_qdrant(
        query_info=query_info,
        top_k=max(candidate_top_k, HYBRID_VECTOR_TOP_K),
    )
    if not lexical_hits and not vector_hits:
        return []

    seed_candidates = _merge_recall_candidates(lexical_hits, vector_hits)

    keyword_hits = _keyword_recall_from_candidates(
        query_info=query_info,
        candidates=seed_candidates,
        top_k=max(candidate_top_k, HYBRID_KEYWORD_TOP_K),
    )
    merged = _merge_recall_candidates(seed_candidates, keyword_hits)

    secondary_hits = _secondary_financial_recall(
        query_info=query_info,
        top_k=max(candidate_top_k, 60),
    )
    if secondary_hits:
        merged = _merge_recall_candidates(merged, secondary_hits)

    reranked_initial = _rerank_hybrid_candidates(
        merged,
        query_info=query_info,
        enable_page_diversity=True,
        enable_page_clustering=False,
    )

    scored_sections = _retrieve_sections_from_candidates(
        reranked_initial[:candidate_top_k],
        query_info=query_info,
        top_k=RETRIEVAL_CANDIDATE_SECTIONS,
    )

    section_doc_pages = set()
    for sec in scored_sections:
        section_doc_pages.add((sec["doc_id"], sec["page_start"]))

    section_aware_candidates: list[dict[str, Any]] = []
    for cand in reranked_initial:
        key = (cand.get("document_id"), cand.get("page_start"))
        if key in section_doc_pages:
            section_aware_candidates.append(cand)

    for cand in reranked_initial[:5]:
        if cand not in section_aware_candidates:
            section_aware_candidates.append(cand)

    reranked = _rerank_with_section_relevance(
        section_aware_candidates,
        query_info=query_info,
        sections=scored_sections,
    )

    primary = _deduplicate_candidates(reranked, top_k=candidate_top_k)
    expanded = _expand_neighbor_chunks(
        primary,
        target_limit=max(candidate_top_k, len(primary) + 2),
    )
    expanded = _cap_page_duplicates(expanded, top_k=candidate_top_k)
    expanded.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    page_cluster_scores = _smooth_page_scores(expanded, alpha=RETRIEVAL_PAGE_CLUSTER_ALPHA)
    if page_cluster_scores:
        for cand in expanded:
            p = cand.get("page_start")
            if p is not None:
                try:
                    cluster_bonus = page_cluster_scores.get(int(p), 0.0) - to_float(cand.get("final_score", 0.0))
                    if cluster_bonus > 0.01:
                        cand["final_score"] += cluster_bonus
                        cand["_page_cluster_bonus"] = round(cluster_bonus, 6)
                except (TypeError, ValueError):
                    pass
        expanded.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    if V4_ENABLE_LLM_RERANK and len(expanded) > 1:
        try:
            reranked = rerank_with_llm(
                query=query,
                intent=intent,
                candidates=expanded,
                top_n=None,
            )
            if reranked and any(c.get("llm_rerank_applied", False) for c in reranked):
                expanded = reranked
        except Exception:
            pass

    final_hits = expanded[:top_k]

    results: list[dict[str, Any]] = []
    for item in final_hits:
        results.append({
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
            "_section_relevance_bonus": round(to_float(item.get("_section_relevance_bonus", 0.0)), 6),
            "_query_intent": item.get("_query_intent", intent),
            "_from_section_retrieval": bool(item.get("_from_section_retrieval", False)),
            "_section_score": round(to_float(item.get("section_score", 0.0)), 6),
            "_rewrite_intent": intent,
            "llm_relevance_score": round(to_float(item.get("llm_relevance_score", 0.0)), 6),
            "llm_rationale": str(item.get("llm_rationale", "")),
            "llm_combined_score": round(to_float(item.get("llm_combined_score", item.get("final_score", 0.0))), 6),
            "llm_rerank_applied": bool(item.get("llm_rerank_applied", False)),
        })

    return results
