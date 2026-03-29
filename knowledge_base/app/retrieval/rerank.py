"""
Hybrid reranking: score fusion with all signals + page clustering + diversity.
"""
from __future__ import annotations

from typing import Any

from .config import (
    RETRIEVAL_WEIGHT_BM25,
    RETRIEVAL_WEIGHT_EMBEDDING,
    RETRIEVAL_WEIGHT_KEYWORD,
    RETRIEVAL_WEIGHT_SECTION,
    RETRIEVAL_WEIGHT_TITLE,
    RETRIEVAL_RATIO_EMBEDDING_WEIGHT,
    RETRIEVAL_RATIO_KEYWORD_WEIGHT,
    RETRIEVAL_RATIO_BM25_WEIGHT,
    RETRIEVAL_RATIO_TITLE_WEIGHT,
    RETRIEVAL_RATIO_SECTION_WEIGHT,
)
from .recall import _compute_keyword_components
from .signals import (
    _compute_anti_noise_penalty,
    _compute_financial_content_bonus,
    _compute_financial_ratio_component_boost,
    _compute_page_diversity_bonus,
    _compute_query_aware_lexical_boost,
    _compute_numeric_density_boost,
    _compute_table_like_boost,
    _length_penalty,
    _metadata_bonus,
    _section_narrative_bonus,
    _smooth_page_scores,
)


def _rerank_hybrid_candidates(
    candidates: list[dict[str, Any]],
    query_info: dict[str, Any],
    enable_page_diversity: bool = True,
    enable_page_clustering: bool = True,
) -> list[dict[str, Any]]:
    from app.services.common import to_float
    from .query_understanding import classify_query_intent
    """
    Rerank hybrid candidates using weighted signals + new rerank signals.

    New signals (P0):
      - numeric_density_boost   : chunk has financial numbers/density
      - table_like_boost        : chunk looks like a financial table
      - query_aware_lexical_boost: intent-based lexical alignment
      - anti_noise_penalty       : penalize TOC/risk-factor/filler chunks
      - page_clustering          : neighborhood smoothing for page scores
    """
    if not candidates:
        return []

    raw_query = query_info.get("raw_query", "")
    intent = classify_query_intent(raw_query)

    reranked: list[dict[str, Any]] = []
    query_terms = query_info.get("important_terms") or query_info.get("unique_terms") or []

    # Compute page counts across ALL candidates (before reranking)
    all_page_counts: dict[int, int] = {}
    for cand in candidates:
        p = cand.get("page_start")
        if p is not None:
            try:
                page = int(p)
                all_page_counts[page] = all_page_counts.get(page, 0) + 1
            except (TypeError, ValueError):
                pass

    page_diversity_bonus: dict[int, float] = {}
    if enable_page_diversity and len(all_page_counts) > 1:
        page_diversity_bonus = _compute_page_diversity_bonus(candidates, all_page_counts)

    for cand in candidates:
        text = cand.get("search_text", "") or cand.get("chunk_text", "")

        embedding_score = to_float(cand.get("embedding_score"))
        keyword_score = to_float(cand.get("keyword_score"))
        bm25_score = to_float(cand.get("bm25_score"))
        title_score = to_float(cand.get("title_match_score"))
        section_score = to_float(cand.get("section_match_score"))
        coverage_score = to_float(cand.get("coverage_score"))

        has_keyword_scores = keyword_score > 0 or coverage_score > 0
        if not has_keyword_scores and bm25_score > 0 and text:
            kw_comps = _compute_keyword_components(query_info, cand)
            keyword_score = to_float(kw_comps.get("keyword_score"))
            title_score = to_float(kw_comps.get("title_match_score"))
            section_score = to_float(kw_comps.get("section_match_score"))
            coverage_score = to_float(kw_comps.get("coverage_score", 0))

        coverage_bonus = 0.08 * coverage_score if query_terms else 0.0

        # For numeric ratio queries (e.g., "quick ratio"), BM25 is less relevant
        # since the answer depends on balance-sheet component data, not term matching.
        query_lower = query_info.get("normalized_query", "").lower()
        is_ratio_query = any(
            ratio in query_lower for ratio in
            ["quick ratio", "current ratio", "liquidity", "working capital"]
        )
        is_ratio_numeric_query = intent == "numeric_fact" and is_ratio_query

        if is_ratio_numeric_query:
            # Ratio queries depend on lexical match of balance-sheet components
            # (current assets, current liabilities, etc.), not semantic similarity.
            embedding_weight = RETRIEVAL_RATIO_EMBEDDING_WEIGHT
            keyword_weight = RETRIEVAL_RATIO_KEYWORD_WEIGHT
            bm25_weight = RETRIEVAL_RATIO_BM25_WEIGHT
            title_weight = RETRIEVAL_RATIO_TITLE_WEIGHT
            section_weight = RETRIEVAL_RATIO_SECTION_WEIGHT
        elif len(query_terms) <= 2:
            embedding_weight = max(0.0, RETRIEVAL_WEIGHT_EMBEDDING - 0.05)
            keyword_weight = RETRIEVAL_WEIGHT_KEYWORD + 0.03
            bm25_weight = RETRIEVAL_WEIGHT_BM25 + 0.02
            title_weight = RETRIEVAL_WEIGHT_TITLE
            section_weight = RETRIEVAL_WEIGHT_SECTION
        else:
            embedding_weight = RETRIEVAL_WEIGHT_EMBEDDING
            keyword_weight = RETRIEVAL_WEIGHT_KEYWORD
            bm25_weight = RETRIEVAL_WEIGHT_BM25
            title_weight = RETRIEVAL_WEIGHT_TITLE
            section_weight = RETRIEVAL_WEIGHT_SECTION

        is_secondary_only = embedding_score == 0 and bm25_score > 0
        secondary_bonus = 0.0
        if is_secondary_only:
            keyword_weight = 0.35
            bm25_weight = 0.25
            secondary_bonus = 0.06

        p = cand.get("page_start")
        page_div = 0.0
        if p is not None:
            try:
                page_div = page_diversity_bonus.get(int(p), 0.0)
            except (TypeError, ValueError):
                pass

        financial_bonus = _compute_financial_content_bonus(cand, query_info)
        numeric_boost = _compute_numeric_density_boost(cand)
        table_boost = _compute_table_like_boost(cand)
        query_aware_boost = _compute_query_aware_lexical_boost(cand, query_info, intent)
        ratio_component_boost = _compute_financial_ratio_component_boost(cand, query_info)
        anti_noise_penalty = _compute_anti_noise_penalty(cand)

        base_score = (
            embedding_weight * embedding_score
            + keyword_weight * keyword_score
            + bm25_weight * bm25_score
            + title_weight * title_score
            + section_weight * section_score
            + coverage_bonus
            + _metadata_bonus(cand)
            + _section_narrative_bonus(cand)
            + page_div
            + secondary_bonus
            + financial_bonus
            + numeric_boost
            + table_boost
            + query_aware_boost
            + ratio_component_boost
            + anti_noise_penalty
            - _length_penalty(text)
        )

        item = dict(cand)
        item["final_score"] = base_score
        item["_numeric_boost"] = round(numeric_boost, 6)
        item["_table_boost"] = round(table_boost, 6)
        item["_query_aware_boost"] = round(query_aware_boost, 6)
        item["_anti_noise_penalty"] = round(anti_noise_penalty, 6)
        item["_query_intent"] = intent
        reranked.append(item)

    reranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    # Apply page clustering as a post-pass
    if enable_page_clustering and len(reranked) > 1:
        smoothed = _smooth_page_scores(reranked, score_key="final_score")
        if smoothed:
            best_per_page: dict[int, dict[str, Any]] = {}
            for cand in reranked:
                p = cand.get("page_start")
                if p is None:
                    continue
                try:
                    page = int(p)
                except (TypeError, ValueError):
                    continue
                if page not in best_per_page:
                    best_per_page[page] = cand
                elif cand.get("final_score", 0) > best_per_page[page].get("final_score", 0):
                    best_per_page[page] = cand

            for page, cand in best_per_page.items():
                cluster_bonus = smoothed.get(page, 0.0) - cand.get("final_score", 0.0)
                if cluster_bonus > 0.01:
                    cand["final_score"] += cluster_bonus
                    cand["_page_cluster_bonus"] = round(cluster_bonus, 6)

            reranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    return reranked
