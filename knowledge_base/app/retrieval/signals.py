"""
Retrieval scoring signals: numeric density, table-like, query-aware lexical, anti-noise, page diversity.
"""
from __future__ import annotations

import re
from typing import Any

from app.retrieval._common import safe_get, to_float

from .config import (
    RETRIEVAL_ANTI_NOISE_PENALTY_WEIGHT,
    RETRIEVAL_NUMERIC_BOOST_WEIGHT,
    RETRIEVAL_PAGE_CLUSTER_ALPHA,
    RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT,
    RETRIEVAL_TABLE_BOOST_WEIGHT,
    _CAUSAL_PHRASE_PATTERNS,
    _NUMERIC_FINANCE_PATTERNS,
    _NOISE_SECTION_PATTERNS,
    _TABLE_STRUCTURE_KEYWORDS,
)


def _length_penalty(text: str) -> float:
    text_len = len(text or "")
    if text_len < 30:
        return 0.05
    if text_len < 60:
        return 0.02
    if text_len > 1800:
        return 0.10
    if text_len > 1200:
        return 0.06
    return 0.0


# Mapping from financial ratio queries to their underlying balance-sheet components.
# When a query asks about a ratio, chunks containing the raw component data
# are highly relevant even if they don't mention the ratio term.
_FINANCIAL_RATIO_COMPONENTS: dict[str, list[str]] = {
    "quick ratio": [
        "cash equivalents", "cash and cash equivalents", "short-term investments",
        "accounts receivable", "receivables", "current liabilities",
    ],
    "current ratio": [
        "current assets", "current liabilities",
    ],
    "liquidity": [
        "current assets", "cash", "cash equivalents", "short-term investments",
        "accounts receivable", "receivables", "current liabilities",
    ],
    "working capital": [
        "current assets", "current liabilities",
    ],
}

# Denominator-only components: chunks with these alone (no numerator) are still critical for ratio queries
_FINANCIAL_RATIO_DENOMINATOR: dict[str, list[str]] = {
    "quick ratio": ["current liabilities", "total current liabilities"],
    "current ratio": ["current liabilities", "total current liabilities"],
    "liquidity": ["current liabilities", "total current liabilities"],
    "working capital": ["current liabilities", "total current liabilities"],
}


def _compute_financial_ratio_component_boost(
    cand: dict[str, Any],
    query_info: dict[str, Any],
) -> float:
    """
    When query asks about a financial ratio/metric (e.g., "quick ratio"),
    boost chunks that contain the raw balance-sheet components needed to compute it.
    This helps table data chunks rank higher for ratio queries.
    """
    query_lower = query_info.get("normalized_query", "").lower()

    # Check if query mentions any known financial ratio terms
    matched_ratio = None
    for ratio_term in _FINANCIAL_RATIO_COMPONENTS:
        if ratio_term in query_lower:
            matched_ratio = ratio_term
            break

    if not matched_ratio:
        return 0.0

    # Check if chunk contains the ratio components
    chunk_text = (cand.get("chunk_text", "") or "").lower()
    section_path = cand.get("section_path", "") or ""
    if isinstance(section_path, list):
        section_path = " ".join(section_path)
    section = ((cand.get("section_title", "") or "") + " " + section_path).lower()
    combined = chunk_text + " " + section

    components = _FINANCIAL_RATIO_COMPONENTS[matched_ratio]
    hit_count = sum(1 for comp in components if comp in combined)

    if hit_count >= 4:
        return 1.20  # Strong boost for chunks with most/all components
    elif hit_count >= 3:
        return 1.00
    elif hit_count >= 2:
        return 0.70
    elif hit_count == 1:
        # Single hit - check if it's a denominator-only boost
        # Only boost if chunk has "total current liabilities" specifically AND numeric table characteristics
        denom_components = _FINANCIAL_RATIO_DENOMINATOR.get(matched_ratio, [])
        denom_hit = any(comp in combined for comp in denom_components)
        if denom_hit and "total current liabilities" in combined:
            # Check if chunk looks like a financial table (has numeric density)
            chunk_text = (cand.get("chunk_text", "") or "")
            numeric_lines = sum(1 for line in chunk_text.split("\n")
                              if line.strip() and any(c.isdigit() for c in line))
            if numeric_lines >= 3:
                # Denominator-only chunks (current liabilities) are critical for ratio queries
                # Give them a VERY significant boost since they're essential for the calculation
                # but don't match any query terms (no BM25/keyword score)
                return 2.5
        return 0.35
    return 0.0


def _metadata_bonus(cand: dict[str, Any]) -> float:
    metadata = cand.get("metadata") or {}

    bonus = 0.0

    block_count = safe_get(metadata, "block_count", 0) or 0
    heading_depth = safe_get(metadata, "heading_path_depth", 0) or 0
    has_list = bool(safe_get(metadata, "has_list", False))
    has_table = bool(safe_get(metadata, "has_table", False))

    if 1 <= to_float(block_count) <= 6:
        bonus += 0.02
    if 1 <= to_float(heading_depth) <= 4:
        bonus += 0.015
    if has_list:
        bonus += 0.01
    if has_table:
        bonus += 0.008

    return bonus


def _compute_numeric_density_boost(cand: dict[str, Any]) -> float:
    """
    Boost chunks with high numeric density.
    Returns a bonus in [0.0, RETRIEVAL_NUMERIC_BOOST_WEIGHT].
    """
    text = (cand.get("chunk_text", "") or "") + " " + (cand.get("search_text", "") or "")
    dollar_count = text.count("$")
    pct_count = text.count("%")
    number_count = len(re.findall(r"[\d,]+\.?\d*", text))

    fin_pattern_matches = 0
    for pat in _NUMERIC_FINANCE_PATTERNS:
        try:
            fin_pattern_matches += len(re.findall(pat, text))
        except re.error:
            pass

    signals = sum([
        1 if dollar_count >= 1 else 0,
        1 if pct_count >= 1 else 0,
        1 if fin_pattern_matches >= 2 else 0,
        1 if number_count >= 5 else 0,
    ])

    if signals >= 3:
        return RETRIEVAL_NUMERIC_BOOST_WEIGHT
    elif signals == 2:
        return RETRIEVAL_NUMERIC_BOOST_WEIGHT * 0.6
    elif signals == 1:
        return RETRIEVAL_NUMERIC_BOOST_WEIGHT * 0.25
    return 0.0


def _compute_table_like_boost(cand: dict[str, Any]) -> float:
    """
    Boost chunks that look like financial table slices.
    """
    text = (cand.get("chunk_text", "") or "").lower()
    section_path = cand.get("section_path", "") or ""
    if isinstance(section_path, list):
        section_path = " ".join(section_path)
    section = ((cand.get("section_title", "") or "") + " " + section_path).lower()

    table_title_hits = sum(1 for kw in _TABLE_STRUCTURE_KEYWORDS if kw in section)
    if table_title_hits >= 2:
        title_bonus = 0.03
    elif table_title_hits == 1:
        title_bonus = 0.015
    else:
        title_bonus = 0.0

    lines = text.split("\n")
    numeric_lines = sum(1 for line in lines if re.search(r"[\d,]+\.?\d*", line))
    if len(lines) >= 3 and numeric_lines / len(lines) >= 0.4:
        structure_bonus = RETRIEVAL_TABLE_BOOST_WEIGHT * 0.7
    elif numeric_lines >= 2:
        structure_bonus = RETRIEVAL_TABLE_BOOST_WEIGHT * 0.35
    else:
        structure_bonus = 0.0

    return min(RETRIEVAL_TABLE_BOOST_WEIGHT, title_bonus + structure_bonus)


def _compute_query_aware_lexical_boost(
    cand: dict[str, Any],
    query_info: dict[str, Any],
    intent: str,
) -> float:
    """
    Boost chunks based on query intent and content alignment.
    """
    text_lower = ((cand.get("chunk_text", "") or "") + " " +
                  (cand.get("search_text", "") or "")).lower()
    section_path = cand.get("section_path", "") or ""
    if isinstance(section_path, list):
        section_path = " ".join(section_path)
    section_lower = ((cand.get("section_title", "") or "") + " " + section_path).lower()
    combined = text_lower + " " + section_lower

    if intent == "numeric_fact":
        numeric_section_kw = [
            "balance sheet", "current assets", "current liabilities",
            "cash equivalents", "short-term investment", "accounts receivable",
            "inventori", "total current", "financial summary",
            "ratio", "quick ratio", "current ratio",
            "effective tax", "provision for income",
            "income tax", "tax rate", "operating income",
            "net income", "gross margin", "revenue",
        ]
        hits = sum(1 for kw in numeric_section_kw if kw in combined)
        if hits >= 3:
            return RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT
        elif hits == 2:
            return RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT * 0.65
        elif hits == 1:
            return RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT * 0.3
        return 0.0

    elif intent == "list_fact":
        list_section_kw = [
            "overview", "products", "services", "segment", "subsidiary",
            "brands", "business", "offering", "we offer", "our products",
            "our business", "principal products", "line of business",
            "geographic", "category",
        ]
        hits = sum(1 for kw in list_section_kw if kw in combined)
        if hits >= 2:
            return RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT
        elif hits == 1:
            return RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT * 0.5
        return 0.0

    elif intent == "descriptive":
        narrative_kw = [
            "management", "discussion", "analysis", "overview",
            "liquidity", "capital resources", "results of operations",
        ]
        hits = sum(1 for kw in narrative_kw if kw in section_lower)
        if hits >= 1:
            return RETRIEVAL_QUERY_AWARE_BOOST_WEIGHT * 0.5
        return 0.0

    return 0.0


def _compute_anti_noise_penalty(cand: dict[str, Any]) -> float:
    """
    Mildly penalize chunks from noise sections (TOC, risk factors, etc.).
    """
    section_path = cand.get("section_path", "") or ""
    if isinstance(section_path, list):
        section_path = " ".join(section_path)
    section = ((cand.get("section_title", "") or "") + " " + section_path).lower()

    for pat in _NOISE_SECTION_PATTERNS:
        if re.search(pat, section):
            return -RETRIEVAL_ANTI_NOISE_PENALTY_WEIGHT

    text = (cand.get("chunk_text", "") or "")
    if len(text) < 80 and not re.search(r"\d", text):
        return -RETRIEVAL_ANTI_NOISE_PENALTY_WEIGHT * 0.5

    return 0.0


def _smooth_page_scores(
    candidates: list[dict[str, Any]],
    score_key: str = "final_score",
    alpha: float | None = None,
) -> dict[int, float]:
    """
    Page-level neighborhood smoothing: for each page p,
      smoothed_score(p) = score(p) + alpha * max(score(neighbor pages))

    Returns a dict {page: smoothed_score}.
    """
    if alpha is None:
        alpha = RETRIEVAL_PAGE_CLUSTER_ALPHA

    page_base: dict[int, float] = {}
    for cand in candidates:
        p = cand.get("page_start")
        if p is None:
            continue
        try:
            page = int(p)
        except (TypeError, ValueError):
            continue
        score = to_float(cand.get(score_key, 0.0))
        if page not in page_base or score > page_base[page]:
            page_base[page] = score

    if not page_base:
        return {}

    smoothed: dict[int, float] = {}
    for page, base_score in page_base.items():
        neighbor_max = 0.0
        for neighbor in (page - 1, page + 1):
            neighbor_score = page_base.get(neighbor, 0.0)
            if neighbor_score > neighbor_max:
                neighbor_max = neighbor_score
        smoothed[page] = base_score + alpha * neighbor_max

    return smoothed


_NARRATIVE_SECTION_KEYWORDS = [
    "management", "discussion", "analysis",
    "liquidity", "capital resources", "financial condition",
    "results of operations", "overview",
    "risk factor", "business",
    "notes to consolidated", "note ",
]
_NOISE_SECTION_KEYWORDS = [
    "schedule", "form 10-k", "table of contents",
    "sec filing", "accession",
]
_FINANCE_TERM_KEYWORDS = [
    "ratio", "revenue", "cash", "debt", "eps", "ebitda",
    "inventory", "capex", "depreciation", "amortization",
    "segment", "gross margin", "operating income", "net income",
    "assets", "liabilities", "equity", "dividend",
    "growth", "decline", "increase", "decrease", "driven",
    "liquidity", "solvency", "working capital",
]


def _section_narrative_bonus(cand: dict[str, Any]) -> float:
    """Boost chunks from narrative/explanatory sections vs pure tables/indices."""
    section = cand.get("section_title", "") or ""
    section_lower = section.lower()
    path = cand.get("section_path", "") or ""
    if isinstance(path, list):
        path = " ".join(path)
    path_lower = path.lower()
    chunk_text = (cand.get("chunk_text", "") or "").lower()
    search_text = (cand.get("search_text", "") or "").lower()
    combined = section_lower + " " + path_lower

    for noise in _NOISE_SECTION_KEYWORDS:
        if noise in combined:
            return -0.05

    narrative_score = 0.0
    for narrative in _NARRATIVE_SECTION_KEYWORDS:
        if narrative in combined:
            narrative_score = 0.06
            break

    combined_text = chunk_text + " " + search_text
    finance_hits = sum(1 for kw in _FINANCE_TERM_KEYWORDS if kw in combined_text)
    if finance_hits >= 3:
        narrative_score += 0.05
    elif finance_hits >= 1:
        narrative_score += 0.025

    return narrative_score


def _compute_financial_content_bonus(
    cand: dict[str, Any],
    query_info: dict[str, Any],
) -> float:
    """
    Boost chunks with financial explanatory content and numeric detail.
    """
    text = (cand.get("chunk_text", "") or "").lower()
    search_text = (cand.get("search_text", "") or "").lower()
    combined = text + " " + search_text

    causal_count = 0
    for pattern in _CAUSAL_PHRASE_PATTERNS:
        if pattern in combined:
            causal_count += 1
    causal_bonus = 0.0
    if causal_count >= 1:
        causal_bonus = min(0.06, 0.025 * causal_count)

    dollar_signs = text.count("$")
    pct_signs = text.count("%")
    numeric_matches = 0
    for pattern in _NUMERIC_FINANCE_PATTERNS:
        try:
            numeric_matches += len(re.findall(pattern, combined))
        except re.error:
            pass

    has_numbers = dollar_signs >= 1 or pct_signs >= 1
    has_causal = causal_count >= 1
    numeric_bonus = 0.0
    if has_numbers and has_causal:
        numeric_bonus = 0.04
    elif has_causal:
        numeric_bonus = 0.02

    query_lower = query_info.get("normalized_query", "").lower()
    is_driver_query = any(
        kw in query_lower for kw in ["driven", "driver", "cause", "why", "due to", "reason", "change", "increase", "decrease", "growth", "decline"]
    )
    if is_driver_query and causal_count >= 1:
        causal_bonus *= 1.5

    return causal_bonus + numeric_bonus


def _compute_page_diversity_bonus(
    candidates: list[dict[str, Any]],
    page_counts: dict[int, int],
    top_n: int = 20,
) -> dict[int, float]:
    """
    Compute a page-level diversity bonus.
    Under-represented pages in top-N get a boost.
    """
    bonus: dict[int, float] = {}
    if not candidates or not page_counts:
        return bonus

    top_candidates = candidates[:top_n]
    top_pages = set()
    for c in top_candidates:
        p = c.get("page_start")
        if p is not None:
            try:
                top_pages.add(int(p))
            except (TypeError, ValueError):
                pass

    total_pages = len(page_counts)
    if total_pages <= 1:
        return bonus

    max_pages_desired = min(total_pages, 5)

    for cand in candidates:
        p = cand.get("page_start")
        if p is None:
            continue
        try:
            page = int(p)
        except (TypeError, ValueError):
            continue

        count_in_top = page_counts.get(page, 0)
        # under-represented pages get a bonus, over-represented get penalized
        ideal_max_per_page = top_n / max_pages_desired
        if count_in_top < ideal_max_per_page:
            bonus[page] = 0.03 * (1 - count_in_top / ideal_max_per_page)
        elif count_in_top > ideal_max_per_page * 1.5:
            bonus[page] = -0.02 * ((count_in_top / ideal_max_per_page) - 1)

    return bonus
