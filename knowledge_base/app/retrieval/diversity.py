"""
Diversity and deduplication: text dedup, page-diversity cap, neighbor chunk expansion.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from app.db import get_neighbor_chunks
from app.retrieval._common import normalize_whitespace

from .config import (
    LEXICAL_CLEAN_RE,
    RETRIEVAL_DEDUP_SIM_THRESHOLD,
    RETRIEVAL_ENABLE_NEIGHBOR_EXPANSION,
    RETRIEVAL_MAX_SAME_SECTION,
    RETRIEVAL_NEIGHBOR_WINDOW,
)


def _normalize_lexical_text(text: str) -> str:
    """Normalize text for lexical comparison. Used by dedup."""
    text = normalize_whitespace(text or "")
    text = text.lower()
    text = LEXICAL_CLEAN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_text(text: str) -> list[str]:
    """Tokenize text into whitespace-separated tokens. Used by dedup similarity."""
    return normalize_whitespace(text).split()


def _text_similarity_for_dedup(a: str, b: str) -> float:
    tokens_a = set(_tokenize_text(_normalize_lexical_text(a)))
    tokens_b = set(_tokenize_text(_normalize_lexical_text(b)))

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))


def _is_table_like_chunk(cand: dict[str, Any]) -> bool:
    """
    Return True if the chunk looks like a financial table slice.
    Table-like chunks should not be deduplicated away since each page
    of a financial table may contain unique numeric data.
    """
    chunk_text = cand.get("chunk_text", "") or ""
    section_title = (cand.get("section_title") or "").lower()
    section_path = cand.get("section_path") or ""
    if isinstance(section_path, list):
        section_path = " ".join(str(x) for x in section_path)
    section_path = section_path.lower()

    # Table column prefix (e.g., "col_0=...")
    if "col_0=" in chunk_text:
        return True

    # Section titles that indicate financial tables
    table_indicators = [
        "balance sheet", "income statement", "cash flow",
        "current assets", "current liabilities", "total assets",
        "statement of", "financial statement",
        "item 7a", "quantitative and qualitative",
        "market risk", "assets and liabilities",
    ]
    combined_section = section_title + " " + section_path
    if any(ind in combined_section for ind in table_indicators):
        # Also check if chunk has substantial numeric content
        dollar_count = chunk_text.count("$")
        number_count = len(re.findall(r"[\d,]+\.?\d*", chunk_text))
        if dollar_count >= 1 and number_count >= 5:
            return True

    return False


def _deduplicate_candidates(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    section_counter: defaultdict[str, int] = defaultdict(int)

    for cand in candidates:
        section_key = f"{cand.get('document_id')}::{cand.get('section_path') or cand.get('section_title') or ''}"
        is_table = _is_table_like_chunk(cand)

        # Table-like chunks bypass section limit — financial tables span pages with unique data
        if not is_table and section_counter[section_key] >= RETRIEVAL_MAX_SAME_SECTION:
            continue

        # Table-like chunks are never deduplicated — each table page may have unique data
        if not is_table:
            is_dup = False
            for chosen in selected:
                sim = _text_similarity_for_dedup(
                    cand.get("search_text", "") or cand.get("chunk_text", ""),
                    chosen.get("search_text", "") or chosen.get("chunk_text", ""),
                )
                if sim >= RETRIEVAL_DEDUP_SIM_THRESHOLD:
                    is_dup = True
                    break

            if is_dup:
                continue

        selected.append(cand)
        section_counter[section_key] += 1

        if len(selected) >= top_k:
            break

    return selected


MAX_CHUNKS_PER_PAGE = 5  # increased for ratio/financial queries that need multiple table sections from same page


def _cap_page_duplicates(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """
    Apply a hard cap on how many chunks from the same page can appear in top_k.
    """
    if not candidates:
        return []

    page_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        p = cand.get("page_start")
        if p is not None:
            try:
                page_groups[int(p)].append(cand)
            except (TypeError, ValueError):
                page_groups[id(cand)].append(cand)
        else:
            page_groups[id(cand)].append(cand)

    for page in page_groups:
        page_groups[page].sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    selected: list[dict[str, Any]] = []
    surplus: list[dict[str, Any]] = []

    for page, chunks in page_groups.items():
        selected.extend(chunks[:MAX_CHUNKS_PER_PAGE])
        surplus.extend(chunks[MAX_CHUNKS_PER_PAGE:])

    selected.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    surplus.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    selected.extend(surplus[: max(0, top_k - len(selected))])

    selected.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    return selected[:top_k]


def _expand_neighbor_chunks(candidates: list[dict[str, Any]], target_limit: int) -> list[dict[str, Any]]:
    if not RETRIEVAL_ENABLE_NEIGHBOR_EXPANSION or not candidates:
        return candidates[:target_limit]

    from app.retrieval._common import safe_get, to_float
    from app.retrieval.recall import _row_to_candidate

    results: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for cand in candidates:
        doc_id = cand.get("document_id")
        if doc_id is not None:
            grouped[int(doc_id)].append(cand)

    # First pass: add ALL center candidates first to ensure they're included
    for doc_id, doc_cands in grouped.items():
        for cand in doc_cands:
            cid = cand.get("chunk_id")
            if cid is None:
                continue
            try:
                cid_int = int(cid)
            except (TypeError, ValueError):
                continue
            if cid_int not in seen_ids:
                results.append(cand)
                seen_ids.add(cid_int)

    # Second pass: add neighbors only if there's remaining capacity
    for doc_id, doc_cands in grouped.items():
        center_indexes = [
            int(c.get("chunk_index"))
            for c in doc_cands
            if c.get("chunk_index") is not None
        ]
        neighbor_rows = get_neighbor_chunks(
            document_id=doc_id,
            center_chunk_indexes=center_indexes,
            window=RETRIEVAL_NEIGHBOR_WINDOW,
        ) or []

        index_map = {safe_get(r, "chunk_index"): r for r in neighbor_rows}

        for cand in doc_cands:
            center_index = cand.get("chunk_index")
            center_section = cand.get("section_path") or cand.get("section_title") or ""
            center_score = to_float(cand.get("final_score"))

            if center_index is None:
                continue

            for offset in range(1, RETRIEVAL_NEIGHBOR_WINDOW + 1):
                if len(results) >= target_limit:
                    break

                for neighbor_idx in (center_index - offset, center_index + offset):
                    if len(results) >= target_limit:
                        break

                    row = index_map.get(neighbor_idx)
                    if not row:
                        continue

                    neighbor_id = safe_get(row, "id")
                    if neighbor_id is None:
                        continue
                    try:
                        neighbor_id_int = int(neighbor_id)
                    except (TypeError, ValueError):
                        continue
                    if neighbor_id_int in seen_ids:
                        continue

                    neighbor_cand = _row_to_candidate(row)
                    neighbor_section = neighbor_cand.get("section_path") or neighbor_cand.get("section_title") or ""

                    if center_section and neighbor_section and center_section != neighbor_section:
                        continue

                    # Only expand if neighbor has meaningful lexical presence
                    neighbor_text = neighbor_cand.get("search_text", "") or neighbor_cand.get("chunk_text", "")
                    if not neighbor_text.strip():
                        continue

                    # Cap neighbor score aggressively to prevent irrelevant neighbors
                    # (from high-boost ratio centers) from outranking gold chunks.
                    # Use: min(center * 0.45, center - 0.7) to ensure neighbors
                    # never displace center-chunks with genuine lexical relevance.
                    neighbor_score = min(center_score * 0.45, center_score - 0.7)
                    neighbor_cand["final_score"] = neighbor_score
                    neighbor_cand["is_neighbor"] = True
                    neighbor_cand["_query_intent"] = cand.get("_query_intent", "unknown")
                    neighbor_cand["_numeric_boost"] = cand.get("_numeric_boost", 0.0)
                    neighbor_cand["_table_boost"] = cand.get("_table_boost", 0.0)
                    neighbor_cand["_query_aware_boost"] = cand.get("_query_aware_boost", 0.0)
                    neighbor_cand["_anti_noise_penalty"] = cand.get("_anti_noise_penalty", 0.0)

                    results.append(neighbor_cand)
                    seen_ids.add(neighbor_id_int)

            if len(results) >= target_limit:
                break
        if len(results) >= target_limit:
            break

    return results
