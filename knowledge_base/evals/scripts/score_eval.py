#!/usr/bin/env python3
"""
score_eval.py – Score a completed eval run at document, chunk, and answer levels.

Supports two input modes:
  1. JSONL run output   (--run)
  2. JSON report file   (--report)

Outputs:
  evals/results/<run_name>_scored.jsonl  (one record per eval sample)
  evals/results/<run_name>_scored.json   (aggregate report)

New P0 metrics:
  - strict_page_hit@k     : exact page match
  - relaxed_page_hit@k    : page within window
  - evidence_semantic_hit@k: chunk-gold semantic similarity (lexical + embedding)

Usage:
    python -m evals.scripts.score_eval --run evals/runs/run_20260326_prompt_v1.jsonl
    python -m evals.scripts.score_eval --run evals/runs/run_20260326_prompt_v1.jsonl --dataset evals/data/financebench_v1_subset_3docs_eval.jsonl
    python -m evals.scripts.score_eval --run evals/runs/run_20260326_prompt_v1.jsonl --dataset evals/data/financebench_v1_subset_3docs_eval.jsonl --evidence-semantic-window 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[\s\u00a0\u3000]+", " ", text)
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    return text.strip()


def normalize_number_friendly(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[$€£¥]", "", text)
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    return text


def normalize_doc_name(x: str) -> str:
    """Normalize document name for comparison."""
    return x.lower().replace(".pdf", "").replace("-", "_").strip()


# ---------------------------------------------------------------------------
# Document-level scoring
# ---------------------------------------------------------------------------

def score_document_level(
    sample: dict,
    result: dict,
) -> dict[str, Any]:
    """
    Document-level metrics (using gold_doc_ids from retrieval block):
      - doc_hit_at_1   : did top-1 doc hit the gold doc?
      - doc_hit_at_5   : did any of top-5 docs hit the gold doc?
      - doc_mrr        : Mean Reciprocal Rank for doc retrieval
    """
    gold_doc_ids = set(sample.get("retrieval", {}).get("gold_doc_ids", []))
    if not gold_doc_ids:
        return {"doc_hit_at_1": None, "doc_hit_at_5": None, "doc_mrr": None}

    retrieved_chunks = result.get("retrieved_chunks", [])
    if not retrieved_chunks:
        return {"doc_hit_at_1": 0, "doc_hit_at_5": 0, "doc_mrr": 0.0}

    # Collect unique doc_ids in retrieval order (deduplicated)
    seen_docs: list[int] = []
    seen_set: set[int] = set()
    for chunk in retrieved_chunks:
        did = chunk.get("document_id")
        if did is None:
            continue
        try:
            did = int(did)
        except (TypeError, ValueError):
            continue
        if did not in seen_set:
            seen_set.add(did)
            seen_docs.append(did)

    # doc_hit_at_1
    doc_hit_at_1 = 1 if seen_docs and seen_docs[0] in gold_doc_ids else 0

    # doc_hit_at_5
    doc_hit_at_5 = 1 if any(d in gold_doc_ids for d in seen_docs[:5]) else 0

    # doc_mrr
    doc_mrr = 0.0
    for rank, did in enumerate(seen_docs[:5], start=1):
        if did in gold_doc_ids:
            doc_mrr = 1.0 / rank
            break

    return {
        "doc_hit_at_1": doc_hit_at_1,
        "doc_hit_at_5": doc_hit_at_5,
        "doc_mrr": round(doc_mrr, 4),
    }


# ---------------------------------------------------------------------------
# Page-level scoring (strict + relaxed)
# ---------------------------------------------------------------------------

def score_page_level(
    sample: dict,
    result: dict,
    page_window: int = 0,
) -> dict[str, Any]:
    """
    Page-level metrics (strict page + relaxed page):

    Strict page (page_window=0):
      chunk.page_start <= gold_page <= chunk.page_end

    Relaxed page (page_window > 0):
      abs(pred_page - gold_page) <= page_window

    Metrics per mode:
      - hit_at_1 : did top-1 chunk's page hit gold page?
      - hit_at_5 : did any of top-5 chunks' pages hit gold page?
      - mrr      : Mean Reciprocal Rank (1/rank of first hit)

    Note: any-match rule — if any gold page is covered, it's a hit.
    """
    gold_pages = set(sample.get("gold_pages", []))
    retrieved_chunks = result.get("retrieved_chunks", [])

    if not gold_pages:
        return {
            f"page_hit_at_1": None,
            f"page_hit_at_5": None,
            f"page_mrr": None,
            f"page_relaxed_hit_at_1": None,
            f"page_relaxed_hit_at_5": None,
            f"page_relaxed_mrr": None,
        }

    def _strict_hit(page_start, page_end, gold_pages):
        """Strict: gold_page must fall within [page_start, page_end]."""
        if page_start is None:
            return False
        try:
            ps = int(page_start)
            pe = int(page_end) if page_end is not None else ps
            return any(gp >= ps and gp <= pe for gp in gold_pages)
        except (TypeError, ValueError):
            return False

    def _relaxed_hit(page_start, page_end, gold_pages, window):
        """
        Relaxed: chunk's page range must overlap with [gold_page - window, gold_page + window].
        Implements user's definition: chunk.page_start <= gold_page + window
                                     AND chunk.page_end >= gold_page - window
        This checks for RANGE OVERLAP, not just single-point proximity.
        """
        if page_start is None:
            return False
        try:
            ps = int(page_start)
            pe = int(page_end) if page_end is not None else ps
            return any(
                ps <= gp + window and pe >= gp - window
                for gp in gold_pages
            )
        except (TypeError, ValueError):
            return False

    # Compute hits for each chunk position
    strict_hits: list[bool] = []
    relaxed_hits: list[bool] = []

    for chunk in retrieved_chunks[:5]:
        ps = chunk.get("page_start")
        pe = chunk.get("page_end")
        strict_hits.append(_strict_hit(ps, pe, gold_pages))
        relaxed_hits.append(_relaxed_hit(ps, pe, gold_pages, page_window))

    # hit_at_1
    page_hit_at_1 = int(strict_hits[0]) if strict_hits else 0
    page_relaxed_hit_at_1 = int(relaxed_hits[0]) if relaxed_hits else 0

    # hit_at_5 (any of top-5 hits)
    page_hit_at_5 = int(any(strict_hits)) if strict_hits else 0
    page_relaxed_hit_at_5 = int(any(relaxed_hits)) if relaxed_hits else 0

    # mrr
    page_mrr = 0.0
    for rank, hit in enumerate(strict_hits, start=1):
        if hit:
            page_mrr = 1.0 / rank
            break

    page_relaxed_mrr = 0.0
    for rank, hit in enumerate(relaxed_hits, start=1):
        if hit:
            page_relaxed_mrr = 1.0 / rank
            break

    return {
        "page_hit_at_1": page_hit_at_1,
        "page_hit_at_5": page_hit_at_5,
        "page_mrr": round(page_mrr, 4),
        "page_relaxed_hit_at_1": page_relaxed_hit_at_1,
        "page_relaxed_hit_at_5": page_relaxed_hit_at_5,
        "page_relaxed_mrr": round(page_relaxed_mrr, 4),
    }


# ---------------------------------------------------------------------------
# Evidence text-level scoring
# ---------------------------------------------------------------------------

def score_evidence_text_level(
    sample: dict,
    result: dict,
) -> dict[str, Any]:
    """
    Evidence text-level metrics — checks if any retrieved chunk contains
    any of the gold evidence texts (any-match rule).

    This measures semantic retrieval quality beyond page numbers.

    Metrics:
      - evidence_text_hit_at_1 : does top-1 chunk contain any gold evidence text?
      - evidence_text_hit_at_5 : do any of top-5 chunks contain any gold evidence text?
      - evidence_text_mrr      : Mean Reciprocal Rank
    """
    gold_evidence_texts = sample.get("gold_evidence_texts") or []
    # Filter out empty strings
    gold_evidence_texts = [t for t in gold_evidence_texts if t]

    if not gold_evidence_texts:
        return {
            "evidence_text_hit_at_1": None,
            "evidence_text_hit_at_5": None,
            "evidence_text_mrr": None,
        }

    retrieved_chunks = result.get("retrieved_chunks", [])

    # Pre-normalize gold evidence texts for faster matching
    norm_gold_texts = [normalize_text(t) for t in gold_evidence_texts]

    def _chunk_contains_evidence(chunk_text: str) -> bool:
        """Return True if chunk contains any gold evidence text (normalized)."""
        if not chunk_text:
            return False
        norm_chunk = normalize_text(chunk_text)
        return any(gold in norm_chunk for gold in norm_gold_texts)

    # Compute hits per chunk position
    evidence_hits: list[bool] = []
    for chunk in retrieved_chunks[:5]:
        chunk_text = chunk.get("chunk_text", "") or chunk.get("search_text", "") or ""
        evidence_hits.append(_chunk_contains_evidence(chunk_text))

    # hit_at_1
    evidence_text_hit_at_1 = int(evidence_hits[0]) if evidence_hits else 0

    # hit_at_5
    evidence_text_hit_at_5 = int(any(evidence_hits)) if evidence_hits else 0

    # mrr
    evidence_text_mrr = 0.0
    for rank, hit in enumerate(evidence_hits, start=1):
        if hit:
            evidence_text_mrr = 1.0 / rank
            break

    return {
        "evidence_text_hit_at_1": evidence_text_hit_at_1,
        "evidence_text_hit_at_5": evidence_text_hit_at_5,
        "evidence_text_mrr": round(evidence_text_mrr, 4),
    }


# ---------------------------------------------------------------------------
# Answer-level scoring
# ---------------------------------------------------------------------------

_REFUSE_PHRASES = {
    "没有足够信息", "无法确认", "当前未看到", "不能确认", "没有证据",
    "信息不足", "无法从", "知识库中未找到", "暂未找到", "没有找到相关",
    "not enough information", "cannot confirm", "insufficient information",
    "no relevant", "知识库里没有", "未提供",
}


def score_answer_level(
    sample: dict,
    result: dict,
) -> dict[str, Any]:
    """
    Answer-level metrics:
      - numeric_match          : does the answer contain the correct number?
      - normalized_exact_match : token-based exact match after normalization
      - answer_label           : exact | partial | wrong | refuse_correct | refuse_wrong
      - must_include_hit_ratio: fraction of required phrases found
    """
    gold_answer = sample.get("answer", {}).get("gold_answer", "")
    must_include = sample.get("answer", {}).get("must_include", [])
    expected_behavior = sample.get("evaluation", {}).get("expected_behavior", "answer")
    pred_answer = result.get("final_answer", "")

    norm_pred = normalize_text(pred_answer)
    norm_gold = normalize_text(gold_answer)
    norm_gold_nf = normalize_number_friendly(gold_answer)
    norm_pred_nf = normalize_number_friendly(pred_answer)

    # numeric_match
    numeric_match = 0
    gold_numbers = re.findall(r'[\d,]+\.?\d*', gold_answer)
    if gold_numbers:
        for num in gold_numbers:
            norm_num = normalize_number_friendly(num)
            if norm_num in norm_pred or norm_num in norm_pred_nf:
                numeric_match = 1
                break

    # answer_label
    if expected_behavior == "refuse":
        is_refuse = any(rp in norm_pred for rp in _REFUSE_PHRASES)
        answer_label = "refuse_correct" if is_refuse else "refuse_wrong"
    elif expected_behavior == "answer":
        if _fuzzy_equal(norm_pred, norm_gold):
            answer_label = "exact"
        elif _fuzzy_equal(norm_pred_nf, norm_gold_nf):
            answer_label = "exact"
        else:
            inc_ratio, _ = _check_must(pred_answer, must_include)
            if inc_ratio >= 0.5:
                answer_label = "partial"
            else:
                answer_label = "wrong"
    else:
        answer_label = "wrong"

    inc_ratio, _ = _check_must(pred_answer, must_include)

    return {
        "numeric_match": numeric_match,
        "normalized_exact_match": 1 if _fuzzy_equal(norm_pred, norm_gold) else 0,
        "answer_label": answer_label,
        "must_include_hit_ratio": round(inc_ratio, 4),
        # LLM judge stub
        "answer_correctness": None,
        "answer_groundedness": None,
        "answer_completeness": None,
    }


# ---------------------------------------------------------------------------
# Gold evidence embedding cache (for evidence_semantic_hit)
# ---------------------------------------------------------------------------

# Module-level cache: gold_evidence_id -> embedding vector
# Avoids re-embedding the same gold evidence text across multiple chunks
_GOLD_EMBEDDING_CACHE: dict[str, list[float]] = {}
_GOLD_EMBEDDING_LOCK = False  # prevents re-entrant calls during cache population


def _get_gold_embedding(gold_text: str, cache_key: str) -> Optional[list[float]]:
    """
    Get embedding for a gold evidence text, using a shared cache.
    If not cached, generates embedding using the embedding service.
    Returns None if embedding generation fails.
    """
    if cache_key in _GOLD_EMBEDDING_CACHE:
        return _GOLD_EMBEDDING_CACHE[cache_key]

    try:
        from app.services.llm_service import get_embedding
        emb = get_embedding(gold_text)
        if emb:
            _GOLD_EMBEDDING_CACHE[cache_key] = emb
        return emb
    except Exception:
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _lexical_fuzzy_score(chunk_text: str, gold_text: str) -> float:
    """
    Compute a [0, 1] lexical overlap score between chunk and gold evidence.
    Uses normalized token overlap (Jaccard-like) + ngram overlap.
    """
    if not chunk_text or not gold_text:
        return 0.0

    norm_chunk = normalize_text(chunk_text)
    norm_gold = normalize_text(gold_text)

    # Exact substring (strongest signal)
    if norm_gold in norm_chunk:
        return 1.0

    # Token-level Jaccard
    tokens_c = set(norm_chunk.split())
    tokens_g = set(norm_gold.split())
    if not tokens_c or not tokens_g:
        return 0.0
    jaccard = len(tokens_c & tokens_g) / max(1, len(tokens_c | tokens_g))

    # 3-gram overlap on big text blocks (>200 chars)
    if len(norm_gold) > 200:
        def ngrams(s: str, n: int = 3):
            s = s.replace(" ", "")
            return set(s[i:i+n] for i in range(len(s) - n + 1))
        ng_chunk = ngrams(norm_chunk)
        ng_gold = ngrams(norm_gold)
        if ng_gold:
            ngram_overlap = len(ng_chunk & ng_gold) / max(1, len(ng_gold))
        else:
            ngram_overlap = 0.0
        return max(jaccard, ngram_overlap * 0.9)

    return jaccard


def _semantic_similarity_score(
    chunk_text: str,
    gold_text: str,
    gold_cache_key: str,
) -> float:
    """
    Compute semantic similarity using embedding cosine similarity.
    Uses cached gold embeddings + chunk embeddings via embedding_service.
    """
    # Get gold embedding (cached)
    gold_emb = _get_gold_embedding(gold_text, gold_cache_key)
    if gold_emb is None:
        return 0.0

    # Get chunk embedding
    try:
        from app.services.llm_service import get_embedding
        chunk_emb = get_embedding(chunk_text[:2000])  # Truncate to avoid huge emb calls
        if chunk_emb is None:
            return 0.0
        return _cosine_similarity(gold_emb, chunk_emb)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Evidence Semantic Hit Scoring (P0)
# ---------------------------------------------------------------------------

def score_evidence_semantic_level(
    sample: dict,
    result: dict,
    semantic_threshold: float = 0.78,
    lexical_threshold: float = 0.42,
    use_embedding: bool = True,
) -> dict[str, Any]:
    """
    Compute evidence_semantic_hit metrics.

    Two complementary signals (both must pass their threshold independently):
      - lexical_fuzzy : token/ngram overlap (always computed, fast)
      - embedding_sim  : cosine similarity of embeddings (opt-in, cached)

    A semantic_hit occurs when EITHER signal passes its threshold.
    The higher threshold is for embedding (more reliable), lower for lexical.

    Metrics:
      - evidence_semantic_hit_at_1 : top-1 chunk hits gold evidence?
      - evidence_semantic_hit_at_5 : any top-5 chunk hits gold evidence?
      - evidence_semantic_mrr       : reciprocal rank
      - evidence_semantic_best_score: best similarity score across top-5
      - evidence_lexical_hit_at_5  : lexical-only hit (for comparison)
    """
    gold_evidence_texts = sample.get("gold_evidence_texts") or []
    gold_evidence_texts = [t for t in gold_evidence_texts if t]
    if not gold_evidence_texts:
        return {
            "evidence_semantic_hit_at_1": None,
            "evidence_semantic_hit_at_5": None,
            "evidence_semantic_mrr": None,
            "evidence_semantic_best_score": None,
            "evidence_lexical_hit_at_5": None,
        }

    retrieved_chunks = result.get("retrieved_chunks", [])

    def _chunk_evidence_score(chunk: dict) -> float:
        chunk_text = chunk.get("chunk_text", "") or chunk.get("search_text", "") or ""
        if not chunk_text:
            return 0.0

        best = 0.0
        for i, gold_text in enumerate(gold_evidence_texts):
            cache_key = f"{sample.get('id', '')}_gold_{i}"

            # Lexical fuzzy (always computed)
            lex_score = _lexical_fuzzy_score(chunk_text, gold_text)
            if lex_score >= lexical_threshold:
                best = max(best, lex_score)

            # Embedding similarity (if enabled)
            if use_embedding:
                emb_score = _semantic_similarity_score(chunk_text, gold_text, cache_key)
                if emb_score >= semantic_threshold:
                    best = max(best, emb_score)

            # Early exit if we already have a strong score
            if best >= 0.9:
                break

        return best

    scores_per_chunk: list[float] = []
    for chunk in retrieved_chunks[:5]:
        scores_per_chunk.append(_chunk_evidence_score(chunk))

    def _any_hit(threshold: float) -> bool:
        return any(s >= threshold for s in scores_per_chunk)

    def _first_hit_rank(threshold: float) -> int:
        for rank, s in enumerate(scores_per_chunk, 1):
            if s >= threshold:
                return rank
        return 0

    best_score = max(scores_per_chunk) if scores_per_chunk else 0.0

    # Use a combined threshold: semantic_hit if lexical >= 0.42 OR embedding >= 0.78
    hit_threshold = min(lexical_threshold, semantic_threshold)
    first_rank = _first_hit_rank(hit_threshold)

    return {
        "evidence_semantic_hit_at_1": int(
            scores_per_chunk[0] >= hit_threshold if scores_per_chunk else 0
        ),
        "evidence_semantic_hit_at_5": int(_any_hit(hit_threshold)),
        "evidence_semantic_mrr": round(1.0 / first_rank, 4) if first_rank > 0 else 0.0,
        "evidence_semantic_best_score": round(best_score, 4),
        "evidence_lexical_hit_at_5": int(_any_hit(lexical_threshold)),
        # Per-chunk scores for debugging
        "_evidence_chunk_scores": [round(s, 4) for s in scores_per_chunk],
    }


# ---------------------------------------------------------------------------
# Section-level scoring (P0)
# ---------------------------------------------------------------------------

def score_section_level(
    sample: dict,
    result: dict,
    page_window: int = 3,
) -> dict[str, Any]:
    """
    Section-level metrics — checks if any retrieved chunk falls within a section
    that contains the gold page.

    A section is defined as a contiguous page range around the gold page:
      section_start = gold_page - page_window
      section_end = gold_page + page_window
      section_hit if chunk.page_start <= gold_page + window AND chunk.page_end >= gold_page - window

    This is a relaxed form of page hit that accounts for section boundaries
    (e.g., table on page 55 is in the same section as pages 53-57).

    Metrics:
      - section_hit_at_1 : does top-1 chunk fall in same section as gold page?
      - section_hit_at_5 : do any of top-5 chunks fall in same section?
      - section_mrr      : Mean Reciprocal Rank
    """
    gold_pages = set(sample.get("gold_pages", []))
    retrieved_chunks = result.get("retrieved_chunks", [])

    if not gold_pages:
        return {
            "section_hit_at_1": None,
            "section_hit_at_5": None,
            "section_mrr": None,
        }

    def _in_section(page_start, page_end, gold_pages, window):
        """Check if any gold page falls within [page_start - window, page_end + window]."""
        if page_start is None:
            return False
        try:
            ps = int(page_start)
            pe = int(page_end) if page_end is not None else ps
            return any(
                ps <= gp + window and pe >= gp - window
                for gp in gold_pages
            )
        except (TypeError, ValueError):
            return False

    section_hits: list[bool] = []
    for chunk in retrieved_chunks[:5]:
        ps = chunk.get("page_start")
        pe = chunk.get("page_end")
        section_hits.append(_in_section(ps, pe, gold_pages, page_window))

    section_hit_at_1 = int(section_hits[0]) if section_hits else 0
    section_hit_at_5 = int(any(section_hits)) if section_hits else 0

    section_mrr = 0.0
    for rank, hit in enumerate(section_hits, start=1):
        if hit:
            section_mrr = 1.0 / rank
            break

    return {
        "section_hit_at_1": section_hit_at_1,
        "section_hit_at_5": section_hit_at_5,
        "section_mrr": round(section_mrr, 4),
    }


# ---------------------------------------------------------------------------
# Failure Reason Auto-Attribution (P1)
# ---------------------------------------------------------------------------

def _attribut_failure_reason(
    sample: dict,
    result: dict,
    page_window: int = 3,
) -> str:
    """
    Heuristic failure reason auto-attribution for failed retrieval samples.

    Categories (v4 taxonomy):
      - doc_miss           : correct doc not in retrieved top-5
      - section_miss       : correct doc found but no chunk within section window of gold page
      - page_near_miss     : correct doc found; nearby page but not in section window
      - chunk_miss         : exact chunk not retrieved (but section/page might be near)
      - semantic_drift     : correct doc+section region found but wrong semantic content
      - answer_error       : retrieval found correct content but answer generation failed
      - unsupported_answer : answer verifier marked answer as not supported by evidence
      - numeric_inconsistency : verifier found numeric values inconsistent with evidence
      - citation_missing   : answer failed to cite evidence IDs properly
      - refine_failed      : self-refine was attempted but did not improve answer
      - rerank_error       : LLM reranker degraded retrieval quality
      - chunk_content_miss : no retrievable content at all
      - no_gold_label      : no gold labels available
      - unknown
    """
    gold_doc_ids = set(sample.get("retrieval", {}).get("gold_doc_ids", []))
    gold_pages = set(sample.get("gold_pages", []))
    pred_chunks = result.get("retrieved_chunks", [])[:5]
    answer_label = result.get("answer_label", "wrong")
    numeric_match = result.get("numeric_match", 0)
    semantic_hit = result.get("evidence_semantic_hit_at_5", 0)
    lexical_hit = result.get("evidence_lexical_hit_at_5", 0)
    section_hit = result.get("section_hit_at_5", 0)
    must_not_include_violations = result.get("must_not_include_violations", [])

    # V4 fields
    v4_answer_supported = result.get("v4_answer_supported")
    v4_numeric_consistent = result.get("v4_numeric_consistent")
    v4_citation_missing = result.get("v4_citation_missing")
    v4_was_refined = result.get("v4_was_refined")
    v4_refinement_round = result.get("v4_refinement_round", 0)

    if not gold_doc_ids or not gold_pages:
        return "no_gold_label"

    # Check doc hit
    pred_doc_ids = [
        c.get("document_id") for c in pred_chunks
        if c.get("document_id") is not None
    ]
    doc_hit = any(int(d) in gold_doc_ids for d in pred_doc_ids if d is not None)

    if not doc_hit:
        return "doc_miss"

    # Check page hit (strict)
    pred_pages = [c.get("page_start") for c in pred_chunks]
    strict_page_hit = any(
        p is not None and int(p) in gold_pages
        for p in pred_pages
    )

    # Section hit uses page_window (default 3)
    # section_hit_at_5 was precomputed and passed in result

    # Primary failure routing
    if strict_page_hit:
        # Page is correct - if semantic hit is missing, it's semantic drift
        if not semantic_hit:
            return "semantic_drift"
        # V4: Check verifier result first
        if v4_answer_supported is False:
            if v4_numeric_consistent is False:
                return "numeric_inconsistency"
            if v4_citation_missing:
                return "citation_missing"
            return "unsupported_answer"
        # Answer generation issue
        if answer_label == "wrong":
            if must_not_include_violations:
                return "answer_error"
            if numeric_match == 1:
                return "answer_error"
        return "unknown"

    # strict_page_hit is False
    if section_hit:
        # Found content in the right section but not at exact page
        if not semantic_hit:
            return "semantic_drift"
        # V4: check verifier
        if v4_answer_supported is False:
            return "unsupported_answer"
        # chunk content exists but wrong exact page → chunk_miss
        return "chunk_miss"

    if numeric_match == 1 or semantic_hit == 1 or lexical_hit == 1:
        # Found relevant content somewhere in the document
        # V4: check if refine failed to help
        if v4_was_refined and v4_refinement_round > 0 and not v4_answer_supported:
            return "refine_failed"
        return "section_miss"

    # V4: check reranker quality (if LLM rerank was applied)
    llm_rerank_applied = any(c.get("llm_rerank_applied", False) for c in pred_chunks)
    if llm_rerank_applied and not section_hit and not semantic_hit:
        # Reranker may have pulled in wrong chunks
        return "rerank_error"

    # No relevant content found near gold page
    return "page_near_miss"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _fuzzy_equal(a: str, b: str, threshold: float = 0.85) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return (overlap / union) >= threshold if union > 0 else False


def _check_must(text: str, phrases: list[str]) -> tuple[float, list[str]]:
    if not phrases:
        return 1.0, []
    norm_text = normalize_text(text)
    hits = [p for p in phrases if normalize_text(p) in norm_text]
    return len(hits) / len(phrases), [p for p in phrases if p not in hits]


# ---------------------------------------------------------------------------
# Main scoring logic
# ---------------------------------------------------------------------------

def load_run_output(path: Path) -> list[dict]:
    """Load run output from JSONL or JSON report."""
    if path.suffix == ".jsonl":
        samples = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    elif path.suffix == ".json":
        with path.open(encoding="utf-8") as f:
            report = json.load(f)
        return report.get("case_results", [])
    else:
        raise ValueError(f"Unsupported file format: {path}")


def score_run(
    case_results: list[dict],
    dataset_samples: list[dict],
    page_window: int = 0,
    use_evidence_embedding: bool = True,
    evidence_semantic_threshold: float = 0.78,
) -> list[dict]:
    """
    Score all case results against the eval dataset samples.

    Returns a list of scored records with document, page (strict+relaxed),
    evidence text (legacy exact + new semantic), and answer metrics.
    """
    sample_by_id = {s.get("id"): s for s in dataset_samples}

    scored: list[dict] = []

    for result in case_results:
        eval_id = result.get("id", "")
        sample = sample_by_id.get(eval_id, {})

        # Fallback: if dataset lookup missed (e.g. empty id from run_financebench_eval),
        # pull gold fields directly from the result dict so scoring still works.
        if not sample:
            sample = {
                "id": eval_id,
                "gold_doc_name": result.get("gold_doc_name", ""),
                "gold_pages": result.get("gold_pages") or [],
                "gold_evidence_texts": result.get("gold_evidence_texts") or [],
                "gold_chunk_ids": result.get("gold_chunk_ids") or [],
                "gold_chunk_status": result.get("gold_chunk_status", "unresolved"),
                "gold_chunk_match_method": result.get("gold_chunk_match_method", ""),
                "retrieval": result.get("retrieval") or {},
                "answer": {
                    "gold_answer": result.get("gold_answer", ""),
                    "must_include": result.get("answer", {}).get("must_include", []),
                },
                "evaluation": {
                    "expected_behavior": result.get("evaluation", {}).get("expected_behavior", "answer"),
                },
            }

        retrieved_chunks = result.get("retrieved_chunks", [])

        # Build pred_doc_names (deduplicated, in score order)
        seen_doc_names: list[str] = []
        seen_doc_set: set[str] = set()
        for chunk in retrieved_chunks[:5]:
            title = chunk.get("title", "")
            if title and title not in seen_doc_set:
                seen_doc_set.add(title)
                seen_doc_names.append(title)

        # Build pred_chunks with full detail (for debugging output)
        pred_chunks = []
        for c in retrieved_chunks[:5]:
            chunk_text = c.get("chunk_text", "") or c.get("search_text", "") or ""
            pred_chunks.append({
                "chunk_id": _safe_int(c.get("chunk_id")),
                "doc_name": c.get("title", ""),
                "doc_name_norm": normalize_doc_name(c.get("title", "")),
                "section_title": c.get("section_title", ""),
                "section_path": c.get("section_path", ""),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "score": c.get("score", 0.0),
                "rerank_score": c.get("rerank_score", 0.0),
                "embedding_score": c.get("embedding_score", 0.0),
                "keyword_score": c.get("keyword_score", 0.0),
                "bm25_score": c.get("bm25_score", 0.0),
                "chunk_text_preview": chunk_text[:200],
                "chunk_text": chunk_text[:500],
                "_retrieval_query": c.get("_retrieval_query"),
                # New P0 rerank signal breakdowns
                "_numeric_boost": round(float(c.get("_numeric_boost") or 0.0), 6),
                "_table_boost": round(float(c.get("_table_boost") or 0.0), 6),
                "_query_aware_boost": round(float(c.get("_query_aware_boost") or 0.0), 6),
                "_anti_noise_penalty": round(float(c.get("_anti_noise_penalty") or 0.0), 6),
                "_page_cluster_bonus": round(float(c.get("_page_cluster_bonus") or 0.0), 6),
                "_query_intent": c.get("_query_intent", "unknown"),
                # Evidence semantic score for this chunk
                "_evidence_chunk_scores": c.get("_evidence_chunk_scores", []) if retrieved_chunks else [],
            })

        scores = {
            "eval_id": eval_id,
            "question": result.get("query", ""),
            "pred_answer": result.get("final_answer", ""),
            "pred_doc_names": seen_doc_names,
            "pred_chunks": pred_chunks,
            "retrieved_chunks": retrieved_chunks,  # needed by _attribut_failure_reason
            "error": result.get("error", ""),
        }

        # Document-level scores
        doc_scores = score_document_level(sample, result)
        scores.update(doc_scores)

        # Page-level scores (strict + relaxed)
        page_scores = score_page_level(sample, result, page_window=page_window)
        scores.update(page_scores)

        # P0: Section-level scores (section = gold page ± window)
        section_scores = score_section_level(sample, result, page_window=page_window)
        scores.update(section_scores)

        # Evidence text-level scores (legacy exact substring)
        evidence_scores = score_evidence_text_level(sample, result)
        scores.update(evidence_scores)

        # P0: Evidence semantic hit (lexical fuzzy + embedding)
        semantic_scores = score_evidence_semantic_level(
            sample,
            result,
            semantic_threshold=evidence_semantic_threshold,
            use_embedding=use_evidence_embedding,
        )
        scores.update(semantic_scores)

        # Answer-level scores
        answer_scores = score_answer_level(sample, result)
        scores.update(answer_scores)

        # P1: Failure reason auto-attribution
        # Use scores dict (which has scored metrics) not raw result
        failure_reason = _attribut_failure_reason(
            sample,
            scores,  # pass scored dict, not raw result
            page_window=page_window,
        )
        scores["_failure_reason"] = failure_reason

        # Gold info (preserved for debugging)
        scores["gold_doc_name"] = sample.get("gold_doc_name", "")
        scores["gold_pages"] = sample.get("gold_pages", [])
        scores["gold_evidence_texts"] = sample.get("gold_evidence_texts", [])
        scores["gold_chunk_ids"] = sample.get("retrieval", {}).get("gold_chunk_ids", [])
        scores["gold_chunk_status"] = sample.get("gold_chunk_status", "unresolved")
        scores["gold_chunk_match_method"] = sample.get("gold_chunk_match_method", "")

        # Transparency: was retrieval actually scored vs. skipped due to missing gold?
        label_status = sample.get("retrieval", {}).get("label_status", "unlabeled")
        scores["retrieval_label_status"] = label_status
        retrieval_scored = label_status in {"labeled_doc", "labeled_chunk"}
        scores["retrieval_scored"] = retrieval_scored
        scores["retrieval_skip_reason"] = (
            "" if retrieval_scored else
            ("no_gold_doc_ids" if not sample.get("retrieval", {}).get("gold_doc_ids") else
             "unlabeled_or_unanswerable")
        )

        # V4 metrics: extract from result metadata if available
        result_metadata = result.get("metadata") or {}
        verification_result = result_metadata.get("verification") or {}
        refine_result = result_metadata.get("refine") or {}

        # V4: verifier results (new diagnostic structure)
        scores["v4_answer_supported"] = verification_result.get("is_supported", True) if verification_result else None
        scores["v4_support_level"] = verification_result.get("support_level", "unknown") if verification_result else None
        scores["v4_numeric_consistent"] = verification_result.get("numeric_consistency") if verification_result else None
        scores["v4_citation_adequate"] = verification_result.get("citation_adequate") if verification_result else None
        scores["v4_failure_reasons"] = verification_result.get("failure_reasons", []) if verification_result else []
        scores["v4_missing_requirements"] = verification_result.get("missing_requirements", []) if verification_result else []
        scores["v4_verifier_method"] = verification_result.get("method", "unknown") if verification_result else None

        # V4: refine results
        scores["v4_was_refined"] = refine_result.get("was_refined", False) if refine_result else None
        scores["v4_refinement_round"] = refine_result.get("refinement_round", 0) if refine_result else 0
        scores["v4_refine_trigger"] = refine_result.get("trigger_reason", "none") if refine_result else None
        scores["v4_refine_applied"] = refine_result.get("refinement_applied", False) if refine_result else None

        # V4: fallback tracking
        scores["v4_verifier_fallback"] = bool(result_metadata.get("verifier_fallback", False))
        scores["v4_refine_fallback"] = bool(result_metadata.get("refine_fallback", False))

        scored.append(scores)

    return scored


def compute_summary(scored: list[dict]) -> dict[str, Any]:
    """Compute aggregate summary statistics."""
    n = len(scored)
    if n == 0:
        return {}

    def _safe_rate(key, val_key="hit_at_1"):
        vals = [s.get(key) for s in scored if s.get(key) is not None]
        if not vals:
            return "N/A"
        cnt = sum(1 for v in vals if v == 1)
        return f"{cnt}/{n} ({cnt/n*100:.1f}%)"

    def _safe_mrr(key):
        vals = [s.get(key, 0) for s in scored if s.get(key) is not None]
        if not vals:
            return 0.0
        return round(sum(vals) / len(vals), 4)

    # Document-level
    doc_hit_at_1 = sum(1 for s in scored if s.get("doc_hit_at_1") == 1)
    doc_hit_at_5 = sum(1 for s in scored if s.get("doc_hit_at_5") == 1)
    doc_mrr_avg = _safe_mrr("doc_mrr")

    # Chunk/page-level
    chunk_hit_at_1 = sum(1 for s in scored if s.get("chunk_hit_at_1") == 1)
    chunk_hit_at_5 = sum(1 for s in scored if s.get("chunk_hit_at_5") == 1)
    chunk_mrr_vals = [s.get("chunk_mrr", 0) for s in scored if s.get("chunk_mrr") is not None]

    # Answer-level
    numeric_match = sum(1 for s in scored if s.get("numeric_match") == 1)
    exact_match = sum(1 for s in scored if s.get("normalized_exact_match") == 1)
    answer_labels = [s.get("answer_label", "wrong") for s in scored]
    label_counts = defaultdict(int)
    for lbl in answer_labels:
        label_counts[lbl] += 1

    return {
        "n": n,
        "document_level": {
            "doc_hit_at_1": f"{doc_hit_at_1}/{n} ({doc_hit_at_1/n*100:.1f}%)",
            "doc_hit_at_5": f"{doc_hit_at_5}/{n} ({doc_hit_at_5/n*100:.1f}%)",
            "doc_mrr_avg": doc_mrr_avg,
        },
        "page_level": {
            # Strict page
            "page_hit_at_1": f"{sum(1 for s in scored if s.get('page_hit_at_1') == 1)}/{n}",
            "page_hit_at_5": f"{sum(1 for s in scored if s.get('page_hit_at_5') == 1)}/{n}",
            "page_mrr_avg": _safe_mrr("page_mrr"),
            # Relaxed page
            "page_relaxed_hit_at_1": f"{sum(1 for s in scored if s.get('page_relaxed_hit_at_1') == 1)}/{n}",
            "page_relaxed_hit_at_5": f"{sum(1 for s in scored if s.get('page_relaxed_hit_at_5') == 1)}/{n}",
            "page_relaxed_mrr_avg": _safe_mrr("page_relaxed_mrr"),
        },
        "section_level": {
            "section_hit_at_1": f"{sum(1 for s in scored if s.get('section_hit_at_1') == 1)}/{n}",
            "section_hit_at_5": f"{sum(1 for s in scored if s.get('section_hit_at_5') == 1)}/{n}",
            "section_mrr_avg": _safe_mrr("section_mrr"),
        },
        "evidence_text_level": {
            "evidence_text_hit_at_1": f"{sum(1 for s in scored if s.get('evidence_text_hit_at_1') == 1)}/{n}",
            "evidence_text_hit_at_5": f"{sum(1 for s in scored if s.get('evidence_text_hit_at_5') == 1)}/{n}",
            "evidence_text_mrr_avg": _safe_mrr("evidence_text_mrr"),
        },
        "evidence_semantic_level": {
            "evidence_semantic_hit_at_1": f"{sum(1 for s in scored if s.get('evidence_semantic_hit_at_1') == 1)}/{n}",
            "evidence_semantic_hit_at_5": f"{sum(1 for s in scored if s.get('evidence_semantic_hit_at_5') == 1)}/{n}",
            "evidence_semantic_mrr_avg": _safe_mrr("evidence_semantic_mrr"),
            "evidence_lexical_hit_at_5": f"{sum(1 for s in scored if s.get('evidence_lexical_hit_at_5') == 1)}/{n}",
        },
        "answer_level": {
            "numeric_match": f"{numeric_match}/{n} ({numeric_match/n*100:.1f}%)",
            "normalized_exact_match": f"{exact_match}/{n} ({exact_match/n*100:.1f}%)",
            "label_distribution": dict(label_counts),
        },
        "gold_chunk_coverage": {
            "resolved": sum(1 for s in scored if s.get("gold_chunk_status") == "resolved"),
            "multi_chunk": sum(1 for s in scored if s.get("gold_chunk_status") == "multi_chunk"),
            "page_only": sum(1 for s in scored if s.get("gold_chunk_status") == "page_only"),
            "unresolved": sum(1 for s in scored if s.get("gold_chunk_status") == "unresolved"),
        },
        # V4 Pipeline metrics
        "v4_metrics": {
            # Answer quality
            "answer_supported_rate": f"{sum(1 for s in scored if s.get('v4_answer_supported') == True)}/{n}",
            "answer_refined_rate": f"{sum(1 for s in scored if s.get('v4_was_refined') == True)}/{n}",
            "numeric_verifier_pass_rate": f"{sum(1 for s in scored if s.get('v4_numeric_consistent') == True)}/{n}",
            "citation_adequate_rate": f"{sum(1 for s in scored if s.get('v4_citation_adequate') == True)}/{n}",
            # Support level distribution
            "support_level_high": f"{sum(1 for s in scored if s.get('v4_support_level') == 'high')}/{n}",
            "support_level_medium": f"{sum(1 for s in scored if s.get('v4_support_level') == 'medium')}/{n}",
            "support_level_low": f"{sum(1 for s in scored if s.get('v4_support_level') == 'low')}/{n}",
            # Fallback rates (lower is better)
            "verifier_fallback_rate": f"{sum(1 for s in scored if s.get('v4_verifier_fallback'))}/{n}",
            "refine_fallback_rate": f"{sum(1 for s in scored if s.get('v4_refine_fallback'))}/{n}",
            "llm_rerank_fallback_rate": f"{sum(1 for s in scored if s.get('llm_rerank_fallback'))}/{n}",
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Score an eval run at document + chunk + answer levels")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path, help="Path to run JSONL output file")
    group.add_argument("--report", type=Path, help="Path to run JSON report file")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to the original eval dataset JSONL (for gold label lookups)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for scored results (default: evals/results/<run_name>_scored.jsonl)",
    )
    parser.add_argument(
        "--page-window",
        default=0,
        type=int,
        help="Relaxed page match window (default: 0 = strict page match)",
    )
    parser.add_argument(
        "--use-evidence-embedding",
        action="store_true",
        default=True,
        help="Use embedding-based semantic similarity for evidence_semantic_hit (default: True). "
             "Disable with --no-evidence-embedding to fall back to lexical-only.",
    )
    parser.add_argument(
        "--no-evidence-embedding",
        dest="use_evidence_embedding",
        action="store_false",
        help="Disable embedding-based semantic evidence scoring.",
    )
    parser.add_argument(
        "--evidence-semantic-threshold",
        default=0.78,
        type=float,
        help="Threshold for embedding-based evidence semantic hit (default: 0.78).",
    )
    parser.add_argument(
        "--failure-reason-report",
        action="store_true",
        help="Also write a failure reason breakdown report to evals/reports/",
    )
    args = parser.parse_args()

    # Load run output
    run_path = args.run or args.report
    print(f"[score_eval] Loading run output from: {run_path}")
    case_results = load_run_output(run_path)
    print(f"[score_eval] Loaded {len(case_results)} case results")

    # Load dataset samples for gold label lookups
    dataset_samples: list[dict] = []
    if args.dataset:
        with args.dataset.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    dataset_samples.append(json.loads(line))
        print(f"[score_eval] Loaded {len(dataset_samples)} dataset samples")

    # Score
    print(f"[score_eval] Scoring (page_window={args.page_window}, "
          f"evidence_embedding={args.use_evidence_embedding}, "
          f"semantic_threshold={args.evidence_semantic_threshold})...")
    scored = score_run(
        case_results,
        dataset_samples,
        page_window=args.page_window,
        use_evidence_embedding=args.use_evidence_embedding,
        evidence_semantic_threshold=args.evidence_semantic_threshold,
    )

    # Write scored JSONL
    output_path = args.output
    if output_path is None:
        stem = run_path.stem
        output_path = run_path.parent / f"{stem}_scored.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in scored:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[score_eval] Scored results written to: {output_path}")

    # Compute and print summary
    summary = compute_summary(scored)
    print("\n=== Score Summary ===")
    print(f"  Total samples: {summary.get('n', 0)}")

    doc = summary.get("document_level", {})
    print(f"  Document – Hit@1: {doc.get('doc_hit_at_1', 'N/A')}  "
          f"Hit@5: {doc.get('doc_hit_at_5', 'N/A')}  "
          f"MRR: {doc.get('doc_mrr_avg', 'N/A')}")

    pg = summary.get("page_level", {})
    print(f"  Page (strict)     – Hit@1: {pg.get('page_hit_at_1', 'N/A')}  "
          f"Hit@5: {pg.get('page_hit_at_5', 'N/A')}  "
          f"MRR: {pg.get('page_mrr_avg', 'N/A')}")
    print(f"  Page (relaxed)    – Hit@1: {pg.get('page_relaxed_hit_at_1', 'N/A')}  "
          f"Hit@5: {pg.get('page_relaxed_hit_at_5', 'N/A')}  "
          f"MRR: {pg.get('page_relaxed_mrr_avg', 'N/A')}")

    sec = summary.get("section_level", {})
    print(f"  Section           – Hit@1: {sec.get('section_hit_at_1', 'N/A')}  "
          f"Hit@5: {sec.get('section_hit_at_5', 'N/A')}  "
          f"MRR: {sec.get('section_mrr_avg', 'N/A')}")

    ev = summary.get("evidence_text_level", {})
    print(f"  Evidence text     – Hit@1: {ev.get('evidence_text_hit_at_1', 'N/A')}  "
          f"Hit@5: {ev.get('evidence_text_hit_at_5', 'N/A')}  "
          f"MRR: {ev.get('evidence_text_mrr_avg', 'N/A')}")

    es = summary.get("evidence_semantic_level", {})
    print(f"  Evidence semantic – Hit@1: {es.get('evidence_semantic_hit_at_1', 'N/A')}  "
          f"Hit@5: {es.get('evidence_semantic_hit_at_5', 'N/A')}  "
          f"MRR: {es.get('evidence_semantic_mrr_avg', 'N/A')}")
    print(f"  Evidence lexical   – Hit@5: {es.get('evidence_lexical_hit_at_5', 'N/A')}")

    ans = summary.get("answer_level", {})
    print(f"  Answer  – Numeric: {ans.get('numeric_match', 'N/A')}  "
          f"Exact: {ans.get('normalized_exact_match', 'N/A')}")
    print(f"  Answer label dist: {ans.get('label_distribution', {})}")

    cov = summary.get("gold_chunk_coverage", {})
    print(f"  Gold chunk coverage – resolved: {cov.get('resolved', 0)}  "
          f"multi: {cov.get('multi_chunk', 0)}  "
          f"page_only: {cov.get('page_only', 0)}  "
          f"unresolved: {cov.get('unresolved', 0)}")

    # Write JSON summary
    summary_path = output_path.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[score_eval] Summary written to: {summary_path}")

    # P1: Write failure reason breakdown
    if args.failure_reason_report:
        failure_reasons = defaultdict(list)
        for s in scored:
            fr = s.get("_failure_reason", "unknown")
            failure_reasons[fr].append({
                "id": s.get("eval_id", ""),
                "question": s.get("question", "")[:80],
                "answer_label": s.get("answer_label", ""),
                "doc_hit_at_5": s.get("doc_hit_at_5"),
                "page_relaxed_hit_at_5": s.get("page_relaxed_hit_at_5"),
                "evidence_semantic_hit_at_5": s.get("evidence_semantic_hit_at_5"),
                "numeric_match": s.get("numeric_match"),
            })

        failure_report_path = output_path.parent.parent / "reports" / f"{output_path.stem}_failures.json"
        failure_report_path.parent.mkdir(parents=True, exist_ok=True)
        failure_report_data = {
            "total_samples": len(scored),
            "failure_counts": {k: len(v) for k, v in failure_reasons.items()},
            "failure_details": dict(failure_reasons),
        }
        with failure_report_path.open("w", encoding="utf-8") as f:
            json.dump(failure_report_data, f, ensure_ascii=False, indent=2)
        print(f"[score_eval] Failure reason report written to: {failure_report_path}")


if __name__ == "__main__":
    main()
