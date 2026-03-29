"""
reranker_service.py — Answerability-first LLM reranker for v4 pipeline.

Design:
  - Answerability-first: scores chunks on whether they DIRECTLY answer the query
  - NOT generic semantic relevance — specifically whether chunk contains evidence
  - Batched single-prompt LLM scoring
  - Robust JSON parsing with full fallback to heuristic on any failure
  - Config-gated with clear on/off switch
  - Cost controlled: only top-N candidates sent to LLM
"""

from __future__ import annotations

import json
import re
import logging
from typing import Any

import app.config as config
from app.services.common import normalize_whitespace, to_float

logger = logging.getLogger(__name__)


def _cfg(name: str, default: Any):
    return getattr(config, name, default)


# =============================================================================
# Config
# =============================================================================

ENABLE_LLM_RERANK = bool(_cfg("V4_ENABLE_LLM_RERANK", True))  # Default: ON
LLM_RERANK_TOP_N = int(_cfg("V4_LLM_RERANK_TOP_N", 8))  # Cost control: default 8
LLM_RERANK_WEIGHT = float(_cfg("V4_LLM_RERANK_WEIGHT", 0.4))
LLM_RERANK_MODEL = str(_cfg("V4_LLM_RERANK_MODEL", "")) or None
LLM_RERANK_TEMPERATURE = float(_cfg("V4_LLM_RERANK_TEMPERATURE", 0.0))

# Query intent constants (shared with retrieval_service)
QUERY_INTENT_NUMERIC = "numeric_fact"
QUERY_INTENT_LIST = "list_fact"
QUERY_INTENT_DESCRIPTIVE = "descriptive"
QUERY_INTENT_HYBRID = "hybrid"

# Answerability labels
ANSWERABILITY_DIRECTLY_ANSWERABLE = "directly_answerable"
ANSWERABILITY_PARTIALLY_USEFUL = "partially_useful"
ANSWERABILITY_TOPICALLY_RELATED = "topically_related"
ANSWERABILITY_NOISE = "noise"


# =============================================================================
# LLM Reranker Prompts — Answerability-first
# =============================================================================

RERANKER_SYSTEM_PROMPT = """You are an expert at evaluating whether a text chunk DIRECTLY PROVIDES EVIDENCE to answer a financial question.

CORE TASK: For each chunk, determine if it CONTAINS THE ACTUAL ANSWER, not just if it's topically related.

Scoring criteria:
  1.0 = DIRECTLY ANSWERABLE: Chunk contains the specific numbers, facts, or statements that directly answer the question
  0.7-0.9 = PARTIALLY USEFUL: Chunk has relevant evidence but requires inference or extraction
  0.4-0.6 = TOPICALLY RELATED: Chunk mentions the topic but does not contain the answer
  0.1-0.3 = MOSTLY NOISE: Chunk mentions keywords but is not useful for this question
  0.0 = IRRELEVANT OR MISLEADING: Chunk is off-topic or contradicts the question

NUMERIC QUESTIONS (ratio, tax rate, revenue, margin):
  → SCORE HIGH: chunks with actual dollar amounts, percentages, ratios, financial figures
  → SCORE HIGH: table cells, financial statements, numerical summaries
  → PENALIZE: narrative paragraphs without numbers, background descriptions, risk factors without data
  → NEVER let a narrative-only chunk outrank a numeric evidence chunk

LIST QUESTIONS (products, services, segments):
  → SCORE HIGH: chunks that explicitly enumerate/list items
  → SCORE HIGH: table rows listing products, segments, business units
  → PENALIZE: general business discussion without explicit lists

DESCRIPTIVE QUESTIONS (why, how, discuss):
  → SCORE HIGH: analytical paragraphs with causal explanations, management discussion
  → PENALIZE: table of contents, page headers, boilerplate descriptions

Output ONLY valid JSON:
{"scores": [{"chunk_id": "<id>", "relevance": 0.0-1.0, "answerability": "directly_answerable|partially_useful|topically_related|noise", "rationale": "<2-3 word reason>"}, ...]}

Do not include any explanation outside the JSON. Evaluate ALL chunks."""


def build_rerank_prompt(
    query: str,
    intent: str,
    candidates: list[dict[str, Any]],
) -> str:
    """
    Build the user prompt for batched LLM reranking.
    Truncates chunk text to 400 chars for cost control.
    """
    intent_label = {
        QUERY_INTENT_NUMERIC: "NUMERIC (ratio, tax, revenue, margin, etc.)",
        QUERY_INTENT_LIST: "LIST (products, services, segments, etc.)",
        QUERY_INTENT_DESCRIPTIVE: "DESCRIPTIVE (why, how, explain, discuss)",
        QUERY_INTENT_HYBRID: "GENERAL",
    }.get(intent, "GENERAL")

    lines = [
        f"Question: {query}",
        f"Question type: {intent_label}",
        f"",
        f"Evaluate {len(candidates)} candidate chunks:",
        "",
    ]

    for i, cand in enumerate(candidates):
        chunk_id = cand.get("chunk_id") or cand.get("id") or f"chunk_{i}"
        doc_title = cand.get("title", "") or ""
        section = cand.get("section_title", "") or ""
        page = cand.get("page_start", "")
        chunk_type = cand.get("chunk_type", "paragraph")

        # Truncate for cost control (400 chars)
        text = (cand.get("chunk_text", "") or cand.get("search_text", "") or "")[:400]
        text = normalize_whitespace(text)

        header = f"[{i+1}] ID={chunk_id}"
        if doc_title:
            header += f" | Doc: {doc_title[:30]}"
        if section:
            header += f" | Section: {section[:40]}"
        if page:
            header += f" | Page: {page}"
        header += f" | Type: {chunk_type}"

        lines.append(header)
        lines.append(f"Content: {text}")
        lines.append("")

    lines.append("Return JSON with scores for ALL chunks listed above.")
    return "\n".join(lines)


# =============================================================================
# Parsing helpers
# =============================================================================

def _parse_rerank_json(raw: str, num_expected: int) -> tuple[list[dict[str, Any]], bool]:
    """
    Parse LLM reranker JSON output.
    Returns (scores_list, parse_success).
    parse_success=False means full fallback to heuristic.
    """
    text = (raw or "").strip()
    if not text:
        return [], False

    answerability_map = {
        "directly_answerable": ANSWERABILITY_DIRECTLY_ANSWERABLE,
        "directly-answerable": ANSWERABILITY_DIRECTLY_ANSWERABLE,
        "partially_useful": ANSWERABILITY_PARTIALLY_USEFUL,
        "partially-useful": ANSWERABILITY_PARTIALLY_USEFUL,
        "topically_related": ANSWERABILITY_TOPICALLY_RELATED,
        "topically-related": ANSWERABILITY_TOPICALLY_RELATED,
        "noise": ANSWERABILITY_NOISE,
        "irrelevant": ANSWERABILITY_NOISE,
    }

    def _normalize_score(s: dict) -> dict:
        relevance = float(max(0.0, min(1.0, to_float(s.get("relevance", 0.0)))))
        raw_label = str(s.get("answerability", "topically_related")).lower()
        label = answerability_map.get(raw_label, ANSWERABILITY_TOPICALLY_RELATED)
        return {
            "chunk_id": str(s.get("chunk_id", "")),
            "relevance": relevance,
            "answerability": label,
            "rationale": str(s.get("rationale", "") or "")[:50],
        }

    # Try direct JSON parse
    for parse_method, data_or_text in [
        ("direct", lambda: json.loads(text)),
        ("fenced", lambda: json.loads(re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S).group(1)) if re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S) else None),
    ]:
        try:
            data = data_or_text()
            if isinstance(data, dict) and "scores" in data:
                scores = data["scores"]
                if isinstance(scores, list):
                    parsed = [_normalize_score(s) for s in scores if s.get("chunk_id") is not None]
                    if parsed:
                        return parsed, True
        except Exception:
            continue

    # Try finding any {...} with scores array
    for brace_match in re.finditer(r"\{.*?\}", text, re.S):
        try:
            data = json.loads(brace_match.group())
            if isinstance(data, dict) and "scores" in data:
                scores = data["scores"]
                if isinstance(scores, list):
                    parsed = [_normalize_score(s) for s in scores if s.get("chunk_id") is not None]
                    if parsed:
                        return parsed, True
        except Exception:
            continue

    return [], False


# =============================================================================
# Main reranker function
# =============================================================================

def rerank_with_llm(
    query: str,
    intent: str,
    candidates: list[dict[str, Any]],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """
    Rerank candidates using answerability-first LLM judgment.

    Returns candidates with new fields:
      - llm_relevance_score : float (0-1)
      - llm_answerability : str (directly_answerable/partially_useful/topically_related/noise)
      - llm_rationale : str
      - llm_combined_score : float (heuristic * (1-w) + llm * w)
      - llm_rerank_applied : bool
      - llm_rerank_fallback : bool (True if LLM failed and heuristic was used)

    Falls back to heuristic-only scoring on any LLM failure.
    """
    if not ENABLE_LLM_RERANK or not candidates:
        result = _mark_reranked(candidates, applied=False, fallback=True)
        for r in result:
            r["llm_rerank_fallback"] = True
            r["llm_answerability"] = ANSWERABILITY_TOPICALLY_RELATED
        return result

    effective_top_n = top_n if top_n is not None else LLM_RERANK_TOP_N
    top_candidates = candidates[:effective_top_n]

    # Build prompt
    user_prompt = build_rerank_prompt(query, intent, top_candidates)

    # Call LLM
    parsed_scores: list[dict[str, Any]] = []
    parse_success = False
    llm_error = None

    try:
        from app.services.llm_service import chat_completion_raw
        raw = chat_completion_raw(
            system_prompt=RERANKER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=LLM_RERANK_MODEL or None,
            temperature=LLM_RERANK_TEMPERATURE,
        )
        parsed_scores, parse_success = _parse_rerank_json(raw, len(top_candidates))
    except Exception as e:
        llm_error = str(e)
        logger.warning("[reranker] LLM call failed: %s", llm_error)

    # Build score lookup
    score_lookup: dict[str, dict[str, Any]] = {}
    for s in parsed_scores:
        cid = s["chunk_id"]
        score_lookup[cid] = s

    # Determine if we fell back
    fallback = not parse_success or not parsed_scores

    # Apply scores to ALL candidates
    reranked: list[dict[str, Any]] = []
    for cand in candidates:
        item = dict(cand)
        cid = str(cand.get("chunk_id") or cand.get("id") or "")

        if cid in score_lookup and not fallback:
            item["llm_relevance_score"] = score_lookup[cid]["relevance"]
            item["llm_answerability"] = score_lookup[cid]["answerability"]
            item["llm_rationale"] = score_lookup[cid]["rationale"]
            item["llm_rerank_fallback"] = False
        else:
            # Not scored by LLM or fallback — assign based on heuristic position
            item["llm_relevance_score"] = 0.1
            item["llm_answerability"] = ANSWERABILITY_NOISE if fallback else ANSWERABILITY_TOPICALLY_RELATED
            item["llm_rationale"] = "not evaluated" if not fallback else "llm_fallback"
            item["llm_rerank_fallback"] = fallback

        # Combined score: heuristic * (1-w) + llm * w
        heuristic = to_float(item.get("final_score", 0.0))
        llm_score = to_float(item.get("llm_relevance_score", 0.0))
        combined = (1.0 - LLM_RERANK_WEIGHT) * heuristic + LLM_RERANK_WEIGHT * llm_score
        item["llm_combined_score"] = round(combined, 6)
        item["llm_rerank_applied"] = ENABLE_LLM_RERANK

        reranked.append(item)

    # Sort by combined score
    reranked.sort(key=lambda x: x.get("llm_combined_score", 0.0), reverse=True)

    if fallback:
        logger.info("[reranker] LLM rerank fallback to heuristic (parse_success=%s, scores=%d, error=%s)",
                     parse_success, len(parsed_scores), llm_error or "none")

    return reranked


def _mark_reranked(
    candidates: list[dict[str, Any]],
    applied: bool,
    fallback: bool = True,
) -> list[dict[str, Any]]:
    """Helper: mark all candidates as not LLM-reranked."""
    result: list[dict[str, Any]] = []
    for cand in candidates:
        item = dict(cand)
        item["llm_relevance_score"] = 0.0
        item["llm_answerability"] = ANSWERABILITY_TOPICALLY_RELATED
        item["llm_rationale"] = ""
        item["llm_combined_score"] = to_float(item.get("final_score", 0.0))
        item["llm_rerank_applied"] = applied
        item["llm_rerank_fallback"] = fallback
        result.append(item)
    return result
