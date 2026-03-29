"""
verifier_service.py — Diagnostic answer verifier for v4 pipeline.

Design:
  - Diagnostic verifier: outputs structured failure_reasons for engineering debugging
  - LLM verifier + lightweight heuristic checks (heuristic runs first as gate)
  - Config-gated; falls back to heuristic on any LLM failure
  - Output structure:
      is_supported: bool
      support_level: "low"|"medium"|"high"
      numeric_consistency: bool|null
      citation_adequate: bool
      failure_reasons: list[str]   # specific engineering-readable reasons
      missing_requirements: list[str]
      summary: str
      method: "llm"|"heuristic"|"disabled"|"fallback"
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import app.config as config
from app.services.common import normalize_whitespace, to_float

logger = logging.getLogger(__name__)


def _cfg(name: str, default: Any):
    return getattr(config, name, default)


# =============================================================================
# Config
# =============================================================================

ENABLE_ANSWER_VERIFIER = bool(_cfg("V4_ENABLE_ANSWER_VERIFIER", True))  # Default: ON
VERIFIER_MODEL = str(_cfg("V4_VERIFIER_MODEL", "")) or None
VERIFIER_TEMPERATURE = float(_cfg("V4_VERIFIER_TEMPERATURE", 0.0))
VERIFIER_THRESHOLD = float(_cfg("V4_VERIFIER_THRESHOLD", 0.5))
VERIFIER_LLM_WEIGHT = float(_cfg("V4_VERIFIER_LLM_WEIGHT", 0.6))


# =============================================================================
# Failure reason taxonomy (engineering-readable)
# =============================================================================

class FailureReason:
    MISSING_NUMERIC_VALUE = "missing_numeric_value"
    NUMERIC_INCONSISTENCY = "numeric_inconsistency"
    CITATION_MISSING = "citation_missing"
    EVIDENCE_TOO_GENERIC = "evidence_too_generic"
    ANSWER_EXCEEDS_CONTEXT = "answer_exceeds_context"
    LIST_ITEMS_NOT_GROUNDED = "list_items_not_grounded"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ANSWER_EMPTY = "answer_empty"
    REFUSAL_DESPITE_EVIDENCE = "refusal_despite_evidence"
    VERIFIER_PARSE_FALLBACK = "verifier_parse_fallback"
    VERIFIER_LLM_FAILURE = "verifier_llm_failure"
    NO_GOLD_LABEL = "no_gold_label"


# =============================================================================
# Numeric helpers
# =============================================================================

def _extract_numbers(text: str) -> list[str]:
    """Extract all numeric-like strings from text."""
    return re.findall(r"[\d,]+\.?\d*%?|[\$€£¥]?\d+\.?\d*%", text)


def _normalize_number(s: str) -> str:
    """Normalize a number string for comparison."""
    s = re.sub(r"[$€£¥%]", "", s)
    s = re.sub(r",", "", s)
    return s.strip()


def _numbers_overlap(answer_text: str, evidence_text: str) -> float:
    """
    Compute [0,1] overlap score between numbers in answer and evidence.
    Returns 1.0 if all answer numbers are found in evidence.
    """
    ans_nums = _extract_numbers(answer_text)
    if not ans_nums:
        return 1.0

    ev_nums = set(_normalize_number(n) for n in _extract_numbers(evidence_text))

    hits = 0
    for num in ans_nums:
        norm = _normalize_number(num)
        if not norm:
            continue
        for ev_norm in ev_nums:
            if norm == ev_norm or norm in ev_norm or ev_norm in norm:
                hits += 1
                break

    return hits / len(ans_nums)


# =============================================================================
# LLM Verifier Prompts — Diagnostic
# =============================================================================

VERIFIER_SYSTEM_PROMPT = """You are an expert at verifying whether a draft answer is fully supported by evidence chunks.

Your task:
1. Check if the answer is GROUNDED in the provided evidence (not just topically related)
2. Check if any NUMBERS in the answer MATCH the evidence
3. Identify SPECIFIC FAILURE REASONS if the answer is not fully supported
4. List MISSING REQUIREMENTS — what the question asked for that the answer does not provide

Output ONLY valid JSON:
{
  "is_supported": true/false,
  "support_level": "low|medium|high",
  "numeric_consistency": true/false/null,
  "citation_adequate": true/false,
  "failure_reasons": ["specific_reason", ...],
  "missing_requirements": ["what is missing", ...],
  "summary": "2-3 sentence explanation"
}

Rules:
- is_supported = true ONLY if answer is fully grounded in evidence
- support_level: "high" if all evidence matches, "medium" if partially supported, "low" if mostly unsupported
- If answer mentions numbers, they MUST appear in evidence (failing this = numeric_inconsistency)
- citation_adequate = does the answer cite evidence that actually supports it?
- failure_reasons MUST include specific engineering-readable reasons from this list:
    missing_numeric_value, numeric_inconsistency, citation_missing, evidence_too_generic,
    answer_exceeds_context, list_items_not_grounded, insufficient_evidence
- missing_requirements = what the question asked for that answer doesn't provide
- NEVER output generic reasons like "not supported" — be specific"""


def build_verifier_prompt(
    query: str,
    draft_answer: str,
    evidence_chunks: list[dict[str, Any]],
    intent: str = "unknown",
) -> tuple[str, str]:
    """Build system + user prompt for verifier."""
    evidence_lines: list[str] = []
    for i, chunk in enumerate(evidence_chunks[:5]):
        chunk_id = chunk.get("chunk_id") or chunk.get("id") or f"E{i+1}"
        text = (chunk.get("chunk_text", "") or chunk.get("search_text", "") or "")[:500]
        page = chunk.get("page_start", "")
        section = chunk.get("section_title", "") or ""
        evidence_lines.append(
            f"[{chunk_id}] Page={page} Section={section}\n{normalize_whitespace(text)}"
        )

    evidence_text = "\n\n---\n\n".join(evidence_lines)

    system = VERIFIER_SYSTEM_PROMPT
    user = (
        f"Question: {query}\n"
        f"Question intent: {intent}\n\n"
        f"Draft Answer: {draft_answer}\n\n"
        f"Evidence Chunks:\n{evidence_text}\n\n"
        f"Return JSON verification result."
    )
    return system, user


# =============================================================================
# Heuristic verifier (fast gate — runs before LLM)
# =============================================================================

def _heuristic_verify(
    query: str,
    draft_answer: str,
    evidence_chunks: list[dict[str, Any]],
    intent: str,
) -> dict[str, Any]:
    """
    Fast heuristic verification. Returns a diagnostic result dict.
    Used as gate before LLM, and as fallback on LLM failure.
    """
    draft_lower = (draft_answer or "").lower()
    combined_evidence = " ".join(
        (c.get("chunk_text", "") or c.get("search_text", "") or "")
        for c in evidence_chunks
    )
    combined_lower = combined_evidence.lower()
    evidence_texts = [c.get("chunk_text", "") or c.get("search_text", "") or "" for c in evidence_chunks]

    failure_reasons: list[str] = []
    missing_requirements: list[str] = []

    # Check: empty answer
    if not draft_answer or len(draft_answer.strip()) < 3:
        return _make_result(
            is_supported=False,
            support_level="low",
            numeric_consistency=None,
            citation_adequate=False,
            failure_reasons=[FailureReason.ANSWER_EMPTY],
            missing_requirements=["no answer provided"],
            summary="Answer is empty or too short to evaluate.",
            method="heuristic",
        )

    # Check: refusal despite evidence available
    refusal_phrases = {
        "insufficient information", "not enough information", "cannot confirm",
        "no relevant", "insufficient context", "知识库中未找到",
    }
    has_refusal = any(r in draft_lower for r in refusal_phrases)
    if has_refusal and len(combined_evidence) > 100:
        failure_reasons.append(FailureReason.REFUSAL_DESPITE_EVIDENCE)
        missing_requirements.append("refusal despite relevant evidence available")
        return _make_result(
            is_supported=False,
            support_level="low",
            numeric_consistency=None,
            citation_adequate=False,
            failure_reasons=failure_reasons,
            missing_requirements=missing_requirements,
            summary="Answer refused despite evidence being available.",
            method="heuristic",
        )

    # Check: citation presence
    cited_ids: list[str] = []
    for chunk in evidence_chunks[:5]:
        cid = str(chunk.get("chunk_id") or chunk.get("id") or "")
        if cid and cid in draft_answer:
            cited_ids.append(cid)
    citation_adequate = len(cited_ids) > 0
    if not citation_adequate:
        failure_reasons.append(FailureReason.CITATION_MISSING)

    # Check: numeric consistency
    num_overlap = _numbers_overlap(draft_answer, combined_evidence)
    has_numbers_in_answer = bool(_extract_numbers(draft_answer))

    numeric_consistency: bool | None = None
    if has_numbers_in_answer:
        numeric_consistency = num_overlap >= 0.8
        if not numeric_consistency:
            failure_reasons.append(FailureReason.NUMERIC_INCONSISTENCY)
            missing_requirements.append("answer numbers not found in evidence")

    # Check: coverage (query terms in evidence)
    terms_in_query = set(re.findall(r"\b\w{4,}\b", query.lower()))
    terms_in_evidence = set(re.findall(r"\b\w{4,}\b", combined_lower))
    coverage = len(terms_in_query & terms_in_evidence) / max(1, len(terms_in_query))

    if coverage < 0.4:
        failure_reasons.append(FailureReason.INSUFFICIENT_EVIDENCE)
        missing_requirements.append("low evidence coverage of query terms")
    elif coverage < 0.7 and failure_reasons:
        failure_reasons.append(FailureReason.EVIDENCE_TOO_GENERIC)

    # Compute support_score
    support_score = (num_overlap * 0.4 + coverage * 0.4 + (0.2 if citation_adequate else 0.0))
    is_supported = support_score >= VERIFIER_THRESHOLD and (num_overlap >= 0.5 or not has_numbers_in_answer)

    # Determine support_level
    if is_supported and num_overlap >= 0.8 and coverage >= 0.7:
        support_level = "high"
    elif is_supported and num_overlap >= 0.5:
        support_level = "medium"
    else:
        support_level = "low"

    if not failure_reasons and not is_supported:
        failure_reasons.append(FailureReason.INSUFFICIENT_EVIDENCE)

    return _make_result(
        is_supported=is_supported,
        support_level=support_level,
        numeric_consistency=numeric_consistency,
        citation_adequate=citation_adequate,
        failure_reasons=failure_reasons,
        missing_requirements=missing_requirements,
        summary=f"heuristic: num_overlap={num_overlap:.2f} coverage={coverage:.2f} citation={citation_adequate}",
        method="heuristic",
    )


def _make_result(
    is_supported: bool,
    support_level: str,
    numeric_consistency: bool | None,
    citation_adequate: bool,
    failure_reasons: list[str],
    missing_requirements: list[str],
    summary: str,
    method: str,
) -> dict[str, Any]:
    """Construct a standardized verification result dict."""
    return {
        "is_supported": is_supported,
        "support_level": support_level,
        "numeric_consistency": numeric_consistency,
        "citation_adequate": citation_adequate,
        "failure_reasons": failure_reasons,
        "missing_requirements": missing_requirements,
        "summary": summary,
        "method": method,
    }


# =============================================================================
# Main verifier
# =============================================================================

def verify_answer(
    query: str,
    draft_answer: str,
    evidence_chunks: list[dict[str, Any]],
    intent: str = "unknown",
) -> dict[str, Any]:
    """
    Verify whether draft_answer is supported by evidence_chunks.

    Returns diagnostic verification_result dict:
      - is_supported: bool
      - support_level: "low"|"medium"|"high"
      - numeric_consistency: bool|null
      - citation_adequate: bool
      - failure_reasons: list[str]
      - missing_requirements: list[str]
      - summary: str
      - method: "llm"|"heuristic"|"disabled"|"fallback"
    """
    if not ENABLE_ANSWER_VERIFIER:
        return _make_result(
            is_supported=True,
            support_level="high",
            numeric_consistency=None,
            citation_adequate=True,
            failure_reasons=[],
            missing_requirements=[],
            summary="verifier disabled",
            method="disabled",
        )

    if not evidence_chunks or not draft_answer:
        return _heuristic_verify(query, draft_answer, evidence_chunks, intent)

    # Heuristic runs first as gate
    heuristic_result = _heuristic_verify(query, draft_answer, evidence_chunks, intent)

    # If heuristic is confident (very low or very high), skip LLM
    if heuristic_result["support_level"] in ("low", "high"):
        return heuristic_result

    # Call LLM verifier for medium-confidence cases
    try:
        from app.services.llm_service import chat_completion_raw

        system_prompt, user_prompt = build_verifier_prompt(
            query, draft_answer, evidence_chunks, intent
        )

        raw = chat_completion_raw(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=VERIFIER_MODEL or None,
            temperature=VERIFIER_TEMPERATURE,
        )

        text = (raw or "").strip()

        # Try JSON parse with multiple strategies
        llm_result: dict[str, Any] = {}
        try:
            llm_result = json.loads(text)
        except json.JSONDecodeError:
            fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
            if fenced:
                try:
                    llm_result = json.loads(fenced.group(1))
                except json.JSONDecodeError:
                    pass

        if llm_result and isinstance(llm_result, dict):
            llm_score = float(max(0.0, min(1.0, to_float(llm_result.get("support_level") == "high" and 1.0 or llm_result.get("support_level") == "medium" and 0.6 or 0.3))))
            heur_score = 1.0 - (heuristic_result["support_level"] == "high" and 0.0 or heuristic_result["support_level"] == "medium" and 0.4 or 0.8)
            combined_support = VERIFIER_LLM_WEIGHT * (1.0 - llm_score) + (1 - VERIFIER_LLM_WEIGHT) * heur_score

            # Normalize to is_supported
            is_supported = llm_result.get("is_supported", False)
            if isinstance(llm_result.get("support_level"), str):
                support_level_map = {"high": "high", "medium": "medium", "low": "low"}
                support_level = support_level_map.get(str(llm_result.get("support_level")).lower(), "medium")
            else:
                support_level = "medium"

            failure_reasons = llm_result.get("failure_reasons") or []
            if not failure_reasons and not is_supported:
                failure_reasons.append(FailureReason.INSUFFICIENT_EVIDENCE)

            return _make_result(
                is_supported=is_supported,
                support_level=support_level,
                numeric_consistency=llm_result.get("numeric_consistency"),
                citation_adequate=bool(llm_result.get("citation_adequate", False)),
                failure_reasons=failure_reasons,
                missing_requirements=llm_result.get("missing_requirements") or [],
                summary=str(llm_result.get("summary", "")),
                method="llm",
            )

    except Exception as e:
        logger.warning("[verifier] LLM call failed: %s, falling back to heuristic", e)

    # Fallback to heuristic
    result = dict(heuristic_result)
    result["method"] = "fallback"
    result["failure_reasons"] = heuristic_result.get("failure_reasons", [])
    result["failure_reasons"].insert(0, FailureReason.VERIFIER_PARSE_FALLBACK)
    return result


# =============================================================================
# Numeric evidence extraction (for numeric QA)
# =============================================================================

def extract_numeric_evidence(
    evidence_chunks: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """
    Extract numeric evidence from chunks for numeric QA.
    Returns list of NumericEvidence dicts.
    """
    results: list[dict[str, Any]] = []

    patterns = [
        (r"([\d,]+\.?\d*%)", "percent"),
        (r"(\$[\d,]+\.?\d*\s*(million|billion|thousand)?)", "dollar"),
        (r"([\d,]+\.?\d*\s*(x|times|percent|%)?)", "ratio"),
        (r"(?:^|\s)([\d,]+\.?\d+)(?:\s|$)", "plain"),
    ]

    for chunk in evidence_chunks:
        text = (chunk.get("chunk_text", "") or chunk.get("search_text", "") or "")
        cid = chunk.get("chunk_id") or chunk.get("id") or ""
        page = chunk.get("page_start", "")
        section = chunk.get("section_title", "")

        for pat, unit in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                value_raw = match.group(0)
                start = max(0, match.start() - 40)
                end = min(len(text), match.end() + 40)
                surrounding = text[start:end]

                norm_str = re.sub(r"[$%,]", "", value_raw)
                try:
                    normalized = float(norm_str)
                except ValueError:
                    normalized = None

                results.append({
                    "value_raw": value_raw,
                    "normalized": normalized,
                    "unit": unit,
                    "surrounding_text": surrounding,
                    "chunk_id": cid,
                    "page": page,
                    "section": section,
                })

    return results
