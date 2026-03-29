"""
refine_service.py — Self-refine answer service for v4 pipeline.

Design:
  - Default OFF: only triggers when verifier explicitly indicates failure
  - Engineering-gated: enable_self_refine must be True
  - Trigger conditions (ALL must be true):
      1. verifier.is_supported = False
      2. OR verifier.numeric_consistency = False
      3. OR verifier.citation_adequate = False
      4. AND evidence chunks are not empty (not a retrieval miss)
      5. AND draft answer is not empty
  - Max 1 round
  - Fallback to draft answer if refine fails
  - Explicit failure_reasons passed through from verifier
"""

from __future__ import annotations

import logging
from typing import Any

import app.config as config
from app.services.common import normalize_whitespace

logger = logging.getLogger(__name__)


def _cfg(name: str, default: Any):
    return getattr(config, name, default)


# =============================================================================
# Config
# =============================================================================

ENABLE_SELF_REFINE = bool(_cfg("V4_ENABLE_SELF_REFINE", False))  # Default: OFF
MAX_REFINE_ROUNDS = int(_cfg("V4_MAX_REFINE_ROUNDS", 1))
REFINE_MODEL = str(_cfg("V4_REFINE_MODEL", "")) or None
REFINE_TEMPERATURE = float(_cfg("V4_REFINE_TEMPERATURE", 0.2))
REFINE_TRIGGER_REQUIRES_VERIFIER_FAIL = bool(_cfg("V4_REFINE_TRIGGER_REQUIRES_VERIFIER_FAIL", True))


# =============================================================================
# Refinement prompt
# =============================================================================

REFINE_SYSTEM_PROMPT = """You are an expert at improving draft answers to financial questions.

You will receive:
1. The original question
2. The current draft answer (which may be incomplete or unsupported)
3. Verification feedback indicating what is missing or wrong
4. Evidence chunks from the knowledge base

Your task:
- Rewrite/improve the draft answer to address the missing_requirements
- Keep the answer grounded ONLY in the provided evidence
- Do NOT fabricate any numbers, facts, or citations not in the evidence
- If numbers are required, extract them from the evidence chunks
- Cite evidence IDs [E1], [E2], etc. when using specific facts from chunks

Output ONLY the improved answer text. No explanation, no JSON."""


def build_refine_prompt(
    query: str,
    draft_answer: str,
    verification_result: dict[str, Any],
    evidence_chunks: list[dict[str, Any]],
) -> tuple[str, str]:
    """Build system + user prompt for self-refine."""
    system = REFINE_SYSTEM_PROMPT

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

    missing = verification_result.get("missing_requirements") or []
    missing_str = ", ".join(missing) if missing else "unspecified issues"

    failure_reasons = verification_result.get("failure_reasons") or []
    failure_str = ", ".join(failure_reasons) if failure_reasons else "none"

    hallucination = verification_result.get("hallucination_flags") or []
    hallucination_str = ""
    if hallucination:
        hallucination_str = "\n\nHallucinated claims to AVOID:\n" + "\n".join(f"  - {h}" for h in hallucination)

    user = (
        f"Original Question: {query}\n\n"
        f"Current Draft Answer:\n{draft_answer}\n\n"
        f"Verification Feedback:\n"
        f"  - is_supported: {verification_result.get('is_supported', False)}\n"
        f"  - support_level: {verification_result.get('support_level', 'unknown')}\n"
        f"  - failure_reasons: {failure_str}\n"
        f"  - missing_requirements: {missing_str}\n"
        f"  - numeric_consistency: {verification_result.get('numeric_consistency')}\n"
        f"  - citation_adequate: {verification_result.get('citation_adequate', False)}\n"
        f"{hallucination_str}\n\n"
        f"Evidence Chunks:\n{evidence_text}\n\n"
        f"Output ONLY the improved answer. Address the missing_requirements explicitly."
    )
    return system, user


# =============================================================================
# Trigger condition check
# =============================================================================

def _should_refine(
    verification_result: dict[str, Any],
    draft_answer: str,
    evidence_chunks: list[dict[str, Any]],
) -> tuple[bool, str]:
    """
    Check if refinement should be triggered.

    Returns (should_refine, reason).
    """
    if not ENABLE_SELF_REFINE:
        return False, "disabled"

    if not evidence_chunks:
        return False, "no_evidence_chunks"

    if not draft_answer or len(draft_answer.strip()) < 3:
        return False, "empty_draft_answer"

    # Verifier must have explicitly indicated a fixable failure
    is_supported = verification_result.get("is_supported", True)
    numeric_consistency = verification_result.get("numeric_consistency")
    citation_adequate = verification_result.get("citation_adequate", True)
    method = verification_result.get("method", "unknown")

    # If verifier is disabled or heuristic-only, don't refine (no reliable signal)
    if method in ("disabled", "unknown"):
        return False, f"verifier_method={method}"

    # Check failure reasons for non-fixable cases
    failure_reasons = verification_result.get("failure_reasons") or []
    non_fixable = {"verifier_llm_failure", "verifier_parse_fallback", "answer_empty",
                   "refusal_despite_evidence", "no_gold_label"}
    if any(r in non_fixable for r in failure_reasons):
        return False, f"non_fixable_failure={failure_reasons}"

    # Core trigger: verifier says not supported
    if not is_supported:
        return True, "verifier_unsupported"

    # Numeric inconsistency trigger
    if numeric_consistency is False:
        return True, "numeric_inconsistency"

    # Missing citation trigger
    if not citation_adequate:
        return True, "citation_missing"

    return False, "no_trigger_condition"


# =============================================================================
# Main refine function
# =============================================================================

def refine_answer(
    query: str,
    draft_answer: str,
    verification_result: dict[str, Any],
    evidence_chunks: list[dict[str, Any]],
    round_num: int = 1,
) -> dict[str, Any]:
    """
    Refine a draft answer based on verification feedback.

    Returns refinement_result dict:
      - refined_answer: str (empty if no refinement)
      - was_refined: bool
      - refinement_round: int
      - refinement_applied: bool
      - trigger_reason: str
      - missing_requirements_addressed: list[str]
      - method: "llm"|"disabled"|"skipped"|"failed"|"no_trigger"
    """
    # Check trigger conditions first
    should, reason = _should_refine(verification_result, draft_answer, evidence_chunks)

    if not should:
        return {
            "refined_answer": "",
            "was_refined": False,
            "refinement_round": 0,
            "refinement_applied": False,
            "trigger_reason": reason,
            "missing_requirements_addressed": [],
            "method": "skipped",
        }

    if round_num > MAX_REFINE_ROUNDS:
        return {
            "refined_answer": "",
            "was_refined": False,
            "refinement_round": round_num,
            "refinement_applied": False,
            "trigger_reason": reason,
            "missing_requirements_addressed": [],
            "method": "skipped",
        }

    # Build refine prompt
    system_prompt, user_prompt = build_refine_prompt(
        query, draft_answer, verification_result, evidence_chunks
    )

    try:
        from app.services.llm_service import chat_completion_raw

        refined = chat_completion_raw(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=REFINE_MODEL or None,
            temperature=REFINE_TEMPERATURE,
        )
        refined = normalize_whitespace(refined or "")

        if refined and refined != draft_answer:
            logger.info("[refine] Successfully refined answer (trigger=%s)", reason)
            return {
                "refined_answer": refined,
                "was_refined": True,
                "refinement_round": round_num,
                "refinement_applied": True,
                "trigger_reason": reason,
                "missing_requirements_addressed": verification_result.get("missing_requirements") or [],
                "method": "llm",
            }

    except Exception as e:
        logger.warning("[refine] Refine LLM call failed: %s", e)

    # Fallback: return draft answer unchanged
    logger.info("[refine] Refine failed, returning draft answer (trigger=%s)", reason)
    return {
        "refined_answer": draft_answer,  # Fallback to draft
        "was_refined": False,
        "refinement_round": round_num,
        "refinement_applied": False,
        "trigger_reason": reason,
        "missing_requirements_addressed": [],
        "method": "failed",
    }
