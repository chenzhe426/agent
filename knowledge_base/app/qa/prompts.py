"""
Prompt construction and answer parsing for QA pipeline.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.services.common import normalize_whitespace, to_float

from .config import (
    QA_HIGHLIGHT_ENABLE,
    _build_highlight_spans,
    _normalize_chunk_source,
    _truncate_text,
)


def _build_answer_prompt(
    question: str,
    context: str,
    history_text: str = "",
) -> tuple[str, str]:
    """Build system + user prompts for free-text answer generation."""
    system_prompt = (
        "You are a financial document Q&A assistant drawing from a trusted knowledge base.\n\n"
        "PRIORITIES (in order):\n"
        "1. NUMBERS FIRST: Always lead with specific figures -- percentages, dollar amounts, "
        "ratios, units, years. Cite them verbatim from context.\n"
        "2. ENGLISH OUTPUT: Answer in English unless the user explicitly asks in Chinese.\n"
        "3. FINANCIAL RATIO QUESTIONS (quick ratio, current ratio, liquidity, margin, etc.):\n"
        "   a) FIRST identify the EXACT row labeled 'Total current liabilities' (or equivalent).\n"
        "      This is ALWAYS the denominator in ratio calculations. NEVER use a subtotal line "
        "(e.g., 'Accrued liabilities' alone) as the denominator.\n"
        "   b) For the numerator, list ALL component rows: cash, short-term investments, "
        "accounts receivable, receivables from related parties.\n"
        "   c) Add up ALL numerator components, then divide by the total (not a subtotal).\n"
        "   d) When financial tables show two columns (e.g., FY2022 | FY2021), ALWAYS use "
        "the FIRST column (most recent fiscal year) for current-year analysis.\n"
        "   e) Format: 'Quick ratio = (component1 + component2 + ...) / Total current liabilities'.\n"
        "4. DRIVER/IMPACT QUESTIONS: If the question asks 'what drove', 'what caused', "
        "'what is the impact of', structure your answer as:\n"
        "   - Positive drivers (+X% / $X / X basis points)\n"
        "   - Negative drivers (-X% / -$X / -X basis points)\n"
        "   - Quantified impact (X% of total change attributable to this factor)\n"
        "5. NO VAGUE REFUSALS: If the context contains relevant evidence, give a grounded answer. "
        "Only say 'insufficient information' if the context genuinely does not address the question. "
        "Do NOT use templates like 'no information in knowledge base' when evidence exists.\n"
        "6. ACCURACY: Never fabricate page numbers, sources, or facts. "
        "If context does not support a conclusion, state what IS supported.\n"
        "7. BREVITY: Short and direct. One short paragraph unless complexity demands more."
    )

    user_parts = []
    if history_text:
        user_parts.append("Conversation history:")
        user_parts.append(history_text)
        user_parts.append("")

    user_parts.append("Knowledge base context:")
    user_parts.append(context or "(no context available)")
    user_parts.append("")
    user_parts.append(f"Question: {question}")
    user_parts.append("")
    user_parts.append("Answer:")

    user_prompt = "\n".join(user_parts)
    return system_prompt, user_prompt


def _build_structured_answer_prompt(
    question: str,
    context: str,
    history_text: str = "",
) -> tuple[str, str]:
    """Build system + user prompts for structured (JSON) answer generation."""
    system_prompt = (
        "You are a financial document Q&A assistant drawing from a trusted knowledge base.\n\n"
        "PRIORITIES:\n"
        "1. NUMBERS FIRST: Lead with specific figures -- percentages, dollar amounts, ratios.\n"
        "2. ENGLISH OUTPUT: Answer in English unless the user explicitly asks in Chinese.\n"
        "3. DRIVER/IMPACT QUESTIONS: For 'what drove' / 'what caused' questions, use these JSON fields:\n"
        "   {'answer':'<brief direct answer with numbers first>',\n"
        "    'positive_drivers':['<driver 1 with quantified impact>',...],\n"
        "    'negative_drivers':['<driver 1 with quantified impact>',...],\n"
        "    'key_points':['<supporting fact 1>',...],\n"
        "    'confidence':0.0}\n"
        "4. NO VAGUE REFUSALS: Only state insufficient information when context genuinely lacks evidence.\n"
        "5. Output ONLY valid JSON. No markdown, no explanation."
    )

    user_parts = []
    if history_text:
        user_parts.append("Conversation history:")
        user_parts.append(history_text)
        user_parts.append("")

    user_parts.append("Knowledge base context:")
    user_parts.append(context or "(no context available)")
    user_parts.append("")
    user_parts.append(f"Question: {question}")
    user_parts.append("")
    user_parts.append("Output JSON:")

    user_prompt = "\n".join(user_parts)
    return system_prompt, user_prompt


def _safe_parse_structured_answer(raw_text: str) -> dict[str, Any]:
    """Parse raw LLM output into structured answer dict, with graceful fallback."""
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return {
            "answer": "",
            "positive_drivers": [],
            "negative_drivers": [],
            "key_points": [],
            "confidence": 0.0,
        }

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, re.S)
    if fenced:
        raw_text = fenced.group(1).strip()

    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return {
                "answer": normalize_whitespace(data.get("answer", "")),
                "positive_drivers": data.get("positive_drivers") if isinstance(data.get("positive_drivers"), list) else [],
                "negative_drivers": data.get("negative_drivers") if isinstance(data.get("negative_drivers"), list) else [],
                "key_points": data.get("key_points") if isinstance(data.get("key_points"), list) else [],
                "confidence": float(data.get("confidence", 0.0) or 0.0),
            }
    except Exception:
        pass

    return {
        "answer": normalize_whitespace(raw_text),
        "positive_drivers": [],
        "negative_drivers": [],
        "key_points": [],
        "confidence": 0.0,
    }


def _extract_cited_evidence_ids(answer_text: str, retrieved_chunks: list[dict[str, Any]]) -> list[str]:
    """Extract cited evidence IDs from answer text."""
    if not answer_text or not retrieved_chunks:
        return []

    cited: list[str] = []
    answer_lower = answer_text.lower()

    for i, chunk in enumerate(retrieved_chunks[:5]):
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or f"E{i+1}")
        if chunk_id.lower() in answer_lower or f"e{i+1}" in answer_lower or f"source {i+1}" in answer_lower:
            if chunk_id not in cited:
                cited.append(chunk_id)

    return cited


def _augment_answer_with_citations(answer_text: str, cited_ids: list[str]) -> str:
    """Add evidence citation suffix to answer text if not already present."""
    if not answer_text or not cited_ids:
        return answer_text

    if re.search(r"\[E\d+\]|\[Source \d+\]|\[\d+\]", answer_text):
        return answer_text

    citation_str = " [Evidence: " + ", ".join(f"[{cid}]" for cid in cited_ids[:3]) + "]"
    return answer_text + citation_str


def _estimate_confidence(retrieved_chunks: list[dict[str, Any]], answer_text: str) -> float:
    """Estimate confidence based on retrieval scores and answer content."""
    if not retrieved_chunks:
        return 0.05

    top_score = max([to_float(c.get("score")) for c in retrieved_chunks] + [0.0])
    avg_top3 = sum(to_float(c.get("score")) for c in retrieved_chunks[:3]) / max(
        1, min(3, len(retrieved_chunks))
    )

    confidence = 0.55 * top_score + 0.45 * avg_top3
    refusal_phrases = {
        "insufficient information", "not enough information", "cannot confirm",
        "no relevant", "not sufficient", "cannot determine", "insufficient context",
    }
    answer_lower = (answer_text or "").lower()
    if any(rp in answer_lower for rp in refusal_phrases):
        confidence *= 0.65

    return round(max(0.0, min(1.0, confidence)), 4)


def _build_sources(
    retrieved_chunks: list[dict[str, Any]],
    highlight: bool,
    highlight_terms: list[str],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Build source list from retrieved chunks with optional highlights."""
    sources: list[dict[str, Any]] = []

    for chunk in retrieved_chunks[:limit]:
        source = _normalize_chunk_source(chunk)
        quote = normalize_whitespace(source.get("quote", ""))
        source["quote"] = _truncate_text(quote, max_len=260)

        if highlight and QA_HIGHLIGHT_ENABLE:
            source["highlight_spans"] = _build_highlight_spans(quote, highlight_terms)

        sources.append(source)

    return sources
