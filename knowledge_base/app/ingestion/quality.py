"""
PDF parse quality scoring.

Given a list of blocks from a PDF parser candidate, compute a 0-1 quality
score and return a detailed breakdown so the multi-strategy selector can
make an informed decision.
"""
from __future__ import annotations

import re
from typing import Any

from app.ingestion.config import PdfParserConfig


# ---------------------------------------------------------------------------
# Noise patterns (language-agnostic)
# ---------------------------------------------------------------------------

_PAGE_NOISE_PATTERNS = [
    re.compile(r"^\s*第?\s*\d+\s*页\s*$", re.UNICODE),
    re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*-\s*\d+\s*$"),
    re.compile(r"^\s*\[\s*\d+\s*\]\s*$"),
]

_HEADER_FOOTER_CANDIDATE_PATTERNS = [
    re.compile(r"^\s*[\u4e00-\u9fff]{2,8}\s*$"),  # Short Chinese header/footer
    re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\s*$"),  # email
    re.compile(r"^https?://", re.IGNORECASE),
    re.compile(r"^www\.", re.IGNORECASE),
    re.compile(r"^[\u00a9\u24b6-\u24fe]\s+\d{4}\s"),  # © 2024
    re.compile(r"^\s*[\u2500-\u257f]{3,}\s*$"),  # box-drawing noise
]

_REPEATED_LINE_THRESHOLD = 3  # count to trigger removal


def _is_noise_line(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    for pat in _PAGE_NOISE_PATTERNS:
        if pat.match(s):
            return True
    # Box-drawing separators
    if re.fullmatch(r"[-_=*·•]{3,}", s):
        return True
    return False


def _is_header_footer_candidate(text: str) -> bool:
    for pat in _HEADER_FOOTER_CANDIDATE_PATTERNS:
        if pat.match(text.strip()):
            return True
    return False


def score_pdf_blocks(
    blocks: list[dict[str, Any]],
    config: PdfParserConfig | None = None,
) -> dict[str, Any]:
    """
    Score a list of PDF blocks for parse quality.

    Returns a dict with:
        quality_score  : float 0-1
        total_chars    : int
        nonempty_blocks: int
        avg_block_len  : float
        short_frag_ratio : float  (0 = no short fragments, 1 = all short)
        repeated_line_ratio : float
        noise_ratio     : float
        heading_paragraph_ratio : float
        table_count     : int
        page_coverage   : float   (0-1 proportion of pages covered by blocks)
        likely_scanned  : bool
        ocr_recommended : bool
        score_breakdown : dict
    """
    if config is None:
        config = PdfParserConfig()

    if not blocks:
        return _empty_score()

    texts = [b.get("text", "") or "" for b in blocks]
    total_chars = sum(len(t) for t in texts)
    nonempty_blocks = sum(1 for t in texts if t.strip())
    avg_block_len = total_chars / len(blocks) if blocks else 0.0

    # Short fragment ratio (blocks < min_block_chars)
    short_frags = sum(1 for t in texts if 0 < len(t.strip()) < config.min_block_chars)
    short_frag_ratio = short_frags / len(blocks) if blocks else 0.0

    # Repeated line ratio
    line_counts: dict[str, int] = {}
    for t in texts:
        line_key = t.strip().lower()
        if line_key:
            line_counts[line_key] = line_counts.get(line_key, 0) + 1
    repeated_lines = sum(1 for cnt in line_counts.values() if cnt >= _REPEATED_LINE_THRESHOLD)
    repeated_line_ratio = repeated_lines / len(line_counts) if line_counts else 0.0

    # Noise ratio
    noise_lines = sum(1 for t in texts if _is_noise_line(t))
    noise_ratio = noise_lines / len(blocks) if blocks else 0.0

    # Heading vs paragraph diversity
    block_types = [b.get("type", "paragraph") for b in blocks]
    heading_count = sum(1 for t in block_types if t in {"heading", "title"})
    heading_paragraph_ratio = heading_count / len(blocks) if blocks else 0.0

    # Table presence
    table_count = sum(1 for t in block_types if t == "table")

    # Page coverage
    pages_with_text = len({b.get("page") for b in blocks if b.get("page") is not None})
    unique_pages = len({b.get("page") for b in blocks if b.get("page") is not None})
    page_count_est = max(pages_with_text, unique_pages or 1)
    page_coverage = pages_with_text / max(page_count_est, 1)

    # Garbage ratio (characters that are garbage-like)
    garbage_chars = sum(1 for c in "".join(texts) if ord(c) < 32 and c not in "\n\t")
    garbage_ratio = garbage_chars / max(total_chars, 1)

    # Combine into weighted score
    w = config
    score = (
        w.weight_total_chars * min(total_chars / 5000, 1.0)
        + w.weight_nonempty_blocks * min(nonempty_blocks / 50, 1.0)
        + w.weight_avg_block_len * min(avg_block_len / 150, 1.0)
        - w.weight_short_frag_ratio * short_frag_ratio
        - w.weight_repeated_line_ratio * repeated_line_ratio
        - w.weight_noise_ratio * noise_ratio
        + w.weight_heading_paragraph_ratio * heading_paragraph_ratio
        + w.weight_table_presence * (1.0 if table_count > 0 else 0.0)
        + w.weight_page_coverage * page_coverage
    )

    # Clamp
    quality_score = max(0.0, min(1.0, score))

    # Likely scanned detection
    chars_per_page_est = total_chars / max(page_count_est, 1)
    likely_scanned = (
        chars_per_page_est < config.min_chars_per_page
        and len(blocks) > 3
        and short_frag_ratio > 0.6
    ) or (garbage_ratio > config.max_garbage_ratio)

    ocr_recommended = likely_scanned and not any(
        b.get("type") == "table" for b in blocks
    )

    breakdown = {
        "total_chars": total_chars,
        "nonempty_blocks": nonempty_blocks,
        "avg_block_len": round(avg_block_len, 2),
        "short_frag_ratio": round(short_frag_ratio, 3),
        "repeated_line_ratio": round(repeated_line_ratio, 3),
        "noise_ratio": round(noise_ratio, 3),
        "garbage_ratio": round(garbage_ratio, 4),
        "heading_paragraph_ratio": round(heading_paragraph_ratio, 3),
        "table_count": table_count,
        "page_coverage": round(page_coverage, 3),
        "likely_scanned": likely_scanned,
        "ocr_recommended": ocr_recommended,
    }

    return {
        "quality_score": round(quality_score, 4),
        **breakdown,
    }


def _empty_score() -> dict[str, Any]:
    return {
        "quality_score": 0.0,
        "total_chars": 0,
        "nonempty_blocks": 0,
        "avg_block_len": 0.0,
        "short_frag_ratio": 1.0,
        "repeated_line_ratio": 0.0,
        "noise_ratio": 1.0,
        "garbage_ratio": 0.0,
        "heading_paragraph_ratio": 0.0,
        "table_count": 0,
        "page_coverage": 0.0,
        "likely_scanned": False,
        "ocr_recommended": False,
    }
