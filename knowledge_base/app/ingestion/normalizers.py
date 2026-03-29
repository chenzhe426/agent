"""
Shared block cleaning / normalization utilities.

These functions are applied after parsing, regardless of source format,
to ensure blocks are clean, deduplicated, and structured.

Design goals:
- Format-aware: can apply shared rules and format-specific rules
- Configurable: all aggressive options can be disabled
- Non-destructive: headings are NEVER silently dropped (only optionally
  excluded from the `content` fallback field)
- Extensible: add new patterns without touching the core logic
"""
from __future__ import annotations

import re
from typing import Any

from app.ingestion.config import CleaningConfig

# ---------------------------------------------------------------------------
# Shared noise patterns
# ---------------------------------------------------------------------------

_PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*第?\s*\d+\s*页\s*$", re.UNICODE),
    re.compile(r"^\s*Page\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*Pages?\s+\d+\s*[-–]\s*\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*-\s*\d+\s*$"),
    re.compile(r"^\s*\[\s*\d+\s*\]\s*$"),
    re.compile(r"^\s*第?\d+/\d+页?\s*$", re.UNICODE),
]

_TEMPLATE_NOISE_PATTERNS = [
    # Emails
    re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\s*$"),
    # URLs
    re.compile(r"^https?://\S+\s*$", re.IGNORECASE),
    re.compile(r"^www\.\S+\s*$", re.IGNORECASE),
    # Copyright lines
    re.compile(r"^\s*[\u00a9\u24b6-\u24fe]\s+\d{4}(\s+\S+)?\s*$"),
    # Box drawing / separator lines
    re.compile(r"^\s*[\u2500-\u257f\u2014\u2015]{3,}\s*$"),
    re.compile(r"^\s*[\u2500-\u257f]{10,}\s*$"),
    # Confidential / copyright headers
    re.compile(r"^\s*(CONFIDENTIAL|Proprietary|内部文件|机密|版权所有).*$/i"),
    # Document ID lines
    re.compile(r"^\s*(Doc\.?\s*ID\s*[:\-]?\s*)?[A-Z0-9]{6,}\s*$", re.IGNORECASE),
]

_SEVERE_NOISE_PATTERNS = [
    # Repeating single characters / garbled
    re.compile(r"^[_\-\.=\*·•]{5,}\s*$"),
    # Emoji / symbol spam
    re.compile(r"^[\U0001F300-\U0001F9FF]{2,}\s*$"),
]

# Characters considered "garbage" (control characters, zero-width, etc.)
_GARBAGE_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """
    Basic text normalization: whitespace, newlines, unicode.
    Applied early in the pipeline before block processing.
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace non-breaking space with regular space
    text = text.replace("\u00a0", " ")
    text = text.replace("\u3000", " ")  # ideographic space

    # Collapse multiple spaces/tabs into one
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace per line
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def normalize_block_text(text: str) -> str:
    """
    Normalize text within a single block.
    """
    text = normalize_text(text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def collapse_whitespace(text: str) -> str:
    """Collapse all runs of whitespace to a single space."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Block-level type normalization
# ---------------------------------------------------------------------------


def normalize_block_type(block_type: str | None) -> str:
    """Map parser-specific type strings to our canonical set."""
    bt = (block_type or "").strip().lower()
    mapping = {
        # Headings
        "title": "heading",
        "heading": "heading",
        "header": "heading",
        "subtitle": "heading",
        "toc": "heading",
        # Paragraphs
        "narrativetext": "paragraph",
        "paragraph": "paragraph",
        "text": "paragraph",
        "p": "paragraph",
        # Lists
        "listitem": "list_item",
        "list_item": "list_item",
        "listitemex": "list_item",
        "bullet": "list_item",
        "numbered": "list_item",
        # Tables
        "table": "table",
        "table_cell": "table",
        "tabledata": "table",
        "table_row": "table",
        # Quotes
        "quote": "quote",
        "blockquote": "quote",
        "citation": "quote",
        # Code
        "code": "code",
        "codeblock": "code",
        "sourcecode": "code",
        # Captions / metadata
        "caption": "caption",
        "image_caption": "caption",
        "table_caption": "caption",
        "figure": "caption",
        # Page furniture
        "header": "header",
        "footer": "footer",
        "page_header": "header",
        "page_footer": "footer",
        "page_number": "header",
        # Misc
        "metadata": "unknown",
        "unknown": "unknown",
        "embedded": "unknown",
    }
    return mapping.get(bt, "paragraph")


# ---------------------------------------------------------------------------
# Noise detection helpers
# ---------------------------------------------------------------------------


def is_noise_line(text: str, max_len: int = 10) -> bool:
    """
    Return True if `text` looks like a page number or noise artifact.
    `max_len` lets callers be more or less strict.
    """
    s = (text or "").strip()
    if not s:
        return True

    for pattern in _PAGE_NUMBER_PATTERNS:
        if pattern.match(s):
            return True

    if re.fullmatch(r"[-_=*·•]{3,}", s):
        return True

    return False


def is_severe_noise(text: str) -> bool:
    """Return True if text is definitely garbage / corrupt."""
    s = (text or "").strip()
    if not s:
        return True
    for pat in _SEVERE_NOISE_PATTERNS:
        if pat.match(s):
            return True
    # Control characters
    if _GARBAGE_CHARS.search(s):
        return True
    return False


def is_template_noise(text: str) -> bool:
    """Return True if text matches known template/header/footer patterns."""
    s = (text or "").strip()
    if not s:
        return True
    for pat in _TEMPLATE_NOISE_PATTERNS:
        if pat.match(s):
            return True
    return False


def is_repeated_line(text: str, seen: set[str], threshold: int = 3) -> bool:
    """
    Return True if `text` (stripped, lowercased) already appears in `seen`
    at least `threshold` times.  Callers should maintain a Counter externally.
    """
    key = s.strip().lower() if (s := text) else ""
    if not key or len(key) < 4:
        return False
    return seen.get(key, 0) >= threshold


# ---------------------------------------------------------------------------
# Hyphenation repair
# ---------------------------------------------------------------------------


_HYPHENATED_WORD = re.compile(r"\b(\w+)-\n(\w+)\b")
_HYPHENATED_SOFT = re.compile(r"\b(\w+)‑\n(\w+)\b")  # non-breaking hyphen


def repair_hyphenation(text: str) -> str:
    """
    Merge hyphenated line breaks:
      'invest-\nment'   → 'investment'
      'invest‑\nment'   → 'investment'  (non-breaking hyphen)
    """
    text = _HYPHENATED_WORD.sub(r"\1\2", text)
    text = _HYPHENATED_SOFT.sub(r"\1\2", text)
    return text


# ---------------------------------------------------------------------------
# Core block cleaner
# ---------------------------------------------------------------------------


def clean_blocks(
    blocks: list[dict[str, Any]],
    config: CleaningConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Apply all cleaning/normalization steps to a list of blocks.

    Steps (in order):
    1. Text normalization
    2. Type normalization
    3. Noise filtering (page numbers, separators)
    4. Template noise filtering
    5. Repeated-line deduplication
    6. Block merging (small fragments → adjacent blocks)
    7. Block filtering (min length, severe garbage)

    Headings are never dropped; they are always kept in the blocks list
    (they are only optionally excluded from the `content` fallback field).
    """
    if config is None:
        config = CleaningConfig()

    if not blocks:
        return []

    # Pre-scan for repeated lines (used during filtering)
    line_counts: dict[str, int] = {}
    if config.remove_repeated_lines:
        for block in blocks:
            text = block.get("text", "") or ""
            key = text.strip().lower()
            if key and len(key) >= 4:
                line_counts[key] = line_counts.get(key, 0) + 1

    cleaned: list[dict[str, Any]] = []

    for idx, block in enumerate(blocks):
        text = normalize_block_text(block.get("text", "") or "")

        # ---- Noise filtering ----
        if config.remove_page_numbers and is_noise_line(text):
            continue
        if config.remove_separator_lines and re.fullmatch(r"[-_=*·•]{3,}", text.strip()):
            continue
        if config.remove_template_noise and is_template_noise(text):
            continue
        if is_severe_noise(text):
            continue

        # ---- Repeated line removal ----
        if config.remove_repeated_lines:
            key = text.strip().lower()
            if key and len(key) >= 4 and line_counts.get(key, 0) >= 3:
                # Only skip if this is a very short repeated line
                if len(text) < 80:
                    continue

        # ---- Block length filter ----
        if len(text) < config.min_block_char_length:
            continue

        # ---- Hyphenation repair ----
        if config.repair_hyphenation:
            text = repair_hyphenation(text)

        item: dict[str, Any] = dict(block)
        item["text"] = text
        item["type"] = normalize_block_type(block.get("type"))
        item["block_index"] = block.get("block_index", idx)
        # Ensure source_format is set
        if "source_format" not in item:
            item["source_format"] = block.get("source_format", "unknown")
        cleaned.append(item)

    # ---- Merge small adjacent fragments ----
    if config.merge_adjacent_small_blocks:
        cleaned = _merge_small_adjacent_blocks(cleaned, min_chars=config.merge_min_chars)

    return cleaned


# ---------------------------------------------------------------------------
# Small-block merging
# ---------------------------------------------------------------------------

_TRANSITION_PHRASES = {
    "例如：", "比如：", "也就是说：", "也就是：", "原因是：",
    "可以拆成：", "如下：", "比如", "例如", "具体来说：",
}


def _merge_small_adjacent_blocks(
    blocks: list[dict[str, Any]],
    min_chars: int = 35,
) -> list[dict[str, Any]]:
    """
    Merge very short fragments (< min_chars) into adjacent paragraph/list
    blocks within the same section.
    """
    if not blocks:
        return []

    merged: list[dict[str, Any]] = []
    i = 0

    while i < len(blocks):
        cur = dict(blocks[i])
        cur_text = cur.get("text", "").strip()
        cur_type = cur.get("type", "paragraph")
        cur_section = cur.get("section_path")

        # Rule 1: tiny "transition" lines merge forward
        is_transition = (
            len(cur_text) < min_chars
            and cur_type in {"paragraph", "list_item", "quote"}
            and cur_text in _TRANSITION_PHRASES
        )

        # Rule 2: extremely short lines (<12 chars) merge forward
        is_tiny = len(cur_text) < 12

        if (is_transition or is_tiny) and i + 1 < len(blocks):
            nxt = dict(blocks[i + 1])
            nxt_type = nxt.get("type", "paragraph")
            nxt_section = nxt.get("section_path")
            if nxt_type in {"paragraph", "list_item", "quote"} and nxt_section == cur_section:
                nxt["text"] = f"{cur_text}\n{nxt.get('text', '').strip()}".strip()
                blocks[i + 1] = nxt
                i += 1
                continue

        # Rule 3: short block merges backward into previous same-section block
        if merged:
            prev = merged[-1]
            same_section = prev.get("section_path") == cur_section
            merge_back = (
                len(cur_text) < min_chars
                and cur_type == prev.get("type")
                and cur_type in {"paragraph", "list_item"}
                and same_section
            )
            if merge_back:
                prev["text"] = f"{prev.get('text', '').strip()}\n{cur_text}".strip()
                i += 1
                continue

        merged.append(cur)
        i += 1

    return merged


# ---------------------------------------------------------------------------
# Utility: build content string from blocks
# ---------------------------------------------------------------------------


def blocks_to_content(
    blocks: list[dict[str, Any]],
    include_headings: bool = False,
) -> str:
    """
    Concatenate block texts into a single content string.
    By default headings are excluded (backward-compatible behavior).

    Parameters
    ----------
    blocks : list[dict]
        Cleaned blocks from clean_blocks()
    include_headings : bool
        If True, prepend heading texts before their section paragraphs.
    """
    if not blocks:
        return ""

    if include_headings:
        # Group blocks by section and prepend heading text to first paragraph
        section_parts: list[str] = []
        current_section: str | None = None
        current_para_parts: list[str] = []

        def flush():
            nonlocal current_para_parts
            if current_para_parts:
                section_parts.append("\n".join(current_para_parts))
            current_para_parts = []

        for block in blocks:
            btype = block.get("type", "paragraph")
            btext = block.get("text", "").strip()
            bsection = block.get("section_path")

            if bsection != current_section:
                flush()
                current_section = bsection

            if btype == "heading":
                flush()
                current_para_parts.append(btext)
            else:
                current_para_parts.append(btext)

        flush()
        return "\n\n".join(section_parts).strip()

    # Default: skip headings (backward-compatible)
    return "\n\n".join(
        b.get("text", "") for b in blocks if b.get("type") != "heading"
    ).strip()
