"""
Parser for plain-text files (.txt, .md, .markdown).

Features
--------
- Robust encoding detection (UTF-8 → GBK → GB18030 → Latin-1)
- Smart paragraph reconstruction (single-newline soft-wrap → sentence merge)
- Markdown-aware structure recognition
- Heading detection via multiple patterns (Markdown, Chinese, numbered)
- List item preservation
- Code block and blockquote detection
- Noise cleaning (page numbers, separators, repeated lines)
- Unified block output (same schema as PDF/DOCX)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.ingestion.config import CleaningConfig, TextParserConfig, ParsingConfig
from app.ingestion.loaders import load_text_file
from app.ingestion.normalizers import (
    blocks_to_content,
    clean_blocks,
    normalize_text,
)
from app.ingestion.parsers.base import BaseParser
from app.ingestion.schemas import ParsedDocument


# ---------------------------------------------------------------------------
# Heading / list pattern compilation
# ---------------------------------------------------------------------------


def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p))
        except re.error:
            pass
    return compiled


# ---------------------------------------------------------------------------
# Smart paragraph reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_paragraphs_smart(
    lines: list[str],
    config: TextParserConfig,
) -> list[str]:
    """
    Reconstruct paragraphs from lines using smart heuristics.

    Strategy
    --------
    - Empty double-newline  → paragraph break (definite boundary)
    - Single newline        → soft wrap; merge unless the previous line
                               ends with sentence-final punctuation
    - If previous line ends with 。！？!?.;:  → paragraph break
    - If next line starts with uppercase (likely a new sentence) → merge
    - Otherwise merge with space
    """
    if not lines:
        return []

    SENTENCE_END_CHARS = frozenset("。！？!?;:.．;")
    PARAGRAPH_END_CHARS = frozenset("。！？!?;.．")
    result: list[str] = []
    buf: list[str] = []

    def _flush():
        nonlocal buf
        if buf:
            result.append(" ".join(buf).strip())
            buf = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line:
            # Empty line → definite paragraph break
            _flush()
            continue

        # Double newline (empty line was already handled above,
        # but check original for safety)
        if not line:
            _flush()
            continue

        if not buf:
            buf.append(line)
            continue

        prev = buf[-1]

        # Hard paragraph break: prev ends with Chinese/sentence-final punct
        if prev and prev[-1] in PARAGRAPH_END_CHARS:
            _flush()
            buf.append(line)
            continue

        # Also break before Markdown headings / list items that look like new items
        if re.match(r"^\s*[-*#>\d]", line) and len(line) < 80:
            _flush()
            buf.append(line)
            continue

        # Merge as soft wrap
        buf.append(line)

    _flush()
    return result


def _reconstruct_paragraphs_double_newline(
    lines: list[str],
) -> list[str]:
    """Classic approach: empty line = paragraph break."""
    paragraphs: list[str] = []
    buf: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buf:
                paragraphs.append(" ".join(buf).strip())
                buf = []
            continue
        buf.append(stripped)

    if buf:
        paragraphs.append(" ".join(buf).strip())

    return [p for p in paragraphs if p]


# ---------------------------------------------------------------------------
# Block builder
# ---------------------------------------------------------------------------


def _build_blocks_from_lines(
    lines: list[str],
    config: TextParserConfig,
) -> list[dict[str, Any]]:
    """
    Convert raw text lines into typed blocks with section tracking.
    """
    heading_patterns = _compile_patterns(config.heading_patterns)
    list_patterns = _compile_patterns(config.list_patterns)

    # Reconstruct paragraphs
    if config.paragraph_mode == "smart":
        para_texts = _reconstruct_paragraphs_smart(lines, config)
    elif config.paragraph_mode == "single_newline":
        para_texts = [ln.strip() for ln in lines if ln.strip()]
    else:
        para_texts = _reconstruct_paragraphs_double_newline(lines)

    blocks: list[dict[str, Any]] = []
    section_stack: list[str] = []
    code_block_buf: list[str] = []
    in_code_block = False
    code_lang = ""

    def _flush_code_block():
        nonlocal code_block_buf, in_code_block
        if code_block_buf:
            blocks.append(
                {
                    "type": "code",
                    "text": "\n".join(code_block_buf).strip(),
                    "section_path": " > ".join(section_stack) if section_stack else None,
                    "parser_source": "text",
                }
            )
            code_block_buf = []
        in_code_block = False

    def _detect_heading_level(text: str) -> int:
        s = text.strip()
        if s.startswith("#"):
            return len(s) - len(s.lstrip("#"))
        if re.match(r"^第[一二三四五六七八九十百千万零0-9]+章", s):
            return 1
        if re.match(r"^第[一二三四五六七八九十百千万零0-9]+节", s):
            return 2
        if re.match(r"^[一二三四五六七八九十]+[、.．]", s):
            return 2
        if re.match(r"^\d+(\.\d+){0,3}\s+", s):
            dots = re.match(r"^(\d+(?:\.\d+)*)", s).group(1).count(".")
            return min(4, max(2, dots + 2))
        return 1

    def _section_push(text: str, level: int):
        nonlocal section_stack
        while len(section_stack) >= level:
            section_stack.pop()
        section_stack.append(text)

    for para in para_texts:
        stripped = para.strip()
        if not stripped:
            continue

        # ---- Code block delimiter ----
        if "```" in stripped:
            if not in_code_block:
                # Opening fence
                m = re.match(r"```(\w*)", stripped)
                code_lang = m.group(1) if m else ""
                in_code_block = True
                code_block_buf = []
            else:
                # Closing fence
                _flush_code_block()
            continue

        if in_code_block:
            code_block_buf.append(para)
            continue

        # ---- Markdown heading ----
        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            _section_push(heading_text, level)
            blocks.append(
                {
                    "type": "heading",
                    "text": heading_text,
                    "heading_level": level,
                    "section_path": " > ".join(section_stack),
                    "parser_source": "text",
                }
            )
            continue

        # ---- Delimiter / separator line ----
        if re.match(r"^\s*[=-]{3,}\s*$", stripped):
            blocks.append(
                {
                    "type": "heading",
                    "text": stripped,
                    "heading_level": 1,
                    "section_path": " > ".join(section_stack) if section_stack else None,
                    "parser_source": "text",
                }
            )
            continue

        # ---- Check heading patterns ----
        is_heading = False
        heading_text = stripped
        heading_level = 1

        for pat in heading_patterns:
            m = pat.match(stripped)
            if m:
                is_heading = True
                heading_text = m.group(1).strip() if m.groups() else stripped
                heading_level = _detect_heading_level(stripped)
                break

        if is_heading:
            _section_push(heading_text, heading_level)
            blocks.append(
                {
                    "type": "heading",
                    "text": heading_text,
                    "heading_level": heading_level,
                    "section_path": " > ".join(section_stack),
                    "parser_source": "text",
                }
            )
            continue

        # ---- List items ----
        is_list = False
        for pat in list_patterns:
            if pat.match(stripped):
                is_list = True
                break

        if is_list:
            blocks.append(
                {
                    "type": "list_item",
                    "text": stripped,
                    "section_path": " > ".join(section_stack) if section_stack else None,
                    "parser_source": "text",
                }
            )
            continue

        # ---- Blockquote ----
        if stripped.startswith(">"):
            blocks.append(
                {
                    "type": "quote",
                    "text": stripped.lstrip(">").strip(),
                    "section_path": " > ".join(section_stack) if section_stack else None,
                    "parser_source": "text",
                }
            )
            continue

        # ---- Markdown table ----
        if stripped.startswith("|") and "|" in stripped[1:]:
            blocks.append(
                {
                    "type": "table",
                    "text": stripped,
                    "section_path": " > ".join(section_stack) if section_stack else None,
                    "parser_source": "text",
                }
            )
            continue

        # ---- Regular paragraph ----
        blocks.append(
            {
                "type": "paragraph",
                "text": stripped,
                "section_path": " > ".join(section_stack) if section_stack else None,
                "parser_source": "text",
            }
        )

    # Flush any remaining code block
    if in_code_block:
        _flush_code_block()

    return blocks


# ---------------------------------------------------------------------------
# Text Parser – public interface
# ---------------------------------------------------------------------------


class TextParser(BaseParser):
    """
    Parser for plain-text files with smart structure detection.

    The parser:
    1. Detects encoding (not just UTF-8)
    2. Reconstructs paragraphs using the configured mode
    3. Identifies headings, lists, tables, code blocks
    4. Cleans noise and produces unified blocks
    """

    file_type = "text"

    def __init__(self, config: ParsingConfig | None = None):
        self._cfg = config.text if config else TextParserConfig()
        self._clean_cfg = config.cleaning if config else CleaningConfig()

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)

        # ---- Load with encoding detection ----
        try:
            raw = load_text_file(path, self._cfg)
        except Exception as e:
            raise RuntimeError(f"failed to read {path}: {e}") from e

        if not raw:
            return ParsedDocument(
                title=path.stem,
                content="",
                blocks=[],
                source_path=str(path),
                file_type=self.file_type,
                metadata={
                    "source_format": "txt",
                    "source_file": str(path),
                    "errors": [],
                },
            )

        # ---- Hyphenation repair (before line splitting) ----
        if self._cfg.repair_hyphenation:
            raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)
            raw = re.sub(r"(\w)­\n(\w)", r"\1\2", raw)  # non-breaking hyphen

        # ---- Split into lines ----
        lines = [ln.rstrip() for ln in raw.split("\n")]

        # ---- Build typed blocks ----
        blocks = _build_blocks_from_lines(lines, self._cfg)

        # Tag source format
        for block in blocks:
            block["source_format"] = "txt"

        # ---- Clean ----
        cleaned_blocks = clean_blocks(blocks, self._clean_cfg)
        for b in cleaned_blocks:
            b.setdefault("source_format", "txt")

        content = blocks_to_content(
            cleaned_blocks,
            include_headings=self._clean_cfg.include_headings_in_content,
        )

        table_count = sum(1 for b in cleaned_blocks if b.get("type") == "table")
        heading_count = sum(1 for b in cleaned_blocks if b.get("type") == "heading")
        list_count = sum(1 for b in cleaned_blocks if b.get("type") == "list_item")

        quality_score = (
            0.30 * min(len(content) / 2000, 1.0)
            + 0.25 * min(heading_count / 5, 1.0)
            + 0.20 * min(list_count / 10, 1.0)
            + 0.25 * (1.0 if table_count > 0 else 0.0)
        )

        metadata = {
            "parser_used": "text",
            "quality_score": round(quality_score, 4),
            "table_count": table_count,
            "heading_count": heading_count,
            "list_count": list_count,
            "source_format": "txt",
            "source_file": str(path),
            "paragraph_mode": self._cfg.paragraph_mode,
            "encoding_used": self._cfg.encoding_order[0],
            "errors": [],
        }

        return ParsedDocument(
            title=path.stem,
            content=content,
            blocks=cleaned_blocks,
            source_path=str(path),
            file_type=self.file_type,
            metadata=metadata,
        )
