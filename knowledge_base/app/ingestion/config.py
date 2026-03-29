"""
Configuration for the ingestion / parsing pipeline.
All values here are the defaults; they can be overridden via
environment variables or by instantiating ParsingConfig with custom values.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Parser enable/disable
# ---------------------------------------------------------------------------


@dataclass
class PdfParserConfig:
    """PDF parser strategy configuration."""

    # Enable / disable individual strategies
    use_docling: bool = True
    use_pdfplumber: bool = True
    use_pymupdf: bool = True  # PyMuPDF / pypdfium2 fallback

    # Quality score thresholds (0-1)
    min_quality_score: float = 0.30

    # Scanned PDF detection thresholds
    min_chars_per_page: int = 50        # below this → likely scanned
    max_garbage_ratio: float = 0.40     # above this → garbled text
    min_page_coverage_ratio: float = 0.10  # text coverage per page

    # Block-level quality weights (for scoring)
    weight_total_chars: float = 0.15
    weight_nonempty_blocks: float = 0.15
    weight_avg_block_len: float = 0.10
    weight_short_frag_ratio: float = 0.10   # penalty for short fragments
    weight_repeated_line_ratio: float = 0.15  # penalty for repeated lines
    weight_noise_ratio: float = 0.15          # penalty for noise lines
    weight_heading_paragraph_ratio: float = 0.10  # bonus for heading diversity
    weight_table_presence: float = 0.05        # bonus for tables
    weight_page_coverage: float = 0.05         # bonus for good page coverage

    # Table extraction
    extract_tables: bool = True

    # Heading detection for pdfplumber (font-size based heuristics)
    heading_font_size_threshold: float = 12.0  # points; above this → potential heading

    # Block text limits
    min_block_chars: int = 3          # blocks shorter than this are noise candidates
    max_block_chars_for_merge: int = 120  # blocks longer than this won't be merged away

    # Hyphenation repair
    repair_hyphenation: bool = True

    # Page header/footer removal (coordinate-based for PDF)
    remove_header_footer_by_coords: bool = True
    header_footer_y_threshold: float = 60.0   # points from top/bottom edge

    # Logging
    verbose: bool = False


@dataclass
class DocxParserConfig:
    """DOCX parser configuration."""

    # Paragraph style → heading level mapping (lowercase style name → level)
    style_to_level: dict[str, int] = field(default_factory=lambda: {
        "title": 1,
        "heading 1": 1,
        "heading 2": 2,
        "heading 3": 3,
        "heading 4": 4,
        "heading 5": 5,
        "heading 6": 6,
        "subtitle": 2,
        "heading": 1,
        "toc": 1,
    })

    # Preserve headings in content field (backward compat)
    include_headings_in_content: bool = False  # keep off; content is paragraph-only by default

    # List item detection
    list_pattern: str = r"^\s*([-*•·]|\d+[.)、，])\s+"

    # Table extraction
    extract_tables: bool = True

    # Metadata extraction
    extract_doc_props: bool = True

    # Remove template noise (page headers/footers that repeat in docx)
    remove_template_noise: bool = True

    # Logging
    verbose: bool = False


@dataclass
class TextParserConfig:
    """Plain-text (.txt, .md) parser configuration."""

    # Encoding fallback order (first one that decodes without errors wins)
    encoding_order: tuple[str, ...] = ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1")

    # Allow ignoring encoding errors
    encoding_errors: str = "replace"  # replace | ignore | strict

    # Paragraph reconstruction
    # - "double_newline" : empty line = paragraph break (default)
    # - "single_newline"  : treat single newlines as soft wraps, rebuild sentences
    # - "smart"           : combine single-newline lines that don't end with sentence punctuation
    paragraph_mode: str = "smart"

    # Minimum paragraph length to be kept (chars)
    min_paragraph_chars: int = 10

    # Merge very short fragments (< this many chars) into adjacent paragraph
    merge_fragments_shorter_than: int = 30

    # Heading detection patterns (regex strings or empty to disable)
    heading_patterns: list[str] = field(default_factory=lambda: [
        r"^\s{0,3}#{1,6}\s+(.+)$",               # Markdown # ## ###
        r"^\s*第[一二三四五六七八九十百千万零0-9]+[章节部分篇]\s+(.+)$",  # 第1章, 第1节
        r"^\s*[一二三四五六七八九十零]+[、.．]\s*(.+)$",               # 一、 二、
        r"^\s*\d+(\.\d+){0,3}\s+(.+)$",          # 1. 1.2 1.2.3
        r"^\s*[（(][一二三四五六七八九十0-9]+[)）]\s*(.+)$",           # （一）
        r"^\s*[=-]{3,}\s*$",                      # ---- or ==== divider line
        r"^\s*[\u2014\u2018\u2019]{1,2}\s*(.+)$",  # — title or "title
    ])

    # List item patterns
    list_patterns: list[str] = field(default_factory=lambda: [
        r"^\s*[-*•·]\s+",
        r"^\s*\d+[.)、，]\s+",
        r"^\s*[（(]\d+[)）]\s+",
        r"^\s*[a-zA-Z][.)]\s+",
    ])

    # Code block delimiters (markdown style)
    code_block_delimiters: tuple[tuple[str, str], ...] = (("```", "```"),)

    # Noise cleaning
    clean_page_numbers: bool = True
    clean_template_noise: bool = True
    clean_separator_lines: bool = True
    clean_orphaned_digits: bool = True

    # Repeated line detection
    detect_repeated_lines: bool = True
    repeated_line_threshold: int = 3  # remove if same line appears ≥ this many times

    # Hyphenation repair
    repair_hyphenation: bool = True

    # Logging
    verbose: bool = False


@dataclass
class CleaningConfig:
    """
    Shared cleaning/normalization configuration applied after parsing
    regardless of source format.
    """

    # Global noise patterns (applied to all formats)
    remove_page_numbers: bool = True
    remove_separator_lines: bool = True
    remove_template_noise: bool = True
    remove_repeated_lines: bool = True
    remove_orphaned_digits: bool = True

    # Whitespace
    normalize_whitespace: bool = True
    collapse_newlines: bool = True

    # Block filtering
    min_block_char_length: int = 2
    max_noise_line_length: int = 10  # lines ≤ this that match noise patterns are removed

    # Merge small blocks
    merge_adjacent_small_blocks: bool = True
    merge_min_chars: int = 35

    # Hyphenation repair
    repair_hyphenation: bool = True

    # Heading preservation in content (backward compat)
    include_headings_in_content: bool = False

    # Logging
    verbose: bool = False


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class ParsingConfig:
    """
    Combined configuration for the entire parsing pipeline.
    Pass an instance to parse_document(..., config=...) to override defaults.
    """

    pdf: PdfParserConfig = field(default_factory=PdfParserConfig)
    docx: DocxParserConfig = field(default_factory=DocxParserConfig)
    text: TextParserConfig = field(default_factory=TextParserConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)

    @classmethod
    def from_env(cls) -> "ParsingConfig":
        """Build config from environment variables (INGEST_* prefix)."""
        cfg = cls()
        prefix = "INGEST_"
        for var, value in os.environ.items():
            if not var.startswith(prefix):
                continue
            key = var[len(prefix):].lower()

            # Top-level toggles
            if key == "pdf_verbose":
                cfg.pdf.verbose = value.lower() in ("1", "true", "yes")
            elif key == "docx_verbose":
                cfg.docx.verbose = value.lower() in ("1", "true", "yes")
            elif key == "text_verbose":
                cfg.text.verbose = value.lower() in ("1", "true", "yes")
            elif key == "cleaning_verbose":
                cfg.cleaning.verbose = value.lower() in ("1", "true", "yes")
            elif key == "pdf_min_quality":
                try:
                    cfg.pdf.min_quality_score = float(value)
                except ValueError:
                    pass

        return cfg
