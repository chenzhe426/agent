from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedDocument:
    """
    Unified structured document representation after parsing.

    Attributes
    ----------
    title : str
        Document title (usually filename stem).
    content : str
        Plain text fallback — all block texts concatenated (headings excluded
        to keep backward-compatible with existing chunk pipeline).
    blocks : list[dict[str, Any]]
        Structured list of blocks. Each block carries:
        - text          : str   — block text content
        - type          : str   — block kind (heading / paragraph / table /
                                   list_item / quote / code / caption /
                                   header / footer / unknown)
        - source_format : str   — pdf | docx | txt
        - page          : int | None — page number (PDF only; None for docx/txt)
        - block_index   : int   — position of this block in the document
        - section_path  : str | None — dotted heading path, e.g. "1. Overview > 3.2 Key Metrics"
        - heading_level : int | None — 1-6 for headings, None otherwise
        - bbox          : tuple | None — (x0, y0, x1, y1) in PDF points, if available
        - parser_source : str   — name of the sub-parser that produced this block
        - confidence    : float | None — 0-1 quality signal from the parser
        - style_meta    : dict  — additional style info (bold, italic, font, etc.)
    metadata : dict[str, Any]
        Document-level metadata:
        - parser_used        : str   — primary parser selected (e.g. "docling", "pdfplumber")
        - candidate_scores   : dict  — raw quality scores for each candidate parser
        - selection_reason   : str   — human-readable why this parser was chosen
        - page_count         : int   — number of pages (PDF)
        - table_count        : int   — number of table blocks
        - heading_count      : int   — number of heading blocks
        - quality_score      : float — 0-1 overall quality score of the parse
        - likely_scanned_pdf : bool  — true if PDF appears to have no extractable text
        - ocr_recommended    : bool  — true if OCR is suggested for this PDF
        - source_format      : str   — pdf | docx | txt
        - source_file        : str   — original file path
        - errors             : list[str] — errors encountered during parsing
    """

    title: str
    content: str
    blocks: list[dict[str, Any]]
    source_path: str
    file_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
