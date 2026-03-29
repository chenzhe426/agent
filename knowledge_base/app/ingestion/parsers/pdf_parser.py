"""
PDF parser using PyMuPDF (fitz) only.

Fast, process-safe. Table detection via block structure heuristics
(no find_tables() call = no slow layout analysis).
"""

from __future__ import annotations

import logging
import re
import statistics
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from app.ingestion.config import CleaningConfig, PdfParserConfig, ParsingConfig
from app.ingestion.normalizers import blocks_to_content, clean_blocks, normalize_text
from app.ingestion.parsers.base import BaseParser
from app.ingestion.schemas import ParsedDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section_path_push(section_stack: list[str], heading_text: str, level: int) -> str:
    while len(section_stack) >= level:
        section_stack.pop()
    section_stack.append(heading_text)
    return " > ".join(section_stack)


def _looks_like_heading(text: str) -> tuple[bool, int | None]:
    """Lightweight heading heuristic."""
    s = text.strip()
    if not s or len(s) > 120:
        return False, None
    if re.match(r"^(item|section|part)\s+[A-Z0-9IVX.\-]+", s, flags=re.I):
        return True, 2
    if re.match(r"^\d+(\.\d+){0,3}\s+\S+", s):
        return True, 3
    alpha_chars = [c for c in s if c.isalpha()]
    if alpha_chars:
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if upper_ratio > 0.8 and len(s) < 100:
            return True, 2
    return False, None


def _is_table_by_structure(block: dict) -> bool:
    """Detect if a dict block is a table based on layout structure.

    Table indicators:
    - Many lines (>= 3) packed in small vertical space
    - Word x-positions cluster at multiple distinct columns
    - Contains pipe separators
    """
    lines = block.get("lines", [])
    if len(lines) < 2:
        return False

    bbox = block.get("bbox")
    if not bbox:
        return False

    height = bbox[3] - bbox[1]
    if height <= 0:
        return False

    # Dense block = many lines in small height → table
    avg_line_height = height / len(lines)
    if avg_line_height < 12 and len(lines) >= 3:
        return True

    # Check for column alignment (multiple distinct x clusters)
    all_word_x: list[float] = []
    for line in lines:
        for span in line.get("spans", []):
            all_word_x.append(span.get("bbox", [0, 0, 0, 0])[0])

    if len(all_word_x) >= 6:
        try:
            std_x = statistics.stdev(all_word_x)
            # High std dev → words spread across columns
            if std_x > 80:
                return True
        except statistics.stdev:
            pass

    # Pipe separators
    all_text = "".join(
        "".join(s.get("text", "") for s in line.get("spans", []))
        for line in lines
    )
    if " | " in all_text and len(lines) >= 2:
        return True

    return False


# ---------------------------------------------------------------------------
# Per-page worker (runs in thread pool)
# ---------------------------------------------------------------------------

def _parse_page_worker(page_idx: int, page) -> list[dict[str, Any]]:
    """Parse one PDF page. Called in parallel via ThreadPoolExecutor."""
    import fitz

    blocks: list[dict[str, Any]] = []

    try:
        dict_output = page.get_text("dict")
    except Exception:
        return []

    page_rect = page.rect
    page_width, page_height = page_rect.width, page_rect.height

    for block in dict_output.get("blocks", []):
        if block.get("type") != 0:  # ignore images / vector drawings
            continue

        bbox = block.get("bbox")
        if not bbox:
            continue

        lines = block.get("lines", [])
        if not lines:
            continue

        # Build text from spans (fitz 1.27+ uses spans, older versions use words)
        all_spans = []
        for line in lines:
            for span in line.get("spans", []):
                all_spans.append(span)

        full_text = "".join(s.get("text", "") for s in all_spans)
        full_text = normalize_text(full_text).strip()
        if not full_text or len(full_text) < 20:
            continue

        is_table = _is_table_by_structure(block)

        if is_table:
            # Table: preserve line structure with newlines
            table_lines = []
            for line in lines:
                line_text = "".join(s.get("text", "") for s in line.get("spans", []))
                table_lines.append(line_text)
            text = "\n".join(table_lines)
        else:
            text = full_text

        is_heading, heading_level = _looks_like_heading(text) if not is_table else (False, None)

        blocks.append({
            "type": "heading" if is_heading else ("table" if is_table else "paragraph"),
            "text": text,
            "page": page_idx + 1,
            "heading_level": heading_level,
            "section_path": None,  # filled in later
            "parser_source": "fitz",
            "bbox": (
                round(bbox[0], 1), round(bbox[1], 1),
                round(bbox[2], 1), round(bbox[3], 1),
            ),
        })

    return blocks


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------

def _parse_with_fitz(path: Path, config: PdfParserConfig) -> list[dict[str, Any]]:
    """
    Parse a PDF using PyMuPDF (fitz) with parallel page processing.
    Table detection via block structure heuristics (no find_tables()).
    """
    import fitz

    logger.info("[PdfParser][fitz] parsing %s", path)

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        logger.exception("[PdfParser][fitz] fitz open failed for %s: %s", path, e)
        return []

    page_count = len(doc)
    all_blocks: list[dict[str, Any]] = []
    section_stack: list[str] = []

    # Parallel page processing
    workers = min(8, page_count)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_parse_page_worker, page_idx, doc[page_idx]): page_idx
            for page_idx in range(page_count)
        }
        for future in as_completed(futures):
            try:
                page_blocks = future.result()
            except Exception as e:
                logger.warning("[PdfParser] page worker failed: %s", e)
                continue

            # Update section stack for headings, fill section_path
            for block in page_blocks:
                if block["type"] == "heading":
                    text = block["text"]
                    level = block.get("heading_level") or 3
                    block["section_path"] = _section_path_push(section_stack, text, level)
                elif section_stack:
                    block["section_path"] = " > ".join(section_stack)

            all_blocks.extend(page_blocks)

    doc.close()

    logger.info("[PdfParser][fitz] extracted %d blocks from %s", len(all_blocks), path)
    return all_blocks


# ---------------------------------------------------------------------------
# Candidate runner & PdfParser class
# ---------------------------------------------------------------------------

def _run_fitz_candidate(
    parser_name: str,
    parser_func,
    path: Path,
    config: PdfParserConfig,
) -> dict[str, Any] | None:
    """Run a single candidate and score it."""
    try:
        blocks = parser_func(path, config)
        if not blocks:
            return {
                "parser_name": parser_name,
                "blocks": [],
                "quality_score": 0.0,
                "score_details": {},
                "errors": [f"{parser_name}: returned 0 blocks"],
            }

        from app.ingestion.quality import score_pdf_blocks
        score_result = score_pdf_blocks(blocks, config)
        logger.info(
            "[PdfParser] parser=%s succeeded for %s | score=%.4f blocks=%d chars=%s tables=%s",
            parser_name, path,
            score_result.get("quality_score", 0.0),
            score_result.get("nonempty_blocks", len(blocks)),
            score_result.get("total_chars"),
            score_result.get("table_count"),
        )
        return {
            "parser_name": parser_name,
            "blocks": blocks,
            "quality_score": score_result["quality_score"],
            "score_details": score_result,
            "errors": [],
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("[PdfParser] parser=%s failed for %s: %s\n%s", parser_name, path, e, tb)
        return {
            "parser_name": parser_name,
            "blocks": [],
            "quality_score": 0.0,
            "score_details": {},
            "errors": [f"{parser_name}: {e}", tb],
        }


class PdfParser(BaseParser):
    """PDF parser using PyMuPDF (fitz) exclusively."""

    file_type = "pdf"

    def __init__(self, config: ParsingConfig | None = None):
        self._cfg = config.pdf if config else PdfParserConfig()
        self._clean_cfg = config.cleaning if config else CleaningConfig()

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        logger.info("[PdfParser] starting parse for %s", path)

        file_size = path.stat().st_size if path.exists() else 0

        result = _run_fitz_candidate("fitz", _parse_with_fitz, path, self._cfg)

        if not result or not result.get("blocks"):
            raise RuntimeError(
                f"PDF parsing failed for {path}. errors={result.get('errors') if result else 'no result'}"
            )

        blocks = result["blocks"]
        score_details = result["score_details"]
        errors = result.get("errors", [])

        # Clean blocks
        for block in blocks:
            block["source_format"] = "pdf"
            block.setdefault("page", None)

        cleaned_blocks = clean_blocks(blocks, self._clean_cfg)
        for b in cleaned_blocks:
            b.setdefault("source_format", "pdf")

        content = blocks_to_content(
            cleaned_blocks,
            include_headings=self._clean_cfg.include_headings_in_content,
        )

        table_count = sum(1 for b in cleaned_blocks if b.get("type") == "table")
        heading_count = sum(1 for b in cleaned_blocks if b.get("type") == "heading")
        page_count = len({b.get("page") for b in cleaned_blocks if b.get("page") is not None})

        metadata = {
            "parser_used": "fitz",
            "candidate_scores": {"fitz": result["quality_score"]},
            "selection_reason": f"fitz-only mode, score={score_details.get('quality_score', 0.0):.3f}",
            "quality_score": score_details.get("quality_score", 0.0),
            "score_breakdown": score_details,
            "page_count": page_count,
            "table_count": table_count,
            "heading_count": heading_count,
            "likely_scanned_pdf": False,
            "ocr_recommended": False,
            "source_format": "pdf",
            "source_file": str(path),
            "file_size_bytes": file_size,
            "errors": errors,
            "fallback_used": False,
        }

        return ParsedDocument(
            title=path.stem,
            content=content,
            blocks=cleaned_blocks,
            source_path=str(path),
            file_type=self.file_type,
            metadata=metadata,
        )
