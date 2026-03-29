"""
Finance-aware chunk service.

Provides:
- Structured embedding input templates for financial documents
- Finance-aware text chunking (larger chunks, heading-aligned)
- Table-aware chunking with table_linearized_chunk type
- Incremental indexing via chunk_hash diff
- Clear indexing pipeline phases
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import app.config as config
from app.config import (
    DEFAULT_TEXT_CHUNK_SIZE,
    DEFAULT_TEXT_CHUNK_OVERLAP,
    INDEX_INCREMENTAL_ENABLED,
)
from app.db import (
    delete_chunks_by_ids,
    ensure_chunk_search_indexes,
    get_chunks_by_document_id,
    insert_chunks_batch,
)
from app.db.repositories.document_repository import get_document_by_id
from app.services.common import (
    last_section_title,
    normalize_section_path,
    normalize_whitespace,
    safe_get,
    safe_json_loads,
    section_path_to_str,
    to_int,
)
from app.services.llm_service import get_embeddings_batch

# ---------------------------------------------------------------------------
# Constants & patterns
# ---------------------------------------------------------------------------

SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[。！？!?；;])\s+|(?<=\.)\s+(?=[A-Z0-9])|(?<=\n)"
)
HEADING_CN_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百千万0-9]+[章节部分节]|[一二三四五六七八九十]+[、.．]|[0-9]+[.、])"
)
LIST_ITEM_PATTERN = re.compile(r"^([-*•]\s+|[0-9]+[.)、]\s+)")
WHITESPACE_RE = re.compile(r"\s+")
LEXICAL_CLEAN_RE = re.compile(r"[^\w\u4e00-\u9fff]+")

# Table patterns
TABLE_ROW_SEP = re.compile(r"\n")
TABLE_CELL_SEP = re.compile(r"\s*\|\s*")
TABLE_NUMERIC_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\.\d+%?")
TABLE_PERCENT_RE = re.compile(r"\d+(\.\d+)?%")
TABLE_YEAR_RE = re.compile(r"(?<![.\d])(19|20)\d{2}(?![.\d])")
TABLE_UNIT_RE = re.compile(
    r"(?:单位|币种|货币|单位：|币种：|货币：)\s*([^\n\|]+)",
    re.IGNORECASE,
)
TABLE_TITLE_RE = re.compile(
    r"(?:表格|表[题名]?|数据来源|source)\s*[:：]?\s*([^\n\|]{2,50})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Hash utilities
# ---------------------------------------------------------------------------

def _hash_text(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def _estimate_token_count(text: str) -> int:
    text = text or ""
    if not text:
        return 0
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_words = len(re.findall(r"[A-Za-z0-9_./:-]+", text))
    return int(cjk_chars + ascii_words * 1.3)


def _normalize_lexical_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = text.lower()
    text = LEXICAL_CLEAN_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Block utilities
# ---------------------------------------------------------------------------

def _coalesce_page_range(
    page_start: int | None,
    page_end: int | None,
    block: dict[str, Any],
) -> tuple[int | None, int | None]:
    bps = to_int(block.get("page_start"))
    bpe = to_int(block.get("page_end"))
    if page_start is None:
        page_start = bps
    if bpe is not None:
        page_end = bpe
    elif page_end is None:
        page_end = bps
    return page_start, page_end


def block_to_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        metadata = block.get("metadata") or {}
        page_start = to_int(block.get("page_start"))
        page_end = to_int(block.get("page_end"))
        if page_start is None:
            page_start = to_int(block.get("page"))
        if page_end is None:
            page_end = page_start
        if page_start is None:
            page_start = to_int(metadata.get("page"))
        if page_end is None:
            page_end = to_int(metadata.get("page_end")) or page_start
        return {
            "type": normalize_whitespace(str(block.get("type", "paragraph") or "paragraph")).lower(),
            "text": normalize_whitespace(block.get("text", "")),
            "section_path": normalize_section_path(block.get("section_path")),
            "page_start": page_start,
            "page_end": page_end,
            "metadata": metadata,
        }

    metadata = getattr(block, "metadata", {}) or {}
    page_start = to_int(getattr(block, "page_start", None))
    page_end = to_int(getattr(block, "page_end", None))
    if page_start is None:
        page_start = to_int(getattr(block, "page", None))
    if page_end is None:
        page_end = page_start
    if page_start is None:
        page_start = to_int(metadata.get("page"))
    if page_end is None:
        page_end = to_int(metadata.get("page_end")) or page_start
    return {
        "type": normalize_whitespace(str(getattr(block, "type", "paragraph") or "paragraph")).lower(),
        "text": normalize_whitespace(getattr(block, "text", "")),
        "section_path": normalize_section_path(getattr(block, "section_path", [])),
        "page_start": page_start,
        "page_end": page_end,
        "metadata": metadata,
    }


def _guess_heading_level(line: str) -> int:
    line = (line or "").strip()
    if not line:
        return 1
    if re.match(r"^#{1,6}\s+", line):
        return min(6, len(line.split(" ")[0]))
    if re.match(r"^第[一二三四五六七八九十百千万0-9]+章", line):
        return 1
    if re.match(r"^第[一二三四五六七八九十百千万0-9]+节", line):
        return 2
    if re.match(r"^[一二三四五六七八九十]+[、.．]", line):
        return 2
    if re.match(r"^[0-9]+[.、]", line):
        dots = line.split(" ")[0].count(".")
        return min(4, max(2, dots + 2))
    return 1


# ---------------------------------------------------------------------------
# Content -> blocks
# ---------------------------------------------------------------------------

def build_blocks_from_content(content: str) -> list[dict[str, Any]]:
    content = (content or "").strip()
    if not content:
        return []

    lines = [line.rstrip() for line in content.splitlines()]
    blocks: list[dict[str, Any]] = []
    current_para: list[str] = []
    section_path: list[str] = []

    def flush_paragraph() -> None:
        nonlocal current_para
        text = normalize_whitespace("\n".join(current_para))
        if text:
            blocks.append({
                "type": "paragraph",
                "text": text,
                "section_path": list(section_path),
                "page_start": None,
                "page_end": None,
                "metadata": {},
            })
        current_para = []

    for raw in lines:
        line = raw.strip()
        if not line:
            flush_paragraph()
            continue

        if re.match(r"^#{1,6}\s+", line):
            flush_paragraph()
            level = _guess_heading_level(line)
            title = re.sub(r"^#{1,6}\s+", "", line).strip()
            if title:
                section_path = section_path[:max(0, level - 1)]
                section_path.append(title)
                blocks.append({
                    "type": "heading",
                    "text": title,
                    "section_path": list(section_path),
                    "page_start": None,
                    "page_end": None,
                    "metadata": {"level": level},
                })
            continue

        if HEADING_CN_PATTERN.match(line):
            flush_paragraph()
            level = _guess_heading_level(line)
            section_path = section_path[:max(0, level - 1)]
            section_path.append(line)
            blocks.append({
                "type": "heading",
                "text": line,
                "section_path": list(section_path),
                "page_start": None,
                "page_end": None,
                "metadata": {"level": level},
            })
            continue

        if LIST_ITEM_PATTERN.match(line):
            flush_paragraph()
            blocks.append({
                "type": "list_item",
                "text": line,
                "section_path": list(section_path),
                "page_start": None,
                "page_end": None,
                "metadata": {},
            })
            continue

        current_para.append(line)

    flush_paragraph()
    return blocks


# ---------------------------------------------------------------------------
# Block merging
# ---------------------------------------------------------------------------

def _can_merge_blocks(prev: dict[str, Any], curr: dict[str, Any], min_len: int) -> bool:
    prev_type = prev.get("type")
    curr_type = curr.get("type")
    prev_text = prev.get("text", "")
    curr_text = curr.get("text", "")
    same_section = normalize_section_path(prev.get("section_path")) == normalize_section_path(curr.get("section_path"))

    if not same_section:
        return False

    if prev_type in {"heading", "table", "code"} or curr_type in {"heading", "table", "code"}:
        return False

    if prev_type == "list_item" and curr_type == "list_item":
        return True

    if prev_type in {"paragraph", "list_item"} and curr_type in {"paragraph", "list_item"}:
        if len(prev_text) < min_len:
            return True
        if len(curr_text) < max(20, min_len // 2):
            return True
        if prev_text.endswith(("：", ":", "；", ";", "，", ",")):
            return True

    return False


def _merge_short_adjacent_blocks(blocks: list[dict[str, Any]], min_len: int = 80) -> list[dict[str, Any]]:
    if not blocks:
        return []

    merged: list[dict[str, Any]] = []
    for raw_block in blocks:
        block = block_to_dict(raw_block)
        text = block.get("text", "")
        if not text:
            continue

        if not merged:
            merged.append(block)
            continue

        prev = merged[-1]
        if _can_merge_blocks(prev, block, min_len=min_len):
            prev["text"] = normalize_whitespace(prev.get("text", "") + "\n" + text)
            prev_start = prev.get("page_start")
            prev_end = prev.get("page_end")
            curr_start = block.get("page_start")
            curr_end = block.get("page_end")
            if prev_start is None:
                prev["page_start"] = curr_start
            if curr_end is not None:
                prev["page_end"] = curr_end
            elif prev_end is None:
                prev["page_end"] = curr_start
        else:
            merged.append(block)

    return merged


# ---------------------------------------------------------------------------
# Sentence / paragraph splitting
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"\n{2,}", text)
    return [normalize_whitespace(p) for p in parts if normalize_whitespace(p)]


def _split_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = re.split(SENTENCE_BOUNDARY_PATTERN, text)
    return [normalize_whitespace(p) for p in parts if normalize_whitespace(p)] or [text]


def _best_cut_position(text: str, chunk_size: int) -> int:
    if len(text) <= chunk_size:
        return len(text)
    window = text[:chunk_size]
    preferred_marks = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", "，", ",", " "]
    best = -1
    for mark in preferred_marks:
        pos = window.rfind(mark)
        if pos > int(chunk_size * 0.55):
            best = pos + len(mark)
            break
    if best == -1:
        best = chunk_size
    return max(1, best)


def split_text(text: str, chunk_size: int = DEFAULT_TEXT_CHUNK_SIZE, overlap: int = DEFAULT_TEXT_CHUNK_OVERLAP) -> list[str]:
    text = normalize_whitespace((text or "").strip())
    if not text:
        return []
    if chunk_size <= 0 or len(text) <= chunk_size:
        return [text]
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    paragraphs = _split_paragraphs(text)
    if len(paragraphs) > 1:
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            candidate = para if not current else f"{current}\n{para}"
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(normalize_whitespace(current))
                current = ""
            if len(para) <= chunk_size:
                current = para
                continue
            chunks.extend(split_text(para, chunk_size=chunk_size, overlap=overlap))
        if current:
            chunks.append(normalize_whitespace(current))
        return [c for c in chunks if c]

    sentences = _split_sentences(text)
    if len(sentences) > 1:
        chunks: list[str] = []
        current = ""
        for sent in sentences:
            candidate = sent if not current else f"{current} {sent}"
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(normalize_whitespace(current))
            if len(sent) <= chunk_size:
                current = sent
            else:
                start = 0
                while start < len(sent):
                    remaining = sent[start:]
                    cut = _best_cut_position(remaining, chunk_size)
                    part = normalize_whitespace(remaining[:cut])
                    if part:
                        chunks.append(part)
                    if start + cut >= len(sent):
                        break
                    start = max(start + 1, start + cut - overlap)
                current = ""
        if current:
            chunks.append(normalize_whitespace(current))
        return [c for c in chunks if c]

    chunks = []
    start = 0
    while start < len(text):
        remaining = text[start:]
        cut = _best_cut_position(remaining, chunk_size)
        part = normalize_whitespace(remaining[:cut])
        if part:
            chunks.append(part)
        if start + cut >= len(text):
            break
        start = max(start + 1, start + cut - overlap)

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Table parsing
# ---------------------------------------------------------------------------

def _parse_table_rows(table_text: str) -> list[list[str]]:
    lines = TABLE_ROW_SEP.split(table_text.strip())
    rows: list[list[str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in TABLE_CELL_SEP.split(line)]
        while cells and not cells[-1]:
            cells.pop()
        if cells:
            rows.append(cells)
    return rows


def _extract_table_metadata(rows: list[list[str]], table_text: str, section_path: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "table_title": "",
        "table_headers": [],
        "table_units": "",
        "row_start": 0,
        "row_end": len(rows) - 1,
        "contains_numeric_values": False,
        "has_percent": False,
        "has_year": False,
        "col_count": 0,
        "year_columns": [],
    }

    if not rows:
        return metadata

    # Table title
    for line in table_text.split("\n")[:5]:
        m = TABLE_TITLE_RE.search(line)
        if m:
            metadata["table_title"] = normalize_whitespace(m.group(1))
            break

    # Determine header row
    if len(rows) >= 2:
        row0 = rows[0]
        row1 = rows[1]
        row0_cells = len(row0)
        row1_cells = len(row1)

        if row0_cells < row1_cells and row0_cells >= 1:
            metadata["table_headers"] = row0
        else:
            r0_non_first = [c for c in row0[1:] if c]
            r1_non_first = [c for c in row1[1:] if c]
            r0_looks_like_header = (
                r0_non_first
                and all(
                    TABLE_YEAR_RE.match(c.strip()) or len(c.strip()) <= 5
                    for c in r0_non_first
                )
            )
            r1_looks_like_data = r1_non_first and any(
                "." in c or "%" in c or TABLE_NUMERIC_RE.match(c.strip())
                for c in r1_non_first
            )
            if r0_looks_like_header and r1_looks_like_data:
                metadata["table_headers"] = row0
    elif len(rows) == 1:
        metadata["table_headers"] = rows[0]

    metadata["col_count"] = max((len(r) for r in rows), default=0)

    # Units / currency
    unit_candidates: set[str] = set()
    for cell in (metadata["table_headers"] or []):
        m = TABLE_UNIT_RE.search(cell)
        if m:
            unit_candidates.add(normalize_whitespace(m.group(1)))
    pre_lines = table_text.split("\n")[:3]
    for line in pre_lines:
        m = TABLE_UNIT_RE.search(line)
        if m:
            unit_candidates.add(normalize_whitespace(m.group(1)))
    if unit_candidates:
        metadata["table_units"] = " | ".join(sorted(unit_candidates))

    # Numeric signals & year columns
    flat_text = " ".join(" ".join(r) for r in rows)
    if TABLE_NUMERIC_RE.search(flat_text):
        metadata["contains_numeric_values"] = True
    if TABLE_PERCENT_RE.search(flat_text):
        metadata["has_percent"] = True

    year_matches = TABLE_YEAR_RE.findall(flat_text)
    unique_years: list[str] = []
    seen_years: set[str] = set()
    for y in year_matches:
        if y not in seen_years:
            seen_years.add(y)
            unique_years.append(y)
    if unique_years:
        metadata["has_year"] = True
        metadata["year_columns"] = unique_years[:10]

    return metadata


# ---------------------------------------------------------------------------
# Finance-aware search text builder
# ---------------------------------------------------------------------------

def _build_finance_search_text(
    title: str,
    section_path: list[str],
    heading_buffer: list[str],
    body_text: str,
    chunk_type: str,
    chunk_index: int,
    metadata: dict[str, Any],
) -> str:
    """Build structured finance-aware search text for embedding.

    Template:
    [Document] {title}
    [Company] {company}
    [Filing] {filing_type}
    [Section] {section_path}
    [Pages] {page_start}-{page_end}
    [ChunkType] {chunk_type}
    [TableCaption] {table_caption}
    [Units] {table_units}
    [YearColumns] {year_columns}
    {body_text}
    """
    parts: list[str] = []

    title = normalize_whitespace(title)
    if title:
        parts.append(f"[Document] {title}")

    company = normalize_whitespace(metadata.get("company", ""))
    if company:
        parts.append(f"[Company] {company}")

    filing_type = normalize_whitespace(metadata.get("filing_type", ""))
    if filing_type:
        parts.append(f"[Filing] {filing_type}")

    section_title = last_section_title(section_path)
    if section_title:
        parts.append(f"[Section] {section_title}")

    if heading_buffer:
        unique: list[str] = []
        seen = set()
        for h in heading_buffer:
            h = normalize_whitespace(h)
            if h and h not in seen:
                unique.append(h)
                seen.add(h)
        if unique:
            parts.append("[SectionPath] " + " / ".join(unique[-3:]))

    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    if page_start is not None:
        if page_end and page_end != page_start:
            parts.append(f"[Pages] {page_start}-{page_end}")
        else:
            parts.append(f"[Page] {page_start}")

    parts.append(f"[ChunkType] {chunk_type}")

    if chunk_type in ("table", "table_linearized"):
        table_caption = normalize_whitespace(metadata.get("table_caption", ""))
        if table_caption:
            parts.append(f"[TableCaption] {table_caption}")
        table_units = normalize_whitespace(metadata.get("table_units", ""))
        if table_units:
            parts.append(f"[Units] {table_units}")
        year_columns = metadata.get("year_columns", [])
        if year_columns:
            parts.append(f"[YearColumns] {', '.join(year_columns)}")

    body_text = normalize_whitespace(body_text)
    if body_text:
        parts.append(body_text)

    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Table linearization
# ---------------------------------------------------------------------------

def _linearize_table_row(row: list[str], headers: list[str]) -> str:
    """Convert a table row to key=value pairs, matching against headers."""
    pairs: list[str] = []
    for i, cell in enumerate(row):
        cell = cell.strip()
        if not cell:
            continue
        header = headers[i].strip() if i < len(headers) else f"col_{i}"
        pairs.append(f"{header}={cell}")
    return " | ".join(pairs)


def _build_table_linearized(
    title: str,
    section_path: list[str],
    heading_buffer: list[str],
    table_text: str,
    rows: list[list[str]],
    table_meta: dict[str, Any],
    page_start: int | None,
    page_end: int | None,
    block_idx: int,
    chunk_index: int,
) -> dict[str, Any]:
    """Build a table_linearized_chunk from a parsed table.

    Linearized format:
    - Preserves header row as column labels
    - Each data row becomes key=value pairs
    - Includes fiscal year / period columns prominently
    """
    headers = table_meta.get("table_headers") or []
    table_title = table_meta.get("table_title", "")
    table_units = table_meta.get("table_units", "")
    year_columns = table_meta.get("year_columns", [])
    has_year = table_meta.get("has_year", False)

    # Build linearized rows
    data_start_idx = 0
    if headers and rows and rows[0] == headers:
        data_start_idx = 1

    linearized_parts: list[str] = []
    if headers:
        linearized_parts.append("[Headers] " + " | ".join(h.strip() for h in headers))

    for row in rows[data_start_idx:]:
        linearized = _linearize_table_row(row, headers)
        if linearized:
            linearized_parts.append(linearized)

    body_text = "\n".join(linearized_parts)

    metadata = {
        "table_title": table_title,
        "table_headers": headers,
        "table_units": table_units,
        "year_columns": year_columns,
        "has_year": has_year,
        "contains_numeric_values": table_meta.get("contains_numeric_values", False),
        "has_percent": table_meta.get("has_percent", False),
        "col_count": table_meta.get("col_count", 0),
        "company": "",
        "filing_type": "",
        "page_start": page_start,
        "page_end": page_end,
        "heading_path_depth": len(section_path),
        "heading_buffer": list(heading_buffer[-3:]) if heading_buffer else [],
        "numeric_density": _compute_numeric_density(rows[data_start_idx:]),
    }

    section_title = last_section_title(section_path)
    search_text = _build_finance_search_text(
        title=title,
        section_path=section_path,
        heading_buffer=heading_buffer,
        body_text=body_text,
        chunk_type="table_linearized",
        chunk_index=chunk_index,
        metadata=metadata,
    )

    lexical_text = _normalize_lexical_for_chunk(search_text)

    return {
        "title": title,
        "chunk_text": body_text,
        "search_text": search_text,
        "lexical_text": lexical_text,
        "section_path": list(section_path),
        "section_title": section_title,
        "page_start": page_start,
        "page_end": page_end,
        "block_start_index": block_idx,
        "block_end_index": block_idx,
        "chunk_type": "table_linearized",
        "chunk_index": chunk_index,
        "token_count": _estimate_token_count(search_text),
        "chunk_hash": _hash_text(search_text),
        "metadata": metadata,
    }


def _compute_numeric_density(rows: list[list[str]]) -> float:
    """Fraction of cells that contain numeric values."""
    if not rows:
        return 0.0
    total_cells = sum(len(row) for row in rows)
    if total_cells == 0:
        return 0.0
    numeric_cells = 0
    for row in rows:
        for cell in row:
            if TABLE_NUMERIC_RE.search(cell or ""):
                numeric_cells += 1
    return numeric_cells / total_cells


def _normalize_lexical_for_chunk(search_text: str) -> str:
    """Build lexical_text from search_text for fulltext indexing."""
    return _normalize_lexical_text(search_text)


# ---------------------------------------------------------------------------
# Table chunk builder
# ---------------------------------------------------------------------------

def _split_table_by_rows(
    title: str,
    section_path: list[str],
    heading_buffer: list[str],
    table_text: str,
    table_meta: dict[str, Any],
    max_chars: int,
    overlap_chars: int,
    block_idx: int,
    page_start: int | None,
    page_end: int | None,
    chunk_index: int,
) -> list[dict[str, Any]]:
    """Split table into row-group chunks."""
    rows = _parse_table_rows(table_text)
    if not rows:
        return []

    header_row = table_meta.get("table_headers") or []
    col_count = table_meta.get("col_count", 0)
    table_title = table_meta.get("table_title", "")
    table_units = table_meta.get("table_units", "")
    has_year = table_meta.get("has_year", False)
    year_columns = table_meta.get("year_columns", [])

    chunks: list[dict[str, Any]] = []

    header_block = ""
    meta_parts: list[str] = []
    if table_title:
        meta_parts.append(f"表：{table_title}")
    if table_units:
        meta_parts.append(f"单位：{table_units}")
    if has_year and year_columns:
        meta_parts.append(f"年份：{', '.join(year_columns)}")
    if meta_parts:
        header_block = "\n".join(meta_parts)

    header_line = " | ".join(header_row) if header_row else ""

    # Data rows (skip header)
    data_start_idx = 0
    if header_row and rows and rows[0] == header_row:
        data_start_idx = 1
    data_rows = rows[data_start_idx:]

    if not data_rows:
        body_lines = [header_line] if header_line else []
        body_text = "\n".join(body_lines)
        chunk = _make_table_chunk(
            title=title,
            section_path=section_path,
            heading_buffer=heading_buffer,
            body_text=body_text,
            header_block=header_block,
            row_start=0,
            row_end=0,
            table_meta=table_meta,
            block_idx=block_idx,
            chunk_index=chunk_index,
            page_start=page_start,
            page_end=page_end,
        )
        return [chunk]

    def table_chunk_size(heading_parts: list[str], body_lines: list[str]) -> int:
        prefix = header_block + "\n" + "\n".join(heading_parts)
        body = "\n".join(body_lines)
        return len(prefix) + len(body)

    current_lines: list[str] = []
    current_start_row = data_start_idx
    chunk_i = chunk_index

    for i, row in enumerate(data_rows):
        row_line = " | ".join(row)
        projected = table_chunk_size([], current_lines + [row_line])

        if projected > max_chars and current_lines:
            row_end = current_start_row + len(current_lines) - 1
            chunk = _make_table_chunk(
                title=title,
                section_path=section_path,
                heading_buffer=heading_buffer,
                body_text="\n".join(current_lines),
                header_block=header_block,
                row_start=current_start_row,
                row_end=row_end,
                table_meta=table_meta,
                block_idx=block_idx,
                chunk_index=chunk_i,
                page_start=page_start,
                page_end=page_end,
            )
            chunks.append(chunk)
            chunk_i += 1

            overlap_rows = overlap_chars // (col_count * 4) if col_count > 0 else 1
            overlap_rows = max(1, min(overlap_rows, 3))
            overlap_start = max(current_start_row + len(current_lines) - overlap_rows, data_start_idx)
            current_lines = [" | ".join(r) for r in rows[overlap_start:i]]
            current_start_row = overlap_start
        else:
            if not current_lines:
                current_start_row = data_start_idx + i
            current_lines.append(row_line)

    if current_lines:
        row_end = current_start_row + len(current_lines) - 1
        chunk = _make_table_chunk(
            title=title,
            section_path=section_path,
            heading_buffer=heading_buffer,
            body_text="\n".join(current_lines),
            header_block=header_block,
            row_start=current_start_row,
            row_end=row_end,
            table_meta=table_meta,
            block_idx=block_idx,
            chunk_index=chunk_i,
            page_start=page_start,
            page_end=page_end,
        )
        chunks.append(chunk)

    return chunks


def _make_table_chunk(
    title: str,
    section_path: list[str],
    heading_buffer: list[str],
    body_text: str,
    header_block: str,
    row_start: int,
    row_end: int,
    table_meta: dict[str, Any],
    block_idx: int,
    chunk_index: int,
    page_start: int | None,
    page_end: int | None,
) -> dict[str, Any]:
    """Build a single table chunk dict."""
    section_title = last_section_title(section_path)
    headers = table_meta.get("table_headers") or []
    table_title = table_meta.get("table_title", "")
    table_units = table_meta.get("table_units", "")
    year_columns = table_meta.get("year_columns", [])

    metadata = {
        "table_title": table_title,
        "table_headers": headers,
        "table_units": table_units,
        "row_start": row_start,
        "row_end": row_end,
        "contains_numeric_values": table_meta.get("contains_numeric_values", False),
        "has_percent": table_meta.get("has_percent", False),
        "has_year": table_meta.get("has_year", False),
        "year_columns": year_columns,
        "col_count": table_meta.get("col_count", 0),
        "company": "",
        "filing_type": "",
        "page_start": page_start,
        "page_end": page_end,
        "heading_path_depth": len(section_path),
        "heading_buffer": list(heading_buffer[-3:]) if heading_buffer else [],
        "numeric_density": table_meta.get("numeric_density", 0.0),
    }

    search_parts: list[str] = []
    if title:
        search_parts.append(f"[Document] {title}")
    if table_title:
        search_parts.append(f"[TableCaption] {table_title}")
    if header_block:
        search_parts.append(header_block)
    if body_text:
        search_parts.append("---数据行---")
        for line in (body_text or "").splitlines():
            line = normalize_whitespace(line)
            if line:
                search_parts.append(line)

    search_text = "\n".join(search_parts).strip()
    lexical_text = _normalize_lexical_for_chunk(search_text)

    return {
        "title": title,
        "chunk_text": body_text,
        "search_text": search_text,
        "lexical_text": lexical_text,
        "section_path": list(section_path),
        "section_title": section_title,
        "page_start": page_start,
        "page_end": page_end,
        "block_start_index": block_idx,
        "block_end_index": block_idx,
        "chunk_type": "table",
        "chunk_index": chunk_index,
        "token_count": _estimate_token_count(search_text),
        "chunk_hash": _hash_text(search_text),
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Text chunk builder (finance-aware)
# ---------------------------------------------------------------------------

def _build_text_chunk(
    title: str,
    body_text: str,
    section_path: list[str],
    heading_buffer: list[str],
    page_start: int | None,
    page_end: int | None,
    block_start: int,
    block_end: int,
    chunk_type: str,
    chunk_index: int,
    metadata_base: dict[str, Any],
) -> dict[str, Any]:
    """Build a single text/mixed paragraph chunk with finance-aware search text."""
    metadata = dict(metadata_base)
    metadata["page_start"] = page_start
    metadata["page_end"] = page_end

    section_title = last_section_title(section_path)
    search_text = _build_finance_search_text(
        title=title,
        section_path=section_path,
        heading_buffer=heading_buffer,
        body_text=body_text,
        chunk_type=chunk_type,
        chunk_index=chunk_index,
        metadata=metadata,
    )
    lexical_text = _normalize_lexical_for_chunk(search_text)

    return {
        "title": title,
        "chunk_text": body_text,
        "search_text": search_text,
        "lexical_text": lexical_text,
        "section_path": list(section_path),
        "section_title": section_title,
        "page_start": page_start,
        "page_end": page_end,
        "block_start_index": block_start,
        "block_end_index": block_end,
        "chunk_type": chunk_type,
        "chunk_index": chunk_index,
        "token_count": _estimate_token_count(search_text),
        "chunk_hash": _hash_text(search_text),
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Main chunking entry point
# ---------------------------------------------------------------------------

def split_blocks_into_chunks(
    title: str,
    blocks: list[dict[str, Any]],
    max_chars: int = DEFAULT_TEXT_CHUNK_SIZE,
    overlap: int = DEFAULT_TEXT_CHUNK_OVERLAP,
    metadata_base: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Split document blocks into finance-aware chunks.

    Outputs three chunk types:
    - text_chunk: narrative/paragraph content
    - table_chunk: tabular data with row-windowing
    - table_linearized_chunk: key=value linearized table rows
    """
    metadata_base = metadata_base or {}
    normalized_blocks = [block_to_dict(b) for b in blocks]
    normalized_blocks = [b for b in normalized_blocks if b.get("text")]
    normalized_blocks = _merge_short_adjacent_blocks(normalized_blocks)

    if not normalized_blocks:
        return []

    chunks: list[dict[str, Any]] = []
    chunk_index = 0

    # State for accumulating text chunks
    current_texts: list[str] = []
    current_types: list[str] = []
    current_section_path: list[str] = []
    current_page_start: int | None = None
    current_page_end: int | None = None
    current_block_start: int | None = None
    current_block_end: int | None = None
    current_heading_buffer: list[str] = []

    def flush_text_chunk() -> None:
        nonlocal current_texts, current_types, current_section_path
        nonlocal current_page_start, current_page_end
        nonlocal current_block_start, current_block_end, chunk_index

        body_text = normalize_whitespace("\n".join(current_texts))
        if not body_text:
            return

        unique_types = {t for t in current_types if t}
        chunk_type = "mixed"
        if len(unique_types) == 1:
            chunk_type = list(unique_types)[0]

        chunk = _build_text_chunk(
            title=title,
            body_text=body_text,
            section_path=current_section_path,
            heading_buffer=current_heading_buffer,
            page_start=current_page_start,
            page_end=current_page_end,
            block_start=current_block_start or 0,
            block_end=current_block_end or 0,
            chunk_type=chunk_type,
            chunk_index=chunk_index,
            metadata_base=metadata_base,
        )
        chunks.append(chunk)
        chunk_index += 1

        current_texts = []
        current_types = []
        current_section_path = []
        current_page_start = None
        current_page_end = None
        current_block_start = None
        current_block_end = None

    for idx, block in enumerate(normalized_blocks):
        btype = (block.get("type") or "paragraph").lower()
        btext = normalize_whitespace(block.get("text", ""))
        if not btext:
            continue

        section_path = normalize_section_path(block.get("section_path"))
        metadata = block.get("metadata") or {}

        if btype == "heading":
            flush_text_chunk()
            current_heading_buffer.append(btext)
            level = to_int(metadata.get("level")) or len(section_path) or 1
            if len(current_heading_buffer) > 6:
                current_heading_buffer = current_heading_buffer[-6:]
            if level <= 1:
                current_heading_buffer = current_heading_buffer[-1:]
            continue

        # Table blocks: emit table_chunk + table_linearized_chunk
        if btype == "table":
            flush_text_chunk()
            parsed_rows = _parse_table_rows(btext)
            table_meta = _extract_table_metadata(parsed_rows, btext, section_path)

            # Add numeric_density
            data_rows = parsed_rows[1:] if parsed_rows else []
            table_meta["numeric_density"] = _compute_numeric_density(data_rows)

            # Table chunk (row-windowed)
            table_chunks = _split_table_by_rows(
                title=title,
                section_path=section_path,
                heading_buffer=current_heading_buffer,
                table_text=btext,
                table_meta=table_meta,
                max_chars=max_chars,
                overlap_chars=overlap,
                block_idx=idx,
                page_start=to_int(block.get("page_start")),
                page_end=to_int(block.get("page_end")),
                chunk_index=chunk_index,
            )
            for tc in table_chunks:
                tc["metadata"]["company"] = metadata_base.get("company", "")
                tc["metadata"]["filing_type"] = metadata_base.get("filing_type", "")
            chunks.extend(table_chunks)
            chunk_index += len(table_chunks)

            # Also emit a table_linearized_chunk (one per table, not row-windowed)
            if parsed_rows and len(parsed_rows) > 1:
                linearized = _build_table_linearized(
                    title=title,
                    section_path=section_path,
                    heading_buffer=current_heading_buffer,
                    table_text=btext,
                    rows=parsed_rows,
                    table_meta=table_meta,
                    page_start=to_int(block.get("page_start")),
                    page_end=to_int(block.get("page_end")),
                    block_idx=idx,
                    chunk_index=chunk_index,
                )
                linearized["metadata"]["company"] = metadata_base.get("company", "")
                linearized["metadata"]["filing_type"] = metadata_base.get("filing_type", "")
                chunks.append(linearized)
                chunk_index += 1
            continue

        # For oversized non-table blocks, hard-split
        estimated = _build_finance_search_text(
            title=title,
            section_path=section_path,
            heading_buffer=current_heading_buffer,
            body_text=btext,
            chunk_type=btype,
            chunk_index=0,
            metadata=dict(metadata_base, page_start=block.get("page_start"), page_end=block.get("page_end")),
        )
        if len(estimated) > max_chars:
            flush_text_chunk()
            sub_chunks = split_text(btext, chunk_size=max_chars, overlap=overlap)
            for sub_i, sub_text in enumerate(sub_chunks):
                chunk = _build_text_chunk(
                    title=title,
                    body_text=sub_text,
                    section_path=section_path,
                    heading_buffer=current_heading_buffer,
                    page_start=to_int(block.get("page_start")),
                    page_end=to_int(block.get("page_end")),
                    block_start=idx,
                    block_end=idx,
                    chunk_type=btype,
                    chunk_index=chunk_index,
                    metadata_base=metadata_base,
                )
                chunks.append(chunk)
                chunk_index += 1
            continue

        # Accumulate into current text chunk
        should_flush = False
        if current_texts:
            if normalize_section_path(current_section_path) != section_path:
                should_flush = True
            else:
                projected_text = normalize_whitespace("\n".join(current_texts + [btext]))
                projected = _build_finance_search_text(
                    title=title,
                    section_path=current_section_path,
                    heading_buffer=current_heading_buffer,
                    body_text=projected_text,
                    chunk_type=btype,
                    chunk_index=0,
                    metadata=dict(metadata_base),
                )
                if len(projected) > max_chars:
                    should_flush = True

        if should_flush:
            flush_text_chunk()

        if not current_texts:
            current_section_path = list(section_path)
            current_page_start = to_int(block.get("page_start"))
            current_block_start = idx

        current_texts.append(btext)
        current_types.append(btype)
        current_page_start, current_page_end = _coalesce_page_range(
            current_page_start, current_page_end, block
        )
        current_block_end = idx

    flush_text_chunk()
    return chunks


# ---------------------------------------------------------------------------
# Chunk diff utilities (for incremental indexing)
# ---------------------------------------------------------------------------

def _load_existing_chunks(document_id: int) -> dict[str, dict[str, Any]]:
    """Load existing chunks for a document, keyed by chunk_hash."""
    rows = get_chunks_by_document_id(document_id)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        h = row.get("chunk_hash")
        if h:
            result[h] = row
    return result


def _diff_chunks(
    existing: dict[str, dict[str, Any]],
    new_chunks: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],  # unchanged_chunks (existing rows to keep)
    list[dict[str, Any]],  # new_chunks to insert
    list[int],             # chunk_ids to delete
]:
    """Diff existing vs new chunks by chunk_hash.

    Returns:
    - unchanged: existing chunk rows whose hash matches a new chunk
    - to_insert: new chunks with no matching existing hash
    - to_delete: existing chunk_ids whose hash is not in new chunks
    """
    existing_hashes = set(existing.keys())
    new_hashes = {c.get("chunk_hash") for c in new_chunks if c.get("chunk_hash")}

    # Unchanged: existing hash appears in new chunks
    unchanged = [existing[h] for h in (existing_hashes & new_hashes)]

    # New: hash not in existing
    to_insert = [c for c in new_chunks if c.get("chunk_hash") not in existing_hashes]

    # Delete: existing hash not in new
    to_delete_ids = [
        existing[h]["id"]
        for h in (existing_hashes - new_hashes)
        if existing[h].get("id")
    ]

    return unchanged, to_insert, to_delete_ids


# ---------------------------------------------------------------------------
# Index document (refactored into clear phases)
# ---------------------------------------------------------------------------

def _to_indexed_chunk_item(chunk_id: int | None, chunk_index: int, chunk: dict[str, Any]) -> dict[str, Any]:
    search_text = normalize_whitespace(chunk.get("search_text", ""))
    return {
        "id": chunk_id,
        "chunk_index": chunk_index,
        "chunk_type": chunk.get("chunk_type"),
        "section_path": section_path_to_str(chunk.get("section_path")),
        "section_title": chunk.get("section_title"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "token_count": chunk.get("token_count", 0),
        "block_start_index": chunk.get("block_start_index"),
        "block_end_index": chunk.get("block_end_index"),
        "preview": search_text[:200],
    }


def index_document(
    document_id: int,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> dict[str, Any]:
    """Index a document with incremental chunk diff support.

    Phases:
    1. Load existing chunks from MySQL
    2. Build new chunks from document content
    3. Diff chunks by hash
    4. Batch embed new/changed chunks
    5. Batch upsert to MySQL (without embedding)
    6. Batch upsert vectors to Qdrant
    7. Delete removed chunks
    """
    row = get_document_by_id(document_id)
    if not row:
        raise ValueError("document not found")

    title = safe_get(row, "title", "") or ""
    content = safe_get(row, "content", "") or ""
    blocks = safe_json_loads(safe_get(row, "blocks_json"), default=[])

    if not blocks:
        blocks = build_blocks_from_content(content)

    # Extract finance metadata for chunk templates
    metadata_json = safe_get(row, "metadata_json", {}) or {}
    metadata_base = {
        "company": safe_get(metadata_json, "company", ""),
        "filing_type": safe_get(metadata_json, "filing_type", ""),
        "fiscal_year": safe_get(metadata_json, "fiscal_year", ""),
    }

    # Ensure fulltext index exists
    ensure_chunk_search_indexes()

    # Phase 1: load existing chunks
    existing_chunks: dict[str, dict[str, Any]] = {}
    if INDEX_INCREMENTAL_ENABLED:
        existing_chunks = _load_existing_chunks(document_id)

    use_chunk_size = chunk_size if chunk_size is not None else DEFAULT_TEXT_CHUNK_SIZE
    use_overlap = overlap if overlap is not None else DEFAULT_TEXT_CHUNK_OVERLAP

    # Phase 2: build new chunks
    new_chunks = split_blocks_into_chunks(
        title=title,
        blocks=blocks,
        max_chars=use_chunk_size,
        overlap=use_overlap,
        metadata_base=metadata_base,
    )

    # Phase 3: diff
    if INDEX_INCREMENTAL_ENABLED and existing_chunks:
        unchanged_existing, to_insert_new, to_delete_ids = _diff_chunks(existing_chunks, new_chunks)
    else:
        # Full rebuild
        unchanged_existing = []
        to_insert_new = new_chunks
        to_delete_ids = [existing_chunks[h]["id"] for h in existing_chunks if existing_chunks[h].get("id")] if existing_chunks else []

    # Phase 4: batch embed new chunks
    embeddings_map: dict[int, list[float]] = {}
    if to_insert_new:
        non_empty: list[tuple[int, str]] = []
        for i, c in enumerate(to_insert_new):
            t = normalize_whitespace(c["search_text"])
            if t:
                non_empty.append((i, t))

        if non_empty:
            texts = [t for _, t in non_empty]
            embs = get_embeddings_batch(texts)
            for (i, _), emb in zip(non_empty, embs):
                if emb:
                    embeddings_map[i] = emb

    # Phase 5a: MySQL - only insert new chunks (NOT unchanged ones)
    new_mysql_payloads: list[dict[str, Any]] = []
    for i, chunk in enumerate(to_insert_new):
        search_text = normalize_whitespace(chunk.get("search_text", ""))
        if not search_text:
            continue
        new_mysql_payloads.append({
            "document_id": document_id,
            "chunk_text": normalize_whitespace(chunk.get("chunk_text", "")),
            "search_text": search_text,
            "lexical_text": normalize_whitespace(chunk.get("lexical_text", "")),
            "chunk_index": chunk.get("chunk_index"),
            "section_path": chunk.get("section_path") or [],
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "block_start_index": chunk.get("block_start_index"),
            "block_end_index": chunk.get("block_end_index"),
            "chunk_type": chunk.get("chunk_type"),
            "title": chunk.get("title"),
            "section_title": chunk.get("section_title"),
            "token_count": chunk.get("token_count", 0),
            "chunk_hash": chunk.get("chunk_hash"),
            "metadata_json": chunk.get("metadata", {}),
        })

    # MySQL INSERT (ON DUPLICATE KEY UPDATE for new/changed chunks)
    new_chunk_ids: list[int] = []
    if new_mysql_payloads:
        new_chunk_ids = insert_chunks_batch(document_id, new_mysql_payloads)

    # Build id_by_chunk_index for new chunks
    id_by_chunk_index: dict[int, int] = {}
    for payload, cid in zip(new_mysql_payloads, new_chunk_ids):
        idx = payload.get("chunk_index")
        if idx is not None and cid:
            id_by_chunk_index[idx] = cid

    # Phase 5b: Build ALL vector payloads
    vector_payloads: list[dict[str, Any]] = []
    saved_chunks: list[dict[str, Any]] = []

    # Unchanged chunks: use existing IDs from the database, compute embedding if needed
    for existing_row in unchanged_existing:
        existing_id = existing_row.get("id")
        if not existing_id:
            continue
        chunk_hash = existing_row.get("chunk_hash")
        # Find the matching new chunk by hash
        matching_new = next((c for c in new_chunks if c.get("chunk_hash") == chunk_hash), None)
        if not matching_new:
            continue

        new_idx = new_chunks.index(matching_new)
        # Try to use already-computed embedding from the map
        emb = embeddings_map.get(new_idx, [])
        if not emb:
            st = normalize_whitespace(matching_new.get("search_text", ""))
            if st:
                from app.services.llm_service import get_embedding
                emb = get_embedding(st)

        if emb:
            vector_payloads.append({
                "chunk_id": existing_id,
                "document_id": document_id,
                "chunk_index": matching_new.get("chunk_index"),
                "title": matching_new.get("title", ""),
                "section_title": matching_new.get("section_title", ""),
                "section_path": section_path_to_str(matching_new.get("section_path")),
                "page_start": matching_new.get("page_start"),
                "page_end": matching_new.get("page_end"),
                "chunk_type": matching_new.get("chunk_type"),
                "embedding": emb,
            })

        saved_chunks.append(_to_indexed_chunk_item(existing_id, matching_new.get("chunk_index", 0), matching_new))

    # New chunks: use returned IDs from MySQL insert
    for i, chunk in enumerate(to_insert_new):
        cid = id_by_chunk_index.get(chunk.get("chunk_index"))
        if not cid:
            saved_chunks.append(_to_indexed_chunk_item(None, chunk.get("chunk_index", i), chunk))
            continue

        emb = embeddings_map.get(i, [])
        if emb:
            vector_payloads.append({
                "chunk_id": cid,
                "document_id": document_id,
                "chunk_index": chunk.get("chunk_index"),
                "title": chunk.get("title", ""),
                "section_title": chunk.get("section_title", ""),
                "section_path": section_path_to_str(chunk.get("section_path")),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "chunk_type": chunk.get("chunk_type"),
                "embedding": emb,
            })

        saved_chunks.append(_to_indexed_chunk_item(cid, chunk.get("chunk_index", i), chunk))

    # Phase 6: Delete removed chunks from MySQL and Qdrant
    if to_delete_ids:
        try:
            delete_chunks_by_ids(to_delete_ids)
        except Exception:
            pass
        try:
            from app.services.vector_store import vector_store
            vector_store.delete_chunk_vectors(to_delete_ids)
        except Exception:
            pass

    # Phase 7: Upsert vectors to Qdrant
    valid_vector_payloads = [vp for vp in vector_payloads if vp.get("chunk_id") and vp.get("embedding")]
    if valid_vector_payloads:
        try:
            from app.services.vector_store import vector_store
            vector_store.upsert_chunks(valid_vector_payloads)
        except Exception as e:
            logging.error(f"Vector store upsert failed for doc {document_id}: {e}")

    return {
        "document_id": document_id,
        "title": title,
        "chunk_count": len(saved_chunks),
        "chunks": saved_chunks,
        "vector_count": len(valid_vector_payloads),
        "status": "indexed",
    }
