# Document Parsing Architecture

## Overview

The ingestion pipeline transforms raw files (PDF, DOCX, TXT) into a unified, structured representation (`ParsedDocument`) with typed blocks, rich metadata, and quality signals. All downstream stages (chunking, indexing, retrieval, evaluation) consume this unified format.

## Unified Document Representation

```python
ParsedDocument(
    title: str,                    # filename stem
    content: str,                   # backward-compatible plain text
    blocks: list[Block],            # structured typed blocks
    source_path: str,               # original file path
    file_type: str,                 # pdf | docx | txt
    metadata: dict                  # document-level quality & stats
)

Block = {
    "text": str,                    # block text content
    "type": str,                    # heading | paragraph | table |
                                   # list_item | quote | code | caption |
                                   # header | footer | unknown
    "source_format": str,           # pdf | docx | txt
    "page": int | None,            # page number (PDF only)
    "block_index": int,            # position in document
    "section_path": str | None,    # "1. Overview > 3.2 Key Metrics"
    "heading_level": int | None,   # 1-6 for headings
    "bbox": tuple | None,          # (x0, y0, x1, y1) in PDF points
    "parser_source": str,           # which sub-parser produced this
    "confidence": float | None,     # quality signal 0-1
    "style_meta": dict             # bold, italic, font_name, etc.
}
```

**Key design principle**: `content` is a backward-compatible plain-text fallback (headings excluded by default). The `blocks` field is the authoritative structured representation.

## Parser Selection Logic

### PDF — Multi-Strategy with Quality Scoring

Three strategies are tried (if enabled):

| Strategy | Library | Strengths | Weaknesses |
|----------|---------|-----------|------------|
| `docling` | docling | Best structure (headings, tables, layout) | Heavier dependency |
| `pdfplumber` | pdfplumber | Reliable text, coordinate access, table extraction | Simpler layout understanding |
| `pymupdf` | pypdfium2 | Good text ordering for complex layouts | No built-in table extraction |

For each candidate that returns blocks:

1. Run **quality scoring** (see below)
2. Pick the **highest-scoring** candidate that meets `min_quality_score`
3. Record scores and selection reason in metadata

#### Quality Scoring Formula

```
score = (
  + 0.15 * min(total_chars / 5000, 1.0)
  + 0.15 * min(nonempty_blocks / 50, 1.0)
  + 0.10 * min(avg_block_len / 150, 1.0)
  - 0.10 * short_frag_ratio         # penalty: tiny fragments
  - 0.15 * repeated_line_ratio       # penalty: repeated lines
  - 0.15 * noise_ratio              # penalty: page numbers, etc.
  + 0.10 * heading_paragraph_ratio  # bonus: heading diversity
  + 0.05 * (1.0 if table_count > 0) # bonus: tables present
  + 0.05 * page_coverage            # bonus: good page spread
)
clamped to [0.0, 1.0]
```

**Scanned PDF detection** — if chars_per_page < 50 AND short_frag_ratio > 0.6, the document is flagged `likely_scanned_pdf = true` and `ocr_recommended = true`.

### DOCX — Dual Parser

| Strategy | Best for |
|----------|----------|
| `docling` | Complex Word docs with tables, styles |
| `python-docx` | Reliable paragraph/heading extraction |

Scored by: character count, heading count, table presence, list presence.

### TXT — Single Optimized Parser

No strategy selection needed. The text parser applies:
1. Encoding detection cascade
2. Hyphenation repair
3. Smart paragraph reconstruction (`paragraph_mode`: `smart` | `double_newline` | `single_newline`)
4. Block typing via pattern matching

## Processing Pipeline

```
Raw File
    │
    ▼
Detect File Type
    │
    ▼
Parser Selection (PDF only)
    │  Run enabled strategies → score each → pick best
    ▼
Block Extraction
    │  docling: markdown → blocks
    │  pdfplumber: per-page text + tables + font-size heuristics
    │  pymupdf: layout-ordered text
    │  python-docx: styles + tables
    │  text: line-based → typed blocks
    ▼
Source Format Tagging
    │  Every block tagged with source_format
    ▼
Noise Cleaning
    │  Page numbers, separators, template noise, repeated lines
    ▼
Block Normalization
    │  Type canonicalization, hyphenation repair, merge small blocks
    ▼
ParsedDocument
    │
    ▼
[chunk_service]  ← split_blocks_into_chunks() consumes blocks + metadata
```

## Configuration

All parameters are in `app/ingestion/config.py`:

| Config class | Key knobs |
|--------------|-----------|
| `PdfParserConfig` | `use_docling`, `use_pdfplumber`, `use_pymupdf`; `min_quality_score`; `heading_font_size_threshold`; `extract_tables`; `remove_header_footer_by_coords`; `likely_scanned_pdf` thresholds |
| `DocxParserConfig` | `style_to_level` dict; `extract_tables`; `include_headings_in_content` |
| `TextParserConfig` | `encoding_order`; `paragraph_mode`; `heading_patterns`; `list_patterns`; `repair_hyphenation`; `detect_repeated_lines` |
| `CleaningConfig` | All noise cleaning switches; `merge_adjacent_small_blocks`; `include_headings_in_content` |

**Environment variable overrides** — prefix with `INGEST_`:

```bash
export INGEST_PDF_VERBOSE=true
export INGEST_PDF_MIN_QUALITY=0.25
```

**Programmatic override**:

```python
from app.ingestion.config import ParsingConfig
from app.ingestion.pipeline import parse_document

config = ParsingConfig(
    pdf=PdfParserConfig(use_docling=False),  # disable docling
    cleaning=CleaningConfig(include_headings_in_content=True),
)
doc = parse_document("report.pdf", config=config)
```

## Metadata Emitted

Each `ParsedDocument.metadata` includes:

| Key | Type | Description |
|-----|------|-------------|
| `parser_used` | str | Name of selected strategy |
| `candidate_scores` | dict | Score per candidate strategy |
| `selection_reason` | str | Human-readable why this parser was picked |
| `quality_score` | float | 0-1 quality score |
| `page_count` | int | Number of pages (PDF) |
| `table_count` | int | Number of table blocks |
| `heading_count` | int | Number of heading blocks |
| `likely_scanned_pdf` | bool | True if PDF appears scanned |
| `ocr_recommended` | bool | True if OCR is suggested |
| `source_format` | str | pdf \| docx \| txt |
| `source_file` | str | Original file path |
| `errors` | list[str] | Errors encountered |

## Chunk Integration

`chunk_service.split_blocks_into_chunks()` consumes `ParsedDocument.blocks` directly. Block types are preserved in `chunk.metadata["source_block_types"]`, so you can trace each chunk back to the original blocks that produced it.

For eval alignment, key fields in each block are retained:
- `page` — page number (PDF)
- `block_index` — block position
- `section_path` — heading hierarchy
- `parser_source` — which parser produced this block

## Known Limitations

1. **Scanned PDF OCR**: `likely_scanned_pdf` detection is implemented but OCR is not integrated. Recommend `pytesseract` + `pdf2image` as optional dependencies if needed.
2. **DOCX revision/tracked changes**: Not currently tracked; python-docx reads the final state.
3. **PDF password-protected files**: Not handled; will raise an error.
4. **Complex multi-column PDF layouts**: May produce suboptimal reading order with pdfplumber/pymupdf. docling handles these better.
5. **Excel/CSV inside DOCX**: Tables are extracted as plain text; cell-level structure is lost.
6. **TXT `paragraph_mode="smart"`**: Sentence-boundary heuristics are heuristic-based and may mis-split edge cases (e.g. abbreviations like "e.g.").

## Enabling/Disabling Parsers

PDF parsers are individually togglable:

```python
from app.ingestion.config import ParsingConfig, PdfParserConfig

# Only pdfplumber (no docling, no pymupdf)
config = ParsingConfig(
    pdf=PdfParserConfig(use_docling=False, use_pymupdf=False)
)
```

For very large deployments where memory is constrained, disable `docling` to avoid its heavier model loading.
