"""
Unified Document Ingestion Orchestrator.

Provides a single entry point for document ingestion with optional auto-indexing.

Public API
----------
ingest_document(file_path, auto_index=True, incremental=True) -> dict
    Parse + optionally index a single document.

ingest_folder(folder_path, auto_index=False, incremental=True) -> list[dict]
    Parse + optionally index multiple documents.

The orchestrator unifies the previously split import/index workflow:
    parse -> enrich metadata -> build chunks -> diff chunks -> embed in batch -> write mysql -> upsert qdrant
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import app.config as config
from app.config import (
    DEFAULT_TEXT_CHUNK_SIZE,
    DEFAULT_TEXT_CHUNK_OVERLAP,
    INDEX_INCREMENTAL_ENABLED,
    INGEST_AUTO_INDEX,
    INGEST_WORKERS,
)
from app.db import get_chunks_by_document_id, insert_document
from app.db.repositories.document_repository import get_document_by_id
from app.ingestion.config import ParsingConfig
from app.ingestion.detectors import detect_file_type
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.pdf_parser import PdfParser
from app.ingestion.parsers.text_parser import TextParser
from app.ingestion.pipeline import parse_document as _parse_document
from app.ingestion.pipeline import parse_documents_from_folder as _parse_folder
from app.ingestion.schemas import ParsedDocument
from app.services.chunk_service import index_document as _index_document
from app.services.common import (
    normalize_section_path,
    normalize_whitespace,
    safe_get,
    safe_json_loads,
)
from app.services.document_service import parsed_document_to_db_payload

logger = logging.getLogger(__name__)

_PARSER_MAP = {
    "pdf": PdfParser,
    "docx": DocxParser,
    "text": TextParser,
}

# ---------------------------------------------------------------------------
# Document-level content hash
# ---------------------------------------------------------------------------

def _compute_content_hash(content: str, raw_text: str | None = None) -> str:
    text = content or raw_text or ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Finance metadata enrichment
# ---------------------------------------------------------------------------

COMPANY_RE = re.compile(r"(?:公司|Company|Corporation|Corp|Inc|Ltd)\s*[:：]?\s*([A-Za-z0-9\u4e00-\u9fff（）()（）]+?)(?:[,，]|公司|$)", re.IGNORECASE)
FILING_TYPE_RE = re.compile(r"(?:年报|半年报|季报|10-K|10-Q|8-K|Form\s*([A-Za-z0-9-]+))", re.IGNORECASE)
FISCAL_YEAR_RE = re.compile(r"(?:财年|Fiscal\s*Year|FY)\s*[:：]?\s*(20\d{2})", re.IGNORECASE)


def _enrich_finance_metadata(
    title: str,
    content: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Extract finance-specific metadata for chunk templates."""
    enriched = dict(metadata)

    # Company name
    if "company" not in enriched or not enriched["company"]:
        company_match = COMPANY_RE.search(title + " " + content[:500])
        if company_match:
            enriched["company"] = normalize_whitespace(company_match.group(1))

    # Filing type
    if "filing_type" not in enriched or not enriched["filing_type"]:
        filing_match = FILING_TYPE_RE.search(title + " " + content[:1000])
        if filing_match:
            enriched["filing_type"] = normalize_whitespace(filing_match.group(0))

    # Fiscal year
    if "fiscal_year" not in enriched or not enriched["fiscal_year"]:
        year_match = FISCAL_YEAR_RE.search(content[:2000])
        if year_match:
            enriched["fiscal_year"] = year_match.group(1)

    return enriched


# ---------------------------------------------------------------------------
# Unified ingest entry point
# ---------------------------------------------------------------------------

def ingest_document(
    file_path: str | Path,
    auto_index: bool = INGEST_AUTO_INDEX,
    incremental: bool = INDEX_INCREMENTAL_ENABLED,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> dict[str, Any]:
    """Parse and optionally index a single document.

    Parameters
    ----------
    file_path : str | Path
        Path to the document file.
    auto_index : bool
        If True, run index_document after import.
    incremental : bool
        If True, use incremental indexing (chunk hash diff).
    chunk_size : int | None
        Override default text chunk size.
    overlap : int | None
        Override default text chunk overlap.

    Returns
    -------
    dict with keys:
        document_id: int
        title: str
        import_status: str ("imported" or "indexed")
        chunk_count: int or None
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    if not path.is_file():
        raise ValueError(f"not a file: {path}")

    # Parse
    parsed_doc = _parse_document(str(path))
    if not parsed_doc:
        raise ValueError(f"failed to parse document: {path}")

    is_scanned = parsed_doc.metadata.get("likely_scanned_pdf", False)
    ocr_hint = parsed_doc.metadata.get("hint", "")

    # Build DB payload
    payload = parsed_document_to_db_payload(parsed_doc, source_path=str(path))

    # Check for scanned PDF
    has_content = bool(payload.get("content") or payload.get("raw_text"))
    if not has_content:
        if is_scanned:
            raise ValueError(
                f"document appears to be a scanned/image PDF with no extractable text "
                f"(likely_scanned_pdf=true). {ocr_hint}"
            )
        raise ValueError(f"document has no content: {path}")

    # Enrich finance metadata
    content_for_meta = payload.get("content", "") or payload.get("raw_text", "")
    payload["metadata_json"] = _enrich_finance_metadata(
        title=payload["title"],
        content=content_for_meta,
        metadata=payload.get("metadata_json") or {},
    )

    # Insert document
    doc_id = insert_document(**payload)

    result = {
        "document_id": doc_id,
        "id": doc_id,
        "title": payload["title"],
        "file_path": str(path),
        "file_type": payload["file_type"],
        "source_type": payload["source_type"],
        "lang": payload.get("lang"),
        "author": payload.get("author"),
        "block_count": payload["block_count"],
        "metadata": payload["metadata_json"],
        "tags": payload.get("tags_json") or [],
        "import_status": "imported",
        "chunk_count": None,
    }

    # Auto-index
    if auto_index:
        try:
            index_result = _index_document(
                document_id=doc_id,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            result["import_status"] = "indexed"
            result["chunk_count"] = index_result.get("chunk_count", 0)
        except Exception as e:
            logger.error(f"Auto-index failed for doc {doc_id}: {e}")
            result["import_status"] = "imported"
            result["index_error"] = str(e)

    return result


def _worker_ingest_one(args: tuple[str, bool, bool, int | None, int | None]) -> dict[str, Any] | None:
    """Top-level worker for ProcessPoolExecutor."""
    (path_str, auto_index, incremental, chunk_size, overlap) = args
    try:
        return ingest_document(
            path_str,
            auto_index=auto_index,
            incremental=incremental,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except Exception as e:
        logging.getLogger(__name__).warning("[orchestrator] ingest failed: %s | %s", path_str, e)
        return None


def ingest_folder(
    folder_path: str | Path,
    auto_index: bool = False,
    incremental: bool = INDEX_INCREMENTAL_ENABLED,
    recursive: bool = True,
    chunk_size: int | None = None,
    overlap: int | None = None,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Parse and optionally index all documents in a folder.

    Parameters
    ----------
    folder_path : str | Path
    auto_index : bool
        If True, index each document after import.
    incremental : bool
        If True, use incremental indexing.
    recursive : bool
        If True, descend into subdirectories.
    chunk_size : int | None
        Override default chunk size.
    overlap : int | None
        Override default overlap.
    max_workers : int | None
        Override default worker count (default: min(cpu_count, file_count, 8)).

    Returns
    -------
    list of result dicts (one per successfully imported document).
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"folder not found: {folder}")
    if not folder.is_dir():
        raise ValueError(f"not a folder: {folder}")

    pattern = "**/*" if recursive else "*"
    file_paths = [p for p in sorted(folder.glob(pattern)) if p.is_file()]

    if not file_paths:
        return []

    # Determine worker count
    if max_workers is None or max_workers <= 0:
        cpu_count = multiprocessing.cpu_count()
        workers = min(cpu_count, len(file_paths), 8)
        if INGEST_WORKERS > 0:
            workers = min(INGEST_WORKERS, len(file_paths))
    else:
        workers = min(max_workers, len(file_paths))

    # Prepare worker args
    args_list = [
        (str(p), auto_index, incremental, chunk_size, overlap)
        for p in file_paths
    ]

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_worker_ingest_one, args): args[0]
            for args in args_list
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                pass

    # Restore original sorted order
    results.sort(key=lambda d: file_paths.index(Path(d.get("file_path", ""))) if d.get("file_path") else 0)
    return results


# ---------------------------------------------------------------------------
# Re-export parse functions for compatibility
# ---------------------------------------------------------------------------

def parse_document(file_path: str | Path, config: ParsingConfig | None = None) -> ParsedDocument:
    """Parse a single document (re-exports from pipeline)."""
    return _parse_document(file_path, config)


def parse_documents_from_folder(
    folder_path: str | Path,
    config: ParsingConfig | None = None,
    recursive: bool = True,
) -> list[ParsedDocument]:
    """Parse all documents in a folder (re-exports from pipeline)."""
    return _parse_folder(folder_path, config, recursive)
