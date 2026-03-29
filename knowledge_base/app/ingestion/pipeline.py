"""
Document ingestion pipeline.

Public API
----------
parse_document(file_path, config=None)
    → ParsedDocument

parse_documents_from_folder(folder_path, config=None, max_workers=None)
    → list[ParsedDocument]
"""
from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import app.config as config
from app.ingestion.config import ParsingConfig
from app.ingestion.detectors import detect_file_type
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.pdf_parser import PdfParser
from app.ingestion.parsers.text_parser import TextParser
from app.ingestion.schemas import ParsedDocument

logger = logging.getLogger(__name__)

_INGEST_WORKERS = int(config.INGEST_WORKERS) if hasattr(config, "INGEST_WORKERS") else 0

# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

_PARSER_MAP = {
    "pdf": PdfParser,
    "docx": DocxParser,
    "text": TextParser,
}


def _worker_parse_one(path_str: str) -> ParsedDocument | None:
    """
    Top-level worker function for ProcessPoolExecutor.
    Must be at module level to be picklable with 'spawn' context.
    Each subprocess re-imports this module, so imports are local.
    """
    import logging
    from pathlib import Path

    try:
        return parse_document(Path(path_str), config=None)
    except Exception as e:
        logging.getLogger(__name__).warning("[pipeline] parse failed: %s | %s", path_str, e)
        return None


def parse_document(
    file_path: str | Path,
    config: ParsingConfig | None = None,
) -> ParsedDocument:
    """
    Parse a single document into a unified ParsedDocument.

    Parameters
    ----------
    file_path : str | Path
        Path to the document file.
    config : ParsingConfig | None
        Parser configuration.  If None, defaults are used.

    Returns
    -------
    ParsedDocument
        Structured document with blocks and metadata.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file type is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    file_type = detect_file_type(path)
    parser_cls = _PARSER_MAP.get(file_type)
    if not parser_cls:
        raise ValueError(f"unsupported file type: {path.suffix!r} (detected as {file_type})")

    parser = parser_cls(config=config)
    return parser.parse(path)


def parse_documents_from_folder(
    folder_path: str | Path,
    config: ParsingConfig | None = None,
    recursive: bool = True,
    max_workers: int | None = None,
) -> list[ParsedDocument]:
    """
    Recursively parse all supported documents in a folder.

    Parameters
    ----------
    folder_path : str | Path
    config      : ParsingConfig | None
    recursive   : bool
        If True (default), descend into subdirectories.
    max_workers : int | None
        Override default worker count (default: auto = min(cpu_count, file_count, 8)).

    Returns
    -------
    list[ParsedDocument]
        Successfully parsed documents (failures are logged and skipped,
        not raised).
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"folder not found: {folder}")

    results: list[ParsedDocument] = []
    pattern = "**/*" if recursive else "*"
    file_paths = [p for p in sorted(folder.glob(pattern)) if p.is_file()]

    if not file_paths:
        return results

    if max_workers is None or max_workers <= 0:
        cpu_count = multiprocessing.cpu_count()
        if _INGEST_WORKERS > 0:
            workers = min(_INGEST_WORKERS, len(file_paths))
        else:
            workers = min(cpu_count, len(file_paths), 8)
    else:
        workers = min(max_workers, len(file_paths))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_worker_parse_one, str(p)): p for p in file_paths
        }
        for future in as_completed(futures):
            try:
                doc = future.result()
                if doc is not None:
                    results.append(doc)
            except Exception as e:
                pass

    # Restore original sorted order for deterministic output
    results.sort(key=lambda d: file_paths.index(Path(d.source_path)) if d.source_path else 0)
    return results
