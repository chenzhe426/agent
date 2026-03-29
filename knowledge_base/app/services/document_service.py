import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from app.db import get_all_documents, get_chunks_by_document_id, insert_document
from app.ingestion.pipeline import parse_document, parse_documents_from_folder
from app.services.common import (
    normalize_section_path,
    normalize_whitespace,
    safe_get,
    safe_json_loads,
)


def _hash_text(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [normalize_whitespace(str(x)) for x in value if normalize_whitespace(str(x))]

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if "," in value:
            return [normalize_whitespace(x) for x in value.split(",") if normalize_whitespace(x)]
        return [normalize_whitespace(value)]

    return [normalize_whitespace(str(value))]


def _parse_datetime_like(value: Any):
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y/%m/%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(value, fmt)
            except Exception:
                pass

        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None

    return None


def _normalize_block(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        metadata = block.get("metadata") or {}
        return {
            "type": normalize_whitespace(str(block.get("type", "paragraph") or "paragraph")).lower(),
            "text": normalize_whitespace(block.get("text", "")),
            "section_path": normalize_section_path(block.get("section_path")),
            "page_start": block.get("page_start", block.get("page")),
            "page_end": block.get("page_end", block.get("page")),
            "metadata": metadata,
        }

    metadata = getattr(block, "metadata", {}) or {}
    return {
        "type": normalize_whitespace(str(getattr(block, "type", "paragraph") or "paragraph")).lower(),
        "text": normalize_whitespace(getattr(block, "text", "")),
        "section_path": normalize_section_path(getattr(block, "section_path", [])),
        "page_start": getattr(block, "page_start", getattr(block, "page", None)),
        "page_end": getattr(block, "page_end", getattr(block, "page", None)),
        "metadata": metadata,
    }


def _normalize_blocks(blocks: Any) -> list[dict[str, Any]]:
    if not blocks:
        return []

    normalized: list[dict[str, Any]] = []
    for block in blocks:
        item = _normalize_block(block)
        if item.get("text"):
            normalized.append(item)
    return normalized


def _to_document_import_result(payload: dict[str, Any], doc_id: int) -> dict[str, Any]:
    """
    统一 document import 类接口的 service 输出结构。
    对外统一使用 document_id；为兼容旧代码，暂时保留 id。
    """
    return {
        "document_id": doc_id,
        "id": doc_id,  # backward compatible
        "title": payload["title"],
        "file_path": payload["file_path"],
        "file_type": payload["file_type"],
        "source_type": payload["source_type"],
        "lang": payload["lang"],
        "author": payload["author"],
        "block_count": payload["block_count"],
        "metadata": payload["metadata_json"] or {},
        "tags": payload["tags_json"] or [],
    }


def parsed_document_to_db_payload(parsed_doc: Any, source_path: str | None = None) -> dict[str, Any]:
    title = normalize_whitespace(
        safe_get(parsed_doc, "title") or (Path(source_path).stem if source_path else "") or "Untitled"
    )
    content = normalize_whitespace(
        safe_get(parsed_doc, "content") or safe_get(parsed_doc, "raw_text") or ""
    )
    raw_text = normalize_whitespace(
        safe_get(parsed_doc, "raw_text") or safe_get(parsed_doc, "content") or ""
    )

    file_path = safe_get(parsed_doc, "source_path") or source_path
    file_type = normalize_whitespace(safe_get(parsed_doc, "file_type") or "")

    metadata = safe_get(parsed_doc, "metadata", {}) or {}
    blocks = _normalize_blocks(safe_get(parsed_doc, "blocks") or [])

    source_type = normalize_whitespace(
        metadata.get("source_type") or ("folder_import" if source_path and Path(source_path).exists() else "upload")
    )
    lang = normalize_whitespace(metadata.get("lang") or "")
    author = normalize_whitespace(metadata.get("author") or "")
    published_at = _parse_datetime_like(metadata.get("published_at"))
    tags = _normalize_tags(metadata.get("tags"))

    payload = {
        "title": title,
        "content": content,
        "raw_text": raw_text,
        "file_path": file_path,
        "file_type": file_type,
        "source_type": source_type or "upload",
        "lang": lang or None,
        "author": author or None,
        "published_at": published_at,
        "content_hash": _hash_text(content or raw_text),
        "block_count": len(blocks),
        "blocks_json": blocks,
        "metadata_json": metadata,
        "tags_json": tags,
    }
    return payload


def import_single_document(file_path: str) -> dict[str, Any]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"not a file: {file_path}")

    parsed_doc = parse_document(str(path))
    if not parsed_doc:
        raise ValueError(f"failed to parse document: {file_path}")

    # Detect scanned / image-only PDFs that have no extractable text
    is_scanned = parsed_doc.metadata.get("likely_scanned_pdf", False)
    ocr_hint = parsed_doc.metadata.get("hint", "")

    payload = parsed_document_to_db_payload(parsed_doc, source_path=str(path))
    has_content = bool(payload.get("content") or payload.get("raw_text"))

    if not has_content:
        if is_scanned:
            raise ValueError(
                f"document appears to be a scanned/image PDF with no extractable text "
                f"(likely_scanned_pdf=true). {ocr_hint}"
            )
        raise ValueError(f"document has no content: {file_path}")

    doc_id = insert_document(**payload)
    return _to_document_import_result(payload, doc_id)


def import_documents(folder: str) -> list[dict[str, Any]]:
    folder_path = Path(folder)

    if not folder_path.exists():
        raise FileNotFoundError(f"folder not found: {folder}")
    if not folder_path.is_dir():
        raise ValueError(f"not a folder: {folder}")

    parsed_docs = parse_documents_from_folder(str(folder_path))
    results: list[dict[str, Any]] = []

    for parsed_doc in parsed_docs:
        source_path = safe_get(parsed_doc, "source_path")
        payload = parsed_document_to_db_payload(parsed_doc, source_path=source_path)

        if not payload["content"] and not payload["raw_text"]:
            is_scanned = parsed_doc.metadata.get("likely_scanned_pdf", False)
            if is_scanned:
                import logging
                logging.warning(
                    "[import] skipping scanned/image PDF with no extractable text: %s",
                    source_path,
                )
            continue

        doc_id = insert_document(**payload)
        results.append(_to_document_import_result(payload, doc_id))

    return results


def list_documents() -> list[dict[str, Any]]:
    rows = get_all_documents() or []
    results: list[dict[str, Any]] = []

    for row in rows:
        blocks = row.get("blocks_json") or []
        metadata = row.get("metadata_json") or {}
        tags = row.get("tags_json") or []

        results.append(
            {
                "document_id": row.get("id"),
                "id": row.get("id"),
                "title": row.get("title"),
                "file_path": row.get("file_path"),
                "file_type": row.get("file_type"),
                "source_type": row.get("source_type"),
                "lang": row.get("lang"),
                "author": row.get("author"),
                "published_at": row.get("published_at"),
                "content_hash": row.get("content_hash"),
                "block_count": row.get("block_count") or len(blocks),
                "metadata": metadata,
                "tags": tags,
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            }
        )

        

    return results


def get_document_chunks(document_id: int) -> list[dict[str, Any]]:
    rows = get_chunks_by_document_id(document_id) or []
    results: list[dict[str, Any]] = []

    for row in rows:
        metadata = row.get("metadata") or {}
        embedding = row.get("embedding")
        if isinstance(embedding, str):
            embedding = safe_json_loads(embedding, default=[])

        results.append(
            {
                "id": row.get("id"),
                "document_id": row.get("document_id"),
                "chunk_index": row.get("chunk_index"),
                "chunk_text": row.get("chunk_text"),
                "search_text": row.get("search_text"),
                "lexical_text": row.get("lexical_text"),
                "embedding": embedding,
                "chunk_type": row.get("chunk_type"),
                "title": row.get("title"),
                "section_title": row.get("section_title"),
                "section_path": row.get("section_path"),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "block_start_index": row.get("block_start_index"),
                "block_end_index": row.get("block_end_index"),
                "token_count": row.get("token_count"),
                "chunk_hash": row.get("chunk_hash"),
                "metadata": metadata,
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            }
        )

    return results