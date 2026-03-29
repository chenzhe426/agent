from __future__ import annotations

from typing import Any, Dict

from app.services import index_document
from app.tools.base import run_tool
from app.tools.schemas import KBIndexDocumentInput, KBIndexDocumentOutput

TOOL_INDEX_DOCUMENT = "kb_index_document"


def kb_index_document(input_data: KBIndexDocumentInput | Dict[str, Any]) -> Dict[str, Any]:
    payload = input_data if isinstance(input_data, KBIndexDocumentInput) else KBIndexDocumentInput(**input_data)

    def _execute() -> Dict[str, Any]:
        result = index_document(
            document_id=payload.document_id,
            chunk_size=payload.chunk_size,
            overlap=payload.overlap,
        )

        raw = result if isinstance(result, dict) else vars(result)

        normalized = KBIndexDocumentOutput(
            document_id=payload.document_id,
            chunk_count=raw.get("chunk_count"),
            vector_count=raw.get("vector_count"),
            status=raw.get("status", "indexed"),
            raw=raw,
        )
        return normalized.model_dump()

    return run_tool(TOOL_INDEX_DOCUMENT, _execute)