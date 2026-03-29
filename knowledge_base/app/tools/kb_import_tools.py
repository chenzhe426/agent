from __future__ import annotations

from typing import Any, Dict, List

from app.services import import_documents, import_single_document
from app.tools.base import run_tool
from app.tools.schemas import (
    DocumentBrief,
    KBImportFileInput,
    KBImportFileOutput,
    KBImportFolderInput,
    KBImportFolderOutput,
)

TOOL_IMPORT_FILE = "kb_import_file"
TOOL_IMPORT_FOLDER = "kb_import_folder"


def _normalize_document_brief(item: Any) -> DocumentBrief:
    if hasattr(item, "model_dump"):
        raw = item.model_dump()
    elif isinstance(item, dict):
        raw = item
    else:
        raw = vars(item)

    return DocumentBrief(
        document_id=raw.get("document_id"),
        title=raw.get("title"),
        status=raw.get("status", "imported"),
        message=raw.get("message"),
    )


def kb_import_file(input_data: KBImportFileInput | Dict[str, Any]) -> Dict[str, Any]:
    payload = input_data if isinstance(input_data, KBImportFileInput) else KBImportFileInput(**input_data)

    def _execute() -> Dict[str, Any]:
        result = import_single_document(payload.file_path)

        if hasattr(result, "model_dump"):
            raw = result.model_dump()
        elif isinstance(result, dict):
            raw = result
        else:
            raw = vars(result)

        normalized = KBImportFileOutput(
            document_id=raw["document_id"],
            title=raw.get("title"),
            status=raw.get("status", "imported"),
            message=raw.get("message", "document imported successfully"),
        )
        return normalized.model_dump()

    return run_tool(TOOL_IMPORT_FILE, _execute)


def kb_import_folder(input_data: KBImportFolderInput | Dict[str, Any]) -> Dict[str, Any]:
    payload = input_data if isinstance(input_data, KBImportFolderInput) else KBImportFolderInput(**input_data)

    def _execute() -> Dict[str, Any]:
        results = import_documents(payload.folder) or []
        documents: List[DocumentBrief] = [_normalize_document_brief(item) for item in results]

        normalized = KBImportFolderOutput(
            count=len(documents),
            documents=documents,
        )
        return normalized.model_dump()

    return run_tool(TOOL_IMPORT_FOLDER, _execute)