from __future__ import annotations

from typing import Any, Dict

from app.qa.session import summarize_document
from app.tools.base import run_tool
from app.tools.schemas import KBSummarizeDocumentInput, KBSummarizeDocumentOutput

TOOL_SUMMARIZE_DOCUMENT = "kb_summarize_document"


def kb_summarize_document(input_data: KBSummarizeDocumentInput | Dict[str, Any]) -> Dict[str, Any]:
    payload = (
        input_data
        if isinstance(input_data, KBSummarizeDocumentInput)
        else KBSummarizeDocumentInput(**input_data)
    )

    def _execute() -> Dict[str, Any]:
        result = summarize_document(payload.document_id)
        raw = result if isinstance(result, dict) else vars(result)

        normalized = KBSummarizeDocumentOutput(
            document_id=payload.document_id,
            summary=raw["summary"],
        )
        return normalized.model_dump()

    return run_tool(TOOL_SUMMARIZE_DOCUMENT, _execute)