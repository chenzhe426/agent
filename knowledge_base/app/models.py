from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# -----------------------------
# documents / chunks
# -----------------------------
class DocumentImportResponse(BaseModel):
    id: int
    title: str
    file_path: str | None = None
    file_type: str | None = None
    source_type: str | None = None
    lang: str | None = None
    author: str | None = None
    block_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class ChunkResult(BaseModel):
    chunk_id: int | None = None
    document_id: int | None = None
    chunk_index: int | None = None
    score: float | None = None
    embedding_score: float | None = None
    keyword_score: float | None = None
    bm25_score: float | None = None
    title_match_score: float | None = None
    section_match_score: float | None = None
    coverage_score: float | None = None
    matched_term_count: int | None = None
    title: str = ""
    section_title: str = ""
    section_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    chunk_type: str | None = None
    chunk_text: str = ""
    term_hits: dict[str, int] = Field(default_factory=dict)
    term_hit_detail: dict[str, Any] = Field(default_factory=dict)
    is_neighbor: bool = False


class SourceHighlightSpan(BaseModel):
    start: int
    end: int
    text: str


class AnswerSource(BaseModel):
    chunk_id: int | None = None
    document_id: int | None = None
    title: str = ""
    section_title: str = ""
    section_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    quote: str = ""
    score: float | None = None
    highlight_spans: list[SourceHighlightSpan] = Field(default_factory=list)


class StructuredAnswer(BaseModel):
    answer: str
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    sources: list[AnswerSource] = Field(default_factory=list)
    confidence: float | None = None


class AskResponse(BaseModel):
    question: str
    rewritten_query: str | None = None
    answer: str
    structured: StructuredAnswer | None = None
    sources: list[AnswerSource] = Field(default_factory=list)
    retrieved_chunks: list[ChunkResult] = Field(default_factory=list)
    confidence: float | None = None
    session_id: str | None = None


class IndexedChunkItem(BaseModel):
    id: int | None = None
    chunk_index: int
    chunk_type: str | None = None
    section_path: str = ""
    section_title: str = ""
    page_start: int | None = None
    page_end: int | None = None
    token_count: int = 0
    block_start_index: int | None = None
    block_end_index: int | None = None
    preview: str = ""


class ImportFolderResponse(BaseModel):
    ok: bool = True
    count: int = 0
    documents: list[DocumentImportResponse] = Field(default_factory=list)


class ImportFileResponse(BaseModel):
    ok: bool = True
    document: DocumentImportResponse


class IndexResponse(BaseModel):
    ok: bool = True
    document_id: int
    title: str
    chunk_count: int = 0
    chunks: list[IndexedChunkItem] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    ok: bool = True
    document_id: int
    title: str
    summary: str


# -----------------------------
# API requests
# -----------------------------
class ImportFolderRequest(BaseModel):
    folder: str


class ImportFileRequest(BaseModel):
    file_path: str


class IndexRequest(BaseModel):
    document_id: int
    chunk_size: int = 1200
    overlap: int = 150


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    response_mode: Literal["text", "structured"] = "text"
    highlight: bool = True
    session_id: str | None = None
    use_chat_context: bool = True


class SummaryRequest(BaseModel):
    document_id: int


# -----------------------------
# chat
# -----------------------------
class ChatSessionCreateRequest(BaseModel):
    session_id: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatMessageItem(BaseModel):
    id: int | None = None
    session_id: str
    role: Literal["user", "assistant", "system"]
    message: str
    rewritten_query: str | None = None
    sources: list[AnswerSource] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class ChatSessionResponse(BaseModel):
    session_id: str
    title: str | None = None
    summary_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ChatHistoryResponse(BaseModel):
    session: ChatSessionResponse | None = None
    messages: list[ChatMessageItem] = Field(default_factory=list)


class AgentAskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question for the agent.")
    session_id: str | None = Field(default=None, description="Chat session id for multi-turn conversation.")