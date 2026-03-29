from contextlib import asynccontextmanager
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from fastapi.responses import Response

from app.agent.service import agent_ask, agent_ask_stream

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.db import create_chat_session, get_chat_session, init_db
from app.models import (
    AskRequest,
    AskResponse,
    ChatHistoryResponse,
    ChatSessionCreateRequest,
    ChatSessionResponse,
    DocumentImportResponse,
    ImportFileRequest,
    ImportFileResponse,
    ImportFolderRequest,
    ImportFolderResponse,
    IndexRequest,
    IndexResponse,
    SummaryRequest,
    SummaryResponse,
    AgentAskRequest,
)


from app.services import (
    import_documents,
    import_single_document,
    index_document,
    ingest_document,
    ingest_folder,
)
from app.qa.pipeline import answer_question
from app.qa.session import get_chat_history, summarize_document


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Knowledge Base API",
    version="2.0.0",
    description="Local RAG knowledge base with hybrid retrieval, structured output, highlighting, and chat context.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/import/folder", response_model=ImportFolderResponse)
def import_folder(req: ImportFolderRequest):
    try:
        result = import_documents(req.folder)
        documents = [
            item if isinstance(item, DocumentImportResponse) else DocumentImportResponse(**item)
            for item in (result or [])
        ]
        return ImportFolderResponse(
            ok=True,
            count=len(documents),
            documents=documents,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/import/file", response_model=ImportFileResponse)
def import_file(req: ImportFileRequest):
    try:
        result = import_single_document(req.file_path)
        document = result if isinstance(result, DocumentImportResponse) else DocumentImportResponse(**result)
        return ImportFileResponse(
            ok=True,
            document=document,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=IndexResponse)
def build_index(req: IndexRequest):
    try:
        result = index_document(
            document_id=req.document_id,
            chunk_size=req.chunk_size,
            overlap=req.overlap,
        )
        return IndexResponse(ok=True, **result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        result = answer_question(
            question=req.question,
            top_k=req.top_k,
            response_mode=req.response_mode,
            highlight=req.highlight,
            session_id=req.session_id,
            use_chat_context=req.use_chat_context,
        )
        return AskResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summary", response_model=SummaryResponse)
def summary(req: SummaryRequest):
    try:
        result = summarize_document(req.document_id)
        return SummaryResponse(ok=True, **result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/session", response_model=ChatSessionResponse)
def create_session(req: ChatSessionCreateRequest):
    try:
        session_id = create_chat_session(
            session_id=req.session_id,
            title=req.title,
            metadata=req.metadata,
        )
        session = get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=500, detail="failed to create session")

        return ChatSessionResponse(
            session_id=session.get("session_id"),
            title=session.get("title"),
            summary_text=session.get("summary_text"),
            metadata=session.get("metadata") or session.get("metadata_json") or {},
            created_at=session.get("created_at"),
            updated_at=session.get("updated_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{session_id}", response_model=ChatHistoryResponse)
def chat_history(session_id: str, limit: int = 20):
    try:
        result = get_chat_history(session_id=session_id, limit=limit)
        return ChatHistoryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/ask")
def agent_ask_api(req: AgentAskRequest):
    return agent_ask(req.question, session_id=req.session_id)


@app.post("/agent/ask/stream")
def agent_ask_stream_api(req: AgentAskRequest):
    return StreamingResponse(
        agent_ask_stream(req.question, session_id=req.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/demo")
def agent_demo_page():
    return FileResponse("app/frontend/index.html")

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)