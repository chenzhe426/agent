"""
EvalAdapter: unified interface for running evaluation samples against
the internal Python services or the FastAPI HTTP interface.

Design goals:
  - Do NOT rewrite or copy RAG/retrieval logic here.
  - Wrap existing service functions with a thin, predictable output schema.
  - The adapter layer is the only place that knows about internal interfaces.
"""
from __future__ import annotations

import time
import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Internal imports – only used when mode == "internal"
# ---------------------------------------------------------------------------
# We import lazily so the module can be loaded without triggering
# the full app initialization (e.g. when checking --help).
# ---------------------------------------------------------------------------

_RETRIEVAL_RETRIES = 1


def _normalize_text(text: str) -> str:
    """Lightweight text normalization for comparison."""
    if not text:
        return ""
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


class EvalAdapter:
    """
    Unified evaluation adapter.

    Parameters
    ----------
    mode : str
        "internal" – call Python functions directly (default, no HTTP dependency).
        "api"      – call FastAPI endpoints over HTTP.
    top_k : int
        Number of chunks to return AND to pass to LLM context.
    retrieve_top_k : int, optional
        If > top_k, retrieves this many candidates first, then reranks and returns top_k.
        Enables two-stage retrieval.
    enable_query_enhance : bool
        If True, applies lightweight financial query enhancement before retrieval.
        The original question is still used for answer generation.
    api_base_url : str
        FastAPI base URL (only used when mode == "api").
    session_id : str, optional
        Optional fixed session ID for multi-turn evaluation.
    """

    def __init__(
        self,
        mode: str = "internal",
        top_k: int = 5,
        retrieve_top_k: int | None = None,
        enable_query_enhance: bool = False,
        use_multistage: bool = False,
        api_base_url: str = "http://127.0.0.1:8000",
        session_id: Optional[str] = None,
        llm_timeout: int | None = None,
    ) -> None:
        if mode not in {"internal", "api"}:
            raise ValueError(f"mode must be 'internal' or 'api', got: {mode!r}")
        self.mode = mode
        self.top_k = top_k
        self.retrieve_top_k = retrieve_top_k
        self.enable_query_enhance = enable_query_enhance
        self.use_multistage = use_multistage
        self.api_base_url = api_base_url.rstrip("/")
        self.session_id = session_id
        self.llm_timeout = llm_timeout

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, conversation_history: Optional[list[dict]] = None, top_k: Optional[int] = None) -> dict[str, Any]:
        """
        Run retrieval only.

        Returns
        -------
        dict with keys:
          - query           : str  – the input query
          - retrieved_chunks: list[dict]  – each dict has chunk_id, document_id, score, chunk_text, …
          - latency_ms      : float
          - raw_response     : Any  – original return value from the underlying service
        """
        tk = top_k if top_k is not None else self.top_k
        start = time.perf_counter()

        if self.mode == "internal":
            result = self._retrieve_internal(query, tk)
        else:
            result = self._retrieve_api(query, tk)

        latency_ms = (time.perf_counter() - start) * 1000

        chunks = self._extract_chunks(result)
        return {
            "query": query,
            "retrieved_chunks": chunks,
            "latency_ms": latency_ms,
            "raw_response": result,
        }

    def answer(self, query: str, conversation_history: Optional[list[dict]] = None) -> dict[str, Any]:
        """
        Run full RAG pipeline (retrieve + generate answer).

        Returns
        -------
        dict with keys:
          - query             : str
          - retrieved_chunks  : list[dict]
          - final_answer      : str
          - raw_response      : Any  – original return from qa service
          - latency_ms        : float
        """
        start = time.perf_counter()
        print(f"[DEBUG EvalAdapter.answer] Starting answer for query: {query[:80]}...")

        # Detect company name from query and resolve to doc_id for filtering
        doc_filter = self._detect_company_doc_id(query)

        if self.mode == "internal":
            result = self._answer_internal(query, conversation_history, doc_filter=doc_filter)
        else:
            result = self._answer_api(query, conversation_history)

        latency_ms = (time.perf_counter() - start) * 1000
        print(f"[DEBUG EvalAdapter.answer] Got result after {latency_ms:.0f}ms")

        answer_text = self._extract_answer(result)
        chunks = self._extract_chunks(result)

        # Extract stage timings if present
        stage_timings = {}
        if isinstance(result, dict):
            stage_timings = result.get("stage_timings", {})
            total_pipeline_ms = result.get("total_pipeline_ms")

        return {
            "query": query,
            "retrieved_chunks": chunks,
            "final_answer": answer_text,
            "raw_response": result,
            "latency_ms": latency_ms,
            "stage_timings": stage_timings,
            "total_pipeline_ms": total_pipeline_ms,
        }

    # ------------------------------------------------------------------
    # Internal mode implementations
    # ------------------------------------------------------------------

    def _retrieve_internal(self, query: str, top_k: int) -> list[dict[str, Any]]:
        # Import here to avoid circular imports and to lazy-load
        from app.retrieval.service import retrieve_chunks
        return retrieve_chunks(query, top_k=top_k)

    def _answer_internal(self, query: str, history: Optional[list[dict]], doc_filter: int | None = None) -> dict[str, Any]:
        from app.qa.pipeline import answer_question
        print(f"[DEBUG EvalAdapter] _answer_internal called with query: {query[:80]}...")
        result = answer_question(
            question=query,
            top_k=self.top_k,
            response_mode="text",
            session_id=self.session_id,
            use_chat_context=bool(history),
            retrieve_top_k=self.retrieve_top_k,
            enable_query_enhance=self.enable_query_enhance,
            use_multistage=self.use_multistage,
            llm_timeout=self.llm_timeout,
            doc_filter=doc_filter,
        )
        print(f"[DEBUG EvalAdapter] answer_question returned, result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        return result

    # ------------------------------------------------------------------
    # Company detection for doc filtering
    # ------------------------------------------------------------------

    _COMPANY_KEYWORDS = {
        "AMD": "AMD_2022_10K",
        "BOEING": "BOEING_2022_10K",
        "AMERICAN EXPRESS": "AMERICANEXPRESS_2022_10K",
        "AMEX": "AMERICANEXPRESS_2022_10K",
    }

    _doc_id_cache: dict[str, int] = {}

    def _detect_company_doc_id(self, query: str) -> int | None:
        """
        Detect company name from query and return the corresponding doc_id.
        Returns None if no company detected or not found.
        """
        query_upper = query.upper()

        for keyword, doc_title in self._COMPANY_KEYWORDS.items():
            if keyword in query_upper:
                # Lazy-load doc_id mapping
                if not self._doc_id_cache:
                    from app.db.bootstrap import init_db
                    from app.db import get_all_documents
                    init_db()
                    for d in get_all_documents():
                        self._doc_id_cache[d["title"]] = d["id"]

                doc_id = self._doc_id_cache.get(doc_title)
                if doc_id is not None:
                    print(f"[DEBUG] Detected company '{keyword}' -> doc_id={doc_id}")
                    return doc_id
        return None

    # ------------------------------------------------------------------
    # API mode implementations
    # ------------------------------------------------------------------

    def _retrieve_api(self, query: str, top_k: int) -> list[dict[str, Any]]:
        import requests
        try:
            resp = requests.get(
                f"{self.api_base_url}/retrieval",
                params={"q": query, "top_k": top_k},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("chunks", [])
        except Exception as e:
            return [{"error": str(e)}]

    def _answer_api(self, query: str, history: Optional[list[dict]]) -> dict[str, Any]:
        import requests
        payload: dict[str, Any] = {"question": query, "top_k": self.top_k}
        if history:
            payload["conversation_history"] = history
        try:
            resp = requests.post(
                f"{self.api_base_url}/ask",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e), "question": query}

    # ------------------------------------------------------------------
    # Output normalization helpers
    # ------------------------------------------------------------------

    def _extract_chunks(self, result: Any) -> list[dict[str, Any]]:
        """
        Extract a uniform list of chunk dicts from whatever the underlying
        service returns.

        Handles:
          - list[dict]  (direct retrieval result)
          - dict with "retrieved_chunks" key
          - dict with "sources" key  (qa_service format)
        """
        if isinstance(result, list):
            return [self._normalize_chunk_item(c) for c in result]

        if not isinstance(result, dict):
            return []

        # qa_service returns "retrieved_chunks"
        raw = result.get("retrieved_chunks")
        if isinstance(raw, list):
            return [self._normalize_chunk_item(c) for c in raw]

        # qa_service also has "sources" which has similar info
        sources = result.get("sources", [])
        if isinstance(sources, list) and sources:
            return [self._normalize_chunk_item(s) for s in sources]

        return []

    def _normalize_chunk_item(self, item: Any) -> dict[str, Any]:
        """Map whatever chunk dict format we get to a consistent shape."""
        if not isinstance(item, dict):
            return {}
        return {
            "chunk_id": self._to_int(item.get("chunk_id")),
            "document_id": self._to_int(item.get("document_id")),
            "score": float(item.get("score") or 0.0),
            "rerank_score": float(item.get("rerank_score") or item.get("score") or 0.0),
            "embedding_score": float(item.get("embedding_score") or 0.0),
            "keyword_score": float(item.get("keyword_score") or 0.0),
            "bm25_score": float(item.get("bm25_score") or 0.0),
            "title": str(item.get("title") or ""),
            "section_title": str(item.get("section_title") or ""),
            "section_path": str(item.get("section_path") or ""),
            "chunk_text": str(item.get("chunk_text") or item.get("quote") or ""),
            "search_text": str(item.get("search_text") or ""),
            "page_start": self._to_int(item.get("page_start")),
            "page_end": self._to_int(item.get("page_end")),
            "term_hits": item.get("term_hits") or {},
            "_retrieval_query": item.get("_retrieval_query"),
            # New P0 rerank signal breakdowns
            "_numeric_boost": float(item.get("_numeric_boost") or 0.0),
            "_table_boost": float(item.get("_table_boost") or 0.0),
            "_query_aware_boost": float(item.get("_query_aware_boost") or 0.0),
            "_anti_noise_penalty": float(item.get("_anti_noise_penalty") or 0.0),
            "_page_cluster_bonus": float(item.get("_page_cluster_bonus") or 0.0),
            "_query_intent": str(item.get("_query_intent") or "unknown"),
        }

    def _extract_answer(self, result: Any) -> str:
        """Extract answer string from service result."""
        if isinstance(result, dict):
            return str(result.get("answer") or result.get("error") or "")
        return str(result) if result else ""

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
