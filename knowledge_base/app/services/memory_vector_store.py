"""
memory_vector_store.py — Qdrant-backed vector memory for successful agent outcomes.

Stores (question, answer, chunks) triplets in Qdrant collection "kb_memory".
On retrieval, finds similar past cases by question embedding cosine similarity.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import EMBEDDING_DIM
from app.services.llm_service import get_embedding


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _normalize_embedding(emb: Any) -> list[float]:
    if not emb:
        return []
    try:
        vector = [float(x) for x in emb]
    except (TypeError, ValueError):
        return []
    if not vector:
        return []
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        return []
    norm = np.linalg.norm(arr)
    if norm == 0:
        return []
    return (arr / norm).astype(np.float32).tolist()


MEMORY_COLLECTION_NAME = "kb_memory"
MEMORY_VECTOR_NAME = "dense"


class MemoryVectorStore:
    """
    Qdrant-backed vector memory for successful (question, answer, chunks) triplets.

    Payload schema:
      - question: str
      - answer: str
      - retrieved_chunk_ids: list[int]
      - session_id: str
      - tool_chain: list[dict]  # [{tool, args}, ...]
      - is_supported: bool
      - created_at: str (ISO)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self.url = _env_str("QDRANT_URL", "http://localhost:6333")
        self.api_key = _env_str("QDRANT_API_KEY", "")
        self.collection_name = _env_str("QDRANT_MEMORY_COLLECTION", MEMORY_COLLECTION_NAME)
        self.vector_name = MEMORY_VECTOR_NAME
        self.embedding_dim = _env_int("EMBEDDING_DIM", EMBEDDING_DIM)
        self.enabled = bool(self.url)

        self.client: QdrantClient | None = None
        if self.enabled:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key or None,
            )
        self._initialized = True

    def _collection_exists(self) -> bool:
        if not self.client:
            return False
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception:
            return False

    def ensure_collection(self) -> None:
        """Create kb_memory collection if it doesn't exist."""
        if not self.client:
            return
        if self._collection_exists():
            return

        actual_dim = self.embedding_dim
        try:
            emb = get_embedding("dimension probe")
            normalized = _normalize_embedding(emb)
            if normalized:
                actual_dim = len(normalized)
        except Exception:
            pass

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.vector_name: VectorParams(
                    size=actual_dim,
                    distance=Distance.COSINE,
                )
            },
        )
        self.embedding_dim = actual_dim

    def search_similar_cases(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k similar past cases by question embedding.

        Returns list of dicts with keys: question, answer, retrieved_chunk_ids,
        tool_chain, score, created_at.
        """
        if not self.client or not query_text or top_k <= 0:
            return []

        emb = get_embedding(query_text)
        query_embedding = _normalize_embedding(emb)
        if not query_embedding:
            return []

        self.ensure_collection()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            using=self.vector_name,
            limit=top_k,
            with_payload=True,
        )

        points = getattr(response, "points", None) or []
        results = []
        for point in points:
            payload = point.payload or {}
            results.append({
                "question": payload.get("question", ""),
                "answer": payload.get("answer", ""),
                "retrieved_chunk_ids": payload.get("retrieved_chunk_ids", []),
                "tool_chain": payload.get("tool_chain", []),
                "session_id": payload.get("session_id", ""),
                "is_supported": payload.get("is_supported", True),
                "created_at": payload.get("created_at", ""),
                "score": float(point.score),
            })
        return results

    def store_successful_case(
        self,
        question: str,
        answer: str,
        retrieved_chunk_ids: list[int],
        tool_chain: list[dict[str, Any]],
        session_id: str,
    ) -> bool:
        """
        Store a successful (question, answer, chunks) triplet.
        Returns True on success.
        """
        if not self.client or not question:
            return False

        try:
            emb = get_embedding(question)
            query_embedding = _normalize_embedding(emb)
            if not query_embedding:
                return False

            self.ensure_collection()

            # Deterministic point ID from session + question
            raw_id = f"{session_id}:{question}".encode()
            point_id_int = int(hashlib.md5(raw_id).hexdigest()[:16], 16) % (2**63)

            point = PointStruct(
                id=point_id_int,
                vector={self.vector_name: query_embedding},
                payload={
                    "question": question,
                    "answer": answer,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "session_id": session_id,
                    "tool_chain": tool_chain,
                    "is_supported": True,
                    "created_at": datetime.utcnow().isoformat(),
                },
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )
            return True
        except Exception as e:
            print(f"[MemoryVectorStore] store_successful_case failed: {e}")
            return False


# Singleton accessor
memory_vector_store = MemoryVectorStore()
