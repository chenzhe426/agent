"""
memory_service.py — Unified memory service combining vector memory and tool chain memory.

Provides:
  - store_success(): Store successful agent outcomes to both stores
  - retrieve_context(): Retrieve relevant memory context for injection into prompts
"""

from __future__ import annotations

from typing import Any

from app.services.memory_vector_store import memory_vector_store
from app.db.repositories.tool_chain_repository import (
    get_memory_context,
    upsert_tool_chain,
)


class MemoryService:
    """
    Unified memory service combining vector memory (Qdrant) and tool chain memory (MySQL).
    """

    @staticmethod
    def store_success(
        question: str,
        answer: str,
        retrieved_chunk_ids: list[int],
        reasoning_trace: list[dict[str, Any]],
        session_id: str,
    ) -> None:
        """
        Store a successful agent outcome to both vector memory and tool chain memory.

        Called after verification passes (is_supported=True).
        """
        # Build tool chain from reasoning trace
        tool_chain = []
        for entry in reasoning_trace:
            action = entry.get("action", "")
            if action and action != "final":
                tool_chain.append({
                    "tool": action,
                    "args": entry.get("action_input", {}),
                })

        # Store to vector memory (Qdrant)
        try:
            memory_vector_store.store_successful_case(
                question=question,
                answer=answer,
                retrieved_chunk_ids=retrieved_chunk_ids,
                tool_chain=tool_chain,
                session_id=session_id,
            )
        except Exception as e:
            print(f"[MemoryService] store_success vector memory failed: {e}")

        # Store to tool chain memory (MySQL)
        try:
            upsert_tool_chain(
                question=question,
                reasoning_trace=reasoning_trace,
                session_id=session_id,
            )
        except Exception as e:
            print(f"[MemoryService] store_success tool chain failed: {e}")

    @staticmethod
    def retrieve_context(question: str, top_k: int = 3) -> dict[str, str]:
        """
        Retrieve memory context for a question.

        Returns dict with keys:
          - vector_memory: Past cases formatted for injection
          - tool_chain_memory: Tool chain suggestion formatted for injection
        """
        # Retrieve from vector memory
        vector_context = ""
        try:
            cases = memory_vector_store.search_similar_cases(question, top_k=top_k)
            if cases:
                parts = []
                for case in cases:
                    q = case.get("question", "")
                    a = case.get("answer", "")
                    score = case.get("score", 0)
                    if q and a:
                        parts.append(
                            f"相似案例(相关度:{score:.2f}): 问:{q} 答:{a[:100]}..."
                        )
                if parts:
                    vector_context = "\n".join(parts)
        except Exception as e:
            print(f"[MemoryService] retrieve_context vector memory failed: {e}")

        # Retrieve from tool chain memory
        tool_chain_context = ""
        try:
            tool_chain_context = get_memory_context(question, min_success_count=2)
        except Exception as e:
            print(f"[MemoryService] retrieve_context tool chain failed: {e}")

        return {
            "vector_memory": vector_context,
            "tool_chain_memory": tool_chain_context,
        }


memory_service = MemoryService()
