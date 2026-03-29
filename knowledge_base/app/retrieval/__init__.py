"""
Retrieval module: split from the original monolithic retrieval_service.py.

Public API lives in:
    app.retrieval.service     - retrieve_chunks()
    app.retrieval.query_understanding - enhance_financial_query, classify_query_intent, rewrite_query

For backward compatibility, use app.services.retrieval_service which re-exports retrieve_chunks.
"""
from __future__ import annotations
