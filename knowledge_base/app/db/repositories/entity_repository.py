"""
entity_repository.py — Agent entity memory for tracking referenced entities across sessions.

Tracks documents, concepts, metrics, and other entities referenced during agent conversations
to enable better context understanding in subsequent interactions.
"""

from __future__ import annotations

import re
from typing import Any

from app.db.connection import get_cursor
from app.db.utils import safe_json_dumps


# Entity types for categorization
ENTITY_TYPE_DOCUMENT = "document"
ENTITY_TYPE_CONCEPT = "concept"
ENTITY_TYPE_METRIC = "metric"
ENTITY_TYPE_PERSON = "person"
ENTITY_TYPE_COMPANY = "company"
ENTITY_TYPE_OTHER = "other"


def _normalize_entity_key(key: str) -> str:
    """Normalize entity key for consistent matching."""
    return key.lower().strip()


def _extract_entities_from_text(text: str) -> list[tuple[str, str]]:
    """
    Extract potential entities from text using simple patterns.

    Returns list of (entity_type, entity_key) tuples.
    """
    entities = []

    if not text:
        return entities

    # Extract potential company names (capitalized words)
    company_pattern = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    for name in company_pattern:
        if len(name) > 2 and name not in {"The", "This", "That", "These", "Those", "Apple", "Google", "Microsoft"}:
            entities.append((ENTITY_TYPE_COMPANY, _normalize_entity_key(name)))

    # Extract metrics/percentages (e.g., "50%", "$100 million", "1.5x")
    metric_patterns = [
        r'\$[\d,]+\.?\d*\s*(?:million|billion|thousand)?',
        r'[\d,]+\.?\d*\s*%',
        r'[\d,]+\.?\d*\s*(?:x|times)',
    ]
    for pattern in metric_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append((ENTITY_TYPE_METRIC, _normalize_entity_key(match)))

    # Extract quoted terms (potential concepts)
    quoted = re.findall(r'"([^"]+)"', text)
    for term in quoted:
        if len(term) > 2:
            entities.append((ENTITY_TYPE_CONCEPT, _normalize_entity_key(term)))

    return entities


def track_entities(
    session_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    """
    Extract and track entities from the given text for a session.

    Returns the number of entities tracked.
    """
    extracted = _extract_entities_from_text(text)
    if not extracted:
        return 0

    tracked = 0
    with get_cursor(commit=True) as (_, cursor):
        for entity_type, entity_key in extracted:
            # Try to update existing entity
            cursor.execute(
                """
                UPDATE agent_entities
                SET reference_count = reference_count + 1,
                    last_seen_at = CURRENT_TIMESTAMP
                WHERE session_id = %s AND entity_type = %s AND entity_key = %s
                """,
                (session_id, entity_type, entity_key),
            )

            # If no existing row, insert new one
            if cursor.rowcount == 0:
                cursor.execute(
                    """
                    INSERT INTO agent_entities (session_id, entity_type, entity_key, reference_count, metadata)
                    VALUES (%s, %s, %s, 1, %s)
                    """,
                    (session_id, entity_type, entity_key, safe_json_dumps(metadata)),
                )
            tracked += 1

    return tracked


def track_document_entity(
    session_id: str,
    document_id: int,
    title: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Track a document reference."""
    with get_cursor(commit=True) as (_, cursor):
        entity_key = f"doc:{document_id}"
        cursor.execute(
            """
            UPDATE agent_entities
            SET reference_count = reference_count + 1,
                last_seen_at = CURRENT_TIMESTAMP,
                display_name = COALESCE(VALUES(display_name), display_name)
            WHERE session_id = %s AND entity_type = %s AND entity_key = %s
            """,
            (session_id, ENTITY_TYPE_DOCUMENT, entity_key),
        )

        if cursor.rowcount == 0:
            cursor.execute(
                """
                INSERT INTO agent_entities (session_id, entity_type, entity_key, display_name, reference_count, metadata)
                VALUES (%s, %s, %s, %s, 1, %s)
                """,
                (session_id, ENTITY_TYPE_DOCUMENT, entity_key, title, safe_json_dumps(metadata)),
            )


def get_session_entities(
    session_id: str,
    entity_type: str | None = None,
    min_reference_count: int = 1,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Get entities tracked for a session.

    Args:
        session_id: The session ID
        entity_type: Optional filter by entity type
        min_reference_count: Minimum reference count to include
        limit: Maximum number of entities to return

    Returns:
        List of entity dictionaries
    """
    with get_cursor() as (_, cursor):
        query = """
            SELECT id, session_id, entity_type, entity_key, display_name,
                   reference_count, first_seen_at, last_seen_at, metadata
            FROM agent_entities
            WHERE session_id = %s AND reference_count >= %s
        """
        params: list[Any] = [session_id, min_reference_count]

        if entity_type:
            query += " AND entity_type = %s"
            params.append(entity_type)

        query += " ORDER BY reference_count DESC, last_seen_at DESC LIMIT %s"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall() or []

        result = []
        for row in rows:
            result.append({
                "id": row[0],
                "session_id": row[1],
                "entity_type": row[2],
                "entity_key": row[3],
                "display_name": row[4],
                "reference_count": row[5],
                "first_seen_at": row[6],
                "last_seen_at": row[7],
                "metadata": row[8],
            })
        return result


def get_entity_memory_context(session_id: str, limit: int = 10) -> str:
    """
    Get entity memory as a formatted context string for LLM prompt injection.

    Returns a string like:
    "Previously discussed: Apple (company, 3 references), $100M (metric, 2 references)"
    """
    entities = get_session_entities(session_id, min_reference_count=2, limit=limit)
    if not entities:
        return ""

    parts = []
    for entity in entities:
        display = entity.get("display_name") or entity.get("entity_key")
        entity_type = entity.get("entity_type", "other")
        ref_count = entity.get("reference_count", 1)
        parts.append(f"{display} ({entity_type}, {ref_count} references)")

    return "Previously discussed: " + ", ".join(parts)


def clear_session_entities(session_id: str) -> int:
    """
    Clear all entities for a session.

    Returns the number of entities deleted.
    """
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute(
            "DELETE FROM agent_entities WHERE session_id = %s",
            (session_id,),
        )
        return cursor.rowcount
