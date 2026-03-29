from __future__ import annotations

from typing import Any

import app.config as config
from app.db.connection import get_cursor
from app.db.utils import normalize_row_json_fields, normalize_rows_json_fields, safe_json_dumps


CHAT_SESSION_JSON_FIELDS = ("metadata",)
CHAT_MESSAGE_JSON_FIELDS = ("citations", "metadata")


def _column_exists(cursor, table_name: str, column_name: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
        LIMIT 1
        """,
        (table_name, column_name),
    )
    return cursor.fetchone() is not None


def _chat_message_text_column(cursor) -> str:
    if _column_exists(cursor, "chat_messages", "content"):
        return "content"
    if _column_exists(cursor, "chat_messages", "message"):
        return "message"
    raise RuntimeError("chat_messages table has neither 'content' nor 'message' column")


def _with_session_aliases(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None

    normalized = dict(row)

    if "summary_text" not in normalized:
        metadata = normalized.get("metadata") or {}
        if isinstance(metadata, dict):
            normalized["summary_text"] = metadata.get("summary_text")

    return normalized


def _with_message_aliases(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None

    normalized = dict(row)

    # Normalize message/content - DB uses 'message' column, API uses 'content'
    if "message" not in normalized:
        if "content" in normalized:
            normalized["message"] = normalized["content"]
        else:
            normalized["message"] = ""

    if "content" not in normalized and "message" in normalized:
        normalized["content"] = normalized["message"]

    if "rewritten_query" not in normalized:
        metadata = normalized.get("metadata") or {}
        if isinstance(metadata, dict):
            normalized["rewritten_query"] = metadata.get("rewritten_query")
        else:
            normalized["rewritten_query"] = None

    if "sources" not in normalized:
        metadata = normalized.get("metadata") or {}
        if isinstance(metadata, dict):
            normalized["sources"] = metadata.get("sources") or []
        else:
            normalized["sources"] = []

    return normalized


def _rows_with_message_aliases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_with_message_aliases(row) for row in rows]


def create_chat_session(
    session_id: str,
    title: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | list[Any] | None = None,
) -> str:
    with get_cursor(commit=True) as (_, cursor):
        has_title = _column_exists(cursor, "chat_sessions", "title")
        has_user_id = _column_exists(cursor, "chat_sessions", "user_id")
        has_metadata = _column_exists(cursor, "chat_sessions", "metadata")
        has_last_message_at = _column_exists(cursor, "chat_sessions", "last_message_at")

        columns = ["session_id"]
        values = [session_id]

        if has_title:
            columns.append("title")
            values.append(title)
        if has_user_id:
            columns.append("user_id")
            values.append(user_id)
        if has_metadata:
            columns.append("metadata")
            values.append(safe_json_dumps(metadata))
        if has_last_message_at:
            columns.append("last_message_at")
            values.append(None)

        placeholders = ", ".join(["%s"] * len(columns))
        column_sql = ", ".join(columns)

        update_parts = []
        if has_title:
            update_parts.append("title = COALESCE(VALUES(title), title)")
        if has_user_id:
            update_parts.append("user_id = COALESCE(VALUES(user_id), user_id)")
        if has_metadata:
            update_parts.append("metadata = COALESCE(VALUES(metadata), metadata)")
        if has_last_message_at:
            update_parts.append("last_message_at = CURRENT_TIMESTAMP")
        if _column_exists(cursor, "chat_sessions", "updated_at"):
            update_parts.append("updated_at = CURRENT_TIMESTAMP")

        update_sql = ", ".join(update_parts) if update_parts else "session_id = VALUES(session_id)"

        cursor.execute(
            f"""
            INSERT INTO chat_sessions ({column_sql})
            VALUES ({placeholders})
            ON DUPLICATE KEY UPDATE {update_sql}
            """,
            values,
        )
        return session_id


def get_chat_session(session_id: str) -> dict[str, Any] | None:
    with get_cursor() as (_, cursor):
        cursor.execute(
            """
            SELECT *
            FROM chat_sessions
            WHERE session_id = %s
            LIMIT 1
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        row = normalize_row_json_fields(row, CHAT_SESSION_JSON_FIELDS)
        return _with_session_aliases(row)


def list_chat_sessions(limit: int = 50) -> list[dict[str, Any]]:
    with get_cursor() as (_, cursor):
        order_by = "created_at DESC"
        if _column_exists(cursor, "chat_sessions", "last_message_at"):
            order_by = "last_message_at DESC, created_at DESC"

        cursor.execute(
            f"""
            SELECT *
            FROM chat_sessions
            ORDER BY {order_by}
            LIMIT %s
            """,
            (int(limit),),
        )
        rows = cursor.fetchall() or []
        rows = normalize_rows_json_fields(rows, CHAT_SESSION_JSON_FIELDS)
        return [_with_session_aliases(row) for row in rows]


def update_chat_session(
    session_id: str,
    title: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | list[Any] | None = None,
    summary_text: str | None = None,
    last_message_at: Any = None,
) -> bool:
    with get_cursor(commit=True) as (_, cursor):
        update_fields: dict[str, Any] = {}

        if title is not None and _column_exists(cursor, "chat_sessions", "title"):
            update_fields["title"] = title

        if user_id is not None and _column_exists(cursor, "chat_sessions", "user_id"):
            update_fields["user_id"] = user_id

        if last_message_at is not None and _column_exists(cursor, "chat_sessions", "last_message_at"):
            update_fields["last_message_at"] = last_message_at

        if _column_exists(cursor, "chat_sessions", "metadata"):
            base_metadata: dict[str, Any] = {}
            if metadata is not None:
                if isinstance(metadata, dict):
                    base_metadata.update(metadata)
                else:
                    base_metadata["metadata"] = metadata

            if summary_text is not None:
                base_metadata["summary_text"] = summary_text

            if metadata is not None or summary_text is not None:
                update_fields["metadata"] = safe_json_dumps(base_metadata)

        if not update_fields:
            return False

        assignments = ", ".join(f"{key} = %s" for key in update_fields.keys())
        values = list(update_fields.values())

        if _column_exists(cursor, "chat_sessions", "updated_at"):
            assignments = f"{assignments}, updated_at = CURRENT_TIMESTAMP"

        values.append(session_id)

        cursor.execute(
            f"""
            UPDATE chat_sessions
            SET {assignments}
            WHERE session_id = %s
            """,
            values,
        )
        return cursor.rowcount > 0


def insert_chat_message(
    session_id: str,
    role: str,
    message: str,
    rewritten_query: str | None = None,
    sources: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | list[Any] | str | None = None,
    citations: list[Any] | dict[str, Any] | None = None,
) -> int:
    merged_metadata: Any

    if metadata is None:
        merged_metadata = {}
    elif isinstance(metadata, dict):
        merged_metadata = dict(metadata)
    else:
        merged_metadata = metadata

    if isinstance(merged_metadata, dict):
        if rewritten_query is not None:
            merged_metadata["rewritten_query"] = rewritten_query
        if sources is not None:
            merged_metadata["sources"] = sources
    elif rewritten_query is not None or sources is not None:
        merged_metadata = {
            "raw_metadata": merged_metadata,
            "rewritten_query": rewritten_query,
            "sources": sources or [],
        }

    with get_cursor(commit=True) as (_, cursor):
        text_column = _chat_message_text_column(cursor)
        has_citations = _column_exists(cursor, "chat_messages", "citations")
        has_metadata = _column_exists(cursor, "chat_messages", "metadata")

        columns = ["session_id", "role", text_column]
        values = [session_id, role, message]

        if has_citations:
            columns.append("citations")
            values.append(safe_json_dumps(citations))

        if has_metadata:
            columns.append("metadata")
            if isinstance(merged_metadata, str):
                values.append(merged_metadata)
            else:
                values.append(safe_json_dumps(merged_metadata))

        placeholders = ", ".join(["%s"] * len(columns))
        column_sql = ", ".join(columns)

        cursor.execute(
            f"""
            INSERT INTO chat_messages ({column_sql})
            VALUES ({placeholders})
            """,
            values,
        )

        if _column_exists(cursor, "chat_sessions", "last_message_at"):
            cursor.execute(
                """
                UPDATE chat_sessions
                SET last_message_at = CURRENT_TIMESTAMP
                WHERE session_id = %s
                """,
                (session_id,),
            )

        return int(cursor.lastrowid)


def get_chat_messages(session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is None:
        limit = getattr(config, "DEFAULT_TOP_K", 3) * 10

    with get_cursor() as (_, cursor):
        text_column = _chat_message_text_column(cursor)

        if text_column == "message":
            select_sql = """
                SELECT
                    id,
                    session_id,
                    role,
                    message,
                    message AS content,
                    citations,
                    metadata,
                    created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC, id ASC
                LIMIT %s
            """
        else:
            select_sql = """
                SELECT *
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC, id ASC
                LIMIT %s
            """

        cursor.execute(select_sql, (session_id, int(limit)))
        rows = cursor.fetchall() or []
        rows = normalize_rows_json_fields(rows, CHAT_MESSAGE_JSON_FIELDS)
        return _rows_with_message_aliases(rows)


def delete_chat_session(session_id: str) -> bool:
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute(
            """
            DELETE FROM chat_sessions
            WHERE session_id = %s
            """,
            (session_id,),
        )
        return cursor.rowcount > 0