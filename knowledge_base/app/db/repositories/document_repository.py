from __future__ import annotations

from typing import Any

from app.db.connection import get_cursor
from app.db.utils import normalize_row_json_fields, normalize_rows_json_fields, safe_json_dumps


DOCUMENT_JSON_FIELDS = ("blocks_json", "metadata_json", "tags_json")


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


def insert_document(
    title: str,
    content: str | None = None,
    raw_text: str | None = None,
    summary: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_path: str | None = None,
    file_type: str | None = None,
    mime_type: str | None = None,
    lang: str | None = None,
    author: str | None = None,
    published_at: Any = None,
    content_hash: str | None = None,
    block_count: int | None = None,
    blocks_json: list[Any] | dict[str, Any] | None = None,
    metadata_json: dict[str, Any] | list[Any] | None = None,
    tags_json: list[Any] | dict[str, Any] | None = None,
    status: str = "active",
) -> int:
    with get_cursor(commit=True) as (_, cursor):
        columns: list[str] = ["title"]
        values: list[Any] = [title]

        optional_fields = {
            "content": content,
            "raw_text": raw_text,
            "summary": summary,
            "source": source,
            "source_type": source_type,
            "file_path": file_path,
            "file_type": file_type,
            "mime_type": mime_type,
            "lang": lang,
            "author": author,
            "published_at": published_at,
            "content_hash": content_hash,
            "block_count": block_count if block_count is not None else 0,
            "status": status,
        }

        for field, value in optional_fields.items():
            if _column_exists(cursor, "documents", field):
                columns.append(field)
                values.append(value)

        json_optional_fields = {
            "blocks_json": blocks_json,
            "metadata_json": metadata_json,
            "tags_json": tags_json,
        }

        for field, value in json_optional_fields.items():
            if _column_exists(cursor, "documents", field):
                columns.append(field)
                values.append(safe_json_dumps(value))

        placeholders = ", ".join(["%s"] * len(columns))
        column_sql = ", ".join(columns)

        cursor.execute(
            f"""
            INSERT INTO documents ({column_sql})
            VALUES ({placeholders})
            """,
            values,
        )
        return int(cursor.lastrowid)


def get_document_by_id(document_id: int) -> dict[str, Any] | None:
    with get_cursor() as (_, cursor):
        cursor.execute(
            """
            SELECT *
            FROM documents
            WHERE id = %s
            LIMIT 1
            """,
            (document_id,),
        )
        row = cursor.fetchone()
        row = normalize_row_json_fields(row, DOCUMENT_JSON_FIELDS)
        return row


def get_all_documents() -> list[dict[str, Any]]:
    with get_cursor() as (_, cursor):
        cursor.execute(
            """
            SELECT *
            FROM documents
            ORDER BY created_at DESC, id DESC
            """
        )
        rows = cursor.fetchall() or []
        rows = normalize_rows_json_fields(rows, DOCUMENT_JSON_FIELDS)
        return rows


def search_documents(keyword: str) -> list[dict[str, Any]]:
    like = f"%{keyword}%"
    with get_cursor() as (_, cursor):
        cursor.execute(
            """
            SELECT *
            FROM documents
            WHERE title LIKE %s
               OR content LIKE %s
               OR raw_text LIKE %s
               OR summary LIKE %s
               OR source LIKE %s
               OR file_path LIKE %s
            ORDER BY created_at DESC, id DESC
            """,
            (like, like, like, like, like, like),
        )
        rows = cursor.fetchall() or []
        rows = normalize_rows_json_fields(rows, DOCUMENT_JSON_FIELDS)
        return rows


def update_document(document_id: int, **fields: Any) -> bool:
    allowed_fields = {
        "title",
        "content",
        "raw_text",
        "summary",
        "source",
        "source_type",
        "file_path",
        "file_type",
        "mime_type",
        "lang",
        "author",
        "published_at",
        "content_hash",
        "block_count",
        "blocks_json",
        "metadata_json",
        "tags_json",
        "status",
    }

    update_fields = {k: v for k, v in fields.items() if k in allowed_fields}
    if not update_fields:
        return False

    with get_cursor(commit=True) as (_, cursor):
        real_update_fields: dict[str, Any] = {}

        for key, value in update_fields.items():
            if not _column_exists(cursor, "documents", key):
                continue
            if key in {"blocks_json", "metadata_json", "tags_json"}:
                real_update_fields[key] = safe_json_dumps(value)
            else:
                real_update_fields[key] = value

        if not real_update_fields:
            return False

        assignments = ", ".join(f"{key} = %s" for key in real_update_fields.keys())
        values = list(real_update_fields.values()) + [document_id]

        cursor.execute(
            f"""
            UPDATE documents
            SET {assignments}
            WHERE id = %s
            """,
            values,
        )
        return cursor.rowcount > 0


def delete_document(document_id: int) -> bool:
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute(
            """
            DELETE FROM documents
            WHERE id = %s
            """,
            (document_id,),
        )
        return cursor.rowcount > 0