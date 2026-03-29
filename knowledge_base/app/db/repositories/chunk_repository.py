from __future__ import annotations

import re
from typing import Any, Iterable

import app.config as config
from app.db.connection import get_cursor
from app.db.utils import normalize_row_json_fields, normalize_rows_json_fields, safe_json_dumps


CHUNK_JSON_FIELDS = ("metadata_json", "section_path", "embedding")


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


def _index_exists(cursor, table_name: str, index_name: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND INDEX_NAME = %s
        LIMIT 1
        """,
        (table_name, index_name),
    )
    return cursor.fetchone() is not None


def _chunk_text_column(cursor) -> str:
    if _column_exists(cursor, "document_chunks", "chunk_text"):
        return "chunk_text"
    if _column_exists(cursor, "document_chunks", "content"):
        return "content"
    raise RuntimeError("document_chunks table has neither 'chunk_text' nor 'content' column")


def _normalize_chunk_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    row = normalize_row_json_fields(row, CHUNK_JSON_FIELDS)
    if row is None:
        return None

    normalized = dict(row)

    # Normalize doc_title -> title
    if "doc_title" in normalized:
        normalized["title"] = normalized.pop("doc_title")
    elif "title" not in normalized:
        normalized["title"] = ""

    # Normalize content/chunk_text
    if "chunk_text" not in normalized:
        normalized["chunk_text"] = normalized.get("content", "")
    if "content" not in normalized:
        normalized["content"] = normalized.get("chunk_text", "")

    # Normalize metadata_json -> metadata (strip _json suffix)
    if "metadata_json" in normalized:
        normalized["metadata"] = normalized.pop("metadata_json")
    elif "metadata" not in normalized:
        normalized["metadata"] = {}

    if "search_text" not in normalized:
        normalized["search_text"] = normalized.get("chunk_text", "")
    if "lexical_text" not in normalized:
        normalized["lexical_text"] = normalized.get("search_text", "") or normalized.get("chunk_text", "")

    return normalized


def _normalize_chunk_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_chunk_row(row) for row in rows]


def _tokenize_terms(query: str) -> list[str]:
    if not query:
        return []
    return [term for term in re.findall(r"[\w\u4e00-\u9fff]+", query.lower()) if term]


def _build_boolean_query(query: str, *, require_all: bool = False) -> str:
    terms = _tokenize_terms(query)
    if not terms:
        return ""

    cleaned: list[str] = []
    for term in terms:
        term = term.strip()
        if not term:
            continue
        term = re.sub(r'[+\-<>~*"()@]+', " ", term).strip()
        if not term:
            continue
        if require_all:
            cleaned.append(f"+{term}*")
        else:
            cleaned.append(f"{term}*")
    return " ".join(cleaned)


def ensure_chunk_search_indexes() -> None:
    with get_cursor(commit=True) as (_, cursor):
        if not _index_exists(cursor, "document_chunks", "idx_chunks_document_chunk"):
            cursor.execute(
                """
                CREATE INDEX idx_chunks_document_chunk
                ON document_chunks(document_id, chunk_index)
                """
            )

        if not _index_exists(cursor, "document_chunks", "ft_chunks_lexical"):
            cursor.execute(
                """
                ALTER TABLE document_chunks
                ADD FULLTEXT INDEX ft_chunks_lexical (
                    lexical_text,
                    search_text,
                    title,
                    section_title
                )
                """
            )


def insert_chunk(
    document_id: int,
    chunk_text: str,
    embedding: Any = None,
    chunk_index: int | None = None,
    section_path: Any = None,
    page_start: int | None = None,
    page_end: int | None = None,
    block_start_index: int | None = None,
    block_end_index: int | None = None,
    chunk_type: str | None = None,
    metadata_json: dict[str, Any] | list[Any] | str | None = None,
    search_text: str | None = None,
    lexical_text: str | None = None,
    title: str | None = None,
    section_title: str | None = None,
    token_count: int | None = None,
    chunk_hash: str | None = None,
    embedding_model: str | None = None,
) -> int:
    if embedding_model is None:
        embedding_model = getattr(config, "OLLAMA_EMBED_MODEL", None)

    with get_cursor(commit=True) as (_, cursor):
        text_column = _chunk_text_column(cursor)

        columns = ["document_id", text_column]
        values: list[Any] = [document_id, chunk_text]

        def add_if_exists(column_name: str, value: Any, *, json_dump: bool = False) -> None:
            if _column_exists(cursor, "document_chunks", column_name):
                columns.append(column_name)
                values.append(safe_json_dumps(value) if json_dump else value)

        add_if_exists("embedding", embedding, json_dump=True)
        add_if_exists("chunk_index", chunk_index)
        add_if_exists("section_path", section_path, json_dump=True)
        add_if_exists("page_start", page_start)
        add_if_exists("page_end", page_end)
        add_if_exists("block_start_index", block_start_index)
        add_if_exists("block_end_index", block_end_index)
        add_if_exists("chunk_type", chunk_type)
        add_if_exists("metadata_json", metadata_json, json_dump=not isinstance(metadata_json, str))
        add_if_exists("search_text", search_text)
        add_if_exists("lexical_text", lexical_text)
        add_if_exists("title", title)
        add_if_exists("section_title", section_title)
        add_if_exists("token_count", token_count)
        add_if_exists("chunk_hash", chunk_hash)
        add_if_exists("embedding_model", embedding_model)

        placeholders = ", ".join(["%s"] * len(columns))
        column_sql = ", ".join(columns)

        unique_on_document_chunk = (
            chunk_index is not None
            and _column_exists(cursor, "document_chunks", "document_id")
            and _column_exists(cursor, "document_chunks", "chunk_index")
        )

        if unique_on_document_chunk:
            update_columns = [c for c in columns if c not in {"document_id", "chunk_index"}]
            update_sql = ", ".join(f"{c} = VALUES({c})" for c in update_columns)

            if _column_exists(cursor, "document_chunks", "updated_at"):
                if update_sql:
                    update_sql += ", updated_at = CURRENT_TIMESTAMP"
                else:
                    update_sql = "updated_at = CURRENT_TIMESTAMP"

            cursor.execute(
                f"""
                INSERT INTO document_chunks ({column_sql})
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE {update_sql}
                """,
                values,
            )

            cursor.execute(
                """
                SELECT id
                FROM document_chunks
                WHERE document_id = %s AND chunk_index = %s
                LIMIT 1
                """,
                (document_id, chunk_index),
            )
            row = cursor.fetchone()
            return int(row["id"])

        cursor.execute(
            f"""
            INSERT INTO document_chunks ({column_sql})
            VALUES ({placeholders})
            """,
            values,
        )
        return int(cursor.lastrowid)


def insert_chunks_batch(
    document_id: int,
    chunks: list[dict[str, Any]],
) -> list[int]:
    """
    批量插入 chunks，MySQL 单次 INSERT + 单次 SELECT 取 ID + 一次 commit。
    返回插入后的 chunk_id 列表，顺序与输入顺序一致。
    仅支持 document_id + chunk_index 唯一约束的情况（index_document 场景）。
    """
    if not chunks:
        return []

    embedding_model = getattr(config, "OLLAMA_EMBED_MODEL", None)

    with get_cursor(commit=True) as (_, cursor):
        text_column = _chunk_text_column(cursor)

        # ---------------------------------------------------------------
        # 从第一条数据确定列结构（假设所有 chunk 结构一致）
        # ---------------------------------------------------------------
        first = chunks[0]
        columns: list[str] = ["document_id", text_column]
        json_fields: set[str] = {"embedding", "section_path", "metadata_json"}

        def add_col(name: str, value: Any, json_dump: bool = False) -> None:
            if _column_exists(cursor, "document_chunks", name):
                columns.append(name)
                if name in json_fields:
                    json_cols.add(name)
                if name == "metadata_json":
                    json_cols.add("metadata_json")

        json_cols: set[str] = set()
        add_col("embedding", first.get("embedding"), json_dump=True)
        add_col("chunk_index", first.get("chunk_index"))
        add_col("section_path", first.get("section_path"), json_dump=True)
        add_col("page_start", first.get("page_start"))
        add_col("page_end", first.get("page_end"))
        add_col("block_start_index", first.get("block_start_index"))
        add_col("block_end_index", first.get("block_end_index"))
        add_col("chunk_type", first.get("chunk_type"))
        add_col("metadata_json", first.get("metadata_json"), json_dump=True)
        add_col("search_text", first.get("search_text"))
        add_col("lexical_text", first.get("lexical_text"))
        add_col("title", first.get("title"))
        add_col("section_title", first.get("section_title"))
        add_col("token_count", first.get("token_count"))
        add_col("chunk_hash", first.get("chunk_hash"))
        add_col("embedding_model", embedding_model)

        placeholders = ", ".join(["%s"] * len(columns))
        column_sql = ", ".join(columns)

        # ---------------------------------------------------------------
        # 构建批量 values
        # ---------------------------------------------------------------
        rows_values: list[list[Any]] = []
        chunk_indices: list[tuple[int, int | None]] = []  # (document_id, chunk_index)

        for chunk in chunks:
            values: list[Any] = [document_id, chunk.get("chunk_text") or ""]
            for col in columns[2:]:  # skip document_id and text column
                val = chunk.get(col)
                if col in json_cols:
                    val = safe_json_dumps(val) if val is not None else None
                values.append(val)
            rows_values.append(values)
            chunk_indices.append((document_id, chunk.get("chunk_index")))

        # ---------------------------------------------------------------
        # INSERT（带 ON DUPLICATE KEY UPDATE 以支持重建）
        # ---------------------------------------------------------------
        update_columns = [c for c in columns if c not in {"document_id", "chunk_index"}]
        update_sql = ", ".join(f"{c} = VALUES({c})" for c in update_columns)
        if _column_exists(cursor, "document_chunks", "updated_at"):
            update_sql += ", updated_at = CURRENT_TIMESTAMP"

        cursor.execute(
            f"""
            INSERT INTO document_chunks ({column_sql})
            VALUES {", ".join(f"({placeholders})" for _ in rows_values)}
            ON DUPLICATE KEY UPDATE {update_sql}
            """,
            [v for row in rows_values for v in row],
        )

        # ---------------------------------------------------------------
        # 批量取回 inserted / updated ID
        # ---------------------------------------------------------------
        if not chunk_indices:
            return []

        non_null = [(did, ci) for did, ci in chunk_indices if ci is not None]
        null_pairs = [(did,) for did, ci in chunk_indices if ci is None]

        # MySQL 支持 (a, b) IN ((1, 2), (3, 4)) 语法
        conditions: list[str] = []
        values: list[Any] = []

        if non_null:
            in_clause = ", ".join("(%s, %s)" for _ in non_null)
            conditions.append(f"(document_id, chunk_index) IN ({in_clause})")
            values.extend(v for pair in non_null for v in pair)

        if null_pairs:
            in_clause = ", ".join("(%s)" for _ in null_pairs)
            conditions.append(f"(document_id, chunk_index) IN ({in_clause}) AND chunk_index IS NULL")
            values.extend(v for pair in null_pairs for v in pair)

        cursor.execute(
            f"""
            SELECT id, document_id, chunk_index
            FROM document_chunks
            WHERE {' OR '.join(conditions)}
            """,
            values,
        )

        id_map: dict[tuple[int, int | None], int] = {}
        for row in cursor.fetchall():
            id_map[(int(row["document_id"]), row["chunk_index"])] = int(row["id"])

        return [id_map.get(pair, 0) for pair in chunk_indices]


def _select_chunk_columns_sql(cursor) -> str:
    text_column = _chunk_text_column(cursor)

    if text_column == "content":
        return """
            SELECT
                id,
                document_id,
                chunk_index,
                content,
                content AS chunk_text,
                search_text,
                lexical_text,
                embedding,
                chunk_type,
                title,
                section_title,
                section_path,
                page_start,
                page_end,
                block_start_index,
                block_end_index,
                token_count,
                chunk_hash,
                metadata_json,
                embedding_model,
                created_at,
                updated_at
            FROM document_chunks
        """

    return """
        SELECT
            id,
            document_id,
            chunk_index,
            chunk_text,
            chunk_text AS content,
            search_text,
            lexical_text,
            embedding,
            chunk_type,
            title,
            section_title,
            section_path,
            page_start,
            page_end,
            block_start_index,
            block_end_index,
            token_count,
            chunk_hash,
            metadata_json,
            embedding_model,
            created_at,
            updated_at
        FROM document_chunks
    """


def get_chunk_by_id(chunk_id: int) -> dict[str, Any] | None:
    with get_cursor() as (_, cursor):
        select_sql = _select_chunk_columns_sql(cursor) + " WHERE id = %s LIMIT 1"
        cursor.execute(select_sql, (chunk_id,))
        row = cursor.fetchone()
        return _normalize_chunk_row(row)


def get_chunks_by_ids(chunk_ids: Iterable[int]) -> list[dict[str, Any]]:
    ids = [int(chunk_id) for chunk_id in chunk_ids if chunk_id is not None]
    if not ids:
        return []

    seen: set[int] = set()
    unique_ids: list[int] = []
    for chunk_id in ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique_ids.append(chunk_id)

    with get_cursor() as (_, cursor):
        placeholders = ", ".join(["%s"] * len(unique_ids))
        select_sql = _select_chunk_columns_sql(cursor) + f" WHERE id IN ({placeholders})"
        cursor.execute(select_sql, unique_ids)
        rows = cursor.fetchall() or []

    row_map = {int(row["id"]): _normalize_chunk_row(row) for row in rows}
    return [row_map[chunk_id] for chunk_id in unique_ids if chunk_id in row_map]


def get_chunks_by_document_id(document_id: int) -> list[dict[str, Any]]:
    with get_cursor() as (_, cursor):
        select_sql = _select_chunk_columns_sql(cursor) + """
            WHERE document_id = %s
            ORDER BY chunk_index ASC, id ASC
        """
        cursor.execute(select_sql, (document_id,))
        rows = cursor.fetchall() or []
        return _normalize_chunk_rows(rows)


def get_all_chunks() -> list[dict[str, Any]]:
    with get_cursor() as (_, cursor):
        select_sql = _select_chunk_columns_sql(cursor) + """
            ORDER BY document_id ASC, chunk_index ASC, id ASC
        """
        cursor.execute(select_sql)
        rows = cursor.fetchall() or []
        return _normalize_chunk_rows(rows)


def search_chunks_fulltext(
    query_text: str,
    limit: int = 100,
    *,
    document_id: int | None = None,
) -> list[dict[str, Any]]:
    query_text = (query_text or "").strip()
    if not query_text:
        return []

    limit = max(1, min(int(limit), 1000))

    with get_cursor() as (_, cursor):
        where_clauses = [
            "MATCH(lexical_text, search_text, title, section_title) AGAINST (%s IN NATURAL LANGUAGE MODE)"
        ]
        params: list[Any] = [query_text]

        if document_id is not None:
            where_clauses.append("document_id = %s")
            params.append(document_id)

        params.extend([query_text, limit])

        cursor.execute(
            f"""
            SELECT
                id,
                document_id,
                chunk_index,
                title,
                section_title,
                section_path,
                chunk_type,
                page_start,
                page_end,
                token_count,
                MATCH(lexical_text, search_text, title, section_title)
                    AGAINST (%s IN NATURAL LANGUAGE MODE) AS lexical_score
            FROM document_chunks
            WHERE {" AND ".join(where_clauses)}
            ORDER BY lexical_score DESC, id DESC
            LIMIT %s
            """,
            params,
        )
        rows = cursor.fetchall() or []

    return normalize_rows_json_fields(rows, ("section_path",))


def search_chunks_boolean(
    query_text: str,
    limit: int = 100,
    *,
    document_id: int | None = None,
    require_all_terms: bool = False,
) -> list[dict[str, Any]]:
    boolean_query = _build_boolean_query(query_text, require_all=require_all_terms)
    if not boolean_query:
        return []

    limit = max(1, min(int(limit), 1000))

    with get_cursor() as (_, cursor):
        where_clauses = [
            "MATCH(lexical_text, search_text, title, section_title) AGAINST (%s IN BOOLEAN MODE)"
        ]
        params: list[Any] = [boolean_query]

        if document_id is not None:
            where_clauses.append("document_id = %s")
            params.append(document_id)

        params.extend([boolean_query, limit])

        cursor.execute(
            f"""
            SELECT
                id,
                document_id,
                chunk_index,
                title,
                section_title,
                section_path,
                chunk_type,
                page_start,
                page_end,
                token_count,
                MATCH(lexical_text, search_text, title, section_title)
                    AGAINST (%s IN BOOLEAN MODE) AS lexical_score
            FROM document_chunks
            WHERE {" AND ".join(where_clauses)}
            ORDER BY lexical_score DESC, id DESC
            LIMIT %s
            """,
            params,
        )
        rows = cursor.fetchall() or []

    return normalize_rows_json_fields(rows, ("section_path",))


def get_neighbor_chunks(
    document_id: int,
    center_chunk_indexes: Iterable[int],
    window: int = 1,
) -> list[dict[str, Any]]:
    indexes = sorted({int(idx) for idx in center_chunk_indexes if idx is not None})
    if not indexes:
        return []

    window = max(0, int(window))

    ranges: list[tuple[int, int]] = []
    for idx in indexes:
        start = max(0, idx - window)
        end = idx + window
        ranges.append((start, end))

    # 合并重叠区间，减少 OR 子句数量
    merged_ranges: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if not merged_ranges or start > merged_ranges[-1][1] + 1:
            merged_ranges.append((start, end))
        else:
            prev_start, prev_end = merged_ranges[-1]
            merged_ranges[-1] = (prev_start, max(prev_end, end))

    with get_cursor() as (_, cursor):
        range_sql = " OR ".join("(chunk_index BETWEEN %s AND %s)" for _ in merged_ranges)
        params: list[Any] = [document_id]
        for start, end in merged_ranges:
            params.extend([start, end])

        select_sql = _select_chunk_columns_sql(cursor) + f"""
            WHERE document_id = %s
              AND ({range_sql})
            ORDER BY chunk_index ASC, id ASC
        """
        cursor.execute(select_sql, params)
        rows = cursor.fetchall() or []

    return _normalize_chunk_rows(rows)


def clear_chunks_by_document_id(document_id: int) -> int:
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute(
            """
            DELETE FROM document_chunks
            WHERE document_id = %s
            """,
            (document_id,),
        )
        return int(cursor.rowcount)


def clear_all_chunks() -> int:
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute("DELETE FROM document_chunks")
        return int(cursor.rowcount)


def delete_chunks_by_ids(chunk_ids: list[int]) -> int:
    """Delete specific chunks by their IDs."""
    if not chunk_ids:
        return 0
    int_ids = [int(cid) for cid in chunk_ids if cid]
    if not int_ids:
        return 0
    placeholders = ", ".join(["%s"] * len(int_ids))
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute(
            f"DELETE FROM document_chunks WHERE id IN ({placeholders})",
            int_ids,
        )
        return int(cursor.rowcount)