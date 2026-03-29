from __future__ import annotations


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


def _table_exists(cursor, table_name: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
        LIMIT 1
        """,
        (table_name,),
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


def _migration_applied(cursor, version: str) -> bool:
    cursor.execute(
        "SELECT 1 FROM schema_migrations WHERE version = %s LIMIT 1",
        (version,),
    )
    return cursor.fetchone() is not None


def _mark_migration_applied(cursor, version: str) -> None:
    cursor.execute(
        """
        INSERT INTO schema_migrations (version)
        VALUES (%s)
        ON DUPLICATE KEY UPDATE version = VALUES(version)
        """,
        (version,),
    )


def _apply_migration(cursor, version: str, fn) -> None:
    if _migration_applied(cursor, version):
        return
    fn(cursor)
    _mark_migration_applied(cursor, version)


def _migration_001_documents_extra_columns(cursor) -> None:
    if not _column_exists(cursor, "documents", "source_type"):
        cursor.execute("ALTER TABLE documents ADD COLUMN source_type VARCHAR(64) NULL")
    if not _column_exists(cursor, "documents", "file_path"):
        cursor.execute("ALTER TABLE documents ADD COLUMN file_path VARCHAR(1024) NULL")
    if not _column_exists(cursor, "documents", "mime_type"):
        cursor.execute("ALTER TABLE documents ADD COLUMN mime_type VARCHAR(255) NULL")
    if not _column_exists(cursor, "documents", "metadata"):
        cursor.execute("ALTER TABLE documents ADD COLUMN metadata JSON NULL")
    if not _column_exists(cursor, "documents", "extra_metadata"):
        cursor.execute("ALTER TABLE documents ADD COLUMN extra_metadata JSON NULL")
    if not _column_exists(cursor, "documents", "status"):
        cursor.execute("ALTER TABLE documents ADD COLUMN status VARCHAR(64) NOT NULL DEFAULT 'active'")

    if not _index_exists(cursor, "documents", "idx_documents_status"):
        cursor.execute("CREATE INDEX idx_documents_status ON documents(status)")
    if not _index_exists(cursor, "documents", "idx_documents_source_type"):
        cursor.execute("CREATE INDEX idx_documents_source_type ON documents(source_type)")


def _migration_002_chunks_extra_columns(cursor) -> None:
    if not _column_exists(cursor, "document_chunks", "token_count"):
        cursor.execute("ALTER TABLE document_chunks ADD COLUMN token_count INT NULL")
    if not _column_exists(cursor, "document_chunks", "metadata"):
        cursor.execute("ALTER TABLE document_chunks ADD COLUMN metadata JSON NULL")
    if not _column_exists(cursor, "document_chunks", "embedding_model"):
        cursor.execute("ALTER TABLE document_chunks ADD COLUMN embedding_model VARCHAR(255) NULL")
    if not _column_exists(cursor, "document_chunks", "updated_at"):
        cursor.execute(
            """
            ALTER TABLE document_chunks
            ADD COLUMN updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            ON UPDATE CURRENT_TIMESTAMP
            """
        )


def _migration_003_chat_sessions_extra_columns(cursor) -> None:
    if not _table_exists(cursor, "chat_sessions"):
        return

    if not _column_exists(cursor, "chat_sessions", "title"):
        cursor.execute("ALTER TABLE chat_sessions ADD COLUMN title VARCHAR(255) NULL")
    if not _column_exists(cursor, "chat_sessions", "user_id"):
        cursor.execute("ALTER TABLE chat_sessions ADD COLUMN user_id VARCHAR(128) NULL")
    if not _column_exists(cursor, "chat_sessions", "metadata"):
        cursor.execute("ALTER TABLE chat_sessions ADD COLUMN metadata JSON NULL")
    if not _column_exists(cursor, "chat_sessions", "last_message_at"):
        cursor.execute("ALTER TABLE chat_sessions ADD COLUMN last_message_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP")
    if not _column_exists(cursor, "chat_sessions", "updated_at"):
        cursor.execute(
            """
            ALTER TABLE chat_sessions
            ADD COLUMN updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            ON UPDATE CURRENT_TIMESTAMP
            """
        )

    if not _index_exists(cursor, "chat_sessions", "idx_chat_sessions_user_id"):
        cursor.execute("CREATE INDEX idx_chat_sessions_user_id ON chat_sessions(user_id)")
    if not _index_exists(cursor, "chat_sessions", "idx_chat_sessions_last_message_at"):
        cursor.execute("CREATE INDEX idx_chat_sessions_last_message_at ON chat_sessions(last_message_at)")


def _migration_004_chat_messages_extra_columns(cursor) -> None:
    if not _table_exists(cursor, "chat_messages"):
        return

    if not _column_exists(cursor, "chat_messages", "citations"):
        cursor.execute("ALTER TABLE chat_messages ADD COLUMN citations JSON NULL")
    if not _column_exists(cursor, "chat_messages", "metadata"):
        cursor.execute("ALTER TABLE chat_messages ADD COLUMN metadata JSON NULL")

    if not _index_exists(cursor, "chat_messages", "idx_chat_messages_created_at"):
        cursor.execute("CREATE INDEX idx_chat_messages_created_at ON chat_messages(created_at)")


def _migration_005_chunk_search_indexes(cursor) -> None:
    if not _table_exists(cursor, "document_chunks"):
        return

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


def run_migrations(cursor) -> None:
    migrations = [
        ("001_documents_extra_columns", _migration_001_documents_extra_columns),
        ("002_chunks_extra_columns", _migration_002_chunks_extra_columns),
        ("003_chat_sessions_extra_columns", _migration_003_chat_sessions_extra_columns),
        ("004_chat_messages_extra_columns", _migration_004_chat_messages_extra_columns),
        ("005_chunk_search_indexes", _migration_005_chunk_search_indexes),
    ]

    for version, fn in migrations:
        _apply_migration(cursor, version, fn)