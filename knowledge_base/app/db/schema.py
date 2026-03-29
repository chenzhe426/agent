from __future__ import annotations


def init_schema(cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            title VARCHAR(255) NOT NULL,
            content LONGTEXT NULL,
            raw_text LONGTEXT NULL,
            summary TEXT NULL,
            source VARCHAR(1024) NULL,
            source_type VARCHAR(64) NULL,
            file_path VARCHAR(1024) NULL,
            file_type VARCHAR(128) NULL,
            mime_type VARCHAR(255) NULL,
            lang VARCHAR(64) NULL,
            author VARCHAR(255) NULL,
            published_at DATETIME NULL,
            content_hash VARCHAR(64) NULL,
            block_count INT NOT NULL DEFAULT 0,
            blocks_json JSON NULL,
            metadata_json JSON NULL,
            tags_json JSON NULL,
            status VARCHAR(64) NOT NULL DEFAULT 'active',
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_documents_title (title),
            INDEX idx_documents_status (status),
            INDEX idx_documents_source_type (source_type),
            INDEX idx_documents_file_type (file_type),
            INDEX idx_documents_content_hash (content_hash)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            document_id BIGINT NOT NULL,
            chunk_index INT NOT NULL,
            chunk_text LONGTEXT NOT NULL,
            search_text LONGTEXT NULL,
            lexical_text LONGTEXT NULL,
            embedding LONGTEXT NULL,
            chunk_type VARCHAR(64) NULL,
            title VARCHAR(255) NULL,
            section_title VARCHAR(255) NULL,
            section_path JSON NULL,
            page_start INT NULL,
            page_end INT NULL,
            block_start_index INT NULL,
            block_end_index INT NULL,
            token_count INT NULL,
            chunk_hash VARCHAR(64) NULL,
            metadata_json JSON NULL,
            embedding_model VARCHAR(255) NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uk_document_chunk (document_id, chunk_index),
            INDEX idx_chunks_document_id (document_id),
            INDEX idx_chunks_document_chunk (document_id, chunk_index),
            INDEX idx_chunks_chunk_hash (chunk_hash),
            CONSTRAINT fk_chunks_document FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            FULLTEXT KEY ft_chunks_lexical (lexical_text, search_text, title, section_title)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id VARCHAR(128) PRIMARY KEY,
            title VARCHAR(255) NULL,
            user_id VARCHAR(128) NULL,
            metadata JSON NULL,
            last_message_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_chat_sessions_user_id (user_id),
            INDEX idx_chat_sessions_last_message_at (last_message_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            session_id VARCHAR(128) NOT NULL,
            role VARCHAR(32) NOT NULL,
            message LONGTEXT NOT NULL,
            citations JSON NULL,
            metadata JSON NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_chat_messages_session_id (session_id),
            INDEX idx_chat_messages_created_at (created_at),
            CONSTRAINT fk_chat_messages_session FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(128) PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_entities (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            session_id VARCHAR(128) NOT NULL,
            entity_type VARCHAR(64) NOT NULL COMMENT 'document|concept|metric|person|company|etc',
            entity_key VARCHAR(512) NOT NULL COMMENT 'normalized entity name/identifier',
            display_name VARCHAR(512) NULL,
            reference_count INT NOT NULL DEFAULT 1,
            first_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            metadata JSON NULL,
            INDEX idx_agent_entities_session (session_id),
            INDEX idx_agent_entities_type_key (entity_type, entity_key),
            INDEX idx_agent_entities_reference (reference_count DESC)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_tool_chains (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            question_hash VARCHAR(64) NOT NULL COMMENT 'SHA256 of normalized question, first 16 chars',
            question_text TEXT NOT NULL COMMENT 'Full question text for reference',
            tool_sequence JSON NOT NULL COMMENT '[{tool, args}, ...] - sequence of tools that succeeded',
            session_id VARCHAR(128) NULL,
            success_count INT NOT NULL DEFAULT 1 COMMENT 'Number of times this chain succeeded',
            last_used TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uk_tool_chains_question_hash (question_hash),
            INDEX idx_tool_chains_success (success_count DESC),
            INDEX idx_tool_chains_last_used (last_used)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )