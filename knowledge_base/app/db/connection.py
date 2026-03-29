from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pymysql
from pymysql.cursors import DictCursor

import app.config as config


def get_connection():
    return pymysql.connect(
        host=getattr(config, "MYSQL_HOST", "127.0.0.1"),
        port=int(getattr(config, "MYSQL_PORT", 3306)),
        user=getattr(config, "MYSQL_USER", "root"),
        password=getattr(config, "MYSQL_PASSWORD", ""),
        database=getattr(config, "MYSQL_DATABASE", "knowledge_base"),
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
    )


@contextmanager
def get_cursor(commit: bool = False) -> Iterator[tuple[pymysql.connections.Connection, DictCursor]]:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        yield conn, cursor
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()