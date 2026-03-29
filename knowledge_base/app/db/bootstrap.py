from __future__ import annotations

from app.db.connection import get_cursor
from app.db.migrations import run_migrations
from app.db.schema import init_schema


def init_db() -> None:
    with get_cursor(commit=True) as (_, cursor):
        init_schema(cursor)
        run_migrations(cursor)


def reset_database(keep_schema: bool = True) -> None:
    """
    重置当前数据库。

    keep_schema=True:
        保留表结构，仅清空所有表数据（TRUNCATE）

    keep_schema=False:
        删除所有表后重新执行 init_db() 建表
    """
    with get_cursor(commit=True) as (_, cursor):
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")

        cursor.execute("SHOW TABLES")
        rows = cursor.fetchall() or []

        table_names: list[str] = []
        for row in rows:
            table_names.extend(str(v) for v in row.values())

        for table_name in table_names:
            if keep_schema:
                cursor.execute(f"TRUNCATE TABLE `{table_name}`")
            else:
                cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")

        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

    if not keep_schema:
        init_db()