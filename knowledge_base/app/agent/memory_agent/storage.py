"""
memory_agent/storage.py - 记忆存储层

提供三层记忆的存储接口：
- 短期记忆：内存中的结构化摘要
- 中期记忆：Redis (TTL=30min)
- 长期记忆：MySQL
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional
from datetime import datetime

import redis

import app.config as config
from app.db.connection import get_cursor


# =============================================================================
# 短期记忆 - 内存存储
# =============================================================================

class ShortTermStorage:
    """
    短期记忆：存储在内存中，保留最近N轮对话的结构化摘要。
    使用类变量在进程内共享。
    """

    # 类级别存储: session_id -> {turns: [], summary: {...}, updated_at: float}
    _storage: dict[str, dict[str, Any]] = {}
    _lock_key = "_lock"

    # 配置 - 从config读取
    @property
    def RECENT_TURNS(cls):
        return getattr(config, 'MEMORY_SHORT_TERM_TURNS', 3)

    @classmethod
    def store(cls, session_id: str, turns: list[dict[str, Any]], summary: dict[str, Any]) -> None:
        """存储短期记忆"""
        cls._storage[session_id] = {
            "turns": turns[-cls.RECENT_TURNS:] if len(turns) > cls.RECENT_TURNS else turns,
            "summary": summary,
            "updated_at": time.time(),
        }

    @classmethod
    def get(cls, session_id: str) -> Optional[dict[str, Any]]:
        """获取短期记忆"""
        return cls._storage.get(session_id)

    @classmethod
    def append_turn(cls, session_id: str, role: str, message: str) -> list[dict[str, Any]]:
        """追加一轮对话，返回当前保留的轮次"""
        if session_id not in cls._storage:
            cls._storage[session_id] = {
                "turns": [],
                "summary": {"entities": [], "topics": [], "qa_pairs": [], "coreferences": []},
                "updated_at": time.time(),
            }

        storage = cls._storage[session_id]
        storage["turns"].append({
            "role": role,
            "message": message,
            "timestamp": time.time(),
        })

        # 只保留最近3轮
        if len(storage["turns"]) > cls.RECENT_TURNS:
            storage["turns"] = storage["turns"][-cls.RECENT_TURNS:]

        storage["updated_at"] = time.time()
        return storage["turns"]

    @classmethod
    def update_summary(cls, session_id: str, summary: dict[str, Any]) -> None:
        """更新结构化摘要"""
        if session_id in cls._storage:
            cls._storage[session_id]["summary"] = summary
            cls._storage[session_id]["updated_at"] = time.time()

    @classmethod
    def clear(cls, session_id: str) -> None:
        """清除短期记忆"""
        if session_id in cls._storage:
            del cls._storage[session_id]


# =============================================================================
# 中期记忆 - Redis存储
# =============================================================================

class MidTermStorage:
    """
    中期记忆：存储在Redis中，TTL=30分钟。
    存储第4-10轮对话的摘要信息。
    """

    # 配置 - 从config读取
    @property
    def TTL_SECONDS(cls):
        return getattr(config, 'MEMORY_MID_TERM_TTL', 30 * 60)

    KEY_PREFIX = "kb:mid_term:"

    @property
    def TURN_THRESHOLD(cls):
        return getattr(config, 'MEMORY_MID_TERM_START_TURN', 3)

    _redis_client: Optional[redis.Redis] = None

    @classmethod
    def _get_client(cls) -> redis.Redis:
        """获取Redis客户端（懒加载）"""
        if cls._redis_client is None:
            cls._redis_client = redis.Redis(
                host=getattr(config, 'REDIS_HOST', 'localhost'),
                port=getattr(config, 'REDIS_PORT', 6379),
                db=getattr(config, 'REDIS_DB', 0),
                password=getattr(config, 'REDIS_PASSWORD', None),
                decode_responses=True,
            )
        return cls._redis_client

    @classmethod
    def _key(cls, session_id: str) -> str:
        return f"{cls.KEY_PREFIX}{session_id}"

    @classmethod
    def store(cls, session_id: str, turn_range: tuple[int, int], summary: str,
              key_entities: list[str], key_topics: list[str]) -> bool:
        """存储中期记忆到Redis"""
        try:
            client = cls._get_client()
            data = {
                "session_id": session_id,
                "turn_range": turn_range,
                "summary": summary,
                "key_entities": json.dumps(key_entities),
                "key_topics": json.dumps(key_topics),
                "created_at": datetime.now().isoformat(),
            }
            client.hset(cls._key(session_id), mapping=data)
            client.expire(cls._key(session_id), cls.TTL_SECONDS)
            return True
        except Exception as e:
            print(f"[MidTermStorage] store error: {e}")
            return False

    @classmethod
    def get(cls, session_id: str) -> Optional[dict[str, Any]]:
        """获取中期记忆"""
        try:
            client = cls._get_client()
            data = client.hgetall(cls._key(session_id))
            if not data:
                return None
            return {
                "session_id": data.get("session_id"),
                "turn_range": json.loads(data.get("turn_range", "[0, 0]")),
                "summary": data.get("summary", ""),
                "key_entities": json.loads(data.get("key_entities", "[]")),
                "key_topics": json.loads(data.get("key_topics", "[]")),
                "created_at": data.get("created_at"),
            }
        except Exception as e:
            print(f"[MidTermStorage] get error: {e}")
            return None

    @classmethod
    def delete(cls, session_id: str) -> bool:
        """删除中期记忆"""
        try:
            client = cls._get_client()
            client.delete(cls._key(session_id))
            return True
        except Exception as e:
            print(f"[MidTermStorage] delete error: {e}")
            return False

    @classmethod
    def exists(cls, session_id: str) -> bool:
        """检查中期记忆是否存在"""
        try:
            client = cls._get_client()
            return client.exists(cls._key(session_id)) > 0
        except Exception:
            return False


# =============================================================================
# 长期记忆 - MySQL存储
# =============================================================================

class LongTermStorage:
    """
    长期记忆：存储在MySQL中，跨session可查询。
    存储>10轮对话的摘要信息。
    """

    KEY_PREFIX = "kb:long_term:"

    @classmethod
    def _ensure_table(cls) -> None:
        """确保长期记忆表存在"""
        with get_cursor(commit=True) as (_, cursor):
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_long_term_memory (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    summary TEXT NOT NULL,
                    key_entities JSON,
                    key_topics JSON,
                    question_hint VARCHAR(500),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_session_id (session_id),
                    INDEX idx_question_hint (question_hint(100)),
                    UNIQUE KEY uk_session_id (session_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

    @classmethod
    def store(cls, session_id: str, summary: str,
              key_entities: list[str], key_topics: list[str],
              question_hint: str = "") -> bool:
        """存储长期记忆到MySQL"""
        cls._ensure_table()
        try:
            with get_cursor(commit=True) as (_, cursor):
                cursor.execute("""
                    INSERT INTO agent_long_term_memory
                    (session_id, summary, key_entities, key_topics, question_hint)
                    VALUES (%s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        summary = VALUES(summary),
                        key_entities = VALUES(key_entities),
                        key_topics = VALUES(key_topics),
                        question_hint = VALUES(question_hint),
                        updated_at = CURRENT_TIMESTAMP
                """, (session_id, summary, json.dumps(key_entities), json.dumps(key_topics), question_hint))
                return True
        except Exception as e:
            print(f"[LongTermStorage] store error: {e}")
            return False

    @classmethod
    def get(cls, session_id: str) -> Optional[dict[str, Any]]:
        """获取指定session的长期记忆"""
        cls._ensure_table()
        try:
            with get_cursor() as (_, cursor):
                cursor.execute("""
                    SELECT session_id, summary, key_entities, key_topics, created_at, updated_at
                    FROM agent_long_term_memory
                    WHERE session_id = %s
                """, (session_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return {
                    "session_id": row[0],
                    "summary": row[1],
                    "key_entities": json.loads(row[2]) if row[2] else [],
                    "key_topics": json.loads(row[3]) if row[3] else [],
                    "created_at": row[4],
                    "updated_at": row[5],
                }
        except Exception as e:
            print(f"[LongTermStorage] get error: {e}")
            return None

    @classmethod
    def search(cls, keyword: str, limit: int = 5) -> list[dict[str, Any]]:
        """搜索相关长期记忆（基于关键词）"""
        cls._ensure_table()
        try:
            with get_cursor() as (_, cursor):
                cursor.execute("""
                    SELECT session_id, summary, key_entities, key_topics, created_at
                    FROM agent_long_term_memory
                    WHERE question_hint LIKE %s OR summary LIKE %s OR key_entities LIKE %s
                    ORDER BY updated_at DESC
                    LIMIT %s
                """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", limit))
                rows = cursor.fetchall() or []
                return [
                    {
                        "session_id": r[0],
                        "summary": r[1],
                        "key_entities": json.loads(r[2]) if r[2] else [],
                        "key_topics": json.loads(r[3]) if r[3] else [],
                        "created_at": r[4],
                    }
                    for r in rows
                ]
        except Exception as e:
            print(f"[LongTermStorage] search error: {e}")
            return []

    @classmethod
    def delete(cls, session_id: str) -> bool:
        """删除长期记忆"""
        cls._ensure_table()
        try:
            with get_cursor(commit=True) as (_, cursor):
                cursor.execute("DELETE FROM agent_long_term_memory WHERE session_id = %s", (session_id,))
                return True
        except Exception as e:
            print(f"[LongTermStorage] delete error: {e}")
            return False


# =============================================================================
# 导出单例访问接口
# =============================================================================

short_term_storage = ShortTermStorage
mid_term_storage = MidTermStorage
long_term_storage = LongTermStorage
