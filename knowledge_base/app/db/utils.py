from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable


def _json_default(value: Any):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def safe_json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def safe_json_loads(value: Any, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        return fallback

    text = value.strip()
    if not text:
        return fallback

    try:
        return json.loads(text)
    except Exception:
        return fallback


def normalize_row_json_fields(row: dict[str, Any] | None, json_fields: Iterable[str]) -> dict[str, Any] | None:
    if row is None:
        return None

    normalized = dict(row)
    for field in json_fields:
        if field in normalized:
            normalized[field] = safe_json_loads(normalized[field], fallback=normalized[field])
    return normalized


def normalize_rows_json_fields(rows: list[dict[str, Any]], json_fields: Iterable[str]) -> list[dict[str, Any]]:
    return [normalize_row_json_fields(row, json_fields) for row in rows]