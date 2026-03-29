"""
Shared utilities for retrieval submodules.
Avoids importing from app.services to prevent circular import chains.
"""
from __future__ import annotations

import json
import re
from typing import Any

WHITESPACE_RE = re.compile(r"\s+")


def safe_json_loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if not isinstance(value, str):
        return default
    text = value.strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def normalize_whitespace(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def to_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning default on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get a key from an object (dict or attribute access)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        try:
            return getattr(obj, key)
        except AttributeError:
            return default
    return default


def normalize_embedding(value: Any) -> list[float]:
    """Normalize an embedding vector to a list of floats."""
    if value is None:
        return []
    if isinstance(value, list):
        result: list[float] = []
        for item in value:
            try:
                result.append(float(item))
            except Exception:
                continue
        return result
    if isinstance(value, str):
        parsed = safe_json_loads(value, default=[])
        if isinstance(parsed, list):
            return normalize_embedding(parsed)
        return []
    return []
