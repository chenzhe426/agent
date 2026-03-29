import json
import re
from typing import Any


WHITESPACE_RE = re.compile(r"\s+")


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


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    if hasattr(obj, key):
        try:
            return getattr(obj, key)
        except Exception:
            return default

    return default


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


def to_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def normalize_section_path(section_path: Any) -> list[str]:
    if section_path is None:
        return []

    if isinstance(section_path, list):
        result = []
        for item in section_path:
            text = normalize_whitespace(item)
            if text:
                result.append(text)
        return result

    if isinstance(section_path, tuple):
        result = []
        for item in section_path:
            text = normalize_whitespace(item)
            if text:
                result.append(text)
        return result

    if isinstance(section_path, str):
        text = section_path.strip()
        if not text:
            return []

        # 优先尝试把 JSON list 字符串转回来
        parsed = safe_json_loads(text, default=None)
        if isinstance(parsed, list):
            return normalize_section_path(parsed)

        # 兼容 "A > B > C"
        if ">" in text:
            parts = [normalize_whitespace(x) for x in text.split(">")]
            return [x for x in parts if x]

        # 兼容 "A/B/C"
        if "/" in text:
            parts = [normalize_whitespace(x) for x in text.split("/")]
            return [x for x in parts if x]

        text = normalize_whitespace(text)
        return [text] if text else []

    text = normalize_whitespace(section_path)
    return [text] if text else []


def section_path_to_str(section_path: Any) -> str:
    parts = normalize_section_path(section_path)
    return " > ".join(parts) if parts else ""


def last_section_title(section_path: Any) -> str:
    parts = normalize_section_path(section_path)
    return parts[-1] if parts else ""


def normalize_embedding(value: Any) -> list[float]:
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