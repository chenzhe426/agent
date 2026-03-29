import hashlib
import json
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from threading import Lock
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import app.config as config
from app.services.common import normalize_whitespace


OLLAMA_BASE_URL = config.OLLAMA_BASE_URL
OLLAMA_MODEL = config.OLLAMA_MODEL
OLLAMA_EMBED_MODEL = config.OLLAMA_EMBED_MODEL
REQUEST_TIMEOUT = int(config.OLLAMA_TIMEOUT)
EMBEDDING_BATCH_SIZE = int(config.EMBEDDING_BATCH_SIZE)
EMBEDDING_MAX_RETRIES = int(config.EMBEDDING_MAX_RETRIES)
_EMBEDDING_CACHE_MAX_SIZE = int(config.EMBEDDING_CACHE_SIZE)

# Reuse HTTP session with connection pooling for better performance
_session_lock = Lock()
_http_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Get or create a reusable HTTP session with connection pooling."""
    global _http_session
    if _http_session is None:
        with _session_lock:
            if _http_session is None:
                session = requests.Session()
                # Retry adapter for transient errors
                retry_config = Retry(
                    total=2,
                    backoff_factor=0.1,
                    status_forcelist=[502, 503, 504],
                    allowed_methods=["POST"],
                )
                adapter = HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=20,
                    max_retries=retry_config,
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                _http_session = session
    return _http_session


# =============================================================================
# Bounded LRU Cache for embeddings
# =============================================================================

_EMBEDDING_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_EMBEDDING_CACHE_LOCK = Lock()


def _get_cached_embedding(key: str) -> Optional[list[float]]:
    with _EMBEDDING_CACHE_LOCK:
        if key in _EMBEDDING_CACHE:
            _EMBEDDING_CACHE.move_to_end(key)
            return _EMBEDDING_CACHE[key]
    return None


def _set_cached_embedding(key: str, value: list[float]) -> None:
    with _EMBEDDING_CACHE_LOCK:
        if key in _EMBEDDING_CACHE:
            _EMBEDDING_CACHE.move_to_end(key)
        else:
            if len(_EMBEDDING_CACHE) >= _EMBEDDING_CACHE_MAX_SIZE:
                _EMBEDDING_CACHE.popitem(last=False)
        _EMBEDDING_CACHE[key] = value


def _embedding_cache_key(text: str) -> str:
    """Hash-based cache key: normalized text -> md5 hex"""
    normalized = normalize_whitespace(text or "")
    if not normalized:
        return ""
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout: int | None = None,
) -> dict[str, Any]:
    session = _get_session()
    resp = session.post(url, json=payload, timeout=timeout or REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("invalid json response")
    return data


# =============================================================================
# Embedding API calls
# =============================================================================

def _call_embedding_api_single(text: str) -> list[float] | None:
    """Call Ollama single-text embedding API."""
    base_url = OLLAMA_BASE_URL.rstrip("/")

    # Try /api/embeddings (OpenAI-compatible)
    try:
        data = _post_json(
            f"{base_url}/api/embeddings",
            {"model": OLLAMA_EMBED_MODEL, "input": text},
        )
        embedding = data.get("embedding")
        if isinstance(embedding, list) and embedding:
            return embedding
    except Exception:
        pass

    # Try /api/embed (legacy Ollama)
    try:
        data = _post_json(
            f"{base_url}/api/embed",
            {"model": OLLAMA_EMBED_MODEL, "input": text},
        )
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            return embeddings[0]
        embedding = data.get("embedding")
        if isinstance(embedding, list):
            return embedding
    except Exception:
        pass

    return None


def _call_embedding_api_batch(texts: list[str]) -> list[list[float] | None] | None:
    """Call Ollama batch embedding API.

    Returns list of embeddings (None for failed items) on success,
    or None if batch API is not supported.
    """
    base_url = OLLAMA_BASE_URL.rstrip("/")

    # Try /api/embeddings with "inputs" (OpenAI-compatible batch)
    try:
        data = _post_json(
            f"{base_url}/api/embeddings",
            {"model": OLLAMA_EMBED_MODEL, "inputs": texts},
        )
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            return embeddings
    except Exception:
        pass

    # Try /api/embed with "inputs" array
    try:
        data = _post_json(
            f"{base_url}/api/embed",
            {"model": OLLAMA_EMBED_MODEL, "inputs": texts},
        )
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            return embeddings
    except Exception:
        pass

    return None


# =============================================================================
# Public Embedding API
# =============================================================================

@lru_cache(maxsize=4096)
def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text. Cached via lru_cache + LRU dict."""
    text = normalize_whitespace(text or "")
    if not text:
        return []

    cache_key = _embedding_cache_key(text)
    if cache_key:
        cached = _get_cached_embedding(cache_key)
        if cached is not None:
            return cached

    for attempt in range(EMBEDDING_MAX_RETRIES):
        embedding = _call_embedding_api_single(text)
        if embedding is not None:
            if cache_key:
                _set_cached_embedding(cache_key, embedding)
            return embedding
        if attempt < EMBEDDING_MAX_RETRIES - 1:
            time.sleep(0.5 * (attempt + 1))

    return []


def get_embeddings_batch(texts: list[str], max_workers: int = 4) -> list[list[float]]:
    """
    Batch embedding: try Ollama batch API first, fall back to threaded individual calls.
    Returns embeddings in same order as input texts.
    Empty/whitespace-only texts return [].
    """
    if not texts:
        return []

    # Separate empty from non-empty
    non_empty: list[tuple[int, str]] = []  # (original_idx, normalized_text)
    for i, t in enumerate(texts):
        normalized = normalize_whitespace(t)
        if normalized:
            non_empty.append((i, normalized))

    if not non_empty:
        return [[] for _ in texts]

    # Split into cached vs. to-embed
    cached: dict[int, list[float]] = {}
    to_embed: list[tuple[int, str]] = []

    for orig_idx, text in non_empty:
        key = _embedding_cache_key(text)
        if key:
            hit = _get_cached_embedding(key)
            if hit is not None:
                cached[orig_idx] = hit
                continue
        to_embed.append((orig_idx, text))

    # Embed uncached texts
    embedded: dict[int, list[float]] = {}
    if to_embed:
        texts_to_call = [t for _, t in to_embed]
        batch_results = _call_embedding_api_batch(texts_to_call)

        if batch_results is not None:
            # Batch API succeeded
            for (orig_idx, text), emb in zip(to_embed, batch_results):
                if isinstance(emb, list) and emb:
                    embedded[orig_idx] = emb
                    key = _embedding_cache_key(text)
                    if key:
                        _set_cached_embedding(key, emb)
                else:
                    # Individual item failed - fall back inline
                    embedded[orig_idx] = get_embedding(text)
        else:
            # No batch API - threaded individual calls
            def worker(orig_idx: int, text: str) -> tuple[int, list[float]]:
                return orig_idx, get_embedding(text)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(worker, oi, t): oi
                    for oi, t in to_embed
                }
                for future in as_completed(futures):
                    try:
                        oi, emb = future.result()
                        embedded[oi] = emb
                    except Exception:
                        pass

    # Merge results in original order
    all_results = {**cached, **embedded}
    return [all_results.get(i, []) for i in range(len(texts))]


def get_embeddings_batch_api(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """Alias for get_embeddings_batch. batch_size kept for compat."""
    return get_embeddings_batch(texts)


# =============================================================================
# Chat completion
# =============================================================================

def _extract_chat_content(data: dict[str, Any]) -> str:
    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    response = data.get("response")
    if isinstance(response, str) and response.strip():
        return response.strip()
    return ""


def _extract_generate_content(data: dict[str, Any]) -> str:
    response = data.get("response")
    if isinstance(response, str) and response.strip():
        return response.strip()

    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def chat_completion(system_prompt: str, user_prompt: str, timeout: int | None = None) -> str:
    system_text = normalize_whitespace(system_prompt or "")
    user_text = normalize_whitespace(user_prompt or "")

    if not system_text and not user_text:
        return ""

    base_url = OLLAMA_BASE_URL.rstrip("/")
    messages: list[dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    if user_text:
        messages.append({"role": "user", "content": user_text})

    try:
        data = _post_json(
            f"{base_url}/api/chat",
            {"model": OLLAMA_MODEL, "stream": False, "messages": messages, "think": False},
            timeout=timeout,
        )
        content = _extract_chat_content(data)
        if content:
            return content
    except Exception:
        pass

    # Fallback /api/generate
    prompt_parts: list[str] = []
    if system_text:
        prompt_parts.append(f"系统指令：\n{system_text}")
    if user_text:
        prompt_parts.append(f"用户问题：\n{user_text}")
    prompt = "\n\n".join(prompt_parts).strip()

    try:
        data = _post_json(
            f"{base_url}/api/generate",
            {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "think": False},
            timeout=timeout,
        )
        content = _extract_generate_content(data)
        if content:
            return content
        return ""
    except Exception as e:
        raise RuntimeError(f"llm request failed: {e}") from e


def summarize_text(text: str) -> str:
    text = normalize_whitespace(text or "")
    if not text:
        return "没有可摘要的内容。"

    system_prompt = (
        "你是一个擅长阅读文档并生成摘要的助手。"
        "请基于给定内容输出中文摘要，尽量准确、简洁、结构清晰。"
    )
    user_prompt = (
        "请总结下面这段内容，输出：\n"
        "1. 一段总体摘要\n"
        "2. 3到5个关键点\n\n"
        f"内容：\n{text}"
    )
    return chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)


def chat_completion_json(
    system_prompt: str,
    user_prompt: str,
    default: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if default is None:
        default = {}

    try:
        raw = chat_completion(system_prompt, user_prompt)
        if not raw:
            return default

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        brace_match = re.search(r"(\{.*\})", raw, re.S)
        if brace_match:
            try:
                return json.loads(brace_match.group(1))
            except json.JSONDecodeError:
                pass

        return default
    except Exception:
        return default


def chat_completion_raw(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int | None = None,
) -> str:
    system_text = normalize_whitespace(system_prompt or "")
    user_text = normalize_whitespace(user_prompt or "")

    if not system_text and not user_text:
        return ""

    base_url = OLLAMA_BASE_URL.rstrip("/")
    model_name = model or OLLAMA_MODEL

    messages: list[dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    if user_text:
        messages.append({"role": "user", "content": user_text})

    payload: dict[str, Any] = {
        "model": model_name,
        "stream": False,
        "messages": messages,
    }
    if temperature > 0:
        payload["temperature"] = temperature

    try:
        data = _post_json(f"{base_url}/api/chat", payload, timeout=timeout)
        content = _extract_chat_content(data)
        if content:
            return content
    except Exception:
        pass

    prompt_parts: list[str] = []
    if system_text:
        prompt_parts.append(f"系统指令：\n{system_text}")
    if user_text:
        prompt_parts.append(f"用户问题：\n{user_text}")
    prompt = "\n\n".join(prompt_parts).strip()

    fallback_payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }
    if temperature > 0:
        fallback_payload["temperature"] = temperature

    data = _post_json(f"{base_url}/api/generate", fallback_payload, timeout=timeout)
    content = _extract_generate_content(data)
    if content:
        return content

    raise RuntimeError("chat_completion_raw: all backends failed")
