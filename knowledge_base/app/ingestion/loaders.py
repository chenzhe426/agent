"""
Smart text file loading with robust encoding detection.

Instead of blindly reading as UTF-8 with errors='ignore' (which silently
produces garbage), we try a cascade of encodings and only fall back to
replacement when truly necessary. We also strip BOM bytes.
"""
from __future__ import annotations

import codecs
from pathlib import Path

from app.ingestion.config import TextParserConfig


# ---------------------------------------------------------------------------
# Encoding detection & loading
# ---------------------------------------------------------------------------


def load_text_file(
    file_path: str | Path,
    config: TextParserConfig | None = None,
) -> str:
    """
    Load a plain-text file with robust encoding detection.

    Strategy
    --------
    1. Try each encoding from ``config.encoding_order`` in sequence.
    2. The first one that decodes without a ``UnicodeDecodeError`` wins.
    3. On final failure, fall back to ``errors='replace'`` so we get /something/
       rather than crashing the whole pipeline.

    Parameters
    ----------
    file_path : str | Path
    config    : TextParserConfig | None
        If None, reasonable defaults are used (UTF-8 → GBK → GB18030 → Latin-1).

    Returns
    -------
    str
        The file contents, normalized (newlines, BOM stripped).
    """
    path = Path(file_path)
    raw_bytes = path.read_bytes()

    # Strip BOM
    for bom in (
        codecs.BOM_UTF8,
        codecs.BOM_UTF32_BE,
        codecs.BOM_UTF32_LE,
        codecs.BOM_UTF16_BE,
        codecs.BOM_UTF16_LE,
    ):
        if raw_bytes.startswith(bom):
            raw_bytes = raw_bytes[len(bom):]
            break

    if not raw_bytes:
        return ""

    if config is None:
        config = TextParserConfig()

    # Try each encoding in order
    for enc in config.encoding_order:
        try:
            return raw_bytes.decode(enc, errors="strict").strip()
        except (UnicodeDecodeError, LookupError):
            continue

    # Final fallback — replace bad bytes rather than crash
    decoded = raw_bytes.decode("utf-8", errors=config.encoding_errors)
    return decoded.strip()


def load_binary_file(file_path: str | Path) -> bytes:
    """Return raw bytes, no decoding."""
    return Path(file_path).read_bytes()
