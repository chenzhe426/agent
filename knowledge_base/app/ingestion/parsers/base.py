from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.ingestion.schemas import ParsedDocument


class BaseParser(ABC):
    file_type: str = "unknown"

    @abstractmethod
    def parse(self, file_path: str | Path) -> ParsedDocument:
        raise NotImplementedError