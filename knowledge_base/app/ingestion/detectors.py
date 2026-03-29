from pathlib import Path

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
DOCX_EXTENSIONS = {".docx"}
PDF_EXTENSIONS = {".pdf"}


def detect_file_type(file_path: str | Path) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext in DOCX_EXTENSIONS:
        return "docx"
    if ext in PDF_EXTENSIONS:
        return "pdf"
    return "unknown"