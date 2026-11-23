"""File-type detection by magic bytes (never by extension) and upload validation."""

import io
import zipfile

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
JPEG = "image/jpeg"
PNG = "image/png"
WEBP = "image/webp"
TIFF = "image/tiff"
IMAGE_TYPES = {JPEG, PNG, WEBP, TIFF}
SUPPORTED = {PDF, DOCX} | IMAGE_TYPES

MAX_DOCX_UNCOMPRESSED = 300 * 1024 * 1024  # zip-bomb guard


class FileRejected(ValueError):
    """The upload can't be processed. `code` is safe to return to API clients."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def detect_mime(data: bytes) -> str | None:
    head = data[:16]
    if head.startswith(b"%PDF-"):
        return PDF
    if head.startswith(b"\xff\xd8\xff"):
        return JPEG
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return PNG
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return WEBP
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return TIFF
    if head.startswith(b"PK\x03\x04") and _is_docx(data):
        return DOCX
    return None


def _is_docx(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile:
        return False
    return "[Content_Types].xml" in names and "word/document.xml" in names


def validate_upload(data: bytes, max_bytes: int) -> str:
    """Return the MIME type, or raise FileRejected."""
    if not data:
        raise FileRejected("empty", "file is empty")
    if len(data) > max_bytes:
        raise FileRejected("too_large", f"file exceeds {max_bytes} bytes")
    mime = detect_mime(data)
    if mime is None:
        raise FileRejected("unsupported_type", "unsupported file type (PDF, DOCX, JPG, PNG, WEBP, TIFF)")
    if mime == DOCX:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if sum(i.file_size for i in zf.infolist()) > MAX_DOCX_UNCOMPRESSED:
                raise FileRejected("too_large", "DOCX expands beyond the allowed size")
    return mime
