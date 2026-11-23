import io
import zipfile

import pytest

from app.extractors.filetype import DOCX, JPEG, PDF, PNG, TIFF, WEBP, FileRejected, detect_mime, validate_upload


@pytest.mark.parametrize("name,mime", [
    ("id_card.png", PNG), ("id_card.jpg", JPEG), ("id_card.webp", WEBP), ("two_pages.tiff", TIFF),
    ("scanned_id.pdf", PDF), ("letter_invoice_arabic.pdf", PDF), ("delivery_note.docx", DOCX),
])
def test_detects_by_magic_bytes(fixtures, name, mime):
    assert detect_mime(fixtures[name]) == mime


def test_extension_is_ignored(fixtures):
    # Content decides, not the name: PNG bytes stay PNG whatever the file is called.
    assert validate_upload(fixtures["id_card.png"], 10**8) == PNG


def test_plain_zip_is_not_docx():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", "hi")
    assert detect_mime(buf.getvalue()) is None


@pytest.mark.parametrize("data", [b"GIF89a....", b"hello world", b"\x00\x01\x02\x03"])
def test_rejects_unsupported(data):
    with pytest.raises(FileRejected) as e:
        validate_upload(data, 10**6)
    assert e.value.code == "unsupported_type"


def test_rejects_too_large():
    with pytest.raises(FileRejected) as e:
        validate_upload(b"%PDF-" + b"0" * 100, max_bytes=50)
    assert e.value.code == "too_large"


def test_rejects_empty():
    with pytest.raises(FileRejected) as e:
        validate_upload(b"", 10)
    assert e.value.code == "empty"
