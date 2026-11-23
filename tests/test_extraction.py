"""End-to-end extraction on synthetic documents (loads the real PaddleOCR models)."""

import json

import pytest

from app import cli
from app.extractors import FileRejected, extract
from tests.conftest import compact

ID_TOKENS_AR = ["بطاقةشخصية", "الرقمالوطني", "تاريخالميلاد"]
ID_TOKENS_EN = ["NATIONALIDENTITYCARD", "SAMPLETESTER", "01/02/1990", "01234567890"]


def assert_id_card_text(text: str):
    c = compact(text)
    missing = [t for t in ID_TOKENS_AR + ID_TOKENS_EN if t not in c]
    assert not missing, f"missing {missing} in OCR text:\n{text}"


@pytest.mark.parametrize("name", ["id_card.png", "id_card.jpg", "id_card.webp"])
def test_image_ocr_reads_arabic_and_english(fixtures, settings, engine, name):
    r = extract(fixtures[name], settings, engine)
    assert len(r.pages) == 1
    p = r.pages[0]
    assert p.method == "ocr" and p.confidence > 0.85
    assert_id_card_text(p.text)


def test_result_has_normalized_text_and_languages(fixtures, settings, engine):
    r = extract(fixtures["id_card.png"], settings, engine)
    assert r.languages == ["ar", "en"]
    n = compact(r.normalized_text)
    assert "بطاقهشخصيه" in n and "تاريخالميلاد" in n  # ة -> ه for matching
    assert "بطاقةشخصية" in compact(r.full_text)  # original letters kept in text
    d = r.to_json_dict()
    assert {"normalizedText", "languages", "fullText"} <= set(d)


def test_multipage_tiff(fixtures, settings, engine):
    r = extract(fixtures["two_pages.tiff"], settings, engine)
    assert [p.page for p in r.pages] == [1, 2]
    assert "فاتورةضريبية" in compact(r.pages[1].text)


def test_pdf_decides_per_page(fixtures, settings, engine):
    r = extract(fixtures["letter_invoice_arabic.pdf"], settings, engine)
    assert [p.method for p in r.pages] == ["text_layer", "ocr", "text_layer"]
    assert "Thank you for your order" in r.pages[0].text and r.pages[0].confidence is None
    assert "فاتورةضريبية" in compact(r.pages[1].text)
    assert "2024/115" in r.pages[1].text
    assert r.pages[2].text  # Arabic text layer (visual order / presentation forms: fixed in Phase 2)
    assert r.timings_ms["ocrPages"] == 1


def test_scanned_pdf_is_ocrd(fixtures, settings, engine):
    r = extract(fixtures["scanned_id.pdf"], settings, engine)
    assert r.pages[0].method == "ocr"
    assert_id_card_text(r.pages[0].text)


def test_docx_text_tables_and_embedded_image(fixtures, settings, engine):
    r = extract(fixtures["delivery_note.docx"], settings, engine)
    body = r.pages[0]
    assert body.method == "text_layer"
    for expected in ["Example Trading Co. - Internal", "Delivery note", "هذه وثيقة تجريبية", "Item | الكمية",
                     "Paper A4 | 10"]:
        assert expected in body.text
    assert len(r.pages) == 2 and r.pages[1].method == "ocr"
    assert_id_card_text(r.pages[1].text)


def test_text_only_pdf_does_not_load_ocr_models(settings):
    import io

    from reportlab.pdfgen import canvas

    from app.ocr.engine import OcrEngine

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "A plain text letter with enough characters to skip OCR.")
    c.save()
    eng = OcrEngine(settings)
    r = extract(buf.getvalue(), settings, eng)
    assert r.pages[0].method == "text_layer"
    assert eng._det is None  # models never loaded


def test_corrupt_pdf_rejected(settings, engine):
    with pytest.raises(FileRejected) as e:
        extract(b"%PDF-1.7\nthis is not really a pdf", settings, engine)
    assert e.value.code == "corrupt"


def test_cli_extract_prints_json(tmp_path, fixtures, capsys):
    f = tmp_path / "card.bin"  # extension is irrelevant
    f.write_bytes(fixtures["id_card.png"])
    assert cli.main(["extract", str(f)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["file"] == {"name": "card.bin", "mimeType": "image/png", "pages": 1}
    assert out["pages"][0]["method"] == "ocr"
    assert set(out) == {"file", "pages", "fullText", "normalizedText", "languages", "timingsMs"}


def test_cli_rejects_unsupported(tmp_path, capsys):
    f = tmp_path / "x.pdf"
    f.write_bytes(b"not a pdf")
    assert cli.main(["extract", str(f)]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "unsupported_type"


def test_pool_ocrs_pdf_pages_in_parallel_with_same_result(fixtures, settings, engine):
    import io

    from PIL import Image

    from app.ocr.engine import OcrPool

    frames = [Image.open(io.BytesIO(fixtures[n])).convert("RGB") for n in ("id_card.png", "invoice.png", "passport.png")]
    buf = io.BytesIO()
    frames[0].save(buf, format="PDF", save_all=True, append_images=frames[1:], resolution=150)
    pdf = buf.getvalue()  # three image-only pages: all OCR'd
    seq = extract(pdf, settings, engine)
    pool = OcrPool(settings, size=2)
    par = extract(pdf, settings, pool)
    assert [p.method for p in par.pages] == ["ocr"] * 3 and [p.page for p in par.pages] == [1, 2, 3]
    assert [p.text for p in par.pages] == [p.text for p in seq.pages]
    assert par.vision_image is not None and par.vision_image.shape == seq.vision_image.shape  # first page kept
