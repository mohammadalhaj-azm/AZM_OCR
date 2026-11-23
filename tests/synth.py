"""Synthetic test documents: fake IDs (fake names/numbers) and non-ID invoices/letters.

Never use real documents in tests. To write all fixtures to a folder for manual CLI use:
    uv run python -m tests.synth tests/fixtures/synthetic
"""

import io
import sys
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "C:/Windows/Fonts/arial.ttf"

ID_CARD_LINES = [
    "الجمهورية العربية التجريبية",
    "بطاقة شخصية",
    "NATIONAL IDENTITY CARD",
    "الاسم: سامي تجريبي",
    "Name: SAMPLE TESTER",
    "تاريخ الميلاد: 1990/02/01",
    "Date of Birth: 01/02/1990",
    "الرقم الوطني: 01234567890",
]

INVOICE_LINES = [
    "شركة المثال للتجارة",
    "فاتورة ضريبية",
    "رقم الفاتورة: 2024/115",
    "التاريخ: 2024/03/15",
    "المبلغ الإجمالي: 1500 ريال",
    "INVOICE",
]

MENTIONS_ID_LINES = [  # hard negative: talks about IDs but is a letter
    "تحية طيبة،",
    "يرجى إحضار صورة عن جواز السفر وبطاقة الهوية لاستكمال فتح الحساب.",
    "Dear Mr. Sample,",
    "Please bring a copy of your passport and your national ID card to complete your account opening.",
    "Kind regards, Example Bank",
]


def passport_mrz(surname: str = "TESTER", given: str = "SAMPLE", number: str = "AB1234567",
                 birth: str = "900201", expiry: str = "300101", sex: str = "M") -> list[str]:
    """TD3 MRZ for a fake passport of the fictional ICAO state UTO, with correct check digits."""
    from app.detection.mrz import check_digit as cd

    line1 = f"P<UTO{surname}<<{given}".ljust(44, "<")
    personal = "<" * 14
    body = f"{number}{cd(number)}UTO{birth}{cd(birth)}{sex}{expiry}{cd(expiry)}{personal}{cd(personal)}"
    composite = cd(body[0:10] + body[13:20] + body[21:43])
    return [line1, body + composite]


PASSPORT_LINES = [
    "جواز سفر",
    "PASSPORT",
    "الاسم: سامي تجريبي",
    "Name: SAMPLE TESTER",
    "Nationality: UTOPIAN",
]

LETTER_LINES = [
    "Example Trading Co.",
    "Dear Customer,",
    "Thank you for your order. Your delivery is scheduled for next week.",
    "Please contact our support team if you have any questions.",
    "Kind regards, The Example Team",
]


def has_arabic(text: str) -> bool:
    return any("\u0600" <= c <= "\u06ff" for c in text)


def visual(text: str) -> str:
    """Shape + reorder Arabic for renderers without complex-script support (Pillow w/o raqm, reportlab)."""
    return get_display(arabic_reshaper.reshape(text)) if has_arabic(text) else text


def render_lines(lines: list[str], width: int = 1400, font_size: int = 40) -> Image.Image:
    font = ImageFont.truetype(FONT_PATH, font_size)
    line_h = int(font_size * 1.9)
    img = Image.new("RGB", (width, 60 + line_h * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        s = visual(line)
        x = width - 50 - font.getlength(s) if has_arabic(line) else 50  # RTL lines right-aligned
        draw.text((x, 30 + i * line_h), s, fill="black", font=font)
    return img


def id_card_image() -> Image.Image:
    return render_lines(ID_CARD_LINES)


def passport_image() -> Image.Image:
    top = render_lines(PASSPORT_LINES)
    mono = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 34)
    mrz = passport_mrz()
    img = Image.new("RGB", (top.width, top.height + 150), "white")
    img.paste(top, (0, 0))
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(mrz):
        draw.text((50, top.height + 20 + i * 60), line, fill="black", font=mono)
    return img


def mentions_id_image() -> Image.Image:
    return render_lines(MENTIONS_ID_LINES, width=1900, font_size=34)


def invoice_image() -> Image.Image:
    return render_lines(INVOICE_LINES)


def image_bytes(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def multipage_tiff() -> bytes:
    return image_bytes(id_card_image(), "TIFF", save_all=True, append_images=[invoice_image()])


def text_pdf() -> bytes:
    """Page 1: English letter (text layer). Page 2: scanned invoice (image only). Page 3: Arabic text layer."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    pdfmetrics.registerFont(TTFont("Arial", FONT_PATH))
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica", 12)
    for i, line in enumerate(LETTER_LINES):
        c.drawString(72, h - 72 - i * 20, line)
    c.showPage()
    inv = invoice_image()
    c.drawImage(ImageReader(inv), 36, h - 36 - inv.height * 0.35, width=inv.width * 0.35, height=inv.height * 0.35)
    c.showPage()
    c.setFont("Arial", 14)
    for i, line in enumerate(INVOICE_LINES[:5]):
        c.drawRightString(w - 72, h - 72 - i * 24, visual(line))  # visual order, like many real PDFs
    c.showPage()
    c.save()
    return buf.getvalue()


def scanned_pdf() -> bytes:
    """Image-only PDF (no text layer), like a scanner produces."""
    return image_bytes(id_card_image(), "PDF", resolution=150)


def docx_with_image() -> bytes:
    from docx import Document
    from docx.shared import Inches

    doc = Document()
    doc.sections[0].header.paragraphs[0].text = "Example Trading Co. - Internal"
    doc.add_heading("Delivery note", level=1)
    doc.add_paragraph("شكراً لتعاملكم معنا. هذه وثيقة تجريبية.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Item", "الكمية"
    table.cell(1, 0).text, table.cell(1, 1).text = "Paper A4", "10"
    doc.add_picture(io.BytesIO(image_bytes(id_card_image(), "PNG")), width=Inches(5))
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def all_fixtures() -> dict[str, bytes]:
    card = id_card_image()
    return {
        "id_card.png": image_bytes(card, "PNG"),
        "id_card.jpg": image_bytes(card, "JPEG", quality=90),
        "id_card.webp": image_bytes(card, "WEBP", quality=90),
        "two_pages.tiff": multipage_tiff(),
        "invoice.png": image_bytes(invoice_image(), "PNG"),
        "passport.png": image_bytes(passport_image(), "PNG"),
        "letter_mentions_id.png": image_bytes(mentions_id_image(), "PNG"),
        "letter_invoice_arabic.pdf": text_pdf(),
        "scanned_id.pdf": scanned_pdf(),
        "delivery_note.docx": docx_with_image(),
    }


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/synthetic")
    out.mkdir(parents=True, exist_ok=True)
    for name, data in all_fixtures().items():
        (out / name).write_bytes(data)
        print(f"wrote {out / name} ({len(data)} bytes)")
