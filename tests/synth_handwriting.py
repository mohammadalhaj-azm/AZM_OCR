"""Synthetic pages for the handwriting tests: blank, noisy blank, signature, stamp, a few words (BGR arrays).

They are stand-ins for the cases the model must not invent text for; real scans belong in the evaluation set
(eval/run_handwriting_eval.py), not in the repository.
"""

import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tests.synth import FONT_PATH, visual

W, H = 1654, 2339  # A4 at 200 DPI


def _page() -> Image.Image:
    return Image.new("RGB", (W, H), "white")


def bgr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))[:, :, ::-1].copy()


def png(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img[:, :, ::-1]).save(buf, "PNG")
    return buf.getvalue()


def blank() -> np.ndarray:
    return bgr(_page())


def noisy_blank(seed: int = 1) -> np.ndarray:
    """Off-white scanned paper: sensor noise, dust specks and a dark scanner edge."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 236, np.float32) + rng.normal(0, 6, (H, W, 1))
    for _ in range(60):  # dust: 1-3 px specks
        y, x, r = rng.integers(0, H), rng.integers(0, W), rng.integers(1, 3)
        img[max(0, y - r):y + r, max(0, x - r):x + r] = 90
    img[:, :25] = 70  # scanner shadow along the left edge (inside the ignored margin)
    img[:18, :] = 80
    return np.clip(img, 0, 255).astype(np.uint8)


def signature(scale: float = 1.0, width: int = 4) -> np.ndarray:
    """One continuous pen stroke, like a signature, near the bottom of an otherwise empty page."""
    img = _page()
    d = ImageDraw.Draw(img)
    x0, y0 = 1000, 1950
    pts = [(x0 + scale * (t * 3.2), y0 + scale * (40 * math.sin(t / 9) + 25 * math.sin(t / 3.1)))
           for t in range(0, 130)]
    d.line(pts, fill=(20, 30, 90), width=width, joint="curve")
    return bgr(img)


def stamp() -> np.ndarray:
    """A round red stamp with some text inside, nothing else on the page."""
    img = _page()
    d = ImageDraw.Draw(img)
    cx, cy, r = 1200, 1900, 150
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(200, 30, 40), width=7)
    d.ellipse((cx - r + 22, cy - r + 22, cx + r - 22, cy + r - 22), outline=(200, 30, 40), width=3)
    font = ImageFont.truetype(FONT_PATH, 34)
    d.text((cx, cy), visual("مختوم"), fill=(200, 30, 40), font=font, anchor="mm")
    return bgr(img)


def few_words(text: str = "نعم موافق", size: int = 56) -> np.ndarray:
    img = _page()
    ImageDraw.Draw(img).text((900, 500), visual(text), fill=(15, 15, 15),
                             font=ImageFont.truetype(FONT_PATH, size), anchor="mm")
    return bgr(img)


def text_page(lines: list[str], size: int = 44) -> np.ndarray:
    img = _page()
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, size)
    for i, line in enumerate(lines):
        d.text((W - 150, 200 + i * (size * 2)), visual(line), fill=(10, 10, 10), font=font, anchor="ra")
    return bgr(img)
