import numpy as np

from app.extractors.image import pil_to_bgr
from app.ocr.preprocess import estimate_skew, preprocess, rotate
from tests import synth


def test_estimates_skew_of_rotated_page():
    img = pil_to_bgr(synth.id_card_image())
    assert abs(estimate_skew(img)) <= 0.4
    for angle in (4.0, -6.0):
        assert abs(estimate_skew(rotate(img, angle)) + angle) <= 0.6  # correction undoes the rotation


def test_preprocess_returns_bgr_image():
    img = pil_to_bgr(synth.id_card_image())
    out = preprocess(rotate(img, 3.0))
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8
