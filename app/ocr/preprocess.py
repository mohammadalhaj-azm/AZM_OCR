"""Optional OpenCV preprocessing (OCR_PREPROCESS=true): deskew + contrast (CLAHE)."""

import cv2
import numpy as np


def clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a, b = cv2.split(lab)
    l_chan = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_chan, a, b)), cv2.COLOR_LAB2BGR)


def rotation_matrix(shape: tuple[int, ...], angle: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Affine matrix (original -> rotated) for `rotate`, and the expanded canvas size (w, h)."""
    h, w = shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return m, (nw, nh)


def rotate(img: np.ndarray, angle: float, fill: int = 255) -> np.ndarray:
    """Rotate by `angle` degrees (counter-clockwise), expanding the canvas."""
    m, size = rotation_matrix(img.shape, angle)
    border = (fill,) * 3 if img.ndim == 3 else fill
    return cv2.warpAffine(img, m, size, flags=cv2.INTER_LINEAR, borderValue=border)


def estimate_skew(img: np.ndarray, max_angle: float = 15.0) -> float:
    """Angle (degrees, for `rotate`) that makes text lines horizontal. 0.0 if unsure.

    Smears text horizontally into line blobs, measures each elongated blob's long-edge
    angle, and takes the length-weighted median.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = min(1.0, 1500 / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # text = 255
    blobs = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles, weights = [], []
    for c in contours:
        pts = cv2.boxPoints(cv2.minAreaRect(c))
        edges = [pts[(i + 1) % 4] - pts[i] for i in range(2)]
        long_edge, short_edge = sorted(edges, key=lambda e: -np.hypot(*e))
        length, thickness = np.hypot(*long_edge), np.hypot(*short_edge)
        if length < 60 or length < 5 * thickness:
            continue  # not a text line
        a = np.degrees(np.arctan2(long_edge[1], long_edge[0]))
        a = (a + 90) % 180 - 90  # direction-free, in [-90, 90)
        if abs(a) <= max_angle:
            angles.append(a)
            weights.append(length)
    if not angles:
        return 0.0
    order = np.argsort(angles)
    cum = np.cumsum(np.asarray(weights)[order])
    median = np.asarray(angles)[order][np.searchsorted(cum, cum[-1] / 2)]
    return round(float(median), 1)


def preprocess_with_transform(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Deskew + CLAHE. Also returns the affine matrix (input -> output coordinates), or None if not rotated."""
    angle = estimate_skew(img)
    m = None
    if abs(angle) >= 0.3:
        m, size = rotation_matrix(img.shape, angle)
        img = cv2.warpAffine(img, m, size, flags=cv2.INTER_LINEAR, borderValue=(255,) * 3 if img.ndim == 3 else 255)
    return clahe(img), m


def preprocess(img: np.ndarray) -> np.ndarray:
    return preprocess_with_transform(img)[0]
