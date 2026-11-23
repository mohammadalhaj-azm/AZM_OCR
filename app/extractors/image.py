"""Decode images (JPG/PNG/WEBP/TIFF, incl. multi-page TIFF) into BGR arrays."""

import io

import numpy as np
from PIL import Image, ImageOps, ImageSequence

from app.extractors.filetype import FileRejected


def pil_to_bgr(im: Image.Image) -> np.ndarray:
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, rgba)
    return np.asarray(im.convert("RGB"))[:, :, ::-1].copy()


def decode_images(data: bytes, max_frames: int) -> list[np.ndarray]:
    try:
        with Image.open(io.BytesIO(data)) as im:
            frames = []
            for i, frame in enumerate(ImageSequence.Iterator(im)):
                if i >= max_frames:
                    raise FileRejected("too_many_pages", f"more than {max_frames} frames")
                frames.append(pil_to_bgr(ImageOps.exif_transpose(frame)))
            return frames
    except FileRejected:
        raise
    except Exception as e:  # noqa: BLE001 - any decoder error means a corrupt image
        raise FileRejected("corrupt", "image could not be decoded") from e
