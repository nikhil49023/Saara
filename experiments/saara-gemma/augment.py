"""Seeded page-image degradations for OCR robustness (old scans, faded ink, phone photos).

The transcription target never changes; only the image gets harder. Used for a share of the
training pages and for a separate noisy copy of the test pages (robustness metric).
"""
from __future__ import annotations

import io
import random

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def _noise(image: Image.Image, rng: random.Random, amount: float) -> Image.Image:
    noise = Image.effect_noise(image.size, rng.uniform(20, 60)).convert("RGB")
    return Image.blend(image, noise, amount)


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def degrade(image: Image.Image, seed: int) -> tuple[Image.Image, list[str]]:
    """Apply 2-4 random degradations. Same seed -> same output."""
    rng = random.Random(seed)
    ops = [
        ("blur", lambda im: im.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.6)))),
        ("noise", lambda im: _noise(im, rng, rng.uniform(0.05, 0.15))),
        ("jpeg", lambda im: _jpeg(im, rng.randint(15, 40))),
        ("low_contrast", lambda im: ImageEnhance.Contrast(im).enhance(rng.uniform(0.4, 0.7))),
        ("faded_ink", lambda im: ImageEnhance.Brightness(im).enhance(rng.uniform(1.1, 1.35))),
        ("yellowed_paper", lambda im: Image.blend(im, Image.new("RGB", im.size, (226, 208, 160)), rng.uniform(0.15, 0.3))),
        ("skew", lambda im: im.rotate(rng.uniform(-2.5, 2.5), resample=Image.BICUBIC, expand=True, fillcolor=(245, 245, 240))),
        ("downscale", lambda im: im.resize((im.width // 2, im.height // 2)).resize(im.size)),
        ("grayscale", lambda im: ImageOps.grayscale(im).convert("RGB")),
    ]
    chosen = rng.sample(ops, k=rng.randint(2, 4))
    for _, op in chosen:
        image = op(image)
    return image, [name for name, _ in chosen]
