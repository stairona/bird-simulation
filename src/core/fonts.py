"""Shared font loading for Phase 1 rendering modules."""

from __future__ import annotations

from PIL import ImageFont

_FONT_PATHS = [
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "Arial.ttf",
]


def load_font(size: int) -> ImageFont.ImageFont:
    """Try common font paths, falling back to Pillow's default bitmap font."""
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()
