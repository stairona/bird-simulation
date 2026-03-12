"""
Collage assembly for Phase 1 corridor maps.

Merges the functionality of the original make_collage.py,
make_collage_sets.py, and make_collages_ppt.py into one module
with three layout modes:

  full   — 12 months in a 4x3 grid (one per view)
  half   — Two 3x2 grids per view (Jan-Jun, Jul-Dec)
  ppt    — 4K 16:9 PowerPoint-ready half-year collages with labels
"""

from __future__ import annotations

import os
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from ..core.config import SiteConfig


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _find_month_file(input_dir: str, prefix: str, month: int) -> Optional[str]:
    start = f"{prefix}_{month:02d}_"
    for fname in sorted(os.listdir(input_dir)):
        if fname.startswith(start) and fname.endswith(".png"):
            return os.path.join(input_dir, fname)
    return None


def _load_month_images(
    input_dir: str, prefix: str, months: List[int],
) -> List[Image.Image]:
    images = []
    for m in months:
        path = _find_month_file(input_dir, prefix, m)
        if path is None:
            raise FileNotFoundError(
                f"Missing image for {prefix} month {m:02d} in {input_dir}"
            )
        images.append(Image.open(path))
    return images


def _simple_grid(images: List[Image.Image], cols: int, border: int = 6) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size

    total_w = cols * w + (cols + 1) * border
    total_h = rows * h + (rows + 1) * border
    collage = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = border + c * (w + border)
        y = border + r * (h + border)
        collage.paste(img.convert("RGB"), (x, y))

    return collage


def _letterbox_fit(img: Image.Image, tw: int, th: int) -> Image.Image:
    w, h = img.size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    tile = Image.new("RGB", (tw, th), (255, 255, 255))
    tile.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return tile


def _ppt_collage(
    images: List[Image.Image],
    title_text: str,
    month_labels: List[str],
    out_w: int = 3840, out_h: int = 2160,
) -> Image.Image:
    cols, rows = 3, 2
    margin_lr, margin_bottom, title_h = 140, 140, 220
    gap, border_w = 18, 6
    border_color = (230, 230, 230)

    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    usable_w = out_w - 2 * margin_lr
    usable_h = out_h - title_h - margin_bottom
    tile_w = (usable_w - (cols - 1) * gap) // cols
    tile_h = (usable_h - (rows - 1) * gap) // rows

    font_title = _load_font(64)
    font_small = _load_font(44)

    draw.text((margin_lr, 70), title_text, fill=(30, 30, 30), font=font_title)

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        tile = _letterbox_fit(img.convert("RGB"), tile_w, tile_h)
        px = margin_lr + c * (tile_w + gap)
        py = title_h + r * (tile_h + gap)
        canvas.paste(tile, (px, py))
        draw.rectangle([px, py, px + tile_w, py + tile_h], outline=border_color, width=border_w)
        if i < len(month_labels):
            draw.text((px + 22, py + 22), month_labels[i], fill=(20, 20, 20), font=font_small)

    return canvas


# ── Public API ────────────────────────────────────────────────────

def generate_collages(
    cfg: SiteConfig,
    input_dir: str = "outputs/monthly-annotated-maps",
    out_dir: str = "outputs/collages",
    layout: str = "all",
):
    """
    Generate collages from Phase 1 monthly maps.

    Args:
        layout: "full", "half", "ppt", or "all"
    """
    os.makedirs(out_dir, exist_ok=True)
    view_keys = list(cfg.maps.keys())

    do_full = layout in ("full", "all")
    do_half = layout in ("half", "all")
    do_ppt = layout in ("ppt", "all")

    for vk in view_keys:
        if do_full:
            imgs = _load_month_images(input_dir, vk, list(range(1, 13)))
            collage = _simple_grid(imgs, cols=4, border=6)
            path = os.path.join(out_dir, f"collage_{vk}_full.png")
            collage.save(path)
            print(f"Saved: {path}")

        if do_half:
            for label, months in [("Jan-Jun", range(1, 7)), ("Jul-Dec", range(7, 13))]:
                imgs = _load_month_images(input_dir, vk, list(months))
                collage = _simple_grid(imgs, cols=3, border=6)
                path = os.path.join(out_dir, f"collage_{vk}_{label}.png")
                collage.save(path)
                print(f"Saved: {path}")

        if do_ppt:
            for label, months in [("Jan-Jun", range(1, 7)), ("Jul-Dec", range(7, 13))]:
                imgs = _load_month_images(input_dir, vk, list(months))
                month_labels = [MONTH_NAMES[m - 1] for m in months]
                title = f"{cfg.site_name} ({vk}) — {label}"
                collage = _ppt_collage(imgs, title, month_labels)
                path = os.path.join(out_dir, f"ppt_{vk}_{label}_4k.png")
                collage.save(path, "PNG", optimize=False)
                print(f"Saved: {path}")
