#!/usr/bin/env python3
import os
from PIL import Image, ImageDraw, ImageFont

# ---------- SETTINGS (PowerPoint-ready) ----------
# 16:9 slide exports
OUT_W, OUT_H = 3840, 2160   # 4K (best for sharpness on 1080p display)
BG = (255, 255, 255)        # white background

# Collage layout: 2 rows x 3 cols (6 months per collage)
ROWS, COLS = 2, 3

# Margins (in pixels)
MARGIN_LR = 140             # left/right
MARGIN_BOTTOM = 140
TITLE_AREA_H = 220          # set to 0 if you do NOT want space for a slide title

# Thin borders between images
GAP = 18                    # gap between tiles
BORDER_W = 6                # thin border stroke
BORDER_COLOR = (230, 230, 230)

# Optional label inside collage image (helpful if you insert full-bleed)
DRAW_LABEL = True
LABEL_FONT_SIZE = 64
LABEL_COLOR = (30, 30, 30)

IN_DIR = "../outputs/monthly-annotated-maps"
OUT_DIR = "../outputs/ppt-collages"
os.makedirs(OUT_DIR, exist_ok=True)

MONTHS = [
    ("01_Jan", "Jan"), ("02_Feb", "Feb"), ("03_Mar", "Mar"),
    ("04_Apr", "Apr"), ("05_May", "May"), ("06_Jun", "Jun"),
    ("07_Jul", "Jul"), ("08_Aug", "Aug"), ("09_Sep", "Sep"),
    ("10_Oct", "Oct"), ("11_Nov", "Nov"), ("12_Dec", "Dec")
]

def load_font(size):
    # macOS-safe fallback chain
    for path in [
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf"
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()

def letterbox_fit(img, target_w, target_h):
    """Resize to fit inside target, preserving aspect ratio (no cropping)."""
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    # center on a blank tile
    tile = Image.new("RGB", (target_w, target_h), BG)
    ox = (target_w - nw) // 2
    oy = (target_h - nh) // 2
    tile.paste(resized, (ox, oy))
    return tile

def make_one_collage(kind, start_idx, end_idx, title_text):
    """
    kind: "visited" or "whole"
    start_idx/end_idx: slice of MONTHS (0-6, 6-12)
    """
    canvas = Image.new("RGB", (OUT_W, OUT_H), BG)
    draw = ImageDraw.Draw(canvas)

    usable_w = OUT_W - 2 * MARGIN_LR
    usable_h = OUT_H - TITLE_AREA_H - MARGIN_BOTTOM

    tile_w = (usable_w - (COLS - 1) * GAP) // COLS
    tile_h = (usable_h - (ROWS - 1) * GAP) // ROWS

    x0 = MARGIN_LR
    y0 = TITLE_AREA_H

    # Optional label
    if DRAW_LABEL and title_text:
        font = load_font(LABEL_FONT_SIZE)
        draw.text((MARGIN_LR, 70), title_text, fill=LABEL_COLOR, font=font)

    # Place 6 months
    subset = MONTHS[start_idx:end_idx]
    for i, (m_key, m_short) in enumerate(subset):
        r = i // COLS
        c = i % COLS

        path = os.path.join(IN_DIR, f"{kind}_{m_key}_eco.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

        img = Image.open(path).convert("RGB")
        tile = letterbox_fit(img, tile_w, tile_h)

        px = x0 + c * (tile_w + GAP)
        py = y0 + r * (tile_h + GAP)
        canvas.paste(tile, (px, py))

        # Border rectangle (thin)
        draw.rectangle(
            [px, py, px + tile_w, py + tile_h],
            outline=BORDER_COLOR,
            width=BORDER_W
        )

        # Month label (small, top-left inside each tile)
        font_small = load_font(44)
        label_pad = 22
        draw.text((px + label_pad, py + label_pad), m_short, fill=(20, 20, 20), font=font_small)

    return canvas

def main():
    # Four outputs:
    # visited: Jan–Jun, Jul–Dec
    # whole:   Jan–Jun, Jul–Dec
    outputs = [
        ("visited", 0, 6,  "Visited turbines — Jan–Jun"),
        ("visited", 6, 12, "Visited turbines — Jul–Dec"),
        ("whole",   0, 6,  "Whole wind farm — Jan–Jun"),
        ("whole",   6, 12, "Whole wind farm — Jul–Dec"),
    ]

    for kind, a, b, label in outputs:
        img = make_one_collage(kind, a, b, label)
        out_path = os.path.join(OUT_DIR, f"{kind}_{a+1:02d}-{b:02d}_ppt_4k.png")
        img.save(out_path, "PNG", optimize=False)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
