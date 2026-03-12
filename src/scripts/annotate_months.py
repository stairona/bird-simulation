# annotate_months.py
# Generates 24 images (12 months x 2 views) with:
# - Ecologically realistic corridors (Gaussian dispersion + curvature + turbine deflection)
# - Cinematic mode (glow + motion blur + density blending)
# - Publication-ready mode (clean, crisp, restrained styling)
#
# Usage examples:
#   python annotate_months.py --mode eco
#   python annotate_months.py --mode cinematic
#   python annotate_months.py --mode pub
#
# Requirements:
#   pip install pillow numpy
#
# Files expected (edit paths below if needed):
#   whole_base.png   (the full wind farm satellite screenshot)
#   visited_base.png (the close-up satellite screenshot)

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# -----------------------------
# CONFIG YOU MAY EDIT
# -----------------------------
WHOLE_BASE_PATH = "../data/whole_base.png"
VISITED_BASE_PATH = "../data/visited_base.png"
OUT_DIR = "../outputs/monthly-annotated-maps"

# If you have exact turbine pin locations in pixels, put them here (x, y) for each map.
# These are EXAMPLES. Replace with your real pixel coordinates from your screenshots.
TURBINES_WHOLE = [(980, 410), (1060, 470), (1010, 360), (990, 300), (1040, 280)]
TURBINES_VISITED = [(1060, 520), (1180, 600), (1050, 650), (990, 360), (1090, 300)]

# Corridor geometry per view (start/end points in pixels).
# Replace these with your real endpoints if you want exact alignment.
CORRIDORS = {
    "whole": {
        "songbirds":  {"p0": (180, 720), "p3": (1860, 430), "curv": +160},
        "waterfowl":  {"p0": (120, 520), "p3": (1910, 520), "curv":  -40},
        "raptors":    {"p0": (1700, 120), "p3": (1700, 1020), "curv": +60},
        "local":      {"center": (1020, 520), "radius": 260},
    },
    "visited": {
        "songbirds":  {"p0": (140, 710), "p3": (1880, 430), "curv": +140},
        "waterfowl":  {"p0": (90, 520),  "p3": (1930, 520), "curv":  -30},
        "raptors":    {"p0": (1700, 90), "p3": (1700, 1040),"curv": +50},
        "local":      {"center": (1030, 520), "radius": 220},
    }
}

# Month intensities (your current scheme)
MONTHS = [
    ("Jan", 0.10, "Winter: low activity"),
    ("Feb", 0.12, "Winter: low activity"),
    ("Mar", 0.40, "Spring build-up"),
    ("Apr", 0.85, "Spring: northbound peak (Apr–May)"),
    ("May", 1.00, "Spring: northbound peak (Apr–May)"),
    ("Jun", 0.20, "Summer: lower migration period"),
    ("Jul", 0.15, "Summer: lower migration period"),
    ("Aug", 0.35, "Late summer: ramp-up"),
    ("Sep", 0.85, "Fall: southbound peak (Sep–Nov)"),
    ("Oct", 1.00, "Fall: southbound peak (Sep–Nov)"),
    ("Nov", 0.45, "Fall: southbound peak (Sep–Nov)"),
    ("Dec", 0.10, "Winter: low activity"),
]

WINTER_MONTHS = {"Dec", "Jan", "Feb"}

# Publication map styling prefers less-saturated colors.
COLORS = {
    "songbirds": (255, 165, 0),     # orange
    "raptors":   (255,  60,  60),   # red
    "waterfowl": ( 60, 200, 255),   # cyan
    "local":     ( 60, 255, 140),   # green
}
COLORS_PUB = {
    "songbirds": (230, 150, 40),
    "raptors":   (210,  80,  80),
    "waterfowl": ( 80, 170, 210),
    "local":     ( 90, 210, 150),
}

# Try to load a decent font. Falls back safely.
def load_font(size: int) -> ImageFont.ImageFont:
    for name in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
        "Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


# -----------------------------
# MATH / GEOMETRY
# -----------------------------
Point = Tuple[float, float]

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def vec_add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])

def vec_sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])

def vec_mul(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)

def vec_len(a: Point) -> float:
    return math.hypot(a[0], a[1])

def vec_norm(a: Point) -> Point:
    l = vec_len(a)
    if l == 0:
        return (0.0, 0.0)
    return (a[0] / l, a[1] / l)

def vec_perp(a: Point) -> Point:
    return (-a[1], a[0])

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def rotate(v: Point, ang_rad: float) -> Point:
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    return (v[0]*c - v[1]*s, v[0]*s + v[1]*c)

def bezier(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    u = 1 - t
    return (
        (u**3)*p0[0] + 3*(u**2)*t*p1[0] + 3*u*(t**2)*p2[0] + (t**3)*p3[0],
        (u**3)*p0[1] + 3*(u**2)*t*p1[1] + 3*u*(t**2)*p2[1] + (t**3)*p3[1],
    )

def bezier_deriv(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    u = 1 - t
    return (
        3*(u**2)*(p1[0]-p0[0]) + 6*u*t*(p2[0]-p1[0]) + 3*(t**2)*(p3[0]-p2[0]),
        3*(u**2)*(p1[1]-p0[1]) + 6*u*t*(p2[1]-p1[1]) + 3*(t**2)*(p3[1]-p2[1]),
    )


# -----------------------------
# DRAWING PRIMITIVES
# -----------------------------
def draw_arrow(draw: ImageDraw.ImageDraw, p: Point, direction: Point, length: float, width: int, color, alpha: int):
    d = vec_norm(direction)
    start = (p[0] - d[0]*length*0.5, p[1] - d[1]*length*0.5)
    end   = (p[0] + d[0]*length*0.5, p[1] + d[1]*length*0.5)

    # main line
    draw.line([start, end], fill=(*color, alpha), width=width)

    # arrowhead
    head_len = max(8.0, length * 0.22)
    head_w   = max(6.0, head_len * 0.55)
    left = rotate(d, +math.radians(150))
    right= rotate(d, -math.radians(150))
    tip = end
    p1 = (tip[0] + left[0]*head_len,  tip[1] + left[1]*head_len)
    p2 = (tip[0] + right[0]*head_len, tip[1] + right[1]*head_len)
    draw.polygon([tip, p1, p2], fill=(*color, alpha))

def add_label_panel(img: Image.Image, title: str, subtitle: str, mode: str):
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    title_font = load_font(46 if mode != "pub" else 40)
    sub_font   = load_font(28 if mode != "pub" else 24)

    pad = 18
    panel_h = 92 if mode != "pub" else 82
    x0, y0 = 20, h - panel_h - 18
    x1, y1 = w - 20, h - 18

    # panel background
    bg_alpha = 170 if mode != "pub" else 150
    draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=(0,0,0,bg_alpha))

    # text
    draw.text((x0 + pad, y0 + 10), title, fill=(255,255,255,235), font=title_font)
    draw.text((x0 + pad, y0 + 56 if mode != "pub" else y0 + 50), subtitle, fill=(255,255,255,210), font=sub_font)

def add_legend(img: Image.Image, mode: str):
    draw = ImageDraw.Draw(img, "RGBA")
    font = load_font(26 if mode != "pub" else 24)
    pad = 16
    x0, y0 = 20, 20
    items = [
        ("Songbirds (warblers/sparrows)", "songbirds"),
        ("Raptors (hawks)", "raptors"),
        ("Waterfowl / shorebirds", "waterfowl"),
        ("Local/resident birds", "local"),
    ]
    box_w = 460 if mode != "pub" else 440
    box_h = 36 + len(items)*34
    draw.rounded_rectangle([x0, y0, x0+box_w, y0+box_h], radius=18, fill=(0,0,0,170 if mode!="pub" else 150))
    draw.text((x0+pad, y0+10), "Approx corridors (illustrative)", fill=(255,255,255,235), font=font)

    for i, (label, key) in enumerate(items):
        cy = y0 + 42 + i*34
        col = COLORS_PUB[key] if mode == "pub" else COLORS[key]
        draw.rounded_rectangle([x0+pad, cy+6, x0+pad+18, cy+24], radius=4, fill=(*col, 235))
        draw.text((x0+pad+28, cy), label, fill=(255,255,255,220), font=font)


# -----------------------------
# ECO MODEL: dispersion + curvature + turbine deflection
# -----------------------------
@dataclass
class SpeciesStyle:
    base_arrows: int
    base_sigma: float
    base_width: int
    base_len: float
    alpha: int

SPECIES_STYLES: Dict[str, SpeciesStyle] = {
    "songbirds": SpeciesStyle(base_arrows=55, base_sigma=18, base_width=5, base_len=58, alpha=150),
    "waterfowl": SpeciesStyle(base_arrows=45, base_sigma=14, base_width=5, base_len=56, alpha=150),
    "raptors":   SpeciesStyle(base_arrows=26, base_sigma=10, base_width=6, base_len=64, alpha=160),
}

def build_curved_corridor(p0: Point, p3: Point, curv: float, seasonal_shift: float) -> Tuple[Point, Point, Point, Point]:
    # Curvature is applied perpendicular to the corridor direction.
    d = vec_sub(p3, p0)
    dn = vec_norm(d)
    n = vec_perp(dn)

    mid = vec_add(p0, vec_mul(d, 0.5))
    ctrl_offset = curv + seasonal_shift
    ctrl = vec_add(mid, vec_mul(n, ctrl_offset))

    # Use symmetric control points around the mid-control to form a smooth curve
    p1 = vec_add(p0, vec_mul(vec_sub(ctrl, p0), 0.6))
    p2 = vec_add(p3, vec_mul(vec_sub(ctrl, p3), 0.6))
    return p0, p1, p2, p3

def turbine_deflect(pos: Point, direction: Point, turbines: List[Tuple[int,int]], intensity: float, rng: np.random.Generator) -> Tuple[Point, Point]:
    # Deflection zone radius and strength scale with intensity (more birds => more interaction visual signal)
    R = 160.0
    push_max = 24.0 * (0.4 + 0.6*intensity)
    turn_max = math.radians(14.0) * (0.4 + 0.6*intensity)

    best = None
    best_dist = 1e9
    for tx, ty in turbines:
        d = vec_sub(pos, (tx, ty))
        dist = vec_len(d)
        if dist < best_dist:
            best_dist = dist
            best = (tx, ty, d, dist)

    if best is None:
        return pos, direction

    tx, ty, dvec, dist = best
    if dist > R:
        return pos, direction

    # Push away from turbine + slight turn away
    away = vec_norm(dvec)
    strength = (1.0 - dist / R)
    push = push_max * strength
    pos2 = vec_add(pos, vec_mul(away, push))

    # turn direction away, with small random jitter
    dirn = vec_norm(direction)
    # compute signed angle to rotate toward away-vector (limited)
    cross = dirn[0]*away[1] - dirn[1]*away[0]
    sign = 1.0 if cross > 0 else -1.0
    jitter = rng.normal(0.0, math.radians(2.0))
    ang = sign * turn_max * strength + jitter
    dir2 = rotate(dirn, ang)
    return pos2, dir2

def render_corridors(
    base_img: Image.Image,
    view_key: str,
    month_name: str,
    intensity: float,
    period_label: str,
    mode: str,
    turbines: List[Tuple[int,int]],
    rng: np.random.Generator
) -> Image.Image:
    img = base_img.convert("RGBA")
    w, h = img.size

    # winter suppression: force *very* low density
    if month_name in WINTER_MONTHS:
        intensity_eff = intensity * 0.18
    else:
        intensity_eff = intensity

    # slight seasonal phase shift so each month is not identical
    seasonal_shift = rng.normal(0.0, 18.0)  # pixels

    # layers: main, glow/blur, density
    main = Image.new("RGBA", (w, h), (0,0,0,0))
    glow = Image.new("RGBA", (w, h), (0,0,0,0))
    dens = Image.new("RGBA", (w, h), (0,0,0,0))

    d_main = ImageDraw.Draw(main, "RGBA")
    d_glow = ImageDraw.Draw(glow, "RGBA")
    d_dens = ImageDraw.Draw(dens, "RGBA")

    # species corridors
    for key in ["songbirds", "raptors", "waterfowl"]:
        spec = SPECIES_STYLES[key]
        geom = CORRIDORS[view_key][key]
        p0, p3, curv = geom["p0"], geom["p3"], geom["curv"]

        # Build curve with a month-specific shift
        P0, P1, P2, P3 = build_curved_corridor(p0, p3, curv, seasonal_shift * (0.6 if key!="raptors" else 0.35))

        # Ecological dispersion: sigma grows with intensity (wider during peak)
        sigma = spec.base_sigma * (0.55 + 0.95*intensity_eff)
        # Number of arrows scales nonlinearly (peak months grow fast; winter stays tiny)
        n_arrows = int(round(spec.base_arrows * (intensity_eff ** 1.35)))
        if month_name in WINTER_MONTHS:
            n_arrows = max(2, int(round(n_arrows)))  # still show a hint
        else:
            n_arrows = max(6, n_arrows)

        # Style by mode
        if mode == "pub":
            color = COLORS_PUB[key]
            width = max(2, int(round(spec.base_width * (0.65 + 0.35*intensity_eff))))
            alpha = int(round(130 + 70*intensity_eff))
            arrow_len = spec.base_len * (0.85 + 0.20*intensity_eff)
        else:
            color = COLORS[key]
            width = max(3, int(round(spec.base_width * (0.70 + 0.55*intensity_eff))))
            alpha = int(round(spec.alpha * (0.65 + 0.35*intensity_eff)))
            arrow_len = spec.base_len * (0.85 + 0.25*intensity_eff)

        # Draw arrows sampled along curve with Gaussian lateral offsets
        for _ in range(n_arrows):
            t = float(rng.uniform(0.06, 0.94))
            pos = bezier(P0, P1, P2, P3, t)
            deriv = bezier_deriv(P0, P1, P2, P3, t)
            dvec = vec_norm(deriv)
            nvec = vec_norm(vec_perp(dvec))

            # lateral gaussian dispersion
            lateral = float(rng.normal(0.0, sigma))
            # slight along-track jitter
            along = float(rng.normal(0.0, sigma * 0.18))

            pos2 = vec_add(pos, vec_add(vec_mul(nvec, lateral), vec_mul(dvec, along)))

            # turbine interaction (deflect near turbines)
            pos3, dvec2 = turbine_deflect(pos2, dvec, turbines, intensity_eff, rng)

            # cinematic density layer (wider, softer)
            if mode == "cinematic":
                draw_arrow(d_dens, pos3, dvec2, arrow_len*1.05, width + 10, color, int(alpha*0.30))
                draw_arrow(d_glow, pos3, dvec2, arrow_len*1.00, width + 6,  color, int(alpha*0.40))

            # main / pub / eco line
            draw_arrow(d_main, pos3, dvec2, arrow_len, width, color, alpha)

    # Local/resident birds: small near turbines, mostly constant across months
    local_color = COLORS_PUB["local"] if mode == "pub" else COLORS["local"]
    local_alpha = 130 if mode == "pub" else 140
    local_width = 3 if mode == "pub" else 4
    local_len = 28 if mode == "pub" else 30
    local_count = 12 if view_key == "whole" else 14

    # Keep residents mildly variable but not driven by migration intensity
    for _ in range(local_count):
        # choose a turbine and spawn near it
        tx, ty = turbines[int(rng.integers(0, len(turbines)))]
        ang = float(rng.uniform(0, 2*math.pi))
        rad = float(abs(rng.normal(0, 70)))
        pos = (tx + math.cos(ang)*rad, ty + math.sin(ang)*rad)

        # random short movement vector
        dvec = (math.cos(ang + rng.normal(0, 0.6)), math.sin(ang + rng.normal(0, 0.6)))

        if mode == "cinematic":
            draw_arrow(d_glow, pos, dvec, local_len, local_width+4, local_color, int(local_alpha*0.35))
        draw_arrow(d_main, pos, dvec, local_len, local_width, local_color, local_alpha)

    # Apply mode-specific post effects
    if mode == "cinematic":
        # Glow
        glow_blur = glow.filter(ImageFilter.GaussianBlur(radius=10))
        dens_blur = dens.filter(ImageFilter.GaussianBlur(radius=12))

        # Motion blur approximation: shift-blend along corridor direction (global subtle)
        def motion_smear(layer: Image.Image, shifts: int = 6, step: int = 2) -> Image.Image:
            acc = Image.new("RGBA", (w, h), (0,0,0,0))
            for i in range(shifts):
                dx = int(i * step)
                dy = int(i * step * 0.3)
                tmp = Image.new("RGBA", (w, h), (0,0,0,0))
                tmp.paste(layer, (dx, dy))
                acc = Image.alpha_composite(acc, tmp)
            return acc.filter(ImageFilter.GaussianBlur(radius=1.2))

        smear = motion_smear(main, shifts=7, step=2)

        img = Image.alpha_composite(img, dens_blur)
        img = Image.alpha_composite(img, glow_blur)
        img = Image.alpha_composite(img, smear)
        img = Image.alpha_composite(img, main)

        # Slight contrast pop (kept simple to avoid extra deps)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))

    elif mode == "eco":
        # Light smoothing for natural look
        soft = main.filter(ImageFilter.GaussianBlur(radius=1.2))
        img = Image.alpha_composite(img, soft)
        img = Image.alpha_composite(img, main)

    else:  # pub
        # Crisp only
        img = Image.alpha_composite(img, main)

    # Legend + month/title panels
    add_legend(img, mode)
    title_prefix = "Visited turbines area (close-up)" if view_key == "visited" else "Whole wind farm (Isabella Wind)"
    title = f"{title_prefix} — {month_name} | migration intensity={intensity:.2f}"
    add_label_panel(img, title, period_label, mode)

    return img


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eco", "cinematic", "pub"], default="eco")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=OUT_DIR)
    parser.add_argument("--whole", type=str, default=WHOLE_BASE_PATH)
    parser.add_argument("--visited", type=str, default=VISITED_BASE_PATH)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    whole_base = Image.open(args.whole).convert("RGBA")
    visited_base = Image.open(args.visited).convert("RGBA")

    for idx, (mon, inten, period_label) in enumerate(MONTHS, start=1):
        # Use a month-specific RNG stream so each month is stable but different
        month_rng = np.random.default_rng(args.seed * 100 + idx)

        whole_img = render_corridors(
            base_img=whole_base,
            view_key="whole",
            month_name=mon,
            intensity=inten,
            period_label=period_label,
            mode=args.mode,
            turbines=TURBINES_WHOLE,
            rng=month_rng
        )

        visited_img = render_corridors(
            base_img=visited_base,
            view_key="visited",
            month_name=mon,
            intensity=inten,
            period_label=period_label,
            mode=args.mode,
            turbines=TURBINES_VISITED,
            rng=month_rng
        )

        whole_out = os.path.join(args.out, f"whole_{idx:02d}_{mon}_{args.mode}.png")
        visited_out = os.path.join(args.out, f"visited_{idx:02d}_{mon}_{args.mode}.png")
        whole_img.save(whole_out)
        visited_img.save(visited_out)

        print(f"Saved: {visited_out}")
        print(f"Saved: {whole_out}")

if __name__ == "__main__":
    main()
