"""
Phase 1: Monthly corridor map generator.

Renders annotated migration corridor maps for each month and map view
defined in the site config. All geometry, species, and intensities
come from the YAML — no site-specific values are hardcoded here.

Rendering modes:
  eco       — Gaussian dispersion + turbine deflection (ecological realism)
  cinematic — Glow, motion blur, density blending (presentations)
  pub       — Clean, crisp styling (academic papers)
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..core.config import MapView, SiteConfig, SpeciesDef
from ..core.corridors import (
    Point,
    bezier,
    bezier_deriv,
    build_curved_corridor,
    rotate,
    vec_add,
    vec_mul,
    vec_norm,
    vec_perp,
)
from ..core.turbines import turbine_deflect


from ..core.fonts import load_font as _load_font


# ── Drawing primitives ────────────────────────────────────────────

def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    p: Point, direction: Point,
    length: float, width: int,
    color: Tuple[int, int, int], alpha: int,
):
    d = vec_norm(direction)
    start = (p[0] - d[0] * length * 0.5, p[1] - d[1] * length * 0.5)
    end = (p[0] + d[0] * length * 0.5, p[1] + d[1] * length * 0.5)

    draw.line([start, end], fill=(*color, alpha), width=width)

    head_len = max(8.0, length * 0.22)
    left = rotate(d, +math.radians(150))
    right = rotate(d, -math.radians(150))
    tip = end
    p1 = (tip[0] + left[0] * head_len, tip[1] + left[1] * head_len)
    p2 = (tip[0] + right[0] * head_len, tip[1] + right[1] * head_len)
    draw.polygon([tip, p1, p2], fill=(*color, alpha))


def _add_label_panel(
    img: Image.Image,
    title: str, subtitle: str,
    mode: str,
):
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    title_font = _load_font(46 if mode != "pub" else 40)
    sub_font = _load_font(28 if mode != "pub" else 24)

    pad = 18
    panel_h = 92 if mode != "pub" else 82
    x0, y0 = 20, h - panel_h - 18
    x1, y1 = w - 20, h - 18

    bg_alpha = 170 if mode != "pub" else 150
    draw.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=(0, 0, 0, bg_alpha))
    draw.text((x0 + pad, y0 + 10), title, fill=(255, 255, 255, 235), font=title_font)
    draw.text(
        (x0 + pad, y0 + (56 if mode != "pub" else 50)),
        subtitle, fill=(255, 255, 255, 210), font=sub_font,
    )


def _add_legend(
    img: Image.Image,
    species: Dict[str, SpeciesDef],
    mode: str,
):
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font(26 if mode != "pub" else 24)
    pad = 16
    x0, y0 = 20, 20

    items = [(sp.label, sp.key) for sp in species.values()]
    box_w = 460 if mode != "pub" else 440
    box_h = 36 + len(items) * 34

    draw.rounded_rectangle(
        [x0, y0, x0 + box_w, y0 + box_h], radius=18,
        fill=(0, 0, 0, 170 if mode != "pub" else 150),
    )
    draw.text(
        (x0 + pad, y0 + 10),
        "Approx corridors (illustrative)",
        fill=(255, 255, 255, 235), font=font,
    )

    for i, (label, key) in enumerate(items):
        cy = y0 + 42 + i * 34
        sp = species[key]
        col = sp.color_pub if mode == "pub" else sp.color
        draw.rounded_rectangle(
            [x0 + pad, cy + 6, x0 + pad + 18, cy + 24],
            radius=4, fill=(*col, 235),
        )
        draw.text((x0 + pad + 28, cy), label, fill=(255, 255, 255, 220), font=font)


# ── Core renderer ─────────────────────────────────────────────────

def render_corridors(
    base_img: Image.Image,
    view: MapView,
    cfg: SiteConfig,
    month_name: str,
    intensity: float,
    period_label: str,
    mode: str,
    rng: np.random.Generator,
) -> Image.Image:
    """Render migration corridors onto a base satellite image for one month."""
    img = base_img.convert("RGBA")
    w, h = img.size

    is_winter = month_name in cfg.winter_month_names
    intensity_eff = intensity * 0.18 if is_winter else intensity

    seasonal_shift = rng.normal(0.0, 18.0)

    main = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dens = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    d_main = ImageDraw.Draw(main, "RGBA")
    d_glow = ImageDraw.Draw(glow, "RGBA")
    d_dens = ImageDraw.Draw(dens, "RGBA")

    turbines = view.turbine_pixels

    # Migratory species corridors
    for key, sp in cfg.species.items():
        if key == "local":
            continue
        if key not in view.corridor_endpoints:
            continue

        ep = view.corridor_endpoints[key]
        if ep.p0 is None or ep.p3 is None:
            continue

        p0 = (float(ep.p0[0]), float(ep.p0[1]))
        p3 = (float(ep.p3[0]), float(ep.p3[1]))
        curv = ep.curv

        shift_factor = 0.35 if key == "raptors" else 0.6
        P0, P1, P2, P3 = build_curved_corridor(
            p0, p3, curv, seasonal_shift * shift_factor,
        )

        sigma = sp.sigma * (0.55 + 0.95 * intensity_eff)
        n_arrows = int(round(sp.arrow_count * (intensity_eff ** 1.35)))
        if is_winter:
            n_arrows = max(2, n_arrows)
        else:
            n_arrows = max(6, n_arrows)

        if mode == "pub":
            color = sp.color_pub
            width = max(2, int(round(sp.arrow_width * (0.65 + 0.35 * intensity_eff))))
            alpha = int(round(130 + 70 * intensity_eff))
            arrow_len = sp.arrow_length * (0.85 + 0.20 * intensity_eff)
        else:
            color = sp.color
            width = max(3, int(round(sp.arrow_width * (0.70 + 0.55 * intensity_eff))))
            alpha = int(round(sp.alpha * (0.65 + 0.35 * intensity_eff)))
            arrow_len = sp.arrow_length * (0.85 + 0.25 * intensity_eff)

        for _ in range(n_arrows):
            t = float(rng.uniform(0.06, 0.94))
            pos = bezier(P0, P1, P2, P3, t)
            deriv = bezier_deriv(P0, P1, P2, P3, t)
            dvec = vec_norm(deriv)
            nvec = vec_norm(vec_perp(dvec))

            lateral = float(rng.normal(0.0, sigma))
            along = float(rng.normal(0.0, sigma * 0.18))
            pos2 = vec_add(pos, vec_add(vec_mul(nvec, lateral), vec_mul(dvec, along)))

            pos3, dvec2 = turbine_deflect(pos2, dvec, turbines, intensity_eff, rng)

            if mode == "cinematic":
                _draw_arrow(d_dens, pos3, dvec2, arrow_len * 1.05, width + 10, color, int(alpha * 0.30))
                _draw_arrow(d_glow, pos3, dvec2, arrow_len * 1.00, width + 6, color, int(alpha * 0.40))

            _draw_arrow(d_main, pos3, dvec2, arrow_len, width, color, alpha)

    # Local/resident birds
    if "local" in cfg.species and "local" in view.corridor_endpoints:
        local_sp = cfg.species["local"]
        local_color = local_sp.color_pub if mode == "pub" else local_sp.color
        local_alpha = 130 if mode == "pub" else 140
        local_width = 3 if mode == "pub" else 4
        local_len = 28 if mode == "pub" else 30
        local_count = max(8, len(turbines) * 2)

        for _ in range(local_count):
            tx, ty = turbines[int(rng.integers(0, len(turbines)))]
            ang = float(rng.uniform(0, 2 * math.pi))
            rad = float(abs(rng.normal(0, 70)))
            pos = (tx + math.cos(ang) * rad, ty + math.sin(ang) * rad)
            dvec = (math.cos(ang + rng.normal(0, 0.6)), math.sin(ang + rng.normal(0, 0.6)))

            if mode == "cinematic":
                _draw_arrow(d_glow, pos, dvec, local_len, local_width + 4, local_color, int(local_alpha * 0.35))
            _draw_arrow(d_main, pos, dvec, local_len, local_width, local_color, local_alpha)

    # Post-processing by mode
    if mode == "cinematic":
        glow_blur = glow.filter(ImageFilter.GaussianBlur(radius=10))
        dens_blur = dens.filter(ImageFilter.GaussianBlur(radius=12))

        def motion_smear(layer: Image.Image, shifts: int = 6, step: int = 2) -> Image.Image:
            acc = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            for i in range(shifts):
                dx = int(i * step)
                dy = int(i * step * 0.3)
                tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                tmp.paste(layer, (dx, dy))
                acc = Image.alpha_composite(acc, tmp)
            return acc.filter(ImageFilter.GaussianBlur(radius=1.2))

        smear = motion_smear(main, shifts=7, step=2)
        img = Image.alpha_composite(img, dens_blur)
        img = Image.alpha_composite(img, glow_blur)
        img = Image.alpha_composite(img, smear)
        img = Image.alpha_composite(img, main)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))

    elif mode == "eco":
        soft = main.filter(ImageFilter.GaussianBlur(radius=1.2))
        img = Image.alpha_composite(img, soft)
        img = Image.alpha_composite(img, main)

    else:  # pub
        img = Image.alpha_composite(img, main)

    # Overlay legend and labels
    _add_legend(img, cfg.species, mode)
    title = f"{cfg.site_name} ({view.key}) — {month_name} | migration intensity={intensity:.2f}"
    _add_label_panel(img, title, period_label, mode)

    return img


# ── Entry point ───────────────────────────────────────────────────

def generate_monthly_maps(
    cfg: SiteConfig,
    mode: str = "eco",
    seed: int = 42,
    out_dir: str = "outputs/monthly-annotated-maps",
):
    """
    Generate annotated corridor maps for every month x every map view.

    This is the Phase 1 pipeline entry point.
    """
    os.makedirs(out_dir, exist_ok=True)
    base_dir = cfg.config_dir

    for view_key, view in cfg.maps.items():
        img_path = os.path.join(base_dir, view.base_image)
        if not os.path.isabs(view.base_image):
            # Also try relative to project root (one level up from configs/)
            if not os.path.exists(img_path):
                img_path = os.path.join(base_dir, "..", view.base_image)

        if not os.path.exists(img_path):
            print(f"WARNING: Base image not found: {img_path} — skipping view '{view_key}'")
            continue

        base_img = Image.open(img_path).convert("RGBA")

        for idx, cal in enumerate(cfg.monthly_calendar, start=1):
            month_rng = np.random.default_rng(seed * 100 + idx)

            rendered = render_corridors(
                base_img=base_img,
                view=view,
                cfg=cfg,
                month_name=cal.month,
                intensity=cal.intensity,
                period_label=cal.label,
                mode=mode,
                rng=month_rng,
            )

            out_path = os.path.join(out_dir, f"{view_key}_{idx:02d}_{cal.month}_{mode}.png")
            rendered.save(out_path)
            print(f"Saved: {out_path}")
