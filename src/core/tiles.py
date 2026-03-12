"""
Map tile fetching for auto-generated base images.

Uses contextily to download satellite or street map tiles for a given
lat/lon bounding box, producing a base PNG that Phase 1 can render onto.
Requires the optional ``contextily`` dependency.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from .geo import BoundingBox


def _check_contextily():
    try:
        import contextily  # noqa: F401
        return contextily
    except ImportError:
        raise ImportError(
            "contextily is required for map tile fetching. "
            "Install it with: pip install contextily"
        )


def fetch_basemap(
    bbox: BoundingBox,
    out_path: str,
    style: str = "satellite",
    zoom: Optional[int] = None,
    dpi: int = 150,
    figsize: Tuple[float, float] = (16, 12),
) -> Tuple[str, int, int]:
    """
    Fetch map tiles for a bounding box and save as PNG.

    Args:
        bbox: geographic bounding box
        out_path: where to save the PNG
        style: "satellite", "street", or "topo"
        zoom: tile zoom level (None = auto)
        dpi: output resolution
        figsize: matplotlib figure size in inches

    Returns:
        (out_path, width_px, height_px) of the saved image
    """
    ctx = _check_contextily()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    providers = {
        "satellite": ctx.providers.Esri.WorldImagery,
        "street": ctx.providers.OpenStreetMap.Mapnik,
        "topo": ctx.providers.OpenTopoMap,
    }
    source = providers.get(style, ctx.providers.Esri.WorldImagery)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.set_xlim(bbox.lon_min, bbox.lon_max)
    ax.set_ylim(bbox.lat_min, bbox.lat_max)

    zoom_kw = {"zoom": zoom} if zoom is not None else {"zoom": "auto"}
    ctx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=source,
        **zoom_kw,
    )

    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)

    from PIL import Image
    with Image.open(out_path) as img:
        w, h = img.size

    return out_path, w, h
