"""
Auto-generate a site YAML config from a turbine CSV and flyway preset.

Workflow:
  1. Read turbine lat/lon positions from CSV
  2. Compute bounding box and project to normalized [0,1] space
  3. Derive turbine clusters via simple k-means
  4. Pull corridor/species/calendar from flyway preset
  5. Optionally fetch a satellite base map (requires contextily)
  6. Write complete YAML config ready for the pipeline
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from ..core.flyways import SEASON_DEFAULTS, get_flyway
from ..core.geo import BoundingBox, bounding_box, latlon_to_normalized, load_turbine_csv


def _derive_clusters(
    xy: np.ndarray, max_clusters: int = 4, seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Derive turbine clusters from normalized positions using a simple
    iterative k-means. Returns cluster defs suitable for YAML output.
    """
    n = len(xy)
    k = min(max_clusters, max(1, n // 10))
    if k <= 1:
        cx, cy = float(xy[:, 0].mean()), float(xy[:, 1].mean())
        sx = max(float(xy[:, 0].std()), 0.05)
        sy = max(float(xy[:, 1].std()), 0.05)
        return [{"center": [round(cx, 3), round(cy, 3)],
                 "spread": [round(sx, 3), round(sy, 3)],
                 "fraction": 1.0}]

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False)
    centers = xy[indices].copy()

    for _ in range(30):
        dists = np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        new_centers = np.array([
            xy[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
            for j in range(k)
        ])
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers

    clusters = []
    for j in range(k):
        mask = labels == j
        count = mask.sum()
        if count == 0:
            continue
        pts = xy[mask]
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        sx = max(float(pts[:, 0].std()), 0.03)
        sy = max(float(pts[:, 1].std()), 0.03)
        frac = round(count / n, 2)
        clusters.append({
            "center": [round(cx, 3), round(cy, 3)],
            "spread": [round(sx, 3), round(sy, 3)],
            "fraction": frac,
        })

    return clusters


def generate_config(
    turbine_csv: str,
    region: str,
    site_name: Optional[str] = None,
    output_path: Optional[str] = None,
    fetch_map: bool = False,
    map_style: str = "satellite",
    seed: int = 42,
) -> str:
    """
    Generate a complete YAML config from turbine CSV + flyway region.

    Args:
        turbine_csv: path to CSV with lat/lon columns
        region: flyway preset name (atlantic, mississippi, central, pacific, etc.)
        site_name: display name (defaults to CSV filename)
        output_path: where to write the YAML
        fetch_map: if True, download a satellite base map
        map_style: tile style for base map (satellite, street, topo)
        seed: random seed for layout and simulation

    Returns:
        Path to the generated YAML file.
    """
    flyway = get_flyway(region)
    lats, lons = load_turbine_csv(turbine_csv)
    n_turbines = len(lats)
    bbox = bounding_box(lats, lons)
    xy = latlon_to_normalized(lats, lons, bbox)

    if site_name is None:
        site_name = os.path.splitext(os.path.basename(turbine_csv))[0].replace("_", " ").title()

    clusters = _derive_clusters(xy, seed=seed)

    # Build corridor defs with centers at the centroid of turbines
    cx_norm = float(xy[:, 0].mean())
    cy_norm = float(xy[:, 1].mean())
    corridors = []
    for corr in flyway["corridors"]:
        corridors.append({
            **corr,
            "center": [round(cx_norm, 3), round(cy_norm, 3)],
        })

    config: Dict[str, Any] = {
        "site": {"name": site_name, "region": flyway["label"]},
        "turbines": {
            "count": n_turbines,
            "layout_seed": seed,
            "clusters": clusters,
        },
        "corridors": corridors,
        "density_blobs": [],
        "species": flyway["species"],
        "monthly_calendar": flyway["monthly_calendar"],
        "seasons": SEASON_DEFAULTS,
        "maps": {},
        "simulation": {
            "seed": seed,
            "base_rate": 0.75,
            "winter_suppression": 0.35,
            "agent": {
                "birds_per_day_base": max(200, n_turbines * 4),
                "migrant_speed": 2.4,
                "resident_speed": 1.2,
                "steps_per_day": 18,
                "world_size": [100.0, 100.0],
            },
            "collision": {
                "rotor_radius": 2.2,
                "base_strike_prob": 0.0025,
                "avoidance": 0.80,
                "night_risk_mult": 1.5,
                "altitude_match_prob": 0.55,
            },
        },
    }

    if output_path is None:
        safe_name = site_name.lower().replace(" ", "_")
        output_path = f"configs/{safe_name}.yaml"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if fetch_map:
        try:
            from ..core.tiles import fetch_basemap
            map_dir = os.path.join(os.path.dirname(output_path), "..", "data")
            os.makedirs(map_dir, exist_ok=True)
            safe = site_name.lower().replace(" ", "_")
            map_path = os.path.join(map_dir, f"{safe}_base.png")
            _, w, h = fetch_basemap(bbox, map_path, style=map_style)
            from ..core.geo import normalized_to_pixels
            turbine_px = normalized_to_pixels(xy, w, h)
            config["maps"]["overview"] = {
                "base_image": os.path.relpath(map_path, os.path.dirname(output_path)),
                "turbine_pixels": turbine_px[:min(8, len(turbine_px))],
                "corridor_endpoints": _auto_corridor_endpoints(
                    corridors, flyway["species"], w, h
                ),
            }
            print(f"Fetched base map: {map_path} ({w}x{h})")
        except ImportError:
            print("contextily not installed — skipping map tile fetch")

    with open(output_path, "w") as f:
        f.write(f"# Auto-generated config for {site_name}\n")
        f.write(f"# Source: {os.path.basename(turbine_csv)}, region: {region}\n")
        f.write(f"# Turbines: {n_turbines}, Clusters: {len(clusters)}\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


def _auto_corridor_endpoints(
    corridors: List[Dict], species: Dict, img_w: int, img_h: int
) -> Dict[str, Any]:
    """
    Generate pixel-space corridor endpoints for the auto-generated map view.
    Places corridor lines spanning the image based on corridor angles.
    """
    import math
    endpoints = {}
    cx, cy = img_w // 2, img_h // 2
    radius = min(img_w, img_h) * 0.45

    for corr in corridors:
        angle = math.radians(corr["angle_deg"])
        dx = math.cos(angle) * radius
        dy = -math.sin(angle) * radius  # y-axis is inverted in pixel space

        for sp_key in corr.get("species", []):
            p0 = [int(cx - dx), int(cy + dy)]
            p3 = [int(cx + dx), int(cy - dy)]
            endpoints[sp_key] = {
                "p0": p0, "p3": p3,
                "curv": int(corr.get("curvature", 0) * 1000),
            }

    if "local" in species:
        endpoints["local"] = {
            "center": [cx, cy],
            "radius": int(radius * 0.4),
        }

    return endpoints
