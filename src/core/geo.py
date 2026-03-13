"""
Geographic projection utilities.

Projects lat/lon coordinates into the normalized [0, 1] space used by the
simulation core. Uses equirectangular approximation, which is accurate enough
for the <100 km scales typical of a wind farm site.

Also provides `load_turbine_csv` for reading lat/lon positions from CSV files.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class BoundingBox:
    """Lat/lon bounding box with padding."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.lat_min + self.lat_max) / 2,
            (self.lon_min + self.lon_max) / 2,
        )

    @property
    def lat_span(self) -> float:
        return self.lat_max - self.lat_min

    @property
    def lon_span(self) -> float:
        return self.lon_max - self.lon_min


def bounding_box(
    lats: np.ndarray, lons: np.ndarray, pad_fraction: float = 0.15
) -> BoundingBox:
    """
    Compute a padded bounding box around a set of lat/lon points.

    pad_fraction adds relative padding on each side so turbines
    don't sit right at the image edges.
    """
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())

    lat_pad = max((lat_max - lat_min) * pad_fraction, 0.005)
    lon_pad = max((lon_max - lon_min) * pad_fraction, 0.005)

    return BoundingBox(
        lat_min=lat_min - lat_pad,
        lat_max=lat_max + lat_pad,
        lon_min=lon_min - lon_pad,
        lon_max=lon_max + lon_pad,
    )


def latlon_to_normalized(
    lats: np.ndarray, lons: np.ndarray, bbox: BoundingBox
) -> np.ndarray:
    """
    Project lat/lon points into [0, 1] normalized space.

    Uses equirectangular projection with cos(center_lat) correction for
    longitude scaling. Returns (N, 2) array where column 0 = x, column 1 = y.
    Latitude is flipped so that north is up (y=0 is top of image).
    """
    center_lat = bbox.center[0]
    cos_lat = math.cos(math.radians(center_lat))

    x = (lons - bbox.lon_min) / (bbox.lon_span + 1e-12) * cos_lat
    y = 1.0 - (lats - bbox.lat_min) / (bbox.lat_span + 1e-12)

    # Re-normalize to [0, 1] after cos correction
    x_min, x_max = x.min(), x.max()
    if x_max - x_min > 1e-12:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.full_like(x, 0.5)

    y_min, y_max = y.min(), y.max()
    if y_max - y_min > 1e-12:
        y = (y - y_min) / (y_max - y_min)
    else:
        y = np.full_like(y, 0.5)

    return np.column_stack([x, y])


def normalized_to_pixels(
    xy: np.ndarray, img_width: int, img_height: int
) -> List[Tuple[int, int]]:
    """Convert normalized [0, 1] positions to pixel coordinates."""
    px = (xy[:, 0] * (img_width - 1)).astype(int)
    py = (xy[:, 1] * (img_height - 1)).astype(int)
    return list(zip(px.tolist(), py.tolist()))


def load_turbine_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load turbine positions from a CSV with columns ``latitude,longitude``
    (or ``lat,lon``). Returns (lats, lons) arrays.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Turbine CSV not found: {path}")

    lats, lons = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        lat_col = next((c for c in reader.fieldnames or []
                        if c.strip().lower() in ("latitude", "lat")), None)
        lon_col = next((c for c in reader.fieldnames or []
                        if c.strip().lower() in ("longitude", "lon", "lng")), None)

        if lat_col is None or lon_col is None:
            raise ValueError(
                f"CSV must have latitude/lat and longitude/lon/lng columns. "
                f"Found: {reader.fieldnames}"
            )

        for row_num, row in enumerate(reader, start=2):
            lat_val = row[lat_col].strip()
            lon_val = row[lon_col].strip()
            if not lat_val or not lon_val:
                continue
            try:
                lats.append(float(lat_val))
                lons.append(float(lon_val))
            except ValueError as e:
                raise ValueError(
                    f"Invalid number in CSV row {row_num}: "
                    f"lat={lat_val!r}, lon={lon_val!r}"
                ) from e

    if len(lats) == 0:
        raise ValueError(f"No valid rows found in {path}")

    return np.array(lats), np.array(lons)
