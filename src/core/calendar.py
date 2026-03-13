"""
Monthly/seasonal calendar utilities.

Maps month numbers to season definitions and provides the migration
intensity array from config. Also houses shared constants used
across Phase 1 and Phase 2 modules.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import SeasonDef, SiteConfig

MONTH_NAMES: List[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

MONTH_LENGTHS: List[int] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def build_month_to_season(cfg: SiteConfig) -> Dict[int, SeasonDef]:
    """Map month numbers (1-12) to their SeasonDef from config."""
    mapping = {}
    for season in cfg.seasons.values():
        for m in season.months:
            mapping[m] = season
    return mapping


def migration_index_array(cfg: SiteConfig) -> np.ndarray:
    """Return 12-element array of migration_index values from the calendar."""
    return np.array(
        [entry.migration_index for entry in cfg.monthly_calendar],
        dtype=float,
    )
