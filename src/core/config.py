"""
Loads and validates a site YAML configuration into structured Python objects.

Everything site-specific lives in the YAML; the math models read from here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class ClusterDef:
    center: Tuple[float, float]
    spread: Tuple[float, float]
    fraction: float


@dataclass
class CorridorDef:
    name: str
    angle_deg: float
    sigma: float
    curvature: float
    species: List[str]
    center: Optional[Tuple[float, float]] = None
    center_x: Optional[float] = None
    weight_spring: float = 0.50
    weight_fall: float = 0.50
    weight_default: float = 0.50


@dataclass
class BlobDef:
    center: Tuple[float, float]
    spread: Tuple[float, float]
    weight: float


@dataclass
class SpeciesDef:
    key: str
    label: str
    color: Tuple[int, int, int]
    color_pub: Tuple[int, int, int]
    arrow_count: int = 40
    sigma: float = 14.0
    arrow_width: int = 5
    arrow_length: float = 56.0
    alpha: int = 150


@dataclass
class MonthEntry:
    month: str
    migration_index: float
    intensity: float
    label: str
    season: str


@dataclass
class SeasonDef:
    name: str
    months: List[int]
    migration_intensity: float
    resident_fraction: float
    night_fraction: float
    weather_risk: float


@dataclass
class MapCorridorEndpoints:
    """Pixel-space corridor geometry for one species on one map view."""
    p0: Optional[Tuple[int, int]] = None
    p3: Optional[Tuple[int, int]] = None
    curv: float = 0.0
    # For local/resident species: radial pattern instead of corridor
    center: Optional[Tuple[int, int]] = None
    radius: int = 250


@dataclass
class MapView:
    key: str
    base_image: str
    turbine_pixels: List[Tuple[int, int]]
    corridor_endpoints: Dict[str, MapCorridorEndpoints]


@dataclass
class CollisionParams:
    rotor_radius: float = 2.2
    base_strike_prob: float = 0.0025
    avoidance: float = 0.80
    night_risk_mult: float = 1.5
    altitude_match_prob: float = 0.55


@dataclass
class AgentParams:
    birds_per_day_base: int = 600
    migrant_speed: float = 2.4
    resident_speed: float = 1.2
    steps_per_day: int = 18
    world_size: Tuple[float, float] = (100.0, 100.0)


@dataclass
class SimulationParams:
    seed: int = 42
    base_rate: float = 0.75
    winter_suppression: float = 0.35
    heterogeneity_sigma: float = 0.30
    mortality_scaling: float = 0.90
    winter_cap: int = 1
    agent: AgentParams = field(default_factory=AgentParams)
    collision: CollisionParams = field(default_factory=CollisionParams)


@dataclass
class SiteConfig:
    """Top-level configuration for one wind farm site."""
    site_name: str
    region: str
    config_dir: str

    turbine_count: int
    layout_seed: int
    clusters: List[ClusterDef]

    corridors: List[CorridorDef]
    density_blobs: List[BlobDef]
    species: Dict[str, SpeciesDef]
    monthly_calendar: List[MonthEntry]
    seasons: Dict[str, SeasonDef]
    maps: Dict[str, MapView]
    simulation: SimulationParams

    # Optional: lat/lon turbine positions loaded from CSV
    turbine_csv: Optional[str] = None
    turbine_latlon: Optional[Any] = None  # np.ndarray (N, 2) of [lat, lon]

    @property
    def migratory_species_keys(self) -> List[str]:
        """Species that participate in corridor migration (everything except 'local')."""
        return [k for k in self.species if k != "local"]

    @property
    def winter_months(self) -> set:
        s = self.seasons.get("winter")
        if s is None:
            return set()
        return set(s.months)

    @property
    def winter_month_names(self) -> set:
        from .calendar import MONTH_NAMES
        return {MONTH_NAMES[m - 1] for m in self.winter_months}


def _tup2(raw) -> Tuple[float, float]:
    return (float(raw[0]), float(raw[1]))


def _tup2i(raw) -> Tuple[int, int]:
    return (int(raw[0]), int(raw[1]))


def _tup3i(raw) -> Tuple[int, int, int]:
    return (int(raw[0]), int(raw[1]), int(raw[2]))


def _parse_corridor_endpoint(raw: dict) -> MapCorridorEndpoints:
    ep = MapCorridorEndpoints()
    if "p0" in raw:
        ep.p0 = _tup2i(raw["p0"])
    if "p3" in raw:
        ep.p3 = _tup2i(raw["p3"])
    if "curv" in raw:
        ep.curv = float(raw["curv"])
    if "center" in raw:
        ep.center = _tup2i(raw["center"])
    if "radius" in raw:
        ep.radius = int(raw["radius"])
    return ep


def load_config(path: str) -> SiteConfig:
    """Load a site YAML and return a validated SiteConfig."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(path))

    site = raw.get("site", {})
    turb = raw.get("turbines", {})
    clusters = [
        ClusterDef(center=_tup2(c["center"]), spread=_tup2(c["spread"]), fraction=float(c["fraction"]))
        for c in turb.get("clusters", [])
    ]

    corridors = []
    for c in raw.get("corridors", []):
        sigma = float(c["sigma"])
        if sigma <= 0:
            raise ValueError(
                f"Corridor '{c['name']}' has sigma={sigma}; must be > 0"
            )
        corridors.append(CorridorDef(
            name=c["name"],
            angle_deg=float(c["angle_deg"]),
            sigma=sigma,
            curvature=float(c["curvature"]),
            species=c.get("species", []),
            center=_tup2(c["center"]) if "center" in c else None,
            center_x=float(c["center_x"]) if "center_x" in c else None,
            weight_spring=float(c.get("weight_spring", 0.5)),
            weight_fall=float(c.get("weight_fall", 0.5)),
            weight_default=float(c.get("weight_default", 0.5)),
        ))

    blobs = [
        BlobDef(center=_tup2(b["center"]), spread=_tup2(b["spread"]), weight=float(b["weight"]))
        for b in raw.get("density_blobs", []) or []
    ]

    species = {}
    for k, v in raw.get("species", {}).items():
        species[k] = SpeciesDef(
            key=k,
            label=v.get("label", k),
            color=_tup3i(v["color"]),
            color_pub=_tup3i(v.get("color_pub", v["color"])),
            arrow_count=int(v.get("arrow_count", 40)),
            sigma=float(v.get("sigma", 14.0)),
            arrow_width=int(v.get("arrow_width", 5)),
            arrow_length=float(v.get("arrow_length", 56.0)),
            alpha=int(v.get("alpha", 150)),
        )

    calendar = [
        MonthEntry(
            month=m["month"],
            migration_index=float(m["migration_index"]),
            intensity=float(m["intensity"]),
            label=m["label"],
            season=m["season"],
        )
        for m in raw.get("monthly_calendar", [])
    ]

    if len(calendar) != 12:
        raise ValueError(
            f"monthly_calendar must have exactly 12 entries, got {len(calendar)}. "
            f"Config: {path}"
        )

    seasons = {}
    for k, v in raw.get("seasons", {}).items():
        seasons[k] = SeasonDef(
            name=k,
            months=[int(m) for m in v["months"]],
            migration_intensity=float(v["migration_intensity"]),
            resident_fraction=float(v["resident_fraction"]),
            night_fraction=float(v["night_fraction"]),
            weather_risk=float(v["weather_risk"]),
        )

    maps = {}
    for k, v in raw.get("maps", {}).items():
        endpoints = {}
        for sk, sv in v.get("corridor_endpoints", {}).items():
            endpoints[sk] = _parse_corridor_endpoint(sv)
        maps[k] = MapView(
            key=k,
            base_image=v["base_image"],
            turbine_pixels=[_tup2i(t) for t in v.get("turbine_pixels", [])],
            corridor_endpoints=endpoints,
        )

    sim_raw = raw.get("simulation", {})
    agent_raw = sim_raw.get("agent", {})
    col_raw = sim_raw.get("collision", {})

    simulation = SimulationParams(
        seed=int(sim_raw.get("seed", 42)),
        base_rate=float(sim_raw.get("base_rate", 0.75)),
        winter_suppression=float(sim_raw.get("winter_suppression", 0.35)),
        heterogeneity_sigma=float(sim_raw.get("heterogeneity_sigma", 0.30)),
        mortality_scaling=float(sim_raw.get("mortality_scaling", 0.90)),
        winter_cap=int(sim_raw.get("winter_cap", 1)),
        agent=AgentParams(
            birds_per_day_base=int(agent_raw.get("birds_per_day_base", 600)),
            migrant_speed=float(agent_raw.get("migrant_speed", 2.4)),
            resident_speed=float(agent_raw.get("resident_speed", 1.2)),
            steps_per_day=int(agent_raw.get("steps_per_day", 18)),
            world_size=_tup2(agent_raw.get("world_size", [100.0, 100.0])),
        ),
        collision=CollisionParams(
            rotor_radius=float(col_raw.get("rotor_radius", 2.2)),
            base_strike_prob=float(col_raw.get("base_strike_prob", 0.0025)),
            avoidance=float(col_raw.get("avoidance", 0.80)),
            night_risk_mult=float(col_raw.get("night_risk_mult", 1.5)),
            altitude_match_prob=float(col_raw.get("altitude_match_prob", 0.55)),
        ),
    )

    # Optional turbine CSV with lat/lon positions
    turbine_csv_path = turb.get("csv", None)
    turbine_latlon = None
    if turbine_csv_path is not None:
        if not os.path.isabs(turbine_csv_path):
            turbine_csv_path = os.path.normpath(
                os.path.join(config_dir, turbine_csv_path)
            )
        if os.path.exists(turbine_csv_path):
            from .geo import load_turbine_csv
            import numpy as _np
            lats, lons = load_turbine_csv(turbine_csv_path)
            turbine_latlon = _np.column_stack([lats, lons])

    return SiteConfig(
        site_name=site.get("name", "Unnamed Site"),
        region=site.get("region", "Unknown"),
        config_dir=config_dir,
        turbine_count=int(turb.get("count", 50)),
        layout_seed=int(turb.get("layout_seed", 42)),
        clusters=clusters,
        corridors=corridors,
        density_blobs=blobs,
        species=species,
        monthly_calendar=calendar,
        seasons=seasons,
        maps=maps,
        simulation=simulation,
        turbine_csv=turbine_csv_path,
        turbine_latlon=turbine_latlon,
    )
