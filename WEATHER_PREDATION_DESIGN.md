# Weather & Predation Layer Design

## Overview

This document outlines the design for a comprehensive Weather & Predation logic layer to enhance the bird collision simulation toolkit. The current simulation includes a basic `weather_risk` multiplier in `SeasonDef`. This new layer adds:

1. **Granular Weather Modeling**: Daily weather conditions affecting bird flight behavior, visibility, and turbine operations.
2. **Predation Dynamics**: Bird mortality from predators (raptors, ground predators) independent of turbine collisions.

Both layers integrate with the existing agent-based simulation (`phase2_mortality/agent_sim.py`) and statistical models (`simulate.py`).

---

## 1. Weather Layer

### 1.1 Simulation Variables

#### Weather Conditions (daily resolution)
```python
class WeatherCondition:
    """Enum-like class for weather categories."""
    CLEAR = "clear"
    FOG = "fog"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    SNOW = "snow"
    STORM = "storm"  # high winds, lightning
    ICE = "ice"      # icing conditions
```

#### Weather State per Day
```python
@dataclass
class DailyWeather:
    condition: WeatherCondition
    visibility_km: float          # 0.5 (dense fog) to 50+ (clear)
    wind_speed_ms: float          # affects turbine cut-in/out
    wind_direction_deg: float     # downwind vs crosswind
    precipitation_mm: float       # rain/snow amount
    temperature_c: float          # affects bird metabolism, insect activity
    barometric_pressure_hpa: float  # pressure trends affect migration
    turbulence_intensity: float   # 0-1 scale, affects flight stability
```

#### Weather Impact Modifiers
```python
class WeatherImpact:
    """Multipliers derived from daily weather that modify simulation parameters."""

    # Bird behavior modifiers
    flight_activity_mult: float    # 0.2 (severe storm) to 1.2 (optimal)
    migration_delay_prob: float    # probability birds postpone migration
    altitude_shift_meters: float   # birds fly lower/higher in certain weather

    # Collision modifiers
    visibility_collision_mult: float  # fog may increase or decrease collision risk
    # (increased: birds avoid turbines less; decreased: birds ground themselves)

    turbine_operational_mult: float   # 0 (turbines idled) to 1 (fully operational)
    # Wind speed thresholds:
    # - Below cut-in (~3 m/s): turbines off → mult = 0
    # - Between cut-in and rated: mult = 1
    # - Above cut-out (~25 m/s): turbines feathered → mult = 0
    # - In high winds (>20 m/s): increased risk due to blade speed? → mult = 1.2

    # Turbine nacelle yaw may not align perfectly with crosswinds → additional factor
```

### 1.2 Configuration Additions

Add to `config.py`:

```python
@dataclass
class WeatherConfig:
    """Top-level weather configuration for the site."""

    # Climate zone parameters (informed by region)
    climate_zone: str  # e.g., "great_lakes", "coastal", "mountain", "arid"

    # Monthly baseline weather statistics (used to generate daily weather)
    # For each month, define:
    monthly_weather: Dict[int, MonthWeatherStats]
    # Each MonthWeatherStats:
    #   - condition_probabilities: Dict[WeatherCondition, float] (must sum to 1)
    #   - mean_visibility_km: float
    #   - visibility_stddev_km: float
    #   - mean_wind_speed_ms: float
    #   - wind_speed_stddev: float
    #   - prevailing_wind_direction_deg: float  # for wind-roses
    #   - mean_precipitation_mm: float
    #   - probability_of_fog: float  (0-1)
    #   - probability_of_storm: float (0-1)

    # Optional: Base weather pattern for entire year (El Niño/La Niña modifiers)
    annual_pattern: str = "normal"  # "dry", "wet", "normal", "anomalous"
```

Add to YAML config:

```yaml
weather:
  climate_zone: "great_lakes"
  monthly_weather:
    1:  # January
      condition_probs:
        clear: 0.30
        fog: 0.20
        light_rain: 0.20
        snow: 0.25
        heavy_rain: 0.03
        storm: 0.02
      mean_visibility_km: 8.0
      visibility_stddev_km: 3.0
      mean_wind_speed_ms: 8.5
      wind_speed_stddev: 2.5
      prevailing_wind_direction_deg: 270  # westerly
      mean_precipitation_mm: 2.5
      probability_of_fog: 0.25
      probability_of_storm: 0.05
    # ... repeat for months 2-12
```

### 1.3 New Module: `src/core/weather.py`

This module will:

1. **Generate Daily Weather**: For a given day of the year, sample a `DailyWeather` instance based on monthly statistics and temporal coherence (weather tends to persist for days).
   - Use a Markov chain for condition transitions (clear→fog→rain, etc.)
   - Correlate visibility with condition and precipitation
   - Correlate wind speed with direction (e.g., gusts)

2. **Compute Impact Modifiers**: Convert weather to multipliers that affect:
   - `flight_activity_mult`: reduces number of birds spawned on bad weather days
   - `migration_delay_prob`: may skip migration entirely for that day in a region
   - `altitude_shift_meters`: birds fly lower in fog? higher in clear?
   - `visibility_collision_mult`: tension between birds being grounded vs. not seeing turbines
   - `turbine_operational_mult`: cut-in/cut-out logic

3. **Integration Hooks**:
   - In `agent_sim.py`, fetch weather for current day before spawning birds.
   - Modify `birds_today` using `flight_activity_mult`.
   - Pass `weather_impact` to `per_step_collision_prob` as an additional factor (or modify `season.weather_risk` to be dynamically multiplied by daily weather).
   - Influence turbine availability: if operational_mult=0, collisions impossible that day (turbines off).

### 1.4 Integration Points

**File: `src/phase2_mortality/agent_sim.py`**

Modify simulation loop:

```python
from ..core.weather import WeatherSimulator

def simulate_agent(cfg: SiteConfig) -> Tuple[...]:
    # ...
    weather_sim = WeatherSimulator(cfg.weather)  # initialize with config

    for day in range(DAYS):
        # Get today's weather
        daily_weather = weather_sim.get_weather(day_of_year=...)  # 1-365

        # Apply weather impact
        weather_impact = daily_weather.compute_impact()

        # Modify bird count using flight_activity_mult
        base_birds = agent.birds_per_day_base * (0.35 + 0.65 * season.migration_intensity)
        birds_today = int(base_birds * weather_impact.flight_activity_mult * rng.uniform(0.85, 1.15))

        # Later, in per-step collision:
        p_base = col.base_strike_prob
        p_base *= season.weather_risk  # existing seasonal multiplier
        p_base *= weather_impact.visibility_collision_mult  # new daily modifier
        # Also consider turbine operational status:
        if weather_impact.turbine_operational_mult == 0:
            # Turbines idle, no collisions possible
            p_col = 0.0
        else:
            p_col *= weather_impact.turbine_operational_mult
        # Clip p_col as before
```

**File: `src/core/collision.py`**

Optionally extend `per_step_collision_prob` to accept a `weather_impact` parameter and compute the combined probability there, keeping the collision model pure.

---

## 2. Predation Layer

### 2.1 Simulation Variables

#### Predator Types
```python
class PredatorType:
    RAPTOR = "raptor"       # hawks, eagles (attack from above)
    GROUND_PREDATOR = "ground"  # mammals (foxes, cats) - only affect grounded birds
    AVIAN_PREDATOR = "avian"    # larger birds (owl, falcon) - nocturnal/diurnal
```

#### Predator Behavior
```python
@dataclass
class PredatorProfile:
    """Behavioral parameters for a predator category."""
    type: PredatorType
    activity_pattern: str  # "diurnal", "nocturnal", "crepuscular", "cathemeral"
    hunt_radius_meters: float  # how far predators range from nests/perches
    attack_success_rate: float  # 0-1 probability per encounter
    prefer_low_altitude: bool   # if True, only hunt birds below X meters
    group_hunting: bool         # packs increase success rate
    seasonal_activity_mult: Dict[str, float]  # e.g., spring surge for raptors feeding chicks
```

#### Predation Event
```python
@dataclass
class PredationEvent:
    step: int
    bird_id: int
    predator_type: PredatorType
    location: Tuple[float, float]
    cause: str  # "raptor_attack", "ground_predator", etc.
```

### 2.2 Configuration Additions

Add to `config.py`:

```python
@dataclass
class PredationConfig:
    enabled: bool = False  # master switch
    predator_profiles: List[PredatorProfile] = field(default_factory=list)
    density_per_km2: float = 0.5  # average number of active predators per square km
    spawn_near_corridors: bool = True  # predators aware of bird flyways
    corridor_spawn_multiplier: float = 2.0  # higher density along corridors
```

Add to YAML config:

```yaml
predation:
  enabled: true
  density_per_km2: 0.3
  spawn_near_corridors: true
  corridor_spawn_multiplier: 2.5
  predator_profiles:
    - type: raptor
      activity_pattern: diurnal
      hunt_radius_meters: 500
      attack_success_rate: 0.15
      prefer_low_altitude: true
      group_hunting: false
      seasonal_activity_mult:
        spring: 1.5   # more activity nesting season
        summer: 1.3
        fall: 1.0
        winter: 0.2
    - type: ground
      activity_pattern: crepuscular
      hunt_radius_meters: 200
      attack_success_rate: 0.25
      prefer_low_altitude: true
      group_hunting: true
      seasonal_activity_mult:
        spring: 1.0
        summer: 1.2
        fall: 1.0
        winter: 0.5
```

### 2.3 New Module: `src/phase2_mortality/predation_sim.py` (or `src/core/predation.py`)

Responsibilities:

1. **Predator Spawning**: Each simulation day, spawn a Poisson number of predators based on `density_per_km2` and world area.
   - If `spawn_near_corridors=True`, bias spawn locations toward corridor regions (use corridor density map).
   - Assign a random predator type based on config profiles.

2. **Predator Movement**: Predators move differently from birds:
   - Raptors: soar, search patterns, perch at certain altitudes
   - Ground predators: stay on ground (y=0 in world space? or just low altitude)
   - Movement speed slower than birds, but they lurk.

3. **Encounter Check**: At each timestep, for each active bird not already dead:
   - Compute distance to nearby predators (spatial indexing for performance).
   - If predator within attack radius and predator active (based on diurnal/nocturnal pattern), apply attack.
   - Attack success: modify bird's `alive` status.
   - Record `PredationEvent`.

4. **Integration with Agent Sim**:
   - Extend the simulation loop in `agent_sim.py` to run predation alongside collision.
   - Options:
     a) Spawn predators once per day and have them persist throughout the day's steps.
     b) Treat predation as an independent check per step.

5. **Output**:
   - Add `daily_predation` array (365,) to outputs.
   - Extend summary statistics: total predation deaths, by predator type.
   - Modify plots (heatmap) to optionally show predation hotspots.

### 2.4 Data Flow

```
Main simulation loop (agent_sim.py):
  - Initialize PredationSimulator(cfg.predation, cfg.corridors, cfg.species)
  - For each day:
      weather = weather_sim.get_weather(day)
      birds_today_adjusted = base_birds * weather.flight_activity_mult
      spawn_birds()
      spawn_predators()  # daily batch or persistent

      For each step:
          For each bird:
              move()
              check_collision_with_turbines()   # existing
              check_predation()                 # NEW
      Record daily deaths (collision + predation)
```

---

## 3. API and CLI Extensions

### 3.1 Config Model Updates

In `src/core/config.py`:

```python
@dataclass
class WeatherConfig:
    climate_zone: str
    monthly_weather: Dict[int, MonthWeatherStats]
    annual_pattern: str = "normal"

@dataclass
class PredationConfig:
    enabled: bool
    predator_profiles: List[PredatorProfile]
    density_per_km2: float
    spawn_near_corridors: bool
    corridor_spawn_multiplier: float

@dataclass
class SimulationParams:
    # existing fields...
    weather: Optional[WeatherConfig] = None
    predation: Optional[PredationConfig] = None
```

Update `load_config()` to parse `weather:` and `predation:` sections from YAML, with sensible defaults (weather=None, predation with enabled=False) for backward compatibility.

### 3.2 CLI Flags

Extend `src/cli.py`:

```bash
# In agent command, add:
python -m src.cli agent --config config.yaml --enable-predation
python -m src.cli agent --config config.yaml --weather-realization 42  # deterministic weather

# Or better: config-driven, no flags needed. CLI just passes cfg to simulator.
```

### 3.3 Report Enhancements

- `simulate.py` statistical model: Currently uses `season.weather_risk` as a simple multiplier. Could also incorporate daily weather variability by running Monte Carlo with varying weather conditions drawn from distribution. Optional.

- Output CSV for agent simulation: add columns for `daily_predation`, `daily_collision`, `daily_total`.

---

## 4. Implementation Phases

### Phase 1: Weather Core (2-3 days)
1. Add `WeatherConfig` to `config.py`.
2. Create `src/core/weather.py` with:
   - `WeatherCondition` enum
   - `DailyWeather` dataclass
   - `WeatherImpact` computation
   - `WeatherSimulator` class that generates daily weather from monthly stats with Markov coherence.
3. Write unit tests: single-day sampling, coherence over sequences, visibility/wind correlations.
4. Update example config (`example_template.yaml`) with a weather section (using simplified stats).
5. Integrate into `agent_sim.py`: fetch daily weather, modify bird count, pass impact to collision model.

### Phase 2: Turbine Operational Model (1 day)
1. In `WeatherImpact.compute_turbine_operational_mult()`, implement wind-speed cut-in/cut-out thresholds (configurable or standard 3-25 m/s).
2. In collision calculation, skip if turbines not operational.
3. Test: simulate days with high wind → zero collisions.

### Phase 3: Predation Core (2-3 days)
1. Add `PredationConfig` and `PredatorProfile` to `config.py`.
2. Create `src/phase2_mortality/predation_sim.py`:
   - `PredationSimulator` class
   - `spawn_predators()`
   - `update_predators()` (movement)
   - `check_encounters(birds)` → returns death count
3. Add `src/phase2_mortality/agent_sim.py` integration.
4. Extend output arrays and plots to include predation.

### Phase 4: Validation & Documentation (1-2 days)
1. Run baseline scenarios with weather disabled vs enabled to see impact.
2. Document new YAML config fields in `README.md` and `configs/example_template.yaml`.
3. Add diagrams showing weather and predation data flow.
4. Ensure backward compatibility: old configs without weather/predation still work.

---

## 5. Testing Strategy

### Unit Tests
- `test_weather.py`: WeatherSimulator produces expected distributions, transition probabilities.
- `test_impact.py`: WeatherImpact modifiers within [0,2] range, sensible.
- `test_predation.py`: Predator movement, encounter detection boundaries.

### Integration Tests
- Short simulation (7 days) with mock weather/predation configs, verify non-zero weather-modified outputs and predation deaths.

### Regression
- Compare default simulation (without new layer) to ensure same results as before when features disabled.

---

## 6. Performance Considerations

- Weather simulation: O(365) days, negligible cost.
- Predation: adds O(P * B) checks per step where P = number of predators, B = birds. Could be expensive.
  - Mitigation: spatial hashing (grid cells) to only check nearby predators.
  - Or use approximate: only check a random sample of predators per bird if P large.
  - Typical predator density is low (<1 per km²), world space ~100x100 units → few predators.
- Expected overhead: ~10-20% increase in simulation time, acceptable.

---

## 7. Future Enhancements (Post-V1)

- **Acoustic masking**: Loud turbine noise affects bird communication, stress, predator detection.
- **Light pollution**: Turbine lighting affects nocturnal bird behavior.
- **Prey availability**: Insect abundance (linked to weather) influences bird presence.
- **Climate change scenarios**: Shift monthly weather statistics year-over-year.
- **Stochastic turbine downtime**: Random maintenance outages independent of weather.

---

## 8. Deliverables

- Modified `config.py` with new dataclasses.
- New file `src/core/weather.py` (complete implementation).
- New file `src/phase2_mortality/predation_sim.py` (complete implementation).
- Updated `agent_sim.py` and possibly `simulate.py`.
- Updated unit tests (new test files).
- Updated documentation: README, config examples.
- No breaking changes to existing YAML configs (backward compatible).

---

## Conclusion

This design provides a structured, modular extension to the existing simulation, keeping climate/ecological concerns first while maintaining computational feasibility. The two layers (weather and predation) are separable and can be independently enabled.
