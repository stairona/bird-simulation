# Bird Collision Simulation Toolkit

A config-driven Python toolkit for wind energy researchers and ecologists to simulate bird collision risk, visualize migration corridors, and produce mortality analysis — **for any wind farm site**.

## How It Works

The toolkit follows a two-phase pipeline:

```
Phase 1: MIGRATORY PATHS                Phase 2: MORTALITY ANALYSIS
┌─────────────────────────┐              ┌────────────────────────────┐
│  Your site_config.yaml  │─────────────▶│  Uses corridors + layout   │
│                         │              │  from the same config      │
│  Renders monthly maps   │              │                            │
│  with corridors on your │              │  Statistical (Poisson) and │
│  satellite images       │              │  agent-based simulations   │
│                         │              │                            │
│  Outputs:               │              │  Outputs:                  │
│  - 24+ annotated PNGs   │              │  - Mortality CSV           │
│  - Collages (grid, PPT) │              │  - Monthly bar charts      │
└─────────────────────────┘              │  - Heatmaps + layout plots │
                                         └────────────────────────────┘
```

## Typical Outputs

- Annotated corridor PNGs (monthly)
- Mortality CSV summaries
- Comparison charts and heatmaps

## Quick Start

```bash
# Clone and install
git clone https://github.com/stairona/bird-simulation.git
cd bird-simulation
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline with the included Isabella Wind example
python -m src.cli all --config configs/isabella.yaml --mode eco
```

## Using Your Own Site

### Option A: From a Turbine CSV (Recommended for researchers)

If you have turbine positions as lat/lon coordinates in a CSV:

```bash
# Auto-generate a complete config from your turbine CSV
python -m src.cli generate --turbines my_turbines.csv --region atlantic

# Run the full pipeline
python -m src.cli all --config configs/my_turbines.yaml
```

Available flyway regions: `atlantic`, `mississippi`, `central`, `pacific`, `western_palearctic`, `east_asian`

The CSV should have `latitude`/`longitude` (or `lat`/`lon`) columns:
```csv
latitude,longitude
43.745,-84.701
43.748,-84.695
```

### Option B: From the template

```bash
python -m src.cli init --name "My Wind Farm"
# → creates configs/my_wind_farm.yaml
```

Open `configs/my_wind_farm.yaml` and fill in:

- **Turbine layout** — how many turbines, where they cluster
- **Migration corridors** — direction, width, curvature of flyways
- **Species** — which bird groups use each corridor
- **Monthly calendar** — migration intensity by month for your region
- **Map views** — your satellite images + pixel-space corridor geometry
- **Simulation parameters** — collision model tuning knobs

See `configs/isabella.yaml` for a complete worked example.

### Run the pipeline

```bash
# Phase 1 only: corridor maps
python -m src.cli paths --config configs/my_wind_farm.yaml --mode eco

# Phase 1 collages
python -m src.cli collages --config configs/my_wind_farm.yaml

# Phase 2 only: statistical mortality simulation
python -m src.cli mortality --config configs/my_wind_farm.yaml

# Phase 2: agent-based simulation
python -m src.cli agent --config configs/my_wind_farm.yaml

# Everything at once
python -m src.cli all --config configs/my_wind_farm.yaml
```

## Scenario Comparison & Sensitivity Analysis

### Compare multiple scenarios

Run mortality simulations side by side for different configs:

```bash
python -m src.cli compare configs/scenario_a.yaml configs/scenario_b.yaml \
  --names "Low avoidance,High avoidance"
```

Outputs: comparison bar chart, monthly breakdown, summary CSV.

### Sensitivity sweep

Vary a single parameter to see how it affects mortality:

```bash
# Sweep avoidance rate from 0% to 95%
python -m src.cli sweep --config configs/isabella.yaml --param avoidance --steps 10

# Sweep with specific values
python -m src.cli sweep --config configs/isabella.yaml --param base_rate \
  --values "0.5,0.75,1.0,1.5,2.0"

# List available sweep parameters
python -m src.cli sweep --config configs/isabella.yaml --param list
```

Sweepable parameters: `avoidance`, `base_rate`, `turbine_count`, `base_strike_prob`

## CLI Reference

| Command | Description |
|---------|-------------|
| `paths` | Phase 1 — Generate monthly corridor maps for each map view |
| `collages` | Assemble monthly maps into grid collages (full/half/ppt) |
| `mortality` | Phase 2 — Statistical Poisson mortality simulation + charts |
| `agent` | Phase 2 — Agent-based collision simulation + plots |
| `all` | Run full pipeline (Phase 1 + Phase 2) |
| `init` | Create a new site config from the template |
| `generate` | Auto-generate config from turbine CSV + flyway region preset |
| `compare` | Compare mortality across multiple config scenarios |
| `sweep` | Sensitivity sweep of a simulation parameter |

Common flags:
- `--config PATH` — site YAML config (required for all except `init` and `generate`)
- `--mode eco|cinematic|pub` — rendering style for corridor maps
- `--out DIR` — override output directory

## Mathematical Models

All models are site-agnostic and live in `src/core/`:

| Model | File | Description |
|-------|------|-------------|
| Gaussian corridor dispersion | `corridors.py` | Lateral spread from centerline with adjustable sigma |
| Bezier curve corridors | `corridors.py` | Smooth flyway bending with curvature parameter |
| 2D density field | `corridors.py` | Rotated Gaussian cross-sections + hotspot blobs |
| K-NN avoidance factor | `turbines.py` | Sigmoid-based clustering penalty from nearest neighbors |
| Turbine deflection | `turbines.py` | Birds deflect position/heading near turbine clusters |
| Multi-factor collision | `collision.py` | strike_prob × weather × night × altitude × (1 − avoidance) |
| Poisson mortality | `simulate.py` | λ = base_rate × migration × density × (1 − avoid) × season × heterogeneity |
| Agent-based movement | `agent_sim.py` | Migrants follow corridors; residents random-walk; step-by-step proximity checks |

## Project Structure

```
bird-simulation/
├── configs/
│   ├── isabella.yaml              # Worked example (Isabella Wind, Michigan)
│   └── example_template.yaml      # Blank template for new sites
├── src/
│   ├── cli.py                     # Unified command-line interface
│   ├── __main__.py                # python -m src entry point
│   ├── core/                      # Site-agnostic mathematical models
│   │   ├── config.py              # YAML loader + dataclasses
│   │   ├── corridors.py           # Gaussian dispersion, Bezier, density fields
│   │   ├── turbines.py            # Layout, K-NN avoidance, deflection
│   │   ├── collision.py           # Multi-factor collision probability
│   │   ├── calendar.py            # Monthly/seasonal calendar utilities
│   │   ├── geo.py                 # Lat/lon → normalized [0,1] projection
│   │   ├── flyways.py             # Flyway presets (6 major migration routes)
│   │   └── tiles.py               # Auto-fetch satellite map tiles
│   ├── tools/                     # Researcher workflow tools
│   │   ├── generate_config.py     # Auto-generate config from turbine CSV
│   │   ├── compare.py             # Scenario comparison (side-by-side)
│   │   └── sweep.py               # Sensitivity parameter sweeps
│   ├── phase1_paths/              # Phase 1: corridor visualization
│   │   ├── annotate_months.py     # Monthly map renderer
│   │   └── collage.py             # Grid collage assembly (full/half/ppt)
│   └── phase2_mortality/          # Phase 2: mortality simulation
│       ├── simulate.py            # Statistical (Poisson) model
│       ├── agent_sim.py           # Agent-based model
│       └── charts.py              # Bar charts, heatmaps, layout plots
├── tests/                         # pytest test suite (125 tests)
├── data/                          # Base satellite images go here
├── outputs/                       # Generated files (gitignored)
├── requirements.txt
└── README.md
```

## Rendering Modes

Phase 1 corridor maps support three visual styles:

- **eco** — Gaussian dispersion + turbine deflection for ecological realism
- **cinematic** — Glow effects, motion blur, density blending for presentations
- **pub** — Clean, crisp, restrained styling for academic papers

## Config Format

The YAML config has these sections:

```yaml
site:               # Name and region
turbines:           # Count, clusters, layout seed
corridors:          # Flyway definitions (angle, sigma, curvature, species)
density_blobs:      # Optional hotspots (roosting, water)
species:            # Species groups with visual parameters
monthly_calendar:   # 12-month migration intensity + labels
seasons:            # Season definitions for agent sim
maps:               # Satellite images + pixel-space geometry
simulation:         # Collision model tuning knobs
```

## Dependencies

- Python 3.8+
- Pillow (image rendering)
- NumPy (numerical models)
- Matplotlib (chart generation)
- pandas (optional analysis)
- PyYAML (config loading)
- contextily (optional — auto-fetch satellite map tiles)

## License

MIT
