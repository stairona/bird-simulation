# Bird Mortality Visualization Suite

A collection of Python tools for wind energy researchers and ecologists to visualize bird collision risk, simulate mortality patterns, and create publication-ready maps and charts.

## What Problem Does This Solve?

Wind farms need to assess and communicate their impact on bird populations. This project provides a toolkit to:
- Generate realistic monthly bird corridor visualizations based on migration patterns
- Simulate mortality data for analysis and reporting
- Create professional collages and charts for presentations and publications
- Support both ecological realism and cinematic/presentation modes

## Features

### 1. **annotate_months.py**
Generates 24 annotated maps (12 months × 2 views: "whole" wind farm and "visited" close-up):
- **Ecological mode** (`--mode eco`): Gaussian dispersion, turbine deflection, curvature for realistic migration corridors
- **Cinematic mode** (`--mode cinematic`): Glow effects, motion blur, density blending for presentations
- **Publication mode** (`--mode pub`): Clean, crisp styling suitable for academic papers

### 2. **make_collage.py**
Assembles the 24 monthly maps into two 12-image collages (4×3 grid) — one for "visited" view, one for "whole" view.

### 3. **make_collage_sets.py**
Creates first-half (Jan–Jun) and second-half (Jul–Dec) collages (3×2 grid) for manageable seasonal presentations.

### 4. **make_collages_ppt.py**
Generates high-resolution PowerPoint-ready collages (4K, 16:9 aspect ratio) with optional month labels.

### 5. **simulate_isabella_bird_mortality.py**
Simulates bird mortality data for a hypothetical wind farm (136 turbines) across 12 months and produces:
- CSV file with detailed simulation data
- Bar chart of total monthly mortality
- Bar chart of average mortality per turbine per month

## Technologies

- **Python 3.8+**
- **Pillow** (PIL) – Image generation and manipulation
- **NumPy** – Numerical simulations and corridor calculations
- **Matplotlib** – Chart generation
- **pandas** – Data handling (optional for analysis)

## Installation

```bash
# Clone the repository
git clone https://github.com/stairona/bird-simulation.git
cd bird-simulation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generate Monthly Annotated Maps

```bash
python src/scripts/annotate_months.py --mode eco
```

This produces 24 PNG files in `outputs/monthly-annotated-maps/`:
- `whole_01_Jan_eco.png` through `whole_12_Dec_eco.png`
- `visited_01_Jan_eco.png` through `visited_12_Dec_eco.png`

### Create Collages

```bash
# Full 12-month collages
python src/scripts/make_collage.py

# Half-year collages (Jan-Jun, Jul-Dec)
python src/scripts/make_collage_sets.py

# PowerPoint-ready high-res collages
python src/scripts/make_collages_ppt.py
```

Outputs are saved to `outputs/summary-plots/` and `outputs/ppt-collages/`.

### Run Mortality Simulation

```bash
python src/scripts/simulate_isabella_bird_mortality.py
```

Creates a timestamped folder in `outputs/simulation-outputs/` containing:
- `isabella_simulated_mortality.csv`
- `graph_total_monthly_mortality_bar.png`
- `graph_avg_per_turbine_monthly_bar.png`

## Project Structure

```
bird-simulation/
├── src/
│   └── scripts/          # Main Python tools
├── data/                 # Input assets (base satellite images)
├── outputs/             # Generated files (gitignored)
│   ├── monthly-annotated-maps/
│   ├── summary-plots/
│   ├── ppt-collages/
│   └── simulation-outputs/
├── bird-simulation/     # Subproject (original nested repo)
├── collage-creator/     # Subproject (original nested repo)
├── docs/
├── tests/
├── requirements.txt
└── README.md
```

## Notes

- Base images (`whole_base.png`, `visited_base.png`) must exist in `data/` for `annotate_months.py` to run.
- Output directories are created automatically if they don't exist.
- The `bird-simulation/` and `collage-creator/` subfolders are preserved from the original development history but are not required for the main scripts.

## License

MIT
