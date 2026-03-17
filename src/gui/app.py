"""Streamlit GUI for bird-simulation.

Interactive configuration, visualization, and analysis interface.
"""

import sys
from pathlib import Path

# Ensure src is on Python path when running via `streamlit run src/gui/app.py`
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
from typing import Dict, Any

from src.core.config import load_config, SiteConfig
from src.phase1_paths.annotate_months import render_corridors
from src.phase2_mortality.simulate import simulate_dataset, monthly_totals
from src.phase2_mortality.charts import plot_total_monthly_bar, plot_avg_per_turbine_monthly_bar
from src.core.calendar import MONTH_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Page Setup ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bird Collision Simulator",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🦅 Bird Collision Simulator")
st.markdown("Interactive wind farm collision risk assessment")

# ── Session State Initialization ────────────────────────────────────
if "cfg" not in st.session_state:
    st.session_state.cfg = None
if "config_path" not in st.session_state:
    st.session_state.config_path = None
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None
if "selected_month" not in st.session_state:
    st.session_state.selected_month = 0
if "render_mode" not in st.session_state:
    st.session_state.render_mode = "eco"
if "intensity_override" not in st.session_state:
    st.session_state.intensity_override = None

# ── Sidebar: Configuration ──────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    # Config file uploader
    uploaded_file = st.file_uploader(
        "Upload YAML Config",
        type=["yaml", "yml"],
        help="Site configuration file (e.g., isabella.yaml)"
    )

    default_config = Path("configs/isabella.yaml")
    if default_config.exists():
        use_default = st.checkbox("Use default config (Isabella)", value=False)
        if use_default:
            st.session_state.config_path = str(default_config)

    # Load config
    if uploaded_file is not None:
        import tempfile, yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml.safe_load(uploaded_file), f)
            temp_path = f.name
        st.session_state.cfg = load_config(temp_path)
        st.session_state.config_path = temp_path
        st.success(f"Loaded: {st.session_state.cfg.site_name}")
    elif st.session_state.config_path and Path(st.session_state.config_path).exists():
        st.session_state.cfg = load_config(st.session_state.config_path)
        st.info(f"Loaded: {st.session_state.cfg.site_name}")

    # Render mode selector
    if st.session_state.cfg:
        st.session_state.render_mode = st.selectbox(
            "Render Mode",
            ["eco", "cinematic", "pub"],
            index=["eco", "cinematic", "pub"].index(st.session_state.render_mode),
            help="Visualization style: eco (realistic), cinematic (dramatic), pub (clean)"
        )

    st.divider()
    st.caption("v1.0 | Bird Collision Simulator")

# ── Main Content ────────────────────────────────────────────────────
if st.session_state.cfg is None:
    st.info("👈 Please upload a configuration file in the sidebar to get started.")
    st.markdown("""
    **Quick Start:**
    1. Upload a site YAML config (from the `configs/` folder or your own)
    2. Adjust corridor and simulation parameters
    3. Generate maps and run simulations
    4. Explore results with interactive charts

    Example configs are in the `configs/` directory.
    """)
else:
    cfg: SiteConfig = st.session_state.cfg

    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map View", "📊 Simulation", "📈 Results", "⚙️ Parameters"])

    # ── Tab 1: Map View ─────────────────────────────────────────────
    with tab1:
        st.header("Migration Corridors")

        col1, col2, col3 = st.columns(3)
        with col1:
            month_options = [f"{i+1:02d} - {m}" for i, m in enumerate(MONTH_NAMES)]
            month_idx = st.selectbox(
                "Month",
                range(12),
                format_func=lambda i: month_options[i],
                index=st.session_state.selected_month
            )
            st.session_state.selected_month = month_idx

        with col2:
            cal_entry = cfg.monthly_calendar[month_idx]
            st.metric("Migration Intensity", f"{cal_entry.intensity:.2f}")
            st.metric("Migration Index", f"{cal_entry.migration_index:.2f}")

        with col3:
            st.metric("Season", cal_entry.season.title())
            st.metric("Period Label", cal_entry.label)

        # Intensity override slider
        override = st.checkbox("Override intensity", value=st.session_state.intensity_override is not None)
        if override:
            new_intensity = st.slider(
                "Intensity",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.intensity_override or cal_entry.intensity,
                step=0.05
            )
            st.session_state.intensity_override = new_intensity
            intensity = new_intensity
        else:
            st.session_state.intensity_override = None
            intensity = cal_entry.intensity

        # Load and render base map
        if cfg.maps:
            view_key = st.selectbox("Map View", list(cfg.maps.keys()))
            view = cfg.maps[view_key]

            base_img_path = Path(cfg.config_dir) / view.base_image
            if not base_img_path.exists():
                # Try relative to project root
                alt_path = Path(cfg.config_dir) / ".." / view.base_image
                if alt_path.exists():
                    base_img_path = alt_path.resolve()
                else:
                    st.warning(f"Base image not found: {view.base_image}")
                    st.info("You can fetch a satellite map using the CLI: `python -m src.cli generate --fetch-map ...`")
                    base_img = None
            else:
                base_img = Image.open(base_img_path).convert("RGBA")

            if base_img:
                st.write("Rendering corridor map...")
                with st.spinner("Drawing birds..."):
                    rng = np.random.default_rng(42)
                    rendered = render_corridors(
                        base_img=base_img,
                        view=view,
                        cfg=cfg,
                        month_name=cal_entry.month,
                        intensity=intensity,
                        period_label=cal_entry.label,
                        mode=st.session_state.render_mode,
                        rng=rng
                    )

                st.image(rendered, caption=f"{cfg.site_name} — {cal_entry.month}", use_container_width=True)

                # Download button
                buf_path = Path(f"corridor_{view_key}_{month_idx+1:02d}.png")
                rendered.save(buf_path)
                with open(buf_path, "rb") as f:
                    st.download_button(
                        label=f"Download {cal_entry.month} Map",
                        data=f.read(),
                        file_name=buf_path.name,
                        mime="image/png"
                    )
                buf_path.unlink(missing_ok=True)
        else:
            st.warning("No map views defined in config. Add a `maps:` section with base_image and corridor_endpoints.")

    # ── Tab 2: Simulation ────────────────────────────────────────────
    with tab2:
        st.header("Mortality Simulation")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Statistical (Poisson)")
            if st.button("Run Statistical Simulation", type="primary"):
                with st.spinner("Running simulation..."):
                    rows = simulate_dataset(cfg)
                    totals = monthly_totals(rows)
                    annual = sum(totals.values())
                st.session_state.simulation_results = {
                    "type": "statistical",
                    "rows": rows,
                    "totals": totals,
                    "annual": annual
                }
                st.success(f"Simulation complete! Annual mortality: {annual:,}")

            if st.session_state.simulation_results and st.session_state.simulation_results["type"] == "statistical":
                res = st.session_state.simulation_results
                st.metric("Annual Mortality", f"{res['annual']:,}")
                st.dataframe(
                    [[m, res['totals'][m]] for m in MONTH_NAMES],
                    columns=["Month", "Deaths"]
                )

        with col2:
            st.subheader("Agent-Based")
            st.info("Agent-based simulation runs ~2-5 minutes. This feature will be available in a future update.")
            if st.button("Run Agent Simulation (Disabled)", disabled=True):
                pass

    # ── Tab 3: Results ───────────────────────────────────────────────
    with tab3:
        st.header("Results Visualization")

        if st.session_state.simulation_results is None:
            st.info("Run a simulation first (Simulation tab) to see results.")
        else:
            res = st.session_state.simulation_results
            if res["type"] == "statistical":
                rows = res["rows"]

                # Monthly bar chart
                st.subheader("Monthly Mortality")
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                values = [res['totals'][m] for m in MONTH_NAMES]
                bars = ax1.bar(MONTH_NAMES, values, color='#2196F3')
                ax1.set_ylabel("Deaths")
                ax1.set_title("Monthly Mortality")
                for bar, val in zip(bars, values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(val), ha='center', va='bottom', fontsize=8)
                st.pyplot(fig1)

                # Per-turbine average
                st.subheader("Average Deaths per Turbine (Monthly)")
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                turbine_avgs = {}
                for row in rows:
                    tid = row["turbine_id"]
                    turbine_avgs[tid] = turbine_avgs.get(tid, 0) + row["mortality_count"]
                for tid in turbine_avgs:
                    turbine_avgs[tid] /= 12  # average per month

                tids = sorted(turbine_avgs.keys())
                avgs = [turbine_avgs[t] for t in tids]
                ax2.bar(range(len(tids)), avgs, color='#4CAF50')
                ax2.set_xlabel("Turbine ID")
                ax2.set_ylabel("Average Deaths/Month")
                ax2.set_title("Per-Turbine Mortality")
                ax2.set_xticks(range(len(tids))[::max(1, len(tids)//10)])
                ax2.set_xticklabels([str(t) for t in tids][::max(1, len(tids)//10)], rotation=45)
                st.pyplot(fig2)

    # ── Tab 4: Parameters ────────────────────────────────────────────
    with tab4:
        st.header("Configuration Parameters")

        st.subheader("Site Info")
        st.write(f"**Name:** {cfg.site_name}")
        st.write(f"**Region:** {cfg.region}")
        st.write(f"**Config Dir:** {cfg.config_dir}")

        st.subheader("Turbines")
        st.write(f"**Count:** {cfg.turbine_count}")
        st.write(f"**Layout Seed:** {cfg.layout_seed}")
        if cfg.turbine_latlon is not None:
            st.write(f"**Positions:** Loaded from CSV ({len(cfg.turbine_latlon)} turbines)")
        else:
            st.write(f"**Clusters:** {len(cfg.clusters)}")
            for i, c in enumerate(cfg.clusters):
                st.write(f"  {i+1}. Center: {c.center}, Spread: {c.spread}, Fraction: {c.fraction:.2f}")

        st.subheader("Corridors")
        for c in cfg.corridors:
            st.write(f"**{c.name}:** angle={c.angle_deg}°, σ={c.sigma:.3f}, curvature={c.curvature:.3f}")
            st.write(f"  Species: {', '.join(c.species)}")

        st.subheader("Species")
        for key, sp in cfg.species.items():
            st.write(f"**{sp.label}:** color={sp.color}, arrows={sp.arrow_count}, σ={sp.sigma}")

        st.subheader("Simulation")
        st.write(f"Base rate: {cfg.simulation.base_rate}")
        st.write(f"Winter suppression: {cfg.simulation.winter_suppression}")
        st.write(f"Collision avoidance: {cfg.simulation.collision.avoidance:.2%}")
        st.write(f"Night risk multiplier: {cfg.simulation.collision.night_risk_mult}x")
        st.write(f"Altitude match prob: {cfg.simulation.collision.altitude_match_prob:.2%}")

        st.subheader("Download Config")
        import yaml
        config_dict = {
            'site': {'name': cfg.site_name, 'region': cfg.region},
            'turbines': {
                'count': cfg.turbine_count,
                'layout_seed': cfg.layout_seed,
                'clusters': [
                    {'center': list(c.center), 'spread': list(c.spread), 'fraction': c.fraction}
                    for c in cfg.clusters
                ]
            },
            'corridors': [
                {
                    'name': c.name,
                    'angle_deg': c.angle_deg,
                    'sigma': c.sigma,
                    'curvature': c.curvature,
                    'species': c.species,
                    'weight_spring': c.weight_spring,
                    'weight_fall': c.weight_fall,
                    'weight_default': c.weight_default
                }
                for c in cfg.corridors
            ],
            'density_blobs': [
                {'center': list(b.center), 'spread': list(b.spread), 'weight': b.weight}
                for b in cfg.density_blobs
            ],
            'species': {
                k: {
                    'label': v.label,
                    'color': list(v.color),
                    'color_pub': list(v.color_pub),
                    'arrow_count': v.arrow_count,
                    'sigma': v.sigma,
                    'arrow_width': v.arrow_width,
                    'arrow_length': v.arrow_length,
                    'alpha': v.alpha
                }
                for k, v in cfg.species.items()
            },
            'monthly_calendar': [
                {
                    'month': e.month,
                    'migration_index': e.migration_index,
                    'intensity': e.intensity,
                    'label': e.label,
                    'season': e.season
                }
                for e in cfg.monthly_calendar
            ],
            'seasons': {
                k: {
                    'months': v.months,
                    'migration_intensity': v.migration_intensity,
                    'resident_fraction': v.resident_fraction,
                    'night_fraction': v.night_fraction,
                    'weather_risk': v.weather_risk
                }
                for k, v in cfg.seasons.items()
            },
            'simulation': {
                'seed': cfg.simulation.seed,
                'base_rate': cfg.simulation.base_rate,
                'winter_suppression': cfg.simulation.winter_suppression,
                'heterogeneity_sigma': cfg.simulation.heterogeneity_sigma,
                'mortality_scaling': cfg.simulation.mortality_scaling,
                'winter_cap': cfg.simulation.winter_cap,
                'agent': {
                    'birds_per_day_base': cfg.simulation.agent.birds_per_day_base,
                    'migrant_speed': cfg.simulation.agent.migrant_speed,
                    'resident_speed': cfg.simulation.agent.resident_speed,
                    'steps_per_day': cfg.simulation.agent.steps_per_day,
                    'world_size': list(cfg.simulation.agent.world_size)
                },
                'collision': {
                    'rotor_radius': cfg.simulation.collision.rotor_radius,
                    'base_strike_prob': cfg.simulation.collision.base_strike_prob,
                    'avoidance': cfg.simulation.collision.avoidance,
                    'night_risk_mult': cfg.simulation.collision.night_risk_mult,
                    'altitude_match_prob': cfg.simulation.collision.altitude_match_prob
                }
            }
        }
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        st.download_button(
            label="Download Current Config (YAML)",
            data=yaml_str,
            file_name=f"{cfg.site_name.lower().replace(' ', '_')}_config.yaml",
            mime="text/yaml"
        )
