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

from src.core.config import load_config, SiteConfig, SimulationParams, CollisionParams, AgentParams
from src.phase1_paths.annotate_months import render_corridors
from src.phase2_mortality.simulate import simulate_dataset, monthly_totals
from src.core.calendar import MONTH_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import hashlib
import time
from dataclasses import asdict

# ── Phase 3.1: Live Parameter Tuning Utilities ──────────────────────

def _hash_config(cfg: SiteConfig, tuning_params: dict) -> str:
    """Create a hash key for caching simulation results."""
    # Combine base config + tuning overrides
    key_data = {
        'turbine_count': cfg.turbine_count,
        'clusters': [(c.center, c.spread, c.fraction) for c in cfg.clusters],
        'corridors': [(c.angle_deg, c.sigma, c.curvature) for c in cfg.corridors],
        'density_blobs': [(b.center, b.spread, b.weight) for b in cfg.density_blobs],
        'monthly_calendar': [(e.migration_index, e.intensity) for e in cfg.monthly_calendar],
        'seasons': {k: (v.migration_intensity, v.resident_fraction, v.night_fraction, v.weather_risk)
                    for k, v in cfg.seasons.items()},
        'sim_seed': cfg.simulation.seed,
        'tuning': tuning_params,  # overrides applied by sliders
    }
    key_str = str(sorted(key_data.items()))
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def apply_tuning_params(cfg: SiteConfig, params: dict) -> SiteConfig:
    """Apply tuning parameter overrides to a config copy.

    Returns a new SiteConfig with modified simulation parameters.
    """
    import copy
    new_cfg = copy.deepcopy(cfg)

    if 'avoidance' in params:
        new_cfg.simulation.collision.avoidance = float(params['avoidance'])
    if 'base_rate' in params:
        new_cfg.simulation.base_rate = float(params['base_rate'])
    if 'base_strike_prob' in params:
        new_cfg.simulation.collision.base_strike_prob = float(params['base_strike_prob'])
    if 'night_risk_mult' in params:
        new_cfg.simulation.collision.night_risk_mult = float(params['night_risk_mult'])
    if 'winter_suppression' in params:
        new_cfg.simulation.winter_suppression = float(params['winter_suppression'])
    if 'mortality_scaling' in params:
        new_cfg.simulation.mortality_scaling = float(params['mortality_scaling'])

    return new_cfg


def run_simulation_cached(cfg: SiteConfig, tuning_params: dict, use_cache: bool = True):
    """Run simulation with caching support."""
    cache_key = _hash_config(cfg, tuning_params)

    if use_cache and cache_key in st.session_state.sim_cache:
        st.session_state.cache_hits += 1
        st.session_state.simulation_results = st.session_state.sim_cache[cache_key]
        st.toast("✅ Loaded from cache", icon="⚡")
        return st.session_state.simulation_results

    st.session_state.cache_misses += 1
    # Apply tuning overrides
    tuned_cfg = apply_tuning_params(cfg, tuning_params)

    start_time = time.time()
    rows = simulate_dataset(tuned_cfg)
    totals = monthly_totals(rows)
    annual = sum(totals.values())
    elapsed = time.time() - start_time

    results = {
        "type": "statistical",
        "rows": rows,
        "totals": totals,
        "annual": annual,
        "elapsed": elapsed,
        "cached": False,
        "tuning_applied": tuning_params.copy() if tuning_params else {},
    }

    # Cache the results
    st.session_state.sim_cache[cache_key] = results
    st.session_state.simulation_results = results
    st.session_state.last_sim_time = time.time()

    return results


def clear_simulation_cache():
    """Clear the simulation cache."""
    st.session_state.sim_cache.clear()
    st.session_state.cache_hits = 0
    st.session_state.cache_misses = 0
    st.toast("🧹 Cache cleared", icon="🗑️")


def auto_tune_callback():
    """Callback for parameter sliders - marks that an update is needed."""
    # Read current slider values from session state (updated by Streamlit)
    # and store them in tuning_params for simulation and hashing
    if 'avoidance_slider' in st.session_state:
        st.session_state.tuning_params['avoidance'] = st.session_state.avoidance_slider
    if 'base_rate_slider' in st.session_state:
        st.session_state.tuning_params['base_rate'] = st.session_state.base_rate_slider
    if 'base_strike_prob_slider' in st.session_state:
        st.session_state.tuning_params['base_strike_prob'] = st.session_state.base_strike_prob_slider
    if 'night_risk_mult_slider' in st.session_state:
        st.session_state.tuning_params['night_risk_mult'] = st.session_state.night_risk_mult_slider
    if 'winter_suppression_slider' in st.session_state:
        st.session_state.tuning_params['winter_suppression'] = st.session_state.winter_suppression_slider
    if 'mortality_scaling_slider' in st.session_state:
        st.session_state.tuning_params['mortality_scaling'] = st.session_state.mortality_scaling_slider
    st.session_state.needs_sim_update = True


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
# Phase 3.1: Live tuning state
if "tuning_params" not in st.session_state:
    st.session_state.tuning_params = {}
if "sim_cache" not in st.session_state:
    st.session_state.sim_cache = {}
if "cache_hits" not in st.session_state:
    st.session_state.cache_hits = 0
if "cache_misses" not in st.session_state:
    st.session_state.cache_misses = 0
if "last_sim_time" not in st.session_state:
    st.session_state.last_sim_time = None
if "needs_sim_update" not in st.session_state:
    st.session_state.needs_sim_update = False

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

    # ── Tab 2: Simulation (with Live Tuning) ───────────────────────────
    with tab2:
        st.header("Mortality Simulation")

        # Initialize tuning params from current config if not set
        if not st.session_state.tuning_params:
            st.session_state.tuning_params = {
                'avoidance': cfg.simulation.collision.avoidance,
                'base_rate': cfg.simulation.base_rate,
                'base_strike_prob': cfg.simulation.collision.base_strike_prob,
                'night_risk_mult': cfg.simulation.collision.night_risk_mult,
                'winter_suppression': cfg.simulation.winter_suppression,
                'mortality_scaling': cfg.simulation.mortality_scaling,
            }

        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Parameter Tuning")

            st.markdown("**Real-time Preview:** Adjust sliders to see how parameters affect mortality. Changes are automatically simulated with debouncing.")

            # Tuning sliders
            t_col1, t_col2 = st.columns(2)

            with t_col1:
                avoidance = st.slider(
                    "Collision Avoidance",
                    min_value=0.0,
                    max_value=0.99,
                    value=float(st.session_state.tuning_params.get('avoidance', cfg.simulation.collision.avoidance)),
                    step=0.01,
                    format="%.2f",
                    key="avoidance_slider",
                    on_change=auto_tune_callback,
                    help="Fraction of birds that successfully avoid turbines (0=no avoidance, 0.99=near-perfect)"
                )

                base_rate = st.slider(
                    "Base Mortality Rate",
                    min_value=0.1,
                    max_value=5.0,
                    value=float(st.session_state.tuning_params.get('base_rate', cfg.simulation.base_rate)),
                    step=0.05,
                    format="%.2f",
                    key="base_rate_slider",
                    on_change=auto_tune_callback,
                    help="Global scaling factor for collision probability"
                )

                winter_suppression = st.slider(
                    "Winter Suppression",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.tuning_params.get('winter_suppression', cfg.simulation.winter_suppression)),
                    step=0.05,
                    format="%.2f",
                    key="winter_suppression_slider",
                    on_change=auto_tune_callback,
                    help="Reduction factor applied during winter months"
                )

            with t_col2:
                base_strike_prob = st.slider(
                    "Base Strike Probability",
                    min_value=0.0001,
                    max_value=0.01,
                    value=float(st.session_state.tuning_params.get('base_strike_prob', cfg.simulation.collision.base_strike_prob)),
                    step=0.0001,
                    format="%.5f",
                    key="base_strike_prob_slider",
                    on_change=auto_tune_callback,
                    help="Baseline per-step strike probability when bird is in rotor zone"
                )

                night_risk_mult = st.slider(
                    "Night Risk Multiplier",
                    min_value=0.5,
                    max_value=5.0,
                    value=float(st.session_state.tuning_params.get('night_risk_mult', cfg.simulation.collision.night_risk_mult)),
                    step=0.1,
                    format="%.1f",
                    key="night_risk_mult_slider",
                    on_change=auto_tune_callback,
                    help="Multiplier applied during nocturnal flight"
                )

                mortality_scaling = st.slider(
                    "Mortality Scaling",
                    min_value=0.1,
                    max_value=2.0,
                    value=float(st.session_state.tuning_params.get('mortality_scaling', cfg.simulation.mortality_scaling)),
                    step=0.05,
                    format="%.2f",
                    key="mortality_scaling_slider",
                    on_change=auto_tune_callback,
                    help="Global calibration factor for final mortality estimates"
                )

            # Note: tuning_params is updated by auto_tune_callback when sliders change
            # We only need to ensure it's initialized on first load
            if not st.session_state.tuning_params:
                st.session_state.tuning_params = {
                    'avoidance': cfg.simulation.collision.avoidance,
                    'base_rate': cfg.simulation.base_rate,
                    'base_strike_prob': cfg.simulation.collision.base_strike_prob,
                    'night_risk_mult': cfg.simulation.collision.night_risk_mult,
                    'winter_suppression': cfg.simulation.winter_suppression,
                    'mortality_scaling': cfg.simulation.mortality_scaling,
                }

            # Cache control
            st.markdown("---")
            c_col1, c_col2, c_col3 = st.columns(3)
            with c_col1:
                if st.button("🔄 Reset to Config Defaults", help="Reset sliders to values from loaded config"):
                    # Reset tuning params to config defaults
                    st.session_state.tuning_params = {
                        'avoidance': cfg.simulation.collision.avoidance,
                        'base_rate': cfg.simulation.base_rate,
                        'base_strike_prob': cfg.simulation.collision.base_strike_prob,
                        'night_risk_mult': cfg.simulation.collision.night_risk_mult,
                        'winter_suppression': cfg.simulation.winter_suppression,
                        'mortality_scaling': cfg.simulation.mortality_scaling,
                    }
                    # Clear slider state keys so they re-initialize from tuning_params
                    for key in ['avoidance_slider', 'base_rate_slider', 'base_strike_prob_slider',
                               'night_risk_mult_slider', 'winter_suppression_slider', 'mortality_scaling_slider']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.session_state.needs_sim_update = True
                    st.rerun()
            with c_col2:
                if st.button("🗑️ Clear Cache", help="Clear cached simulation results"):
                    clear_simulation_cache()
                    st.rerun()
            with c_col3:
                st.metric("Cache Hits", f"{st.session_state.cache_hits}")
                st.metric("Cache Misses", f"{st.session_state.cache_misses}")

            # Auto-run simulation when sliders change (debounced by Streamlit rerun)
            if st.session_state.needs_sim_update or st.session_state.simulation_results is None:
                with st.spinner("Simulating with tuned parameters..."):
                    results = run_simulation_cached(cfg, st.session_state.tuning_params, use_cache=True)
                st.session_state.needs_sim_update = False

            # Show current results
            if st.session_state.simulation_results:
                res = st.session_state.simulation_results
                cache_tag = "⚡ Cached" if res.get("cached", False) else "🆕 Fresh"
                st.success(f"**Annual Mortality:** {res['annual']:,}  {cache_tag}  (took {res.get('elapsed', 0):.2f}s)")

                # Show which params were applied
                if res.get("tuning_applied"):
                    st.caption("Tuned parameters: " + ", ".join(
                        f"{k}={v:.3f}" for k, v in res["tuning_applied"].items()
                    ))

                st.subheader("Monthly Breakdown")
                df_data = [[m, res['totals'][m]] for m in MONTH_NAMES]
                st.dataframe(
                    df_data,
                    columns=["Month", "Deaths"],
                    use_container_width=True,
                    hide_index=True
                )

        with col2:
            st.subheader("Agent-Based")
            st.info("Agent-based simulation runs ~2-5 minutes. Coming in a future update.")
            if st.button("Run Agent Simulation", disabled=True):
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
