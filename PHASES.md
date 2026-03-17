# Bird-Simulation Development Phases

## Current Status

**V1 (CLI Pipeline)**: ✅ Complete and production-ready
- Core models (corridors, turbines, collision, mortality)
- Two-phase pipeline (visualization + statistical + agent)
- Flyway presets for 6 global regions
- CLI with 9 commands
- Full test coverage
- Comprehensive documentation

**Phase 2 (Interactive GUI)**: ✅ Just completed
- Streamlit web interface
- Real-time config loading and preview
- Live corridor map rendering
- Statistical simulation with progress
- Interactive charts (monthly mortality, per-turbine)
- Config export
- Full integration with existing CLI infrastructure

---

## Total Phase Breakdown

| Phase | Status | Effort | Description |
|-------|--------|--------|-------------|
| **V1** | ✅ Complete | Base | CLI pipeline, core models, tests, docs |
| **Phase 2** | ✅ Complete | 4 weeks | Interactive Streamlit GUI |
| **Phase 3.1** | 🚧 Future | 1 week | Live parameter tuning with debounced updates |
| **Phase 3.2** | 🚧 Future | 1 week | Data export integrations (GeoPackage, JSON, PDF) |
| **Phase 3.3** | 🚧 Future | 2 weeks | Plugin architecture for custom collision models |
| **Phase 3.4** | 🚧 Future | 2 weeks | Uncertainty quantification (bootstrap CI) |
| **Phase 3.5** | 🚧 Future | 3 weeks | GIS integration (GeoJSON) + 3D terrain view |
| **Phase 4** | 🚧 Future | 4 weeks | Collaborative dashboard (multi-user, scenario sharing) |

**Total: 8 logical phases** (V1 + Phase 2 + 5 sub-phases + Phase 4)

---

## Detailed Phase 3+ Roadmap

### Phase 3.1: Live Parameter Tuning (1 week)

**Goal**: Reduce simulation latency for interactive exploration.

**Tasks**:
- Cache corridor density fields on config change
- Implement debounced slider updates (wait for user to stop dragging)
- Add "Quick Preview" mode using approximate formulas
- Show spinner during computation, don't block UI

**Deliverables**:
- Instant visual feedback when adjusting corridor parameters
- "Apply" button for full simulation runs
- Cached intermediate results to avoid recomputation

---

### Phase 3.2: Data Export Integrations (1 week)

**Goal**: Enable export to standard scientific and GIS formats.

**Tasks**:
- Add Export → CSV (already exists in CLI, add to GUI)
- Add Export → GeoPackage (turbine geometries + monthly mortality)
- Add Export → JSON (for web embedding)
- Add Export → PDF report (auto-generated summary with narrative)

**Deliverables**:
- Download buttons in GUI Results tab
- `src/tools/export.py` module with format converters
- PDF report template with company logo, key metrics

---

### Phase 3.3: Advanced Collision Models (2 weeks)

**Goal**: Allow researchers to plug in their own collision formulas.

**Tasks**:
- Define abstract base class `CollisionModel` with `predict(environment, bird, turbine) -> float`
- Move current Poisson model to `StandardPoissonModel`
- Create plugin registry (`src/plugins/__init__.py`)
- Add config option: `collision.model: "standard_poisson"` or `"my_custom"`
- Document plugin development in README

**Deliverables**:
- Plugin API with example custom model (e.g., Weibull)
- Model comparison view in GUI (run same config with different models side-by-side)
- Plugin discovery via entry_points (setuptools)

---

### Phase 3.4: Uncertainty Quantification (2 weeks)

**Goal**: Provide confidence intervals for mortality estimates.

**Tasks**:
- Implement bootstrap resampling of Poisson draws
- Config option: `uncertainty: { bootstrap: true, n_samples: 1000 }`
- Compute mean ± 95% CI across samples
- Update charts to show error bars
- Add uncertainty table to PDF export

**Deliverables**:
- `src/tools/uncertainty.py` with bootstrap engine
- Extended `simulate_dataset` to optionally return distribution
- GUI toggle: "Include uncertainty analysis" (warning: slower)
- CI values in monthly totals and annual summary

---

### Phase 3.5: GIS Integration & 3D View (3 weeks)

**Goal**: Enable GIS workflows and 3D terrain visualization.

**Tasks**:
- Export turbine layouts as GeoJSON (lat/lon positions)
- Import GIS layers (rivers, protected areas) as overlays
- Option: fetch DEM (elevation) via contextily
- Create 3D view using Plotly or PyVista (terrain + corridors + turbines)
- Export 3D scene as standalone HTML

**Deliverables**:
- `src/tools/gis.py` with GeoJSON read/write
- GUI tab: "GIS/3D" with view options
- Export → GeoPackage (multiple layers)
- Interactive 3D scene with camera controls

---

### Phase 4: Collaborative Dashboard (4 weeks)

**Goal**: Multi-user web app with scenario management.

**Tasks**:
- Build FastAPI backend (config storage, job queue)
- Add user authentication (simple or OAuth)
- Create dashboard with project gallery
- Implement scenario comparison UI (multiple configs, versioning)
- Add commenting/annotation on results
- Deploy to cloud (Docker + Kubernetes or Streamlit Cloud with database)

**Deliverables**:
- `backend/` directory with FastAPI + PostgreSQL
- `frontend/` extended Streamlit with shared state
- Dockerfile and docker-compose for local deployment
- Documentation for self-hosting

---

## Phase Completion Criteria

Each phase is considered complete when:
1. All tasks implemented and tested
2. Documentation updated (README + code comments)
3. No regressions in existing test suite
4. Demo run on at least two different configs succeeds
5. Git commit with clear message and no trailing TODOs

---

## Prioritization Notes

**Recommended order:**
1. Phase 3.1 (Live Tuning) — low effort, high UX impact
2. Phase 3.2 (Data Export) — user demand for sharing results
3. Phase 3.4 (Uncertainty) — scientific rigor requirement
4. Phase 3.3 (Plugins) — extensibility for advanced users
5. Phase 3.5 (GIS/3D) — niche but valuable for presentations
6. Phase 4 (Dashboard) — depends on team/user demand for collaboration

The core simulation is already mature. Future work should focus on **usability** (GUI live tuning, exports) and **credibility** (uncertainty quantification). Plugin system is a stretch goal for research groups with custom models.
