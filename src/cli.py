"""
Unified CLI for the bird-simulation pipeline.

Usage:
  python -m src.cli paths     --config configs/isabella.yaml [--mode eco|cinematic|pub]
  python -m src.cli collages  --config configs/isabella.yaml [--layout full|half|ppt|all]
  python -m src.cli mortality  --config configs/isabella.yaml
  python -m src.cli agent      --config configs/isabella.yaml
  python -m src.cli all        --config configs/isabella.yaml
  python -m src.cli init       --name "My Wind Farm" [--output configs/my_site.yaml]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def cmd_paths(args):
    from .core.config import load_config
    from .phase1_paths.annotate_months import generate_monthly_maps

    cfg = load_config(args.config)
    out = args.out or "outputs/monthly-annotated-maps"
    print(f"Phase 1: Generating monthly corridor maps for {cfg.site_name}")
    print(f"  Mode: {args.mode}, Seed: {args.seed}")
    print(f"  Map views: {list(cfg.maps.keys())}")
    generate_monthly_maps(cfg, mode=args.mode, seed=args.seed, out_dir=out)
    print("Phase 1 complete.")


def cmd_collages(args):
    from .core.config import load_config
    from .phase1_paths.collage import generate_collages

    cfg = load_config(args.config)
    input_dir = args.input or "outputs/monthly-annotated-maps"
    out = args.out or "outputs/collages"
    print(f"Generating collages ({args.layout}) for {cfg.site_name}")
    generate_collages(cfg, input_dir=input_dir, out_dir=out, layout=args.layout)
    print("Collages complete.")


def cmd_mortality(args):
    from .core.config import load_config
    from .phase2_mortality.simulate import run_simulation

    cfg = load_config(args.config)
    out = args.out or "outputs/simulation-outputs"
    print(f"Phase 2: Running statistical mortality simulation for {cfg.site_name}")
    run_simulation(cfg, out_dir=out)
    print("Phase 2 (statistical) complete.")


def cmd_agent(args):
    from .core.config import load_config
    from .phase2_mortality.agent_sim import run_agent_simulation

    cfg = load_config(args.config)
    out = args.out or "outputs/agent-sim-outputs"
    print(f"Phase 2: Running agent-based simulation for {cfg.site_name}")
    run_agent_simulation(cfg, out_dir=out)
    print("Phase 2 (agent) complete.")


def cmd_all(args):
    from .core.config import load_config
    from .phase1_paths.annotate_months import generate_monthly_maps
    from .phase1_paths.collage import generate_collages
    from .phase2_mortality.simulate import run_simulation
    from .phase2_mortality.agent_sim import run_agent_simulation

    cfg = load_config(args.config)
    maps_dir = args.out_maps or "outputs/monthly-annotated-maps"
    collage_dir = args.out_collages or "outputs/collages"
    sim_dir = args.out_sim or "outputs/simulation-outputs"
    agent_dir = args.out_agent or "outputs/agent-sim-outputs"

    print(f"=== Full pipeline for {cfg.site_name} ===\n")

    print("Phase 1a: Monthly corridor maps...")
    generate_monthly_maps(cfg, mode=args.mode, seed=args.seed, out_dir=maps_dir)

    print("\nPhase 1b: Collages...")
    generate_collages(cfg, input_dir=maps_dir, out_dir=collage_dir, layout="all")

    print("\nPhase 2a: Statistical mortality simulation...")
    run_simulation(cfg, out_dir=sim_dir)

    print("\nPhase 2b: Agent-based simulation...")
    run_agent_simulation(cfg, out_dir=agent_dir)

    print("\n=== All phases complete ===")


def cmd_init(args):
    """Copy the example template to a new config file."""
    template = os.path.join(os.path.dirname(__file__), "..", "configs", "example_template.yaml")
    template = os.path.normpath(template)

    output = args.output or f"configs/{args.name.lower().replace(' ', '_')}.yaml"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    if os.path.exists(output):
        print(f"File already exists: {output}")
        sys.exit(1)

    shutil.copy2(template, output)

    # Replace placeholder name
    with open(output) as f:
        content = f.read()
    content = content.replace('"My Wind Farm"', f'"{args.name}"')
    with open(output, "w") as f:
        f.write(content)

    print(f"Created config: {output}")
    print(f"Edit it with your site's turbine layout, corridors, and species.")
    print(f"Then run:  python -m src.cli all --config {output}")


def main():
    parser = argparse.ArgumentParser(
        prog="bird-sim",
        description="Bird collision simulation pipeline — config-driven, any site",
    )
    sub = parser.add_subparsers(dest="command", help="Pipeline phase to run")

    # paths
    p_paths = sub.add_parser("paths", help="Phase 1: Generate monthly corridor maps")
    p_paths.add_argument("--config", required=True, help="Path to site YAML config")
    p_paths.add_argument("--mode", choices=["eco", "cinematic", "pub"], default="eco")
    p_paths.add_argument("--seed", type=int, default=42)
    p_paths.add_argument("--out", help="Output directory")
    p_paths.set_defaults(func=cmd_paths)

    # collages
    p_col = sub.add_parser("collages", help="Assemble monthly maps into collages")
    p_col.add_argument("--config", required=True)
    p_col.add_argument("--layout", choices=["full", "half", "ppt", "all"], default="all")
    p_col.add_argument("--input", help="Directory with monthly map PNGs")
    p_col.add_argument("--out", help="Output directory")
    p_col.set_defaults(func=cmd_collages)

    # mortality (statistical)
    p_mort = sub.add_parser("mortality", help="Phase 2: Statistical Poisson mortality simulation")
    p_mort.add_argument("--config", required=True)
    p_mort.add_argument("--out", help="Output directory")
    p_mort.set_defaults(func=cmd_mortality)

    # agent
    p_agent = sub.add_parser("agent", help="Phase 2: Agent-based collision simulation")
    p_agent.add_argument("--config", required=True)
    p_agent.add_argument("--out", help="Output directory")
    p_agent.set_defaults(func=cmd_agent)

    # all
    p_all = sub.add_parser("all", help="Run full pipeline (Phase 1 + Phase 2)")
    p_all.add_argument("--config", required=True)
    p_all.add_argument("--mode", choices=["eco", "cinematic", "pub"], default="eco")
    p_all.add_argument("--seed", type=int, default=42)
    p_all.add_argument("--out-maps", help="Output dir for monthly maps")
    p_all.add_argument("--out-collages", help="Output dir for collages")
    p_all.add_argument("--out-sim", help="Output dir for statistical sim")
    p_all.add_argument("--out-agent", help="Output dir for agent sim")
    p_all.set_defaults(func=cmd_all)

    # init
    p_init = sub.add_parser("init", help="Create a new site config from template")
    p_init.add_argument("--name", required=True, help="Site display name")
    p_init.add_argument("--output", help="Output YAML path")
    p_init.set_defaults(func=cmd_init)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
