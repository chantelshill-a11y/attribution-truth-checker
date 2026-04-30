"""
CLI runner for the geo-lift analysis engine (Component 2).

Reads the four public CSVs from data/synthetic/, runs the TWFE
difference-in-differences estimator with cluster-robust standard errors,
and writes per-channel measured incrementality with confidence intervals
and p-values to data/synthetic/measured_incrementality.csv.

Does NOT read data/synthetic/_ground_truth/.

Run from the project root:
  .venv\\Scripts\\python.exe scripts/run_geolift.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.methods.did import (  # noqa: E402
    build_panel,
    fit_twfe_with_cluster_se,
    results_to_dataframe,
)

DATA = ROOT / "data" / "synthetic"
CONFIG = ROOT / "config" / "default.yaml"


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    channel_names = [c["name"] for c in cfg["channels"]]
    n_weeks = int(cfg["n_weeks"])
    confidence_level = float(cfg.get("confidence_level", 0.90))

    print("Loading public synthetic data...")
    users = pd.read_csv(DATA / "users.csv")
    exposure = pd.read_csv(DATA / "channel_exposure.csv")
    conversions = pd.read_csv(DATA / "conversions.csv")
    print(f"  {len(users):,} users, {len(exposure):,} exposures, {len(conversions):,} conversions")

    print("Building city-week panel...")
    panel = build_panel(users, exposure, conversions, n_weeks, channel_names)
    print(f"  panel shape: {panel.shape[0]:,} city-weeks x {panel.shape[1]} columns")

    print(f"Fitting TWFE OLS with cluster-robust SE (cluster=city, CI={confidence_level*100:.0f}%)...")
    results = fit_twfe_with_cluster_se(panel, channel_names, confidence_level=confidence_level)
    df = results_to_dataframe(results)

    out_path = DATA / "measured_incrementality.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print()

    print("Per-channel measured incrementality:")
    print()
    print(f"  {'channel':<15} {'rate':>10} {'CI low':>10} {'CI high':>10} {'p-val':>8}  {'incr conv':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*8}  {'-'*10}")
    for r in results:
        sig_marker = "*" if r.p_value < 0.10 else " "
        print(
            f"  {r.channel:<15} {r.incremental_rate:>10.5f} {r.ci_low_90:>10.5f} "
            f"{r.ci_high_90:>10.5f} {r.p_value:>7.3f}{sig_marker} {r.measured_incremental_conversions:>10.0f}"
        )
    print()
    print("  '*' marks channels significant at the 90% level (p < 0.10).")


if __name__ == "__main__":
    main()
