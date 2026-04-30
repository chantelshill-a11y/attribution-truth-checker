"""
Simulation harness: runs the full attribution pipeline N times with
different seeds and collects per-channel predicted labels for evaluation.

Each simulation:
  1. Generate synthetic data with a new seed
  2. Run the geo-lift engine
  3. Run the comparison layer
  4. Record per-channel predicted_label

The configured labels (from config/default.yaml) serve as ground truth.
Variance in predicted_label across simulations comes from measurement
noise in the geo-lift engine. The configured pattern is held constant.

Skipping CSV I/O: the public modules are called directly with in-memory
DataFrames so we don't pay a disk roundtrip per simulation.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.comparison import compare_model_to_measured
from src.generator import (
    GeneratorConfig,
    build_conversions_and_ground_truth,
    build_dark_period_mask,
    build_exposure,
    build_users,
    run_last_touch_model,
)
from src.methods.did import (
    build_panel,
    fit_twfe_with_cluster_se,
    results_to_dataframe,
)


def run_one_simulation(cfg: GeneratorConfig, seed: int) -> list[dict]:
    """
    Run the full pipeline once with the given seed.

    Returns one record per channel with:
      simulation_idx     the seed used (de facto unique id)
      channel            channel name
      true_label         label from config (the answer key)
      predicted_label    label assigned by comparison.py
      share_gap_pp       the gap that drove the label decision
      abs_share_gap_pp   absolute value of the gap
      measured_p_value   engine p-value on the measurement
    """
    rng = np.random.default_rng(seed)
    channel_names = [c.name for c in cfg.channels]

    users = build_users(cfg, rng)
    mask = build_dark_period_mask(cfg, rng)
    exposure = build_exposure(users, cfg, mask, rng)
    conversions, _gt = build_conversions_and_ground_truth(users, exposure, cfg, rng)
    model = run_last_touch_model(conversions, exposure, cfg, rng)

    panel = build_panel(users, exposure, conversions, cfg.n_weeks, channel_names)
    results = fit_twfe_with_cluster_se(panel, channel_names, confidence_level=0.90)
    measured = results_to_dataframe(results)

    comparison = compare_model_to_measured(model, measured)

    config_labels = {c.name: c.label for c in cfg.channels}

    rows = []
    for _, r in comparison.iterrows():
        ch = r["channel"]
        rows.append({
            "simulation_idx": seed,
            "channel": ch,
            "true_label": config_labels[ch],
            "predicted_label": r["label"],
            "share_gap_pp": float(r["share_gap_pp"]),
            "abs_share_gap_pp": float(r["abs_share_gap_pp"]),
            "measured_p_value": float(r["measured_p_value"]),
        })
    return rows


def run_simulations(
    cfg: GeneratorConfig,
    n_simulations: int,
    base_seed: int = 1000,
    progress_every: int | None = None,
) -> pd.DataFrame:
    """
    Run N simulations and return all per-channel results.

    progress_every controls how often timing updates print. If None,
    prints roughly every 5% of simulations.
    """
    if progress_every is None:
        progress_every = max(1, n_simulations // 20)

    all_rows: list[dict] = []
    start = time.time()
    for i in range(n_simulations):
        seed = base_seed + i
        rows = run_one_simulation(cfg, seed)
        all_rows.extend(rows)
        if (i + 1) % progress_every == 0 or i == n_simulations - 1:
            elapsed = time.time() - start
            per_sim = elapsed / (i + 1)
            remaining = per_sim * (n_simulations - i - 1)
            print(
                f"  {i + 1:3d}/{n_simulations} simulations | "
                f"{elapsed:5.1f}s elapsed | "
                f"{per_sim:.1f}s/sim | "
                f"~{remaining:5.0f}s remaining"
            )
    return pd.DataFrame(all_rows)
