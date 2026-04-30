"""
CLI runner for the LLM-grades-LLM narrative evaluation extension of Component 6.

For each of N simulations (default 5):
  1. Run the full pipeline to get the comparison DataFrame
  2. Generate an executive narrative via Claude (Component 4)
  3. Grade that narrative against the comparison data via Claude (the grader)
  4. Record per-channel verdicts

Aggregates the grades and writes:
  data/synthetic/_evaluation/narrative_eval_results.json   per-simulation grades
  data/synthetic/_evaluation/narrative_eval_summary.json   aggregated metrics

This costs API tokens. Default N=5 is roughly $0.30 in API spend.

Run from the project root:
  .venv\\Scripts\\python.exe scripts/run_narrative_eval.py [--simulations 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.comparison import compare_model_to_measured  # noqa: E402
from src.evaluation.narrative_eval import (  # noqa: E402
    GradeResult,
    aggregate_grades,
    grade_narrative,
)
from src.generator import (  # noqa: E402
    build_conversions_and_ground_truth,
    build_dark_period_mask,
    build_exposure,
    build_users,
    load_config,
    run_last_touch_model,
)
from src.methods.did import (  # noqa: E402
    build_panel,
    fit_twfe_with_cluster_se,
    results_to_dataframe,
)
from src.narrative import generate_narrative  # noqa: E402

DATA = ROOT / "data" / "synthetic"
EVAL_DIR = DATA / "_evaluation"
CONFIG = ROOT / "config" / "default.yaml"


def _check_api_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
    print("Set it then re-run:")
    print('  $env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable("ANTHROPIC_API_KEY", "User")')
    sys.exit(2)


def run_one_full_simulation(cfg, seed: int) -> pd.DataFrame:
    """Run the full pipeline and return the comparison DataFrame."""
    rng = np.random.default_rng(seed)
    channel_names = [c.name for c in cfg.channels]

    users = build_users(cfg, rng)
    mask = build_dark_period_mask(cfg, rng)
    exposure = build_exposure(users, cfg, mask, rng)
    conversions, _ = build_conversions_and_ground_truth(users, exposure, cfg, rng)
    model = run_last_touch_model(conversions, exposure, cfg, rng)

    panel = build_panel(users, exposure, conversions, cfg.n_weeks, channel_names)
    results = fit_twfe_with_cluster_se(panel, channel_names, confidence_level=0.90)
    measured = results_to_dataframe(results)

    return compare_model_to_measured(model, measured)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LLM-grades-LLM narrative evaluation.")
    parser.add_argument("--simulations", type=int, default=5,
                        help="number of (generate, grade) cycles (default 5; ~$0.30 in API spend)")
    parser.add_argument("--base-seed", type=int, default=2000)
    args = parser.parse_args()

    _check_api_key()

    cfg = load_config(CONFIG)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()

    per_sim_records: list[dict] = []
    grade_results: list[GradeResult] = []
    total_writer_tokens_in = 0
    total_writer_tokens_out = 0
    total_grader_tokens_in = 0
    total_grader_tokens_out = 0

    print(f"Running {args.simulations} simulation+narrative+grade cycles...")
    print(f"Estimated API cost: ~${args.simulations * 0.07:.2f}")
    print()

    for i in range(args.simulations):
        seed = args.base_seed + i
        print(f"=== Simulation {i + 1}/{args.simulations} (seed={seed}) ===")
        t0 = time.time()
        comparison = run_one_full_simulation(cfg, seed)
        print(f"  pipeline: {time.time() - t0:.1f}s")

        total_model = float(comparison["model_attributed_conversions"].sum())
        total_measured = float(comparison["measured_incremental_conversions"].sum())
        over_factor = total_model / total_measured if total_measured > 0 else 0.0

        t1 = time.time()
        narrative = generate_narrative(comparison, total_model, total_measured, over_factor)
        total_writer_tokens_in += narrative.input_tokens
        total_writer_tokens_out += narrative.output_tokens
        print(f"  narrative: {time.time() - t1:.1f}s ({narrative.input_tokens}+{narrative.output_tokens} tokens)")

        t2 = time.time()
        grade = grade_narrative(narrative.text, comparison, client=client)
        total_grader_tokens_in += grade.input_tokens
        total_grader_tokens_out += grade.output_tokens
        grade_results.append(grade)
        print(f"  grade: {time.time() - t2:.1f}s ({grade.input_tokens}+{grade.output_tokens} tokens)")

        for ch in grade.channels:
            per_sim_records.append({
                "simulation_seed": seed,
                **ch,
            })

        print(f"  channels graded: {len(grade.channels)}")
        print(f"  overall: {grade.overall_assessment}")
        print()

    print("=" * 92)
    print("AGGREGATE RESULTS")
    print("=" * 92)
    summary = aggregate_grades(grade_results)
    print(f"  channel-evaluations: {summary['n_channel_evaluations']}")
    print(f"  channel mentioned:           {summary['channel_mentioned_rate']:.3f}")
    print(f"  direction correct:           {summary['direction_correct_rate']:.3f}")
    print(f"  magnitude approx correct:    {summary['magnitude_correct_rate']:.3f}")
    print()
    for label, stats in summary["by_true_label"].items():
        print(f"  {label} (n={stats['n']})")
        print(f"    mentioned:   {stats['channel_mentioned_rate']:.3f}")
        print(f"    direction:   {stats['direction_correct_rate']:.3f}")
        print(f"    magnitude:   {stats['magnitude_correct_rate']:.3f}")
    print()

    writer_cost = (total_writer_tokens_in * 5.0 + total_writer_tokens_out * 25.0) / 1_000_000
    grader_cost = (total_grader_tokens_in * 5.0 + total_grader_tokens_out * 25.0) / 1_000_000
    print(f"  writer cost:  ${writer_cost:.4f} ({total_writer_tokens_in} in + {total_writer_tokens_out} out)")
    print(f"  grader cost:  ${grader_cost:.4f} ({total_grader_tokens_in} in + {total_grader_tokens_out} out)")
    print(f"  total cost:   ${writer_cost + grader_cost:.4f}")
    print()

    # Persist
    results_path = EVAL_DIR / "narrative_eval_results.json"
    results_path.write_text(json.dumps({
        "per_channel_records": per_sim_records,
        "per_simulation_overall": [
            {"seed": args.base_seed + i, "overall_assessment": g.overall_assessment}
            for i, g in enumerate(grade_results)
        ],
    }, indent=2))
    print(f"Wrote {results_path}")

    summary_path = EVAL_DIR / "narrative_eval_summary.json"
    summary_path.write_text(json.dumps({
        **summary,
        "n_simulations": args.simulations,
        "total_api_cost_usd": writer_cost + grader_cost,
        "writer_cost_usd": writer_cost,
        "grader_cost_usd": grader_cost,
    }, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
