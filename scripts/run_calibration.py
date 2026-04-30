"""
CLI runner for the calibration extension of Component 6.

Reads simulations.csv, computes a reliability diagram (stated confidence
vs empirical accuracy), and writes:

  data/synthetic/_evaluation/calibration_table.csv   per-bin breakdown
  data/synthetic/_evaluation/calibration_summary.json ECE + summary stats
  data/synthetic/_verification/calibration.png       reliability diagram

Run from the project root:
  .venv\\Scripts\\python.exe scripts/run_calibration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.branding import (  # noqa: E402
    CHARCOAL,
    FOREST,
    FOREST_DARK,
    FOREST_LIGHT,
    LIGHT,
    MID,
    RULE,
    add_eyebrow,
    add_titles,
    apply_style,
    body_font,
    title_font,
)
from src.evaluation.calibration import (  # noqa: E402
    calibration_table,
    expected_calibration_error,
)

DATA = ROOT / "data" / "synthetic"
EVAL_DIR = DATA / "_evaluation"
PLOT_DIR = DATA / "_verification"


def plot_reliability(cal_df: pd.DataFrame, ece: float, total_n: int, out_path: Path) -> None:
    """
    Reliability diagram: confidence (x) vs accuracy (y) per bin, with
    the 45-degree perfect-calibration reference. Marker size encodes
    bin count so an interviewer can see where the data lives.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Reference: perfect calibration
    ax.plot([0, 1], [0, 1], color=MID, linestyle=":", linewidth=1, alpha=0.8,
            label="Perfect calibration")

    # Per-bin points
    sizes = (cal_df["count"] / cal_df["count"].max()) * 600 + 30
    ax.scatter(
        cal_df["mean_confidence"], cal_df["accuracy"],
        s=sizes, color=FOREST, edgecolors=CHARCOAL, linewidths=1.0,
        alpha=0.85, zorder=5, label="Bin (size = count)",
    )
    # Draw a line through the bin means to make the trend visible.
    ax.plot(
        cal_df["mean_confidence"], cal_df["accuracy"],
        color=FOREST_DARK, linewidth=1.2, alpha=0.5, zorder=4,
    )

    # Annotate each point with its count.
    for _, r in cal_df.iterrows():
        ax.annotate(
            f"n={int(r['count'])}",
            (r["mean_confidence"], r["accuracy"]),
            textcoords="offset points", xytext=(8, -2),
            fontproperties=body_font(size=8),
            color=MID,
        )

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(
        "Stated confidence (margin from decision boundary)",
        fontproperties=body_font(size=10, weight="500"),
    )
    ax.set_ylabel(
        "Empirical accuracy",
        fontproperties=body_font(size=10, weight="500"),
    )
    add_titles(
        fig, ax,
        title="Calibration: does stated confidence match actual accuracy?",
        subtitle=(
            f"Across {total_n:,} predictions, binned by confidence. "
            f"Expected Calibration Error (ECE) = {ece:.3f}. "
            f"Points near the diagonal are well-calibrated; below = over-confident, above = under-confident."
        ),
    )
    add_eyebrow(fig, "Component 6 | Calibration check")
    leg = ax.legend(loc="lower right", prop=body_font(size=9))
    for txt in leg.get_texts():
        txt.set_color(CHARCOAL)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    sims_path = EVAL_DIR / "simulations.csv"
    if not sims_path.exists():
        print(f"ERROR: {sims_path} not found. Run scripts/run_evaluation.py first.")
        sys.exit(1)

    apply_style()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sims_path)
    print(f"Loaded {len(df):,} predictions from {sims_path}")

    cal = calibration_table(df, n_bins=10)
    cal_path = EVAL_DIR / "calibration_table.csv"
    cal.to_csv(cal_path, index=False)
    print(f"Wrote {cal_path}")

    ece = expected_calibration_error(cal)

    summary = {
        "total_predictions": int(len(df)),
        "n_bins_populated": int(len(cal)),
        "expected_calibration_error": ece,
        "interpretation": (
            "ECE is the bin-size-weighted mean absolute gap between stated "
            "confidence and empirical accuracy. 0 is perfect; <0.05 is "
            "well-calibrated; >0.10 is meaningfully miscalibrated."
        ),
    }
    summary_path = EVAL_DIR / "calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")

    print()
    print("=" * 88)
    print("CALIBRATION TABLE")
    print("=" * 88)
    print(f"  {'bin':<14} {'count':>6} {'mean conf':>10} {'accuracy':>10} {'gap':>8}")
    print(f"  {'-' * 14} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 8}")
    for _, r in cal.iterrows():
        bin_label = f"[{r['bin_left']:.2f}, {r['bin_right']:.2f})"
        gap = r["mean_confidence"] - r["accuracy"]
        sign = "+" if gap >= 0 else ""
        print(
            f"  {bin_label:<14} {int(r['count']):>6} "
            f"{r['mean_confidence']:>10.3f} {r['accuracy']:>10.3f} "
            f"{sign}{gap:>7.3f}"
        )
    print()
    print(f"  Expected Calibration Error (ECE): {ece:.3f}")
    if ece < 0.05:
        verdict = "WELL CALIBRATED"
    elif ece < 0.10:
        verdict = "MILDLY MISCALIBRATED"
    else:
        verdict = "MEANINGFULLY MISCALIBRATED"
    print(f"  Verdict: {verdict}")
    print()

    chart_path = PLOT_DIR / "calibration.png"
    plot_reliability(cal, ece, len(df), chart_path)
    print(f"Saved {chart_path}")


if __name__ == "__main__":
    main()
