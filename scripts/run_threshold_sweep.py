"""
CLI runner for the threshold sweep extension of Component 6.

Reads data/synthetic/_evaluation/simulations.csv (produced by
scripts/run_evaluation.py) and computes precision/recall/F1 at every
candidate share-gap threshold from 1.0pp to 15.0pp in 0.5pp steps.

Outputs:
  data/synthetic/_evaluation/threshold_sweep.csv         full sweep
  data/synthetic/_evaluation/optimal_thresholds.json     F1-optimal per class
  data/synthetic/_verification/threshold_sweep.png       chart

Run from the project root:
  .venv\\Scripts\\python.exe scripts/run_threshold_sweep.py
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
    MID,
    add_eyebrow,
    add_titles,
    apply_style,
    body_font,
    title_font,
)
from src.evaluation.threshold_sweep import (  # noqa: E402
    LABEL_ORDER,
    find_optimal_thresholds,
    sweep_thresholds,
)

DATA = ROOT / "data" / "synthetic"
EVAL_DIR = DATA / "_evaluation"
PLOT_DIR = DATA / "_verification"

CLASS_COLORS = {
    "OVER_CREDITED": FOREST_DARK,
    "UNDER_CREDITED": FOREST,
    "ACCURATE": FOREST_LIGHT,
}


def _label_display(label: str) -> str:
    return label.replace("_", " ").title()


def plot_sweep(sweep_df: pd.DataFrame, optimal: dict, out_path: Path) -> None:
    """
    Two-panel chart.
      Left: F1 vs threshold for each class. Optimum dot per class.
      Right: precision/recall/F1 vs threshold for OVER_CREDITED, the
             budget-relevant class.
    Vertical guides on both panels: dotted at the default 5pp, dashed at
    the F1-optimal threshold for OVER_CREDITED.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ----- Left panel: F1 by class -----
    ax = axes[0]
    for class_name in LABEL_ORDER:
        group = sweep_df[sweep_df["class"] == class_name].sort_values("threshold_pp")
        color = CLASS_COLORS[class_name]
        ax.plot(
            group["threshold_pp"], group["f1"],
            color=color, linewidth=1.6,
            label=_label_display(class_name),
        )
        opt = optimal[class_name]
        ax.scatter(
            [opt["threshold_pp"]], [opt["f1"]],
            s=80, color=color, edgecolors=CHARCOAL, linewidths=1.4, zorder=10,
        )

    over_opt_t = optimal["OVER_CREDITED"]["threshold_pp"]
    ax.axvline(5.0, color=MID, linestyle=":", alpha=0.7, linewidth=1)
    ax.axvline(over_opt_t, color=CHARCOAL, linestyle="--", alpha=0.7, linewidth=1)
    ax.text(5.05, 0.04, "default 5pp",
            color=MID, fontproperties=body_font(size=8), va="bottom")
    ax.text(over_opt_t + 0.05, 0.04,
            f"OVER F1-optimum ({over_opt_t:.1f}pp)",
            color=CHARCOAL, fontproperties=body_font(size=8), va="bottom")

    ax.set_xlabel("Share-gap threshold (percentage points)",
                  fontproperties=body_font(size=10, weight="500"))
    ax.set_ylabel("F1 score", fontproperties=body_font(size=10, weight="500"))
    ax.set_ylim(0, 1.02)
    ax.set_title("F1 by class vs threshold",
                 fontproperties=title_font(size=12), color=CHARCOAL, loc="left")
    leg = ax.legend(loc="lower center", ncol=3, prop=body_font(size=9))
    for txt in leg.get_texts():
        txt.set_color(CHARCOAL)

    # ----- Right panel: precision/recall trade-off for OVER_CREDITED -----
    ax = axes[1]
    over = sweep_df[sweep_df["class"] == "OVER_CREDITED"].sort_values("threshold_pp")
    ax.plot(over["threshold_pp"], over["precision"], color=FOREST_DARK,
            linewidth=1.6, label="Precision")
    ax.plot(over["threshold_pp"], over["recall"], color=FOREST,
            linewidth=1.6, label="Recall")
    ax.plot(over["threshold_pp"], over["f1"], color=FOREST_LIGHT,
            linewidth=2.0, label="F1")
    ax.axvline(5.0, color=MID, linestyle=":", alpha=0.7, linewidth=1)
    ax.axvline(over_opt_t, color=CHARCOAL, linestyle="--", alpha=0.7, linewidth=1)

    opt = optimal["OVER_CREDITED"]
    ax.text(over_opt_t + 0.1, 0.06,
            f"F1-optimal: {over_opt_t:.1f}pp\nP={opt['precision']:.2f}, R={opt['recall']:.2f}, F1={opt['f1']:.2f}",
            color=CHARCOAL, fontproperties=body_font(size=8.5), va="bottom")

    ax.set_xlabel("Share-gap threshold (percentage points)",
                  fontproperties=body_font(size=10, weight="500"))
    ax.set_ylabel("Score", fontproperties=body_font(size=10, weight="500"))
    ax.set_ylim(0, 1.02)
    ax.set_title("OVER_CREDITED precision/recall trade-off",
                 fontproperties=title_font(size=12), color=CHARCOAL, loc="left")
    leg = ax.legend(loc="lower center", ncol=3, prop=body_font(size=9))
    for txt in leg.get_texts():
        txt.set_color(CHARCOAL)

    add_eyebrow(fig, "Component 6 | Threshold sweep")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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

    thresholds = np.arange(1.0, 15.01, 0.5)
    print(f"Sweeping thresholds {thresholds[0]:.1f}pp to {thresholds[-1]:.1f}pp "
          f"in 0.5pp steps ({len(thresholds)} candidates)...")

    sweep_df = sweep_thresholds(df, thresholds)
    sweep_path = EVAL_DIR / "threshold_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Wrote {sweep_path}")

    optimal = find_optimal_thresholds(sweep_df)
    optimal_path = EVAL_DIR / "optimal_thresholds.json"
    optimal_path.write_text(json.dumps(optimal, indent=2))
    print(f"Wrote {optimal_path}")

    print()
    print("=" * 88)
    print("F1-OPTIMAL THRESHOLDS")
    print("=" * 88)
    print(f"  {'class':<18} {'opt threshold':>14} {'F1':>6} {'precision':>10} {'recall':>8}")
    print(f"  {'-' * 18} {'-' * 14} {'-' * 6} {'-' * 10} {'-' * 8}")
    for class_name in LABEL_ORDER:
        o = optimal[class_name]
        print(
            f"  {class_name:<18} {o['threshold_pp']:>13.1f}pp "
            f"{o['f1']:>6.3f} {o['precision']:>10.3f} {o['recall']:>8.3f}"
        )
    print()
    macro = optimal["_macro"]
    print(f"  Macro-F1 optimum: threshold={macro['threshold_pp']:.1f}pp, "
          f"macro-F1={macro['macro_f1']:.3f}, accuracy={macro['accuracy']:.3f}")

    # Compare to default 5pp
    at_default = sweep_df[(sweep_df["threshold_pp"] == 5.0)]
    print()
    print("=" * 88)
    print("DEFAULT (5.0pp) VS F1-OPTIMAL")
    print("=" * 88)
    print(f"  {'class':<18} {'F1 @ 5pp':>10} {'F1 @ optimum':>14} {'delta':>8}")
    print(f"  {'-' * 18} {'-' * 10} {'-' * 14} {'-' * 8}")
    for class_name in LABEL_ORDER:
        f1_default = float(at_default[at_default["class"] == class_name]["f1"].iloc[0])
        f1_opt = optimal[class_name]["f1"]
        print(f"  {class_name:<18} {f1_default:>10.3f} {f1_opt:>14.3f} "
              f"{f1_opt - f1_default:>+8.3f}")
    print()

    chart_path = PLOT_DIR / "threshold_sweep.png"
    plot_sweep(sweep_df, optimal, chart_path)
    print(f"Saved {chart_path}")


if __name__ == "__main__":
    main()
