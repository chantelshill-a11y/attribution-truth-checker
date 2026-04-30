"""
CLI runner for Component 6: P/R/F1 self-evaluation harness.

Runs the full pipeline N times with different seeds, then grades each
per-channel prediction against the configured label. Outputs:

  data/synthetic/_evaluation/simulations.csv         per-prediction details
  data/synthetic/_evaluation/metrics_summary.json    overall + per-class metrics
  data/synthetic/_verification/confusion_matrix.png  3x3 confusion grid
  data/synthetic/_verification/per_class_metrics.png precision/recall/F1 bars

Run from the project root:
  .venv\\Scripts\\python.exe scripts/run_evaluation.py [--simulations 50]

Default is 50 simulations, roughly 6-8 minutes at full scale.
Use --simulations 10 for a smoke test (~1 minute).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
    OFF_WHITE,
    add_eyebrow,
    add_titles,
    apply_style,
    body_font,
    make_forest_cmap,
)
from src.evaluation.harness import run_simulations  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.generator import load_config  # noqa: E402

DATA = ROOT / "data" / "synthetic"
EVAL_DIR = DATA / "_evaluation"
PLOT_DIR = DATA / "_verification"
CONFIG = ROOT / "config" / "default.yaml"

LABEL_ORDER = ["OVER_CREDITED", "UNDER_CREDITED", "ACCURATE"]


def _label_display(label: str) -> str:
    return label.replace("_", " ").title()


def plot_confusion_matrix(confusion: pd.DataFrame, out_path: Path) -> None:
    confusion = confusion.reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)
    matrix = confusion.values
    n_total = int(matrix.sum())

    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.imshow(matrix, cmap=make_forest_cmap(), aspect="equal", vmin=0)

    max_val = matrix.max() if matrix.size else 1
    for i, true_label in enumerate(LABEL_ORDER):
        for j, pred_label in enumerate(LABEL_ORDER):
            count = int(matrix[i, j])
            pct = count / n_total * 100 if n_total else 0
            text_color = OFF_WHITE if count > max_val * 0.55 else CHARCOAL
            ax.text(
                j, i, f"{count}\n{pct:.1f}%",
                ha="center", va="center",
                color=text_color,
                fontproperties=body_font(size=11, weight="500"),
            )

    ax.set_xticks(range(len(LABEL_ORDER)))
    ax.set_yticks(range(len(LABEL_ORDER)))
    ax.set_xticklabels([_label_display(l) for l in LABEL_ORDER],
                       fontproperties=body_font(size=9))
    ax.set_yticklabels([_label_display(l) for l in LABEL_ORDER],
                       fontproperties=body_font(size=9))
    ax.set_xlabel("Predicted label",
                  fontproperties=body_font(size=10, weight="500"), labelpad=12)
    ax.set_ylabel("True label",
                  fontproperties=body_font(size=10, weight="500"), labelpad=12)
    ax.grid(False)
    add_titles(
        fig, ax,
        title="Confusion matrix: assigned vs configured labels",
        subtitle=(
            f"Across {n_total:,} channel-level predictions. Diagonal cells are "
            f"correct; off-diagonal cells are confusions to study."
        ),
    )
    add_eyebrow(fig, "Component 6 | Self-evaluation")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_class_metrics(metrics, out_path: Path) -> None:
    classes = LABEL_ORDER
    per_class = {pc.label: pc for pc in metrics.per_class}

    p_vals = [per_class[c].precision for c in classes]
    r_vals = [per_class[c].recall for c in classes]
    f_vals = [per_class[c].f1 for c in classes]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(classes))
    w = 0.27
    ax.bar(x - w, p_vals, w, color=FOREST_DARK, label="Precision")
    ax.bar(x, r_vals, w, color=FOREST, label="Recall")
    ax.bar(x + w, f_vals, w, color=FOREST_LIGHT, label="F1")

    for i in range(len(classes)):
        for offset, val in zip([-w, 0, w], [p_vals[i], r_vals[i], f_vals[i]]):
            ax.text(
                i + offset, val + 0.025, f"{val:.2f}",
                ha="center", color=CHARCOAL,
                fontproperties=body_font(size=8, weight="500"),
            )

    ax.set_xticks(x)
    ax.set_xticklabels([_label_display(c) for c in classes],
                       fontproperties=body_font(size=10))
    ax.set_ylabel("Score", fontproperties=body_font(size=10, weight="500"))
    ax.set_ylim(0, 1.18)
    add_titles(
        fig, ax,
        title="Per-class precision, recall, F1",
        subtitle=(
            f"Macro-F1 {metrics.macro_f1:.3f} | "
            f"Weighted-F1 {metrics.weighted_f1:.3f} | "
            f"Accuracy {metrics.accuracy:.3f} (across {metrics.total:,} predictions)"
        ),
    )
    add_eyebrow(fig, "Component 6 | Self-evaluation")
    leg = ax.legend(loc="upper right", prop=body_font(size=9))
    for txt in leg.get_texts():
        txt.set_color(CHARCOAL)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P/R/F1 self-evaluation harness.")
    parser.add_argument("--simulations", type=int, default=50,
                        help="number of simulations to run (default 50, ~6-8 min)")
    parser.add_argument("--base-seed", type=int, default=1000,
                        help="seed offset; simulation i uses base_seed + i")
    args = parser.parse_args()

    cfg = load_config(CONFIG)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    print(
        f"Running {args.simulations} simulations "
        f"({cfg.n_users:,} users, {cfg.n_cities} cities, {cfg.n_weeks} weeks each)..."
    )
    print(f"Estimate: ~{args.simulations * 7 / 60:.1f} minutes at full scale.")
    print()
    start = time.time()
    df = run_simulations(cfg, args.simulations, base_seed=args.base_seed)
    elapsed = time.time() - start
    print()
    print(f"Total simulation time: {elapsed:.1f}s ({elapsed / args.simulations:.1f}s/sim)")
    print()

    sims_path = EVAL_DIR / "simulations.csv"
    df.to_csv(sims_path, index=False)
    print(f"Wrote {sims_path}")

    metrics = compute_metrics(
        y_true=df["true_label"].tolist(),
        y_pred=df["predicted_label"].tolist(),
        labels=LABEL_ORDER,
    )

    print()
    print("=" * 88)
    print("SELF-EVALUATION SUMMARY")
    print("=" * 88)
    print(f"  Total predictions: {metrics.total:,}")
    print(f"  Accuracy:          {metrics.accuracy:.3f}")
    print(f"  Macro-F1:          {metrics.macro_f1:.3f}")
    print(f"  Weighted-F1:       {metrics.weighted_f1:.3f}")
    print()
    print(f"  {'class':<18} {'support':>8} {'precision':>10} {'recall':>8} {'F1':>6}")
    print(f"  {'-' * 18} {'-' * 8} {'-' * 10} {'-' * 8} {'-' * 6}")
    for pc in metrics.per_class:
        print(f"  {pc.label:<18} {pc.support:>8} "
              f"{pc.precision:>10.3f} {pc.recall:>8.3f} {pc.f1:>6.3f}")
    print()
    print("Confusion matrix (rows=true, columns=predicted):")
    print(metrics.confusion.reindex(LABEL_ORDER, columns=LABEL_ORDER, fill_value=0).to_string())
    print()

    summary = {
        "total_predictions": metrics.total,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "per_class": [
            {
                "label": pc.label, "support": pc.support,
                "precision": pc.precision, "recall": pc.recall, "f1": pc.f1,
                "tp": pc.tp, "fp": pc.fp, "fn": pc.fn,
            }
            for pc in metrics.per_class
        ],
        "confusion": metrics.confusion.reindex(
            LABEL_ORDER, columns=LABEL_ORDER, fill_value=0
        ).to_dict(),
    }
    summary_path = EVAL_DIR / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")

    cm_path = PLOT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(metrics.confusion, cm_path)
    print(f"Saved {cm_path}")

    pc_path = PLOT_DIR / "per_class_metrics.png"
    plot_per_class_metrics(metrics, pc_path)
    print(f"Saved {pc_path}")


if __name__ == "__main__":
    main()
