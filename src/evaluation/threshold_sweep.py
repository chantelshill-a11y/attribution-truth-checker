"""
Threshold tuning analysis.

Given per-prediction share_gap_pp values from a simulation run, vary the
OVER/UNDER decision threshold and compute precision, recall, F1 at each
candidate threshold. Identifies the F1-optimal threshold per class and
overall.

The threshold being swept is the same one in `src/comparison.py`:
`share_gap_threshold_pp`. Default is 5.0pp; this analysis tells us
whether a different choice would give better classification metrics.

No new simulations are required. The sweep operates on the share_gap_pp
column of `data/synthetic/_evaluation/simulations.csv`, re-applying the
labeling rule at each candidate threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics


LABEL_ORDER = ["OVER_CREDITED", "UNDER_CREDITED", "ACCURATE"]


def relabel_at_threshold(share_gaps_pp, threshold_pp: float) -> list[str]:
    """Apply OVER/UNDER/ACCURATE labels at a given share-gap threshold."""
    out = []
    for gap in share_gaps_pp:
        if gap > threshold_pp:
            out.append("OVER_CREDITED")
        elif gap < -threshold_pp:
            out.append("UNDER_CREDITED")
        else:
            out.append("ACCURATE")
    return out


def sweep_thresholds(
    df: pd.DataFrame,
    thresholds: list[float] | np.ndarray,
) -> pd.DataFrame:
    """
    For each threshold, recompute labels and per-class metrics.

    Input columns:  true_label, share_gap_pp
    Output columns: threshold_pp, class, precision, recall, f1, support,
                    plus accuracy / macro_f1 / weighted_f1 on the
                    "_overall" row at each threshold.
    """
    gaps = df["share_gap_pp"].to_numpy()
    y_true = df["true_label"].tolist()

    rows: list[dict] = []
    for t in thresholds:
        y_pred = relabel_at_threshold(gaps, float(t))
        m = compute_metrics(y_true=y_true, y_pred=y_pred, labels=LABEL_ORDER)
        for pc in m.per_class:
            rows.append({
                "threshold_pp": float(t),
                "class": pc.label,
                "precision": pc.precision,
                "recall": pc.recall,
                "f1": pc.f1,
                "support": pc.support,
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
            })
        rows.append({
            "threshold_pp": float(t),
            "class": "_overall",
            "precision": np.nan,
            "recall": np.nan,
            "f1": m.macro_f1,
            "support": m.total,
            "accuracy": m.accuracy,
            "macro_f1": m.macro_f1,
            "weighted_f1": m.weighted_f1,
        })
    return pd.DataFrame(rows)


def find_optimal_thresholds(sweep_df: pd.DataFrame) -> dict:
    """
    For each class and overall, find the threshold that maximizes F1.

    Returns:
      {
        "OVER_CREDITED":  {threshold_pp, f1, precision, recall},
        "UNDER_CREDITED": {threshold_pp, f1, precision, recall},
        "ACCURATE":       {threshold_pp, f1, precision, recall},
        "_macro":         {threshold_pp, macro_f1, accuracy},
      }
    """
    optimal: dict = {}
    for class_name in LABEL_ORDER:
        group = sweep_df[sweep_df["class"] == class_name]
        if group.empty:
            continue
        best = group.loc[group["f1"].idxmax()]
        optimal[class_name] = {
            "threshold_pp": float(best["threshold_pp"]),
            "f1": float(best["f1"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
        }
    overall = sweep_df[sweep_df["class"] == "_overall"]
    if not overall.empty:
        best_overall = overall.loc[overall["macro_f1"].idxmax()]
        optimal["_macro"] = {
            "threshold_pp": float(best_overall["threshold_pp"]),
            "macro_f1": float(best_overall["macro_f1"]),
            "accuracy": float(best_overall["accuracy"]),
        }
    return optimal
