"""
Calibration check for the truth-checker's decision system.

A classifier is well-calibrated when its stated confidence matches its
empirical accuracy. If the system claims 80% confidence and is right 80%
of the time at that confidence level, calibration is good. If it claims
80% and is right 60% of the time, the system is over-confident, which is
dangerous in budget decisions. If it claims 60% and is right 80% of the
time, the system is under-confident.

We define confidence as the **margin from the decision boundary**:

  For OVER or UNDER predictions:
    confidence = (|share_gap_pp| - threshold) / margin_scale, capped at 1.0
    A 15pp gap is highly confident; a 5.5pp gap is barely above threshold.

  For ACCURATE predictions:
    confidence = (threshold - |share_gap_pp|) / threshold, capped at 1.0
    A 0pp gap is highly confident; a 4.5pp gap is barely below threshold.

This is more directly relevant to the OVER/UNDER/ACCURATE decision than
the engine's p-value, which tests the existence of a non-zero effect
(a different question).

The reliability diagram is the standard calibration visualization: bin
predictions by confidence, plot mean confidence vs empirical accuracy in
each bin, with a 45-degree reference line.

The Expected Calibration Error (ECE) summarizes calibration in one
number: the bin-size-weighted mean absolute gap between confidence and
accuracy. Lower is better; 0 is perfect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD_PP = 5.0
DEFAULT_MARGIN_SCALE = 10.0  # 1.0 reached when |gap| is threshold + 10pp = 15pp


def predicted_confidence(
    share_gap_pp: float,
    threshold_pp: float = DEFAULT_THRESHOLD_PP,
    margin_scale: float = DEFAULT_MARGIN_SCALE,
) -> float:
    """
    Margin-based confidence in the assigned label.

    For OVER/UNDER predictions, confidence rises with how far above the
    threshold the gap is. For ACCURATE predictions, confidence rises with
    how far below the threshold the gap is. All in [0, 1].
    """
    abs_gap = abs(share_gap_pp)
    if abs_gap >= threshold_pp:
        # OVER or UNDER assigned. Margin above threshold.
        return min(1.0, (abs_gap - threshold_pp) / margin_scale)
    # ACCURATE assigned. Distance below threshold.
    return min(1.0, (threshold_pp - abs_gap) / threshold_pp)


def calibration_table(
    df: pd.DataFrame,
    n_bins: int = 10,
    threshold_pp: float = DEFAULT_THRESHOLD_PP,
    margin_scale: float = DEFAULT_MARGIN_SCALE,
) -> pd.DataFrame:
    """
    Bin predictions by confidence and compute accuracy per bin.

    Input columns required:
      true_label, predicted_label, share_gap_pp

    Output columns:
      bin (interval), count, mean_confidence, accuracy, ece_contrib
    """
    df = df.copy()
    df["confidence"] = df["share_gap_pp"].apply(
        lambda g: predicted_confidence(g, threshold_pp, margin_scale)
    )
    df["correct"] = (df["predicted_label"] == df["true_label"]).astype(int)

    # Equal-width bins on confidence in [0, 1].
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin_idx"] = np.clip(np.digitize(df["confidence"], edges, right=False) - 1, 0, n_bins - 1)

    rows = []
    n_total = len(df)
    for i in range(n_bins):
        sub = df[df["bin_idx"] == i]
        if len(sub) == 0:
            continue
        bin_left, bin_right = float(edges[i]), float(edges[i + 1])
        mean_conf = float(sub["confidence"].mean())
        acc = float(sub["correct"].mean())
        weight = len(sub) / n_total
        ece_contrib = weight * abs(mean_conf - acc)
        rows.append({
            "bin_left": bin_left,
            "bin_right": bin_right,
            "count": int(len(sub)),
            "mean_confidence": mean_conf,
            "accuracy": acc,
            "weight": weight,
            "ece_contrib": ece_contrib,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(cal_table: pd.DataFrame) -> float:
    """
    Weighted mean of |confidence - accuracy| across bins.
    0 is perfect calibration. >0.10 is meaningfully miscalibrated.
    """
    return float(cal_table["ece_contrib"].sum())
