"""
Classification metrics for attribution decisions.

Treats each (simulation, channel) prediction as a 3-class classification:
  OVER_CREDITED, UNDER_CREDITED, ACCURATE

Computes per-class precision, recall, F1, plus accuracy, macro-F1, and
weighted-F1. No external classification library is used; the math is
visible so an interviewer can read what each number means.

Definitions:
  TP for class C: predicted C, actually C
  FP for class C: predicted C, actually some other class
  FN for class C: actually C, predicted some other class
  precision = TP / (TP + FP)        of those we flagged as C, what fraction were?
  recall    = TP / (TP + FN)        of those that really were C, what fraction did we catch?
  F1        = 2 * P * R / (P + R)   harmonic mean; punishes lopsided P or R
  accuracy  = correct / total       across all classes
  macro F1  = mean of per-class F1, equal weight per class
  weighted F1 = support-weighted mean of per-class F1
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ClassMetrics:
    label: str
    support: int        # count of true examples in this class
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


@dataclass
class OverallMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    total: int
    confusion: pd.DataFrame   # rows = true label, columns = predicted label
    per_class: list[ClassMetrics]


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> OverallMetrics:
    """
    Compute the full classification breakdown for a list of predictions.

    `labels` controls the row/column order of the confusion matrix and the
    set of classes considered. If omitted, the union of true and predicted
    labels is used in alphabetical order.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    n = len(y_true)

    confusion = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            confusion.loc[t, p] += 1

    per_class: list[ClassMetrics] = []
    f1_sum = 0.0
    weighted_f1_sum = 0.0
    correct = 0

    for label in labels:
        tp = int(confusion.loc[label, label])
        # Column sum = total predicted as this label.
        # Row sum = total actually this label.
        fp = int(confusion[label].sum() - tp)
        fn = int(confusion.loc[label].sum() - tp)
        support = int(confusion.loc[label].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class.append(ClassMetrics(
            label=label,
            support=support,
            precision=precision,
            recall=recall,
            f1=f1,
            tp=tp,
            fp=fp,
            fn=fn,
        ))

        correct += tp
        f1_sum += f1
        weighted_f1_sum += f1 * support

    accuracy = correct / n if n > 0 else 0.0
    macro_f1 = f1_sum / len(labels) if labels else 0.0
    weighted_f1 = weighted_f1_sum / n if n > 0 else 0.0

    return OverallMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        total=n,
        confusion=confusion,
        per_class=per_class,
    )
