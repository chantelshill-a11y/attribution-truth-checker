"""
Comparison layer: the truth-check itself.

Takes the attribution model's per-channel claim and the geo-lift engine's
measured incrementality, computes the gap, and assigns one of three labels
to each channel.

The comparison is on **shares**, not absolute conversion counts. Last-touch
attribution has no concept of baseline conversions, so it attributes nearly
all conversions to channels even though most conversions would have happened
without any marketing exposure. In absolute terms, every channel looks
over-credited by 2-5x. The interesting question is which channels are
RELATIVELY over- or under-credited compared to their actual causal share.

Inputs (read by the CLI runner, not this module):
  data/synthetic/model_attribution.csv      — last-touch model claim
  data/synthetic/measured_incrementality.csv — geo-lift engine output

Output:
  data/synthetic/comparison.csv             — per-channel gap and label

This module deliberately does NOT read data/synthetic/_ground_truth/.
The truth-check is made independent of the answer key, on purpose.
"""
from __future__ import annotations

import pandas as pd

LABEL_OVER = "OVER_CREDITED"
LABEL_UNDER = "UNDER_CREDITED"
LABEL_ACCURATE = "ACCURATE"


def _build_recommendation(
    channel: str,
    label: str,
    model_share_pct: float,
    measured_share_pct: float,
    share_gap_pp: float,
    abs_gap: float,
) -> str:
    """
    Templated recommendation per channel. Component 4 will replace these with
    Claude-generated paragraphs that read more naturally and tie multiple
    channels together into a single executive narrative.
    """
    if label == LABEL_OVER:
        return (
            f"The attribution model gives {channel} {model_share_pct:.1f}% of "
            f"channel-driven credit, but measurement supports only "
            f"{measured_share_pct:.1f}%, a gap of +{share_gap_pp:.1f} percentage "
            f"points (about {abs(abs_gap):.0f} extra conversions over 26 weeks). "
            f"The channel is absorbing more credit than its causal share supports. "
            f"Consider trimming {channel}'s share of the marketing mix or running "
            f"a confirmatory holdout test before reallocating."
        )
    if label == LABEL_UNDER:
        return (
            f"The attribution model gives {channel} only {model_share_pct:.1f}% "
            f"of channel-driven credit, but measurement shows it deserves "
            f"{measured_share_pct:.1f}%, a gap of {share_gap_pp:.1f} percentage "
            f"points (about {abs(abs_gap):.0f} conversions of unrecognized "
            f"contribution). {channel.title()} is producing more incremental "
            f"conversions than the model gives it credit for. Consider increasing "
            f"investment or updating the attribution methodology to capture this "
            f"channel's contribution."
        )
    return (
        f"The model gives {channel} {model_share_pct:.1f}% credit and "
        f"measurement supports {measured_share_pct:.1f}%, a gap of "
        f"{share_gap_pp:+.1f}pp that is within the measurement noise threshold. "
        f"No reallocation indicated."
    )


def compare_model_to_measured(
    model: pd.DataFrame,
    measured: pd.DataFrame,
    share_gap_threshold_pp: float = 5.0,
) -> pd.DataFrame:
    """
    Build the per-channel truth-check.

    Each channel's share of the model-attributed pie is compared to its share
    of the measured-incremental pie. If the gap exceeds the threshold (in
    percentage points), the channel is flagged as over- or under-credited.

    Returns a DataFrame sorted by absolute share-gap, descending. Biggest
    stories first.

    The 5pp threshold is the default; Component 6 will sweep it to find the
    F1-optimal cutoff for the OVER_CREDITED class.
    """
    model = model[model["channel"] != "direct"].copy()
    merged = measured.merge(
        model[["channel", "attributed_conversions"]],
        on="channel",
        how="inner",
    )

    total_model = float(merged["attributed_conversions"].sum())
    total_measured = float(merged["measured_incremental_conversions"].sum())
    if total_model <= 0 or total_measured <= 0:
        raise ValueError(
            "Both model_attribution and measured_incrementality must have positive "
            "channel totals; got "
            f"total_model={total_model}, total_measured={total_measured}."
        )

    rows = []
    for _, r in merged.iterrows():
        ch = r["channel"]
        model_claim = float(r["attributed_conversions"])
        measured_pt = float(r["measured_incremental_conversions"])
        ci_low = float(r["incr_conv_ci_low_90"])
        ci_high = float(r["incr_conv_ci_high_90"])
        p_value = float(r["p_value"])

        model_share = model_claim / total_model * 100.0
        measured_share = measured_pt / total_measured * 100.0
        share_gap_pp = model_share - measured_share
        abs_gap = model_claim - measured_pt

        if share_gap_pp > share_gap_threshold_pp:
            label = LABEL_OVER
        elif share_gap_pp < -share_gap_threshold_pp:
            label = LABEL_UNDER
        else:
            label = LABEL_ACCURATE

        rows.append({
            "channel": ch,
            "model_attributed_conversions": model_claim,
            "model_share_pct": model_share,
            "measured_incremental_conversions": measured_pt,
            "measured_share_pct": measured_share,
            "measured_ci_low_90": ci_low,
            "measured_ci_high_90": ci_high,
            "measured_p_value": p_value,
            "absolute_gap": abs_gap,
            "share_gap_pp": share_gap_pp,
            "abs_share_gap_pp": abs(share_gap_pp),
            "label": label,
            "recommendation": _build_recommendation(
                ch, label, model_share, measured_share, share_gap_pp, abs_gap
            ),
        })

    df = pd.DataFrame(rows).sort_values("abs_share_gap_pp", ascending=False).reset_index(drop=True)
    return df


def overall_summary(df: pd.DataFrame, total_model: float, total_measured: float) -> str:
    """
    The global headline that frames the per-channel findings: how badly is
    the attribution model over-attributing in aggregate.
    """
    over_attribution_factor = total_model / total_measured if total_measured > 0 else 0.0
    return (
        f"In absolute terms, the attribution model assigns {total_model:,.0f} "
        f"conversions to channels, but measurement supports only {total_measured:,.0f}. "
        f"The model is over-attributing by {over_attribution_factor:.1f}x in aggregate, "
        f"because last-touch has no concept of baseline conversions. The per-channel "
        f"comparison below is on relative shares, isolating which channels are "
        f"disproportionately over- or under-credited within that pie."
    )
