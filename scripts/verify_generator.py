"""
Verification step for the synthetic data generator.

Reads the five CSVs the generator produced and:
  1. Prints volume sanity stats and a weekly conversion histogram
  2. Prints per-channel exposure counts
  3. Saves three charts to data/synthetic/_verification/
  4. Runs a pass/fail checklist that compares the actual model-vs-truth
     gap per channel against the labels declared in the config

If any channel fails, the generator's parameters need tuning before
analysis is built on top of the data. Tune the channel's reach or
true_incremental_rate in config/default.yaml and rerun.

Run from the project root:
  .venv\\Scripts\\python.exe scripts/verify_generator.py
"""
from __future__ import annotations

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
    REFERENCE,
    RULE,
    SUBJECT,
    add_eyebrow,
    add_titles,
    apply_style,
    body_font,
    make_forest_cmap,
    title_font,
)

DATA = ROOT / "data" / "synthetic"
GT_DIR = DATA / "_ground_truth"
PLOT_DIR = DATA / "_verification"

# Pass/fail threshold: how many percentage points of channel-share
# divergence count as "meaningfully" over- or under-credited.
SHARE_GAP_THRESHOLD_PP = 5.0


def load_data():
    return (
        pd.read_csv(DATA / "users.csv"),
        pd.read_csv(DATA / "channel_exposure.csv"),
        pd.read_csv(DATA / "conversions.csv"),
        pd.read_csv(DATA / "model_attribution.csv"),
        pd.read_csv(GT_DIR / "ground_truth.csv"),
    )


# ---------------------------------------------------------------------------
# Console reports
# ---------------------------------------------------------------------------

def print_volume_sanity(users, exposure, conversions):
    print("=" * 70)
    print("VOLUME SANITY")
    print("=" * 70)
    print(f"users:           {len(users):,}")
    print(f"cities:          {users['city'].nunique()}")
    print(f"exposures:       {len(exposure):,}")
    print(f"conversions:     {len(conversions):,}")
    print(f"convert rate:    {len(conversions) / len(users) * 100:.2f}%")
    print()

    by_city = users["city"].value_counts().sort_values(ascending=False)
    print("largest 5 cities:")
    for city, n in by_city.head(5).items():
        print(f"  {city:10s}  {n:,}")
    print("smallest 5 cities:")
    for city, n in by_city.tail(5).items():
        print(f"  {city:10s}  {n:,}")
    print()

    print("conversions by week (one '#' per 50 conversions):")
    by_week = conversions["conversion_week"].value_counts().sort_index()
    week_min, week_max = int(by_week.index.min()), int(by_week.index.max())
    for w in range(week_min, week_max + 1):
        n = int(by_week.get(w, 0))
        bar = "#" * max(1, n // 50)
        print(f"  week {w:2d}: {n:5d}  {bar}")
    print()


def print_exposure_summary(exposure):
    print("=" * 70)
    print("EXPOSURES PER CHANNEL")
    print("=" * 70)
    for ch, n in exposure["channel"].value_counts().items():
        print(f"  {ch:15s}  {n:>10,}")
    print()


def run_label_checklist(truth: pd.DataFrame, model: pd.DataFrame) -> bool:
    print("=" * 70)
    print("LABEL CHECKLIST (channel-only share comparison)")
    print("=" * 70)
    print("Channels collectively get 99%+ of last-touch credit because the")
    print("model has no concept of baseline conversions. The interesting")
    print("question is which channels get a disproportionate share of that")
    print("credit relative to the causal lift they actually produced.")
    print()
    print(f"PASS criterion (gap = model_share minus true_share, both excluding baseline):")
    print(f"  OVER_CREDITED:  gap > +{SHARE_GAP_THRESHOLD_PP:.1f}pp")
    print(f"  UNDER_CREDITED: gap < -{SHARE_GAP_THRESHOLD_PP:.1f}pp")
    print(f"  ACCURATE:       |gap| <= {SHARE_GAP_THRESHOLD_PP:.1f}pp")
    print()

    truth_ch = truth[truth["channel"] != "_baseline_direct"].copy()
    truth_ch["share_pct"] = (
        truth_ch["true_incremental_conversions"] / truth_ch["true_incremental_conversions"].sum()
    ) * 100
    truth_ch = truth_ch.set_index("channel")

    model_ch = model[model["channel"] != "direct"].copy()
    model_ch["share_pct"] = (
        model_ch["attributed_conversions"] / model_ch["attributed_conversions"].sum()
    ) * 100
    model_ch = model_ch.set_index("channel")

    print(f"  {'channel':<15} {'true %':>8} {'model %':>8} {'gap pp':>8}  {'expected':<14} result")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}  {'-'*14} {'-'*6}")

    n_pass = 0
    for ch in truth_ch.index:
        true_pct = truth_ch.loc[ch, "share_pct"]
        model_pct = model_ch.loc[ch, "share_pct"] if ch in model_ch.index else 0.0
        gap = model_pct - true_pct
        label = truth_ch.loc[ch, "label"]

        if gap > SHARE_GAP_THRESHOLD_PP:
            actual = "OVER_CREDITED"
        elif gap < -SHARE_GAP_THRESHOLD_PP:
            actual = "UNDER_CREDITED"
        else:
            actual = "ACCURATE"

        passed = actual == label
        n_pass += int(passed)
        result = "PASS" if passed else f"FAIL (got {actual})"
        sign = "+" if gap >= 0 else ""
        print(f"  {ch:<15} {true_pct:>7.1f}  {model_pct:>7.1f}  {sign}{gap:>6.1f}   {label:<14} {result}")

    print()
    print(f"  Summary: {n_pass}/{len(truth_ch)} channels match their labels.")
    if n_pass == len(truth_ch):
        print("  Generator is producing the patterns we expect. Safe to build analysis on.")
    else:
        print("  Tune the failing channels' reach or true_incremental_rate in")
        print("  config/default.yaml, then rerun the generator and this script.")
    print()
    return n_pass == len(truth_ch)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_weekly_conversions(conversions, users, out_path):
    by_size = users["city"].value_counts().sort_values(ascending=False)
    large_cities = list(by_size.head(4).index)
    small_cities = list(by_size.tail(4).index)
    cities = large_cities + small_cities

    user_to_city = users.set_index("user_id")["city"]
    df = conversions.assign(city=conversions["user_id"].map(user_to_city))
    pivot = df.groupby(["conversion_week", "city"]).size().unstack(fill_value=0)

    # Large cities in forest variants + charcoal; small cities in mid-gray shades.
    large_palette = [FOREST_DARK, FOREST, FOREST_LIGHT, CHARCOAL]
    small_palette = [MID, "#9C998F", "#BAB7AD", LIGHT]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for city, color in zip(large_cities, large_palette):
        if city in pivot.columns:
            ax.plot(pivot.index, pivot[city], marker="o", markersize=4,
                    label=city, color=color, linewidth=1.6)
    for city, color in zip(small_cities, small_palette):
        if city in pivot.columns:
            ax.plot(pivot.index, pivot[city], marker="o", markersize=4,
                    label=city, color=color, linewidth=1.4)

    ax.set_xlabel("Week", fontproperties=body_font(size=10, weight="500"))
    ax.set_ylabel("Conversions", fontproperties=body_font(size=10, weight="500"))
    add_titles(
        fig, ax,
        title="Weekly conversions across cities of different sizes",
        subtitle="4 largest cities (forest tones) vs 4 smallest (gray tones). Both clusters trend stably with realistic noise.",
    )
    add_eyebrow(fig, "Component 1 verification | Volume sanity")
    leg = ax.legend(loc="best", ncol=2, prop=body_font(size=9))
    for txt in leg.get_texts():
        txt.set_color(CHARCOAL)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)


def plot_exposure_heatmap(exposure, users, out_path):
    user_to_city = users.set_index("user_id")["city"]
    ex = exposure.assign(city=exposure["user_id"].map(user_to_city))
    cities_active = ex.groupby(["channel", "week"])["city"].nunique().unstack(fill_value=0)
    n_cities_total = users["city"].nunique()
    pct = cities_active / n_cities_total * 100

    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(pct.values, aspect="auto", cmap=make_forest_cmap(), vmin=0, vmax=100)
    ax.set_yticks(range(len(pct.index)))
    ax.set_yticklabels(pct.index, fontproperties=body_font(size=9))
    week_max = int(pct.columns.max())
    ax.set_xticks(range(0, week_max + 1, 2))
    ax.set_xticklabels(range(0, week_max + 1, 2), fontproperties=body_font(size=9))
    ax.set_xlabel("Week", fontproperties=body_font(size=10, weight="500"))
    # Heatmap doesn't want gridlines on the data grid.
    ax.grid(False)
    add_titles(
        fig, ax,
        title="Exposure coverage per channel-week",
        subtitle="Cream cells are dark periods (channel held out). Visible, clustered stretches are exactly the natural-experiment structure geo-lift will exploit.",
    )
    add_eyebrow(fig, "Component 1 verification | Dark-period structure")
    cbar = fig.colorbar(im, ax=ax, label="% of cities with at least one exposure")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=MID)
    cbar.set_label("% cities exposed", color=CHARCOAL, fontproperties=body_font(size=9, weight="500"))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)


def plot_truth_vs_model(truth, model, out_path):
    truth_ch = truth[truth["channel"] != "_baseline_direct"].copy()
    truth_ch["share_pct"] = (
        truth_ch["true_incremental_conversions"] / truth_ch["true_incremental_conversions"].sum()
    ) * 100
    truth_ch = truth_ch.set_index("channel")

    model_ch = model[model["channel"] != "direct"].copy()
    model_ch["share_pct"] = (
        model_ch["attributed_conversions"] / model_ch["attributed_conversions"].sum()
    ) * 100
    model_ch = model_ch.set_index("channel")

    channels = list(truth_ch.index)
    truth_pct = truth_ch.loc[channels, "share_pct"].to_numpy()
    model_pct = model_ch.loc[channels, "share_pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(channels))
    w = 0.38
    ax.bar(x - w / 2, truth_pct, w, label="True incremental share (answer key)", color=REFERENCE)
    ax.bar(x + w / 2, model_pct, w, label="Last-touch model claim", color=SUBJECT)
    for i, (t, m) in enumerate(zip(truth_pct, model_pct)):
        gap = m - t
        sign = "+" if gap >= 0 else ""
        ax.text(
            i, max(t, m) + 1.4, f"{sign}{gap:.1f}pp",
            ha="center", color=MID,
            fontproperties=body_font(size=9, weight="500"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=20, ha="right",
                       fontproperties=body_font(size=9))
    ax.set_ylabel("Share among channel-driven conversions (%)",
                  fontproperties=body_font(size=10, weight="500"))
    add_titles(
        fig, ax,
        title="Where last-touch credit diverges from causal reality",
        subtitle="Channel-only shares of conversions. The label above each pair is the gap (model claim minus true share, in percentage points).",
    )
    add_eyebrow(fig, "Component 1 verification | Truth versus model")
    leg = ax.legend(loc="upper right", prop=body_font(size=9))
    for txt in leg.get_texts():
        txt.set_color(CHARCOAL)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not DATA.exists():
        print(f"ERROR: synthetic data not found at {DATA}")
        print("Run the generator first:")
        print("  .venv\\Scripts\\python.exe -m src.generator --config config/default.yaml --output data/synthetic")
        sys.exit(1)

    users, exposure, conversions, model, truth = load_data()
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    apply_style()

    print_volume_sanity(users, exposure, conversions)
    print_exposure_summary(exposure)

    print("Generating charts...")
    plot_weekly_conversions(conversions, users, PLOT_DIR / "weekly_conversions.png")
    plot_exposure_heatmap(exposure, users, PLOT_DIR / "exposure_heatmap.png")
    plot_truth_vs_model(truth, model, PLOT_DIR / "truth_vs_model.png")
    print(f"Charts saved to {PLOT_DIR}")
    print()

    all_pass = run_label_checklist(truth, model)
    sys.exit(0 if all_pass else 2)


if __name__ == "__main__":
    main()
