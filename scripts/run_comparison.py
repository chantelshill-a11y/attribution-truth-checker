"""
CLI runner for Component 3: the comparison layer (the truth-check).

Reads the model claim and the engine's measured incrementality, computes
the per-channel share gap, assigns OVER_CREDITED / UNDER_CREDITED / ACCURATE
labels, generates a polished comparison chart, and writes:

  data/synthetic/comparison.csv
  data/synthetic/_verification/comparison.png

Run from the project root:
  .venv\\Scripts\\python.exe scripts/run_comparison.py
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
    MID,
    REFERENCE,
    SUBJECT,
    add_eyebrow,
    add_titles,
    apply_style,
    body_font,
)
from src.comparison import (  # noqa: E402
    LABEL_ACCURATE,
    LABEL_OVER,
    LABEL_UNDER,
    compare_model_to_measured,
    overall_summary,
)

DATA = ROOT / "data" / "synthetic"
PLOT_DIR = DATA / "_verification"


def _label_color(label: str) -> str:
    if label == LABEL_ACCURATE:
        return MID
    return CHARCOAL


def plot_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """
    Headline truth-check chart. Side-by-side share bars per channel:
    model claim (charcoal) and measured incrementality share (forest).
    Above each pair, the assigned label and the gap in percentage points.
    Channels are ordered by absolute share gap, biggest stories on the left.
    """
    df = df.sort_values("abs_share_gap_pp", ascending=False).reset_index(drop=True)
    channels = df["channel"].tolist()
    n = len(channels)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n)
    w = 0.38

    ax.bar(
        x - w / 2,
        df["model_share_pct"],
        w,
        color=REFERENCE,
        label="Attribution model claim (last-touch share)",
    )
    ax.bar(
        x + w / 2,
        df["measured_share_pct"],
        w,
        color=SUBJECT,
        label="Measured incrementality (geo-lift share)",
    )

    # Annotation above each pair: label name, then gap in pp.
    bar_tops = np.maximum(df["model_share_pct"].to_numpy(), df["measured_share_pct"].to_numpy())
    y_max = bar_tops.max()
    pad = y_max * 0.05
    for i, row in df.iterrows():
        gap = row["share_gap_pp"]
        sign = "+" if gap >= 0 else ""
        label = row["label"]
        color = _label_color(label)
        ax.text(
            i, bar_tops[i] + pad,
            label.replace("_", " "),
            ha="center", color=color,
            fontproperties=body_font(size=9, weight="600"),
        )
        ax.text(
            i, bar_tops[i] + pad * 2.6,
            f"{sign}{gap:.1f}pp",
            ha="center", color=color,
            fontproperties=body_font(size=8, weight="400"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=20, ha="right",
                       fontproperties=body_font(size=10))
    ax.set_ylabel("Share among channel-driven conversions (%)",
                  fontproperties=body_font(size=10, weight="500"))
    add_titles(
        fig, ax,
        title="Where the attribution model misallocates credit",
        subtitle=("Each pair compares the channel's share of the model-attributed "
                  "pie against its share of measured incrementality. Channels are "
                  "sorted by absolute gap; the most-misallocated channel appears first."),
    )
    add_eyebrow(fig, "Component 3 | Truth-check output")
    leg = ax.legend(loc="upper right", prop=body_font(size=9))
    for txt in leg.get_texts():
        txt.set_color(CHARCOAL)
    ax.set_ylim(top=y_max * 1.30)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)


def _wrap(text: str, indent: str = "    ", width: int = 90) -> list[str]:
    """Word-wrap text for console output."""
    words = text.split()
    lines = []
    line = indent
    for word in words:
        if line.strip() and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = indent + word
        else:
            line = (line + " " + word) if line.strip() else (line + word)
    if line.strip():
        lines.append(line)
    return lines


def main() -> None:
    if not (DATA / "model_attribution.csv").exists() or \
       not (DATA / "measured_incrementality.csv").exists():
        print("ERROR: missing inputs. Run the generator and the geo-lift engine first.")
        print("  python -m src.generator --config config/default.yaml --output data/synthetic")
        print("  python scripts/run_geolift.py")
        sys.exit(1)

    model = pd.read_csv(DATA / "model_attribution.csv")
    measured = pd.read_csv(DATA / "measured_incrementality.csv")

    df = compare_model_to_measured(model, measured)

    out_csv = DATA / "comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    apply_style()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_chart = PLOT_DIR / "comparison.png"
    plot_comparison(df, out_chart)
    print(f"Chart saved to {out_chart}")
    print()

    # Global headline.
    total_model = df["model_attributed_conversions"].sum()
    total_measured = df["measured_incremental_conversions"].sum()
    print("=" * 92)
    print("OVERALL: how badly does the model over-attribute in absolute terms?")
    print("=" * 92)
    for line in _wrap(overall_summary(df, total_model, total_measured)):
        print(line)
    print()

    # Per-channel table.
    print("=" * 92)
    print("PER-CHANNEL TRUTH-CHECK (share comparison)")
    print("=" * 92)
    print(f"  {'channel':<14} {'model %':>9} {'measured %':>11} {'gap pp':>9}  "
          f"{'label':<14}")
    print(f"  {'-'*14} {'-'*9} {'-'*11} {'-'*9}  {'-'*14}")
    for _, r in df.iterrows():
        sign = "+" if r["share_gap_pp"] >= 0 else ""
        print(
            f"  {r['channel']:<14} "
            f"{r['model_share_pct']:>9.1f} "
            f"{r['measured_share_pct']:>11.1f} "
            f"{sign}{r['share_gap_pp']:>8.1f}  {r['label']:<14}"
        )
    print()

    # Recommendations: only flagged channels.
    flagged = df[df["label"] != LABEL_ACCURATE]
    n_over = (df["label"] == LABEL_OVER).sum()
    n_under = (df["label"] == LABEL_UNDER).sum()
    n_acc = (df["label"] == LABEL_ACCURATE).sum()
    print("=" * 92)
    print("RECOMMENDATIONS (executive summary)")
    print("=" * 92)
    print(f"  {n_over} channel(s) over-credited, {n_under} under-credited, "
          f"{n_acc} accurate within measurement uncertainty.")
    print()
    if flagged.empty:
        print("  No channels flagged at the 5pp share-gap threshold.")
    else:
        for _, r in flagged.iterrows():
            print(f"  {r['channel'].upper()} | {r['label']}")
            for line in _wrap(r["recommendation"]):
                print(line)
            print()


if __name__ == "__main__":
    main()
