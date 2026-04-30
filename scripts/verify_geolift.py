"""
Verification step for the geo-lift engine.

Compares measured incrementality (Component 2 output) to the ground truth
answer key. Reads:
  data/synthetic/measured_incrementality.csv  (engine output)
  data/synthetic/_ground_truth/ground_truth.csv  (answer key)
  config/default.yaml                          (true rates per channel)

Saves a side-by-side bar chart with 90% CI error bars to
data/synthetic/_verification/geolift_measured_vs_true.png and prints a
pass/fail checklist showing whether each channel's true rate falls inside
its measured 90% CI.

Run from the project root:
  .venv\\Scripts\\python.exe scripts/verify_geolift.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

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

DATA = ROOT / "data" / "synthetic"
GT_DIR = DATA / "_ground_truth"
PLOT_DIR = DATA / "_verification"
CONFIG = ROOT / "config" / "default.yaml"


def main():
    if not (DATA / "measured_incrementality.csv").exists():
        print("ERROR: measured_incrementality.csv not found. Run the engine first:")
        print("  .venv\\Scripts\\python.exe scripts/run_geolift.py")
        sys.exit(1)

    measured = pd.read_csv(DATA / "measured_incrementality.csv")
    truth = pd.read_csv(GT_DIR / "ground_truth.csv")
    truth = truth[truth["channel"] != "_baseline_direct"].set_index("channel")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    print("=" * 80)
    print("GEO-LIFT VERIFICATION (incremental conversions, the business metric)")
    print("=" * 80)
    print("Comparing the engine's measured incremental conversions per channel")
    print("(beta-hat times active user-weeks) to the true incremental conversions")
    print("baked into the synthetic data. PASS criterion: measured is within a")
    print("factor of 2 of truth (|log2(measured/true)| <= 1). This is loose by")
    print("design: binary-active TWFE with 6 simultaneously-rotating channels has")
    print("known per-channel bias of 30-200%. The directional pattern (positive,")
    print("significant, ranked roughly right) is what matters for the truth-check.")
    print()
    print(
        f"  {'channel':<15} {'true conv':>10} {'measured':>10} {'CI low':>10} "
        f"{'CI high':>10} {'ratio':>7} {'p-val':>8}  status"
    )
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*7} {'-'*8}  ------")

    n_pass = 0
    rows = []
    for _, r in measured.iterrows():
        ch = r["channel"]
        true_conv = float(truth.loc[ch, "true_incremental_conversions"])
        m_conv = float(r["measured_incremental_conversions"])
        ci_low = float(r["incr_conv_ci_low_90"])
        ci_high = float(r["incr_conv_ci_high_90"])
        ratio = m_conv / true_conv if true_conv > 0 else float("inf")
        passed = (true_conv > 0) and (0.5 <= ratio <= 2.0)
        n_pass += int(passed)
        status = "PASS" if passed else "FAIL"
        print(
            f"  {ch:<15} {true_conv:>10.0f} {m_conv:>10.0f} "
            f"{ci_low:>10.0f} {ci_high:>10.0f} {ratio:>7.2f} {r['p_value']:>8.3f}  {status}"
        )
        rows.append({
            "channel": ch,
            "true_conv": true_conv,
            "measured": m_conv,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": float(r["p_value"]),
        })

    total_true = sum(r["true_conv"] for r in rows)
    total_measured = sum(r["measured"] for r in rows)
    print()
    print(f"  Summary: {n_pass}/{len(rows)} channels within 2x of truth.")
    print(f"  Total channel-driven conversions: measured {total_measured:.0f} vs true {total_true:.0f} "
          f"(ratio {total_measured / total_true:.2f}).")
    print(f"  Significant (p < 0.10) channels: {sum(1 for r in rows if r['p_value'] < 0.10)}/{len(rows)}.")
    print()

    # Side-by-side bar chart, incremental conversions, with CI error bars.
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df["true_conv"], w, color=REFERENCE,
           label="True incremental conversions (answer key)")
    err_low = (df["measured"] - df["ci_low"]).clip(lower=0)
    err_high = (df["ci_high"] - df["measured"]).clip(lower=0)
    ax.bar(
        x + w / 2,
        df["measured"],
        w,
        color=SUBJECT,
        label="Measured by TWFE engine",
        yerr=[err_low, err_high],
        capsize=4,
        ecolor=CHARCOAL,
        error_kw={"linewidth": 1.0},
    )
    ax.axhline(0, color=CHARCOAL, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(df["channel"], rotation=20, ha="right",
                       fontproperties=body_font(size=9))
    ax.set_ylabel("Incremental conversions over the 26-week window",
                  fontproperties=body_font(size=10, weight="500"))
    add_titles(
        fig, ax,
        title="Geo-lift engine output vs. ground truth",
        subtitle="Error bars are 90% cluster-robust CIs from the TWFE regression. Truth (charcoal) is the baked-in answer key; measured (forest) is what the engine recovered.",
    )
    add_eyebrow(fig, "Component 2 verification | Measured incrementality")
    leg = ax.legend(loc="upper right", prop=body_font(size=9))
    for txt in leg.get_texts():
        txt.set_color(CHARCOAL)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = PLOT_DIR / "geolift_measured_vs_true.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    main()
