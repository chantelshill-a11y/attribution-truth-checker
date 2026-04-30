"""
Build the static site at site/.

Copies the brand-styled chart PNGs from data/synthetic/_verification/ into
site/charts/, and exports the data files the truth-check slider and
self-evaluation pages will consume into site/data/ as JSON.

Run from the project root any time the underlying data or charts change:
  .venv\\Scripts\\python.exe scripts/build_site.py

The HTML files in site/ are hand-edited (one file per page). This script
does NOT regenerate them.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
EVAL = DATA / "_evaluation"
PLOTS = DATA / "_verification"
SITE = ROOT / "site"
SITE_CHARTS = SITE / "charts"
SITE_DATA = SITE / "data"

CHARTS_TO_COPY = [
    "weekly_conversions.png",
    "exposure_heatmap.png",
    "truth_vs_model.png",
    "geolift_measured_vs_true.png",
    "comparison.png",
    "confusion_matrix.png",
    "per_class_metrics.png",
    "threshold_sweep.png",
    "calibration.png",
]


def copy_charts() -> None:
    SITE_CHARTS.mkdir(parents=True, exist_ok=True)
    for name in CHARTS_TO_COPY:
        src = PLOTS / name
        if not src.exists():
            print(f"  WARN: {src.name} missing in {PLOTS}; skipping")
            continue
        shutil.copy2(src, SITE_CHARTS / name)
        size_kb = (SITE_CHARTS / name).stat().st_size // 1024
        print(f"  copied {name} ({size_kb} KB)")


def export_comparison_json() -> None:
    src = DATA / "comparison.csv"
    if not src.exists():
        print(f"  WARN: {src} missing; skipping comparison.json")
        return
    df = pd.read_csv(src)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "channel": r["channel"],
            "model_share_pct": float(r["model_share_pct"]),
            "measured_share_pct": float(r["measured_share_pct"]),
            "model_attributed_conversions": float(r["model_attributed_conversions"]),
            "measured_incremental_conversions": float(r["measured_incremental_conversions"]),
            "share_gap_pp": float(r["share_gap_pp"]),
            "abs_share_gap_pp": float(r["abs_share_gap_pp"]),
            "measured_p_value": float(r["measured_p_value"]),
            "label_at_default_threshold": r["label"],
        })
    total_model = float(df["model_attributed_conversions"].sum())
    total_measured = float(df["measured_incremental_conversions"].sum())
    payload = {
        "total_model_attributed_conversions": total_model,
        "total_measured_incremental_conversions": total_measured,
        "over_attribution_factor": (
            total_model / total_measured if total_measured > 0 else 0.0
        ),
        "channels": rows,
    }
    out = SITE_DATA / "comparison.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {out.name} ({len(rows)} channels)")


def export_simulations_json() -> None:
    src = EVAL / "simulations.csv"
    if not src.exists():
        print(f"  WARN: {src} missing; skipping simulations.json")
        return
    df = pd.read_csv(src)
    # Keep only what the slider needs.
    rows = [
        {
            "true_label": r["true_label"],
            "share_gap_pp": float(r["share_gap_pp"]),
        }
        for _, r in df.iterrows()
    ]
    out = SITE_DATA / "simulations.json"
    out.write_text(json.dumps(rows), encoding="utf-8")
    size_kb = out.stat().st_size // 1024
    print(f"  wrote {out.name} ({len(rows)} predictions, {size_kb} KB)")


def export_metrics_summary_json() -> None:
    for name in [
        "metrics_summary.json",
        "optimal_thresholds.json",
        "calibration_summary.json",
        "narrative_eval_summary.json",
    ]:
        src = EVAL / name
        if not src.exists():
            print(f"  WARN: {src} missing; skipping")
            continue
        shutil.copy2(src, SITE_DATA / name)
        print(f"  copied {name}")


def export_narrative() -> None:
    src = DATA / "narrative.md"
    if not src.exists():
        print(f"  WARN: {src} missing; skipping narrative")
        return
    shutil.copy2(src, SITE_DATA / "narrative.md")
    print(f"  copied narrative.md")


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    SITE_DATA.mkdir(parents=True, exist_ok=True)

    print("Copying charts...")
    copy_charts()
    print()
    print("Exporting JSON data...")
    export_comparison_json()
    export_simulations_json()
    export_metrics_summary_json()
    export_narrative()
    print()
    print(f"Site assets ready at {SITE}")
    print(f"Open site/index.html in any browser, or serve locally with:")
    print(f'  cd "{SITE}" && python -m http.server 8000')


if __name__ == "__main__":
    main()
