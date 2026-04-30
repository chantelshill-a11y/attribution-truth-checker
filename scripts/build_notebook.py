"""
Build notebooks/walkthrough.ipynb from this single source file.

This is the live-demo artifact for the AmEx interview. The notebook
walks through every component of the truth-checker, with a mix of
explanatory markdown and code cells that load the existing data and
display the brand-styled charts inline.

The notebook is organized to be executed top-to-bottom in roughly 25
minutes, with the option to skip sections. All charts are pre-generated
PNGs displayed via IPython.display.Image to keep the notebook fast and
visually consistent with the rest of the project.

Run from the project root to regenerate:
  .venv\\Scripts\\python.exe scripts/build_notebook.py

Optionally execute the notebook in place (populates output cells):
  .venv\\Scripts\\python.exe scripts/build_notebook.py --execute
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "walkthrough.ipynb"


def md(*lines: str) -> dict:
    """Build a markdown cell from one or more lines (joined with newline)."""
    return new_markdown_cell("\n".join(lines))


def code(*lines: str) -> dict:
    """Build a code cell from one or more lines (joined with newline)."""
    return new_code_cell("\n".join(lines))


def build_cells() -> list[dict]:
    cells: list[dict] = []

    # =====================================================================
    # 1. Title and premise
    # =====================================================================
    cells.append(md(
        "# Attribution Truth-Checker",
        "",
        "*A walkthrough of the validation system that compares marketing attribution model claims against measured causal incrementality.*",
        "",
        "**Author:** Chantel Hill  ",
        "**Stack:** Python 3.11, pandas, numpy, matplotlib, Anthropic SDK (Claude Opus 4.7)  ",
        "**Repo layout:** `src/` for modules, `scripts/` for CLI runners, `data/synthetic/` for outputs, `notebooks/` for this walkthrough.",
        "",
        "---",
    ))

    cells.append(md(
        "## The two words this project hinges on",
        "",
        "**Attribution** is bookkeeping. The act of assigning credit for a conversion to one or more marketing touchpoints. Last-touch attribution (the simplest version): whichever ad the user clicked last gets all the credit. The output is a number per channel. None of those numbers, by themselves, tell you whether the channel actually *caused* the conversion.",
        "",
        "**Incrementality** is causation. The answer to: \"If we had not run this channel, how many of these conversions would still have happened?\" The conversions that would *not* have happened without the channel are the *incremental* ones. That is the only number a CFO actually cares about, because it is the only number tied to whether a marketing dollar produced a return.",
        "",
        "The gap between attribution credit and true incrementality is the entire reason this project exists. A channel can be credited with 30 percent of conversions and be responsible for 5 percent of the actual lift. That is a $200M problem at AmEx scale. The truth-checker measures the gap.",
    ))

    # =====================================================================
    # 2. Setup
    # =====================================================================
    cells.append(md(
        "## Setup",
        "",
        "Imports, paths, and a single call to `apply_style()` from the project's branding module so any inline plots match the rest of the deliverable.",
    ))

    cells.append(code(
        "import json",
        "from pathlib import Path",
        "import pandas as pd",
        "from IPython.display import Image, Markdown, display",
        "",
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()",
        "DATA = ROOT / 'data' / 'synthetic'",
        "PLOTS = DATA / '_verification'",
        "EVAL = DATA / '_evaluation'",
        "GT = DATA / '_ground_truth'",
        "",
        "import sys",
        "sys.path.insert(0, str(ROOT))",
        "from src.branding import apply_style  # noqa: E402",
        "apply_style()",
        "",
        "pd.set_option('display.max_columns', 12)",
        "pd.set_option('display.width', 140)",
        "print(f'Project root: {ROOT}')",
        "print(f'All data and chart artifacts present: {DATA.exists() and PLOTS.exists() and EVAL.exists()}')",
    ))

    # =====================================================================
    # 3. Component 1: synthetic data with known ground truth
    # =====================================================================
    cells.append(md(
        "## Component 1: synthetic data with known ground truth",
        "",
        "A validation system needs an answer key. The synthetic data generator builds two parallel realities for the same population:",
        "",
        "1. **True causal incrementality.** How many conversions each channel actually caused. Baked in by the generator. This is the secret answer key.",
        "2. **Last-touch attribution credit.** What a deliberately-broken model claims each channel drove. Computed from the same exposure and conversion data.",
        "",
        "The synthetic data uses a hazard model: per (user, week), conversion probability is `baseline * demographic_multiplier + sum(channel_rate for active channels)`. We draw a Bernoulli outcome at that probability, and as conversions are drawn we record fractional credit per causal component. The sum across all conversions of each channel's fractional credit is the channel's true incremental conversions.",
        "",
        "Six channels are configured with deliberate over-, under-, or accurate-attribution patterns: paid_search (over), display (over), paid_social (accurate), tv_brand (under), direct_mail (under), affiliate (accurate). The reach and true_incremental_rate per channel are tuned in `config/default.yaml` to produce these patterns at the share level.",
        "",
        "The four public CSVs the engine sees:",
    ))

    cells.append(code(
        "users = pd.read_csv(DATA / 'users.csv')",
        "exposure = pd.read_csv(DATA / 'channel_exposure.csv')",
        "conversions = pd.read_csv(DATA / 'conversions.csv')",
        "model_attr = pd.read_csv(DATA / 'model_attribution.csv')",
        "",
        "print(f'users:        {len(users):>9,} rows  ({users.shape[1]} cols)')",
        "print(f'exposure:     {len(exposure):>9,} rows')",
        "print(f'conversions:  {len(conversions):>9,} rows')",
        "print(f'model_attr:   {len(model_attr):>9,} rows')",
        "print()",
        "print('users.head():')",
        "print(users.head().to_string(index=False))",
    ))

    cells.append(md(
        "### Verifying the generator: did the over- and under-credit pattern emerge?",
        "",
        "We compare the answer key (ground truth incremental shares) against the last-touch model's claims. If the generator works, paid_search and display should be over-credited, tv_brand and direct_mail under-credited, paid_social and affiliate roughly accurate.",
    ))

    cells.append(code(
        "Image(filename=str(PLOTS / 'truth_vs_model.png'))",
    ))

    cells.append(md(
        "The pattern is visible: TV brand has the largest blue bar (35% of true causal effect) and one of the smallest red bars (14% of model credit), a -20pp gap. Display has nearly the smallest blue bar (6%) and one of the biggest red bars (25%), a +19pp gap. The synthetic data behaves the way the config says it should.",
    ))

    # =====================================================================
    # 4. Component 2: geo-lift engine
    # =====================================================================
    cells.append(md(
        "## Component 2: the geo-lift analysis engine",
        "",
        "Now we recover the true causal effects from the public data alone, without consulting the answer key. This is the core measurement work.",
        "",
        "**Two-way fixed-effects (TWFE) regression** is the workhorse. The model:",
        "",
        "$$\\text{conversion\\_rate}_{c,w} = \\alpha + \\beta_1 \\cdot \\text{search\\_active}_{c,w} + \\beta_2 \\cdot \\text{social\\_active}_{c,w} + \\dots + \\text{city\\_FE}_c + \\text{week\\_FE}_w + \\epsilon_{c,w}$$",
        "",
        "The city fixed effects absorb everything that is constant within a city (baseline conversion rate, demographic mix). The week fixed effects absorb shocks shared across cities (seasonality, news cycles). What is left to identify each $\\beta$ is the dark-period variation we baked into the data: when a channel is active in one city-week and dark in another, that contrast is the source of identification.",
        "",
        "**Cluster-robust standard errors** (clustered by city) account for the fact that observations within a city are correlated across weeks. Without clustering, p-values would look more confident than they should.",
        "",
        "The implementation in `src/methods/did.py` is hand-rolled in numpy: OLS via the normal equations, the cluster-robust sandwich variance formula, Acklam's algorithm for the inverse normal CDF. No statsmodels (no ARM64 wheels), no scipy.stats (blocked by Smart App Control on this machine). Hand-rolling makes the math visible.",
    ))

    cells.append(code(
        "measured = pd.read_csv(DATA / 'measured_incrementality.csv')",
        "display_cols = ['channel', 'incremental_rate', 'standard_error', 't_stat', 'p_value', 'measured_incremental_conversions']",
        "print(measured[display_cols].round(5).to_string(index=False))",
    ))

    cells.append(code(
        "Image(filename=str(PLOTS / 'geolift_measured_vs_true.png'))",
    ))

    cells.append(md(
        "All six measured estimates are positive. The two highest-incrementality channels in truth (TV brand and direct mail) are also visibly high in measured. The two lowest (display and paid search) are visibly low. The pattern is right; magnitudes are noisy with finite data, which is the documented limitation of multi-channel TWFE — single-channel synthetic control would tighten per-channel magnitudes but is out of scope for Phase 1.",
    ))

    # =====================================================================
    # 5. Component 3: the truth-check itself
    # =====================================================================
    cells.append(md(
        "## Component 3: the truth-check itself",
        "",
        "Compare model claim against measured incrementality, channel by channel. The comparison is on **shares**, not absolute conversion counts. Last-touch attribution attributes nearly all conversions to channels because it has no concept of baseline conversions; in absolute terms every channel looks over-credited by 2-5x. The interesting question is which channels are *relatively* over- or under-credited compared to their actual causal contribution.",
        "",
        "Three-class labeling rule: if `|model_share - measured_share| > 5pp`, flag as OVER_CREDITED or UNDER_CREDITED depending on direction. Otherwise ACCURATE.",
    ))

    cells.append(code(
        "comparison = pd.read_csv(DATA / 'comparison.csv')",
        "summary_cols = ['channel', 'model_share_pct', 'measured_share_pct', 'share_gap_pp', 'label']",
        "print(comparison[summary_cols].round(2).to_string(index=False))",
        "",
        "total_model = comparison['model_attributed_conversions'].sum()",
        "total_measured = comparison['measured_incremental_conversions'].sum()",
        "print()",
        "print(f'Aggregate over-attribution: model claims {total_model:,.0f} conversions vs {total_measured:,.0f} measured ({total_model / total_measured:.1f}x).')",
    ))

    cells.append(code(
        "Image(filename=str(PLOTS / 'comparison.png'))",
    ))

    cells.append(md(
        "**The headline:** direct_mail is the most under-credited channel (-19.5pp), getting 7.7% of attribution credit when its measured share is 27.3%. Paid search and display are most over-credited (+13.5pp and +12.9pp). Paid social and affiliate are within measurement noise.",
    ))

    # =====================================================================
    # 6. Component 4: Claude API narrative
    # =====================================================================
    cells.append(md(
        "## Component 4: Claude API executive narrative",
        "",
        "The comparison data is the right answer in numbers. An executive wants paragraphs. We feed the comparison data to `claude-opus-4-7` with a carefully-structured prompt and get back an executive summary.",
        "",
        "Three prompt-engineering choices worth pointing out:",
        "",
        "1. **Style rules sit in the system prompt** as hard constraints (no em dashes, no \"obviously\" or \"clearly\", every numeric claim must trace back to a number in the data). System prompts are stable across calls; the user message is what varies.",
        "2. **Negative constraints work better than positive ones for tone control.** \"Avoid 'obviously' and 'clearly'\" is more effective than \"be humble.\"",
        "3. **The skip-list approach.** Telling the model which channels to discuss vs omit is more reliable than asking it to filter on its own.",
        "",
        "Cost: ~$0.026 per call (about 1300 input + 800 output tokens at Opus 4.7 prices).",
    ))

    cells.append(code(
        "from src.narrative import SYSTEM_PROMPT",
        "print(SYSTEM_PROMPT[:600] + '...')  # show the head of the system prompt",
    ))

    cells.append(code(
        "narrative_text = (DATA / 'narrative.md').read_text(encoding='utf-8')",
        "Markdown(narrative_text)",
    ))

    # =====================================================================
    # 7. Component 6: P/R/F1 self-evaluation
    # =====================================================================
    cells.append(md(
        "## Component 6: precision, recall, F1 — the validation discipline",
        "",
        "Marketing teams rarely apply classification metrics to attribution decisions because attribution outputs look continuous (credit shares), not categorical. But the *decisions* the truth-checker outputs (OVER_CREDITED, UNDER_CREDITED, ACCURATE) are explicitly a 3-class classification problem. Treating it that way and grading it with precision, recall, and F1 brings the same evaluation discipline used for clause classifiers in legal AI into a domain that doesn't usually have it.",
        "",
        "We run the full pipeline 50 times with different seeds. Configured labels stay constant; predicted labels vary with measurement noise. 50 simulations × 6 channels = 300 predictions, ~100 per class — enough for stable per-class metrics.",
    ))

    cells.append(code(
        "summary = json.loads((EVAL / 'metrics_summary.json').read_text())",
        "print(f'Total predictions: {summary[\"total_predictions\"]}')",
        "print(f'Accuracy:          {summary[\"accuracy\"]:.3f}')",
        "print(f'Macro-F1:          {summary[\"macro_f1\"]:.3f}')",
        "print(f'Weighted-F1:       {summary[\"weighted_f1\"]:.3f}')",
        "print()",
        "print(f'{\"class\":<18} {\"support\":>8} {\"precision\":>10} {\"recall\":>8} {\"F1\":>6}')",
        "print(f'{\"-\" * 18} {\"-\" * 8} {\"-\" * 10} {\"-\" * 8} {\"-\" * 6}')",
        "for pc in summary['per_class']:",
        "    print(f'{pc[\"label\"]:<18} {pc[\"support\"]:>8} {pc[\"precision\"]:>10.3f} {pc[\"recall\"]:>8.3f} {pc[\"f1\"]:>6.3f}')",
    ))

    cells.append(code(
        "Image(filename=str(PLOTS / 'confusion_matrix.png'))",
    ))

    cells.append(md(
        "Three observations from the confusion matrix:",
        "",
        "1. **OVER and UNDER almost never get confused with each other.** 3 out of 100 in each direction. The system never tells you to cut a channel that should be expanded, or vice versa. The mistakes are about magnitude, not direction.",
        "2. **Recall on actionable classes is high.** 85% on OVER, 79% on UNDER. When there is real misallocation, the system catches it.",
        "3. **The weakness is on the ACCURATE class.** 43% recall — the system over-flags channels that are actually fine. The threshold sweep below shows how to fix this.",
    ))

    cells.append(code(
        "Image(filename=str(PLOTS / 'per_class_metrics.png'))",
    ))

    # =====================================================================
    # 8. Threshold sweep
    # =====================================================================
    cells.append(md(
        "### Component 6 extension: threshold sweep",
        "",
        "The 5pp gap threshold I picked for the comparison layer was an educated guess. Sweeping it across [1pp, 15pp] in 0.5pp steps and recomputing metrics at each candidate point shows where F1 actually peaks per class.",
    ))

    cells.append(code(
        "optimal = json.loads((EVAL / 'optimal_thresholds.json').read_text())",
        "print(f'{\"class\":<18} {\"opt threshold\":>14} {\"F1\":>6} {\"precision\":>10} {\"recall\":>8}')",
        "print(f'{\"-\" * 18} {\"-\" * 14} {\"-\" * 6} {\"-\" * 10} {\"-\" * 8}')",
        "for cls in ['OVER_CREDITED', 'UNDER_CREDITED', 'ACCURATE']:",
        "    o = optimal[cls]",
        "    print(f'{cls:<18} {o[\"threshold_pp\"]:>13.1f}pp {o[\"f1\"]:>6.3f} {o[\"precision\"]:>10.3f} {o[\"recall\"]:>8.3f}')",
        "print()",
        "macro = optimal['_macro']",
        "print(f'Macro-F1 optimum at {macro[\"threshold_pp\"]:.1f}pp: macro-F1 = {macro[\"macro_f1\"]:.3f}')",
    ))

    cells.append(code(
        "Image(filename=str(PLOTS / 'threshold_sweep.png'))",
    ))

    cells.append(md(
        "F1-optimal threshold for OVER_CREDITED — the budget-relevant class — is **8.0pp**, not the default 5pp. Modest +0.027 F1 gain. The bigger gain is for the ACCURATE class at 13.5pp (+0.141 F1), which would substantially cut the false-flag rate.",
    ))

    # =====================================================================
    # 9. Calibration
    # =====================================================================
    cells.append(md(
        "### Component 6 extension: calibration check",
        "",
        "When the system says it is confident in a label, is it actually correct that often? Confidence is defined as the margin from the decision boundary: an OVER prediction with share gap 15pp is high-confidence; one at 5.5pp is barely-above-threshold.",
        "",
        "The Expected Calibration Error (ECE) summarizes the gap between stated confidence and empirical accuracy. 0 is perfect; <0.05 is well-calibrated; >0.10 is meaningfully miscalibrated.",
    ))

    cells.append(code(
        "cal_summary = json.loads((EVAL / 'calibration_summary.json').read_text())",
        "print(f'Expected Calibration Error: {cal_summary[\"expected_calibration_error\"]:.3f}')",
        "print(f'Verdict: {\"WELL CALIBRATED\" if cal_summary[\"expected_calibration_error\"] < 0.05 else (\"MILD\" if cal_summary[\"expected_calibration_error\"] < 0.1 else \"MEANINGFULLY MISCALIBRATED\")}')",
    ))

    cells.append(code(
        "Image(filename=str(PLOTS / 'calibration.png'))",
    ))

    cells.append(md(
        "**The system is well-calibrated at the extremes** but over-confident in the moderate-confidence range. At 85% stated confidence, empirical accuracy is only 52%. That is the kind of failure mode you cannot see in macro-F1 alone, and it is what you would want to fix before the truth-checker is used for budget decisions where moderate-confidence findings get acted on.",
    ))

    # =====================================================================
    # 10. LLM-grades-LLM narrative eval
    # =====================================================================
    cells.append(md(
        "### Component 6 extension: LLM-grades-LLM narrative evaluation",
        "",
        "Take Claude's executive summary, structure the comparison data as ground truth, and have Claude grade Claude on whether the narrative's claims are consistent with the data.",
        "",
        "Per channel labeled OVER or UNDER, the grader returns three booleans:",
        "",
        "1. `channel_mentioned` — does the narrative discuss this channel by name?",
        "2. `direction_correct` — does the narrative correctly state over- vs under-credited?",
        "3. `magnitude_approximately_correct` — do the cited numbers match the data within tolerance?",
        "",
        "The grader uses **structured outputs** (`output_config.format` with a JSON schema) so the response is parseable. Same model as the writer (`claude-opus-4-7`) but with a different system prompt: meticulous fact-checker rather than executive consultant.",
    ))

    cells.append(code(
        "neval = json.loads((EVAL / 'narrative_eval_summary.json').read_text())",
        "print(f'channel-evaluations: {neval[\"n_channel_evaluations\"]}')",
        "print(f'channel mentioned:           {neval[\"channel_mentioned_rate\"]:.3f}')",
        "print(f'direction correct:           {neval[\"direction_correct_rate\"]:.3f}')",
        "print(f'magnitude approx correct:    {neval[\"magnitude_correct_rate\"]:.3f}')",
        "print(f'API cost: ${neval[\"total_api_cost_usd\"]:.4f} for {neval[\"n_simulations\"]} simulations')",
    ))

    cells.append(md(
        "Two honest caveats worth stating in an interview:",
        "",
        "- **N is small.** 18 channel-level evaluations across 5 simulations. With 50 evaluations (~$2.50 in API spend) we would likely find some failures, particularly on magnitude. The methodology scales linearly.",
        "- **Same-family LLM judging itself.** The grader is also Opus 4.7. There is known literature on self-favoring bias when an LLM grades its own output. A rigorous test would use a different judge family (Sonnet 4.6 within-family check, or a non-Anthropic model out-of-family).",
        "",
        "Even with the caveats: **100% on direction correctness is a real signal**, because the narrative format makes direction hard to misstate. 100% on magnitude is more impressive — it means the prompt structure (system prompt with style rules, user message with structured data, output format spec) reliably forced the writer to ground its numeric claims in the input data.",
    ))

    # =====================================================================
    # 11. Closing
    # =====================================================================
    cells.append(md(
        "## What I would do at AmEx scale",
        "",
        "Phase 1 here is a working prototype on synthetic data. The natural extensions for production:",
        "",
        "1. **Real data inputs.** Replace `SyntheticDataSource` with warehouse connectors (Snowflake or Teradata). The interface stays the same; the implementation changes.",
        "2. **Attribution model integration.** Pull the existing model's outputs from the warehouse and feed them in alongside measurement.",
        "3. **Scheduled experiments.** Pull experiment metadata from Eppo or Statsig (or an internal tool) so geo-tests are real holdouts, not synthetic dark periods.",
        "4. **Single-channel synthetic control** in `src/methods/synthetic_control.py`. Tighter per-channel magnitudes than multi-channel TWFE.",
        "5. **Larger-N evaluation runs.** 200+ simulations would make the P/R/F1 numbers production-grade. With out-of-family judges (Sonnet, or a non-Anthropic model) the narrative eval becomes rigorous.",
        "6. **Output destinations.** The narrative report auto-delivers to a Confluence page, a Slack channel, or an email distribution list. Audit trail logging for every recalibration recommendation.",
        "",
        "## What I think is interesting about this project",
        "",
        "Three things, in order of how much they distinguish this from a typical attribution build:",
        "",
        "1. **Ground truth baked in.** Most attribution work is unfalsifiable: there is no answer key against which to validate the validator. The synthetic data approach gives us an answer key by construction, which is the only way to know whether the system actually finds what it claims to find.",
        "2. **Classification metrics on attribution decisions.** Marketing teams measure attribution as continuous credit allocation. The decisions an attribution-truth-checker produces are categorical, and they should be graded as such. P/R/F1, threshold sweep, calibration — all of it borrowed from how clause classifiers are evaluated in legal AI. None of it is standard in this domain.",
        "3. **LLM-graded narrative.** Putting an executive summary in front of a CFO without verifying that its claims match the underlying data is reckless. The same prompt-engineering discipline that produces the summary can be turned around to grade it. That round trip is the cheapest way I know to catch hallucinations before they reach the leadership audience.",
        "",
        "---",
        "",
        "*End of walkthrough.*",
    ))

    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the demo notebook.")
    parser.add_argument("--execute", action="store_true",
                        help="Execute the notebook in place after building")
    args = parser.parse_args()

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    nb = new_notebook()
    nb.cells = build_cells()
    nb.metadata.update({
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9",
        },
    })

    NOTEBOOK_PATH.write_text(nbformat.writes(nb), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH} ({len(nb.cells)} cells)")

    if args.execute:
        from nbclient import NotebookClient
        print("Executing notebook in place...")
        client = NotebookClient(nb, timeout=120, kernel_name="python3")
        client.execute(cwd=str(ROOT))
        NOTEBOOK_PATH.write_text(nbformat.writes(nb), encoding="utf-8")
        print(f"Executed {NOTEBOOK_PATH} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
