"""
CLI runner for Component 4: the Claude API narrative.

Reads data/synthetic/comparison.csv (Component 3 output), calls Claude
Opus 4.7 to generate an executive summary, prints the prompt structure
and the result for inspection, and writes the narrative to
data/synthetic/narrative.md.

Requires ANTHROPIC_API_KEY in the environment.

Run from the project root:
  .venv\\Scripts\\python.exe scripts/run_narrative.py

Optional flags:
  --show-prompt   Print the full system + user prompt before the response.
                  Useful for prompt engineering inspection.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anthropic
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.narrative import (  # noqa: E402
    CLAUDE_MODEL,
    SYSTEM_PROMPT,
    NarrativeResult,
    generate_narrative,
)

DATA = ROOT / "data" / "synthetic"
OUT_PATH = DATA / "narrative.md"


def _check_api_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
    print()
    print("Set it for the current PowerShell session with:")
    print('  $env:ANTHROPIC_API_KEY = "sk-ant-..."')
    print()
    print("To set it permanently for your user account:")
    print('  [Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-...", "User")')
    print()
    print("Get a key at https://console.anthropic.com/settings/keys")
    sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the executive narrative via Claude API.")
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the system and user prompts before the response.",
    )
    args = parser.parse_args()

    _check_api_key()

    if not (DATA / "comparison.csv").exists():
        print("ERROR: data/synthetic/comparison.csv not found. Run Component 3 first:")
        print("  .venv\\Scripts\\python.exe scripts/run_comparison.py")
        sys.exit(1)

    df = pd.read_csv(DATA / "comparison.csv")
    total_model = float(df["model_attributed_conversions"].sum())
    total_measured = float(df["measured_incremental_conversions"].sum())
    over_factor = total_model / total_measured if total_measured > 0 else 0.0

    if args.show_prompt:
        print("=" * 92)
        print("SYSTEM PROMPT")
        print("=" * 92)
        print(SYSTEM_PROMPT)
        print()
        print("=" * 92)
        print("USER MESSAGE")
        print("=" * 92)
        # The user message is built inside generate_narrative; reconstruct for display.
        from src.narrative import _build_user_message
        print(_build_user_message(df, total_model, total_measured, over_factor))
        print()

    print(f"Calling {CLAUDE_MODEL} with effort=medium...")
    try:
        result: NarrativeResult = generate_narrative(df, total_model, total_measured, over_factor)
    except anthropic.AuthenticationError as e:
        print(f"ERROR: authentication failed. {e}")
        sys.exit(2)
    except anthropic.RateLimitError:
        print("ERROR: rate limited. Wait a moment and rerun.")
        sys.exit(3)
    except anthropic.APIStatusError as e:
        print(f"ERROR: API status {e.status_code}: {e.message}")
        sys.exit(4)

    print()
    print("=" * 92)
    print("NARRATIVE")
    print("=" * 92)
    print(result.text)
    print()
    print("=" * 92)
    print(
        f"USAGE: {result.input_tokens:,} input + {result.output_tokens:,} output tokens "
        f"(estimated cost ${result.estimated_cost_usd:.4f})"
    )
    print("=" * 92)

    OUT_PATH.write_text(result.text + "\n", encoding="utf-8")
    print(f"\nNarrative saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
