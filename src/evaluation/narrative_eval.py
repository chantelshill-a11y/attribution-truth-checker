"""
LLM-grades-LLM narrative evaluation.

Grades each Claude-generated executive summary against the underlying
comparison data. For every channel labeled OVER_CREDITED or UNDER_CREDITED,
the grader returns three booleans:

  channel_mentioned                  was the channel discussed by name?
  direction_correct                  did the narrative state the right direction?
  magnitude_approximately_correct    did the cited numbers approximately match?

Skips ACCURATE channels (the narrative is supposed to omit them).

We aggregate across simulations to compute per-feature precision/recall
on the narrative output. This adds a second classification layer on top
of the comparison classifier evaluated in Component 6 base.

The grader uses structured outputs (output_config.format) to ensure
parseable JSON. Same model as the writer (claude-opus-4-7) but with a
different system prompt: meticulous fact-checker rather than executive
consultant.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import anthropic
import pandas as pd

CLAUDE_MODEL = "claude-opus-4-7"


GRADER_SYSTEM_PROMPT = """You are a meticulous fact-checker evaluating an AI-generated executive summary against the underlying comparison data.

Your job: verify each claim in the narrative against the structured comparison table provided, and produce a per-channel grade.

For each channel labeled OVER_CREDITED or UNDER_CREDITED in the comparison data, check whether the narrative:

1. channel_mentioned: Does the narrative explicitly discuss this channel by name?

2. direction_correct: Does the narrative correctly state whether the channel is over- or under-credited?
   - For OVER_CREDITED, the narrative should say things like "over-credited", "model attributes more than measured", "the model claims more than reality supports", or recommend reducing budget.
   - For UNDER_CREDITED, opposite: "under-credited", "doing more work than the model recognizes", "should expand investment".

3. magnitude_approximately_correct: Does the narrative include numbers (percentage points, conversion counts, or share percentages) that approximately match the data?
   - Approximate match: within 20% of the true value, or within 2 percentage points absolute for share gaps. Round-number paraphrases are fine.
   - If the narrative does not mention the channel at all, set this to false.

For ACCURATE channels, do NOT include them in the output. The narrative is supposed to skip those.

Be strict but fair. Hedging language ("appears to", "suggests", "the prudent move") is fine when the data supports the claim. Order-of-magnitude wrong numbers, reversed directions, or omissions of clearly-flagged channels are failures.

Output strict JSON exactly matching the provided schema."""


GRADER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "channels_evaluated": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "true_label": {"type": "string", "enum": ["OVER_CREDITED", "UNDER_CREDITED"]},
                    "channel_mentioned": {"type": "boolean"},
                    "direction_correct": {"type": "boolean"},
                    "magnitude_approximately_correct": {"type": "boolean"},
                    "evidence_quote": {
                        "type": "string",
                        "description": "Short verbatim quote from the narrative that justifies the verdict, or 'NOT MENTIONED' if the channel is absent."
                    },
                },
                "required": [
                    "channel",
                    "true_label",
                    "channel_mentioned",
                    "direction_correct",
                    "magnitude_approximately_correct",
                    "evidence_quote",
                ],
                "additionalProperties": False,
            },
        },
        "overall_assessment": {
            "type": "string",
            "description": "One or two sentences on the narrative's overall fidelity to the data."
        },
    },
    "required": ["channels_evaluated", "overall_assessment"],
    "additionalProperties": False,
}


@dataclass
class GradeResult:
    channels: list[dict]      # one dict per flagged channel with the three booleans
    overall_assessment: str
    input_tokens: int
    output_tokens: int

    @property
    def estimated_cost_usd(self) -> float:
        return (self.input_tokens * 5.0 + self.output_tokens * 25.0) / 1_000_000


def _format_comparison_for_grader(comparison_df: pd.DataFrame) -> str:
    """Render the comparison table as a clean reference block for the grader."""
    flagged = comparison_df[comparison_df["label"] != "ACCURATE"].sort_values(
        "abs_share_gap_pp", ascending=False
    )
    accurate = comparison_df[comparison_df["label"] == "ACCURATE"]
    lines = ["# Comparison data (ground truth for this grading task)"]
    lines.append("")
    lines.append("## Channels labeled OVER_CREDITED or UNDER_CREDITED (these MUST be discussed in the narrative):")
    for _, r in flagged.iterrows():
        lines.append(
            f"- **{r['channel']}** ({r['label']}): model claims {r['model_share_pct']:.1f}% of "
            f"channel-driven credit, measurement shows {r['measured_share_pct']:.1f}%, "
            f"gap = {r['share_gap_pp']:+.1f}pp ({r['absolute_gap']:+.0f} conversions absolute). "
            f"p-value = {r['measured_p_value']:.3f}."
        )
    if len(accurate):
        lines.append("")
        lines.append("## Channels labeled ACCURATE (these SHOULD be omitted from the narrative):")
        for _, r in accurate.iterrows():
            lines.append(
                f"- {r['channel']}: model {r['model_share_pct']:.1f}%, "
                f"measured {r['measured_share_pct']:.1f}%, gap {r['share_gap_pp']:+.1f}pp."
            )
    return "\n".join(lines)


def grade_narrative(
    narrative_text: str,
    comparison_df: pd.DataFrame,
    client: anthropic.Anthropic | None = None,
) -> GradeResult:
    """
    Send the narrative + comparison data to Claude and return per-channel grades.

    The model is instructed to use structured outputs (JSON schema) so the
    response is parseable without regex.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    if client is None:
        client = anthropic.Anthropic()

    user_message = (
        f"{_format_comparison_for_grader(comparison_df)}\n\n"
        f"# Narrative to grade\n\n"
        f"{narrative_text}\n\n"
        f"# Task\n"
        f"Grade the narrative per channel as instructed. Return strict JSON."
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=GRADER_SYSTEM_PROMPT,
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": GRADER_OUTPUT_SCHEMA},
        },
        messages=[{"role": "user", "content": user_message}],
    )

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    parsed = json.loads(text)

    return GradeResult(
        channels=parsed["channels_evaluated"],
        overall_assessment=parsed["overall_assessment"],
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def aggregate_grades(grade_results: list[GradeResult]) -> dict:
    """
    Roll grades up across multiple narratives into per-feature accuracy.

    Returns a dict with mention_rate, direction_correct_rate,
    magnitude_correct_rate, plus per-class breakdowns and totals.
    """
    rows = []
    for g in grade_results:
        for c in g.channels:
            rows.append(c)

    if not rows:
        return {
            "n_channel_evaluations": 0,
            "channel_mentioned_rate": 0.0,
            "direction_correct_rate": 0.0,
            "magnitude_correct_rate": 0.0,
            "by_true_label": {},
        }

    df = pd.DataFrame(rows)
    summary = {
        "n_channel_evaluations": len(df),
        "channel_mentioned_rate": float(df["channel_mentioned"].mean()),
        "direction_correct_rate": float(df["direction_correct"].mean()),
        "magnitude_correct_rate": float(df["magnitude_approximately_correct"].mean()),
        "by_true_label": {},
    }
    for label, sub in df.groupby("true_label"):
        summary["by_true_label"][label] = {
            "n": int(len(sub)),
            "channel_mentioned_rate": float(sub["channel_mentioned"].mean()),
            "direction_correct_rate": float(sub["direction_correct"].mean()),
            "magnitude_correct_rate": float(sub["magnitude_approximately_correct"].mean()),
        }
    return summary
