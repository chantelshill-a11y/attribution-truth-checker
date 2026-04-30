"""
Component 4: Claude API narrative integration.

Reads comparison.csv (Component 3 output) and generates a plain-English
executive summary using Claude Opus 4.7. Replaces the templated
recommendations from Component 3 with one cohesive narrative that ties
multiple channels together.

Demonstrates prompt engineering best practices used throughout the project:

  - System prompt defines the stable role and style constraints (cacheable
    if the prefix grew large enough, which it does not here).
  - User message carries the variable per-request task and data, structured
    with headers so the model can parse it reliably.
  - Style constraints sit in the system prompt as hard rules. Negative
    constraints ("avoid 'obviously' and 'clearly'") work better than
    positive ones ("be humble") for tone control.
  - The output format is specified explicitly, reducing variance.
  - The skip-list approach (listing channels to omit) is more reliable than
    asking the model to filter.

The Claude API call uses:
  - claude-opus-4-7              the most capable model
  - output_config.effort=medium  cost/quality sweet spot for this task size
  - no thinking                  task is constrained enough to skip
  - no caching                   payload is below the 4,096 token minimum

Reads ANTHROPIC_API_KEY from the environment. Surface a clear error if missing.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import anthropic
import pandas as pd

CLAUDE_MODEL = "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Prompt structure
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior marketing analytics consultant writing executive summaries that translate technical attribution analysis into plain-language recommendations. Your audience is a CFO, CMO, or CEO reading this on a Monday morning. They want to know what changed, by how much, and what to do.

STYLE RULES (non-negotiable):
- No em dashes anywhere. Use commas, periods, parentheses, or colons instead.
- Confident but not arrogant. Avoid the words "obviously" and "clearly" entirely.
- Every numeric claim must trace back to a specific number in the input data. Do not invent precision or round in ways that contradict the source.
- Conversational tone, not academic. Plain language an executive can absorb in one read.
- No emojis or decorative formatting.
- Do not hedge with "might" or "could" when the data supports a direct claim. Do not state certainty when the p-value is weak.

OUTPUT FORMAT:
1. **Headline** (1-2 sentences). The single most important finding, written so it could stand alone as the email subject line and first paragraph.
2. **Per-channel section**. One paragraph per channel labeled OVER_CREDITED or UNDER_CREDITED, ordered by absolute gap (largest first). Each paragraph: what the model says, what measurement shows, the size of the gap in conversions and percentage points, and the recommended action. Skip channels labeled ACCURATE.
3. **Closing recommendation** (1-2 sentences). The next step the executive should take this week.

Begin the response with the headline directly. No greeting, no "Here is the executive summary," no preamble."""


def _build_user_message(
    df: pd.DataFrame,
    total_model: float,
    total_measured: float,
    over_factor: float,
) -> str:
    """
    The variable, per-request half of the prompt. We pass:
      - The aggregate over-attribution context (frames the headline)
      - A structured per-channel data block the model can cite from
      - An explicit list of which channels to discuss vs skip

    Markdown headers help the model parse the input. Numeric formatting
    matches what the narrative will quote, removing rounding ambiguity.
    """
    flagged = df[df["label"] != "ACCURATE"].sort_values(
        "abs_share_gap_pp", ascending=False
    )
    accurate = df[df["label"] == "ACCURATE"]

    lines = [
        "Generate the executive summary based on the comparison data below.",
        "",
        "# Aggregate context",
        f"Last-touch attribution credits {total_model:,.0f} conversions to channels collectively.",
        f"Geo-lift measurement supports {total_measured:,.0f} conversions of true incremental impact.",
        f"Aggregate over-attribution factor: {over_factor:.1f}x.",
        f"Time window: 26 weeks.",
        "",
        "# Per-channel comparison (model claim vs. measured incrementality)",
    ]

    for _, r in df.iterrows():
        lines.append(
            f"- **{r['channel']}**: model claims {r['model_share_pct']:.1f}% of channel-driven credit; "
            f"measurement shows {r['measured_share_pct']:.1f}%. "
            f"Gap: {r['share_gap_pp']:+.1f} percentage points "
            f"({r['absolute_gap']:+.0f} conversions absolute). "
            f"Label: {r['label']}. "
            f"Engine p-value on the measurement: {r['measured_p_value']:.3f}."
        )

    flagged_names = ", ".join(flagged["channel"].tolist()) or "none"
    accurate_names = ", ".join(accurate["channel"].tolist()) or "none"
    lines.extend([
        "",
        "# Channels to discuss",
        f"Write one paragraph each for these channels (in this order, biggest gap first): "
        f"{', '.join(flagged['channel'].tolist())}.",
        f"Skip these channels in the per-channel section (they are within measurement noise): {accurate_names}.",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

@dataclass
class NarrativeResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    system_prompt: str
    user_message: str

    @property
    def estimated_cost_usd(self) -> float:
        # Opus 4.7: $5 per 1M input tokens, $25 per 1M output tokens
        return (self.input_tokens * 5.0 + self.output_tokens * 25.0) / 1_000_000


def generate_narrative(
    df: pd.DataFrame,
    total_model: float,
    total_measured: float,
    over_factor: float,
) -> NarrativeResult:
    """
    Call Claude Opus 4.7 to generate the executive narrative.

    Raises anthropic.AuthenticationError if ANTHROPIC_API_KEY is not set.
    Caller can catch this and surface an actionable message.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise anthropic.AuthenticationError(
            message="ANTHROPIC_API_KEY environment variable is not set.",
            response=None,  # type: ignore[arg-type]
            body=None,
        )

    client = anthropic.Anthropic()
    user_message = _build_user_message(df, total_model, total_measured, over_factor)

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": user_message}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")

    return NarrativeResult(
        text=text.strip(),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model=CLAUDE_MODEL,
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
    )
