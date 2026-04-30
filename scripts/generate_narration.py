"""
Generate cloned-voice narration MP3s for the project site via ElevenLabs.

Reads ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID from the environment,
then synthesizes one MP3 per walkthrough panel and saves to site/audio/.

Usage:
  scripts/generate_narration.py --sample      # generate only sample-overview.mp3
  scripts/generate_narration.py                # generate all 7 narrations

Voice settings are tuned for content narration: balanced stability,
strong similarity to source recording, neutral style. Tune below if the
sample comes back off in any dimension.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "site" / "audio"

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

VOICE_SETTINGS = {
    "stability": 0.30,        # lower = more expressive (counter-intuitive: 0.5 sounded monotone)
    "similarity_boost": 0.60, # slightly lower than default 0.75 so the model isn't locked to source flatness
    "style": 0.45,             # adds emotional/intonation variety
    "use_speaker_boost": True,
}
MODEL_ID = "eleven_multilingual_v2"


# ---------------------------------------------------------------------------
# Narration scripts. Adapted from the walkthrough panels on each page,
# lightly edited for spoken delivery (no HTML tags, looser pacing).
# ---------------------------------------------------------------------------

NARRATIONS: list[tuple[str, str]] = [
    (
        "walkthrough-overview.mp3",
        """Hi. What you're looking at is a tool that does something most marketing teams don't: hold their attribution model accountable to causal reality. Attribution is the bookkeeping that says "this channel drove that conversion." Incrementality is the harder question of whether the conversion would have happened anyway. Most teams treat attribution as if it answered both. It doesn't.

The chart up top is the entire project compressed into one image. The charcoal bars are what a last-touch attribution model claims each channel contributed. The forest-green bars are what a geo-lift experiment actually measured. Where the two bars disagree, the model is misallocating credit. Direct mail on the far left is the most-misallocated channel in this run, getting only 7.7 percent of credit when reality says it deserves 27.

Below this panel, four headline numbers. Below that, six cards. Each card is a self-contained walkthrough of one piece of how the system works, written in this same plain-language voice. There is no required order. Click whatever looks interesting.""",
    ),
    (
        "walkthrough-synthetic-data.mp3",
        """Real attribution data has no answer key. That's the entire problem. The question "did this channel actually drive this conversion" is unanswerable from production data alone, because you only ever see one universe: the one where the channel was on. To validate any attribution method, we need data where we know the answer in advance. So that's what I built.

The mechanism is what statisticians call a hazard model. For each user-week, the conversion probability is a baseline rate plus a per-channel boost for every channel the user was exposed to that week. We draw a coin flip at that probability, and we keep an honest tally of which channel deserves credit for each conversion. The tally is the answer key.

The chart you see compares the answer key to what a deliberately-broken last-touch attribution model claims. Display is wildly over-credited. TV brand is wildly under-credited. The pattern is exactly what the rest of the system has to detect from public data alone, without ever consulting the answer key.""",
    ),
    (
        "walkthrough-geo-lift.mp3",
        """Now the hard part: recover the answer key from public data without peeking at it. The technique is called geo-lift, and the math underneath it is something called difference-in-differences.

Imagine two cities, Phoenix and Tucson. Phoenix runs a TV campaign. Tucson does not. Phoenix's conversions go up by three percent during the campaign. Tucson's go up by one percent. Both cities probably had some shared trend, like seasonality. The lift attributable to TV is the difference of those differences: two percent. Subtraction twice. Once to remove the city's own baseline. Once to remove the common trend with the control. What is left is the causal effect.

We scale that two-by-two logic up across 50 cities and 26 weeks using a regression that absorbs every city's individual baseline and every week's average shock, leaving only the channel-on-or-off variation as the source of truth. The technical name is two-way fixed effects.""",
    ),
    (
        "walkthrough-truth-check.mp3",
        """This is the headline of the entire project. The charcoal bars are what the attribution model claims each channel contributed. The forest-green bars are what we measured actually happened, using the geo-lift engine on the previous page. The number above each pair is the gap, in percentage points.

The comparison is on shares, not absolute conversion counts. Last-touch attribution attributes nearly all conversions to channels because it has no concept of baseline conversions. In absolute terms, every channel looks over-credited by 2 to 5 times. The interesting question is which channels are relatively over- or under-credited compared to their actual causal contribution.

Below the chart is the interactive part. The default threshold is 5 percentage points. Drag the slider to see how the labels recompute. As you raise the threshold, channels that were borderline drop into the accurate bucket. As you lower it, more get flagged. This is the kind of decision you would actually make in production.""",
    ),
    (
        "walkthrough-narrative.mp3",
        """The truth-check on the previous page is right in numbers. A CFO does not read tables. A CFO reads paragraphs. So the next layer of the system feeds the comparison data to a Claude API call with a carefully-structured prompt and gets back an executive summary.

Three prompt-engineering decisions matter here. First, the style rules sit in the system prompt as hard constraints: no em dashes, no "obviously" or "clearly," every numeric claim must trace back to a specific number in the data. System prompts are stable across calls and set the role; the user message is what varies per request.

Second, negative constraints work better than positive ones for tone control. "Avoid 'obviously' and 'clearly'" is more effective than "be humble." Telling a model what not to do is concrete; telling it what to do is interpretable in too many ways.

Third, the skip-list approach. The user message includes an explicit list of channels to discuss versus channels to omit. Asking the model to filter on its own is unreliable.""",
    ),
    (
        "walkthrough-self-evaluation.mp3",
        """Marketing teams almost never apply classification metrics to attribution decisions. Attribution outputs look continuous, like credit shares, not categorical. But the decisions this system outputs are categorical: over-credited, under-credited, or accurate. So we should grade them the same way we grade any classifier.

What you are looking at on this page is the validator, validated. The confusion matrix shows where the system's labels match the configured ground truth, and where they confuse. The threshold sweep shows where the F1-optimal cutoff is. The calibration check tells us whether the system's stated confidence matches its empirical accuracy. The narrative evaluation has Claude grade Claude on whether the executive summary's claims are consistent with the comparison data.

This page is the methodological discipline most attribution work skips. The framing is borrowed from how clause classifiers are evaluated in legal AI, applied to a domain that does not usually receive it.""",
    ),
    (
        "manifesto.mp3",
        """Hi, I'm Chantel Hill. I'm an AI Consultant.

I help teams ship AI systems that stakeholders actually trust. That means hands-on work, not just advisory. I help define what an AI system should do for a business, build a working version of it, and grade whether it's reliable before anyone deploys it.

I came up in legal AI, specifically contract clause extraction. The discipline I learned there, measuring AI accuracy with precision, recall, and F1, treating ambiguous decisions as classification problems even when they look continuous, that discipline applies anywhere AI has to earn trust before it goes live.

The featured project on this site is a marketing attribution truth-checker. I built it deliberately outside of legal to prove the methodology transfers between domains. Synthetic data with known ground truth, geo-lift validation, classifier-style evaluation on top. The system caught a 19 percentage-point under-credit on a channel the attribution model was telling the business to cut.

If you're shipping AI into production and you need a consultant who measures trust the same way they measure accuracy, that's the work I do.""",
    ),
]


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def generate_one(filename: str, text: str, voice_id: str, api_key: str) -> int:
    """
    Call ElevenLabs TTS, save the resulting MP3 to site/audio/<filename>.
    Returns the size in bytes of the saved file.
    """
    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)
    body = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": VOICE_SETTINGS,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    out_path = AUDIO_DIR / filename
    with urllib.request.urlopen(req) as resp:
        audio_bytes = resp.read()
    out_path.write_bytes(audio_bytes)
    return len(audio_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true",
                        help="Only generate the first (sample-overview.mp3) file")
    args = parser.parse_args()

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID")
    if not api_key or not voice_id:
        print("ERROR: ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID must be set.")
        sys.exit(1)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    targets = NARRATIONS[:1] if args.sample else NARRATIONS
    print(f"Generating {len(targets)} narration{'s' if len(targets) != 1 else ''}...")
    print(f"Voice ID: {voice_id}")
    print(f"Settings: stability={VOICE_SETTINGS['stability']}, "
          f"similarity={VOICE_SETTINGS['similarity_boost']}, "
          f"model={MODEL_ID}")
    print()

    total_chars = 0
    total_bytes = 0
    for filename, text in targets:
        char_count = len(text)
        total_chars += char_count
        t0 = time.time()
        try:
            size = generate_one(filename, text, voice_id, api_key)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            print(f"  FAILED {filename}: HTTP {e.code} - {error_body[:200]}")
            continue
        elapsed = time.time() - t0
        total_bytes += size
        kb = size // 1024
        print(f"  {filename:<42} {char_count:>5} chars -> {kb:>4} KB in {elapsed:.1f}s")

    print()
    print(f"Total: {total_chars:,} characters -> {total_bytes // 1024:,} KB")
    print(f"Audio saved to {AUDIO_DIR}")


if __name__ == "__main__":
    main()
