"""
Synthetic data generator for the Attribution Truth-Checker.

Produces five CSV files. Four are public to the rest of the system, one is
the answer key that downstream analysis must not read.

  data/synthetic/users.csv
  data/synthetic/channel_exposure.csv
  data/synthetic/conversions.csv
  data/synthetic/model_attribution.csv      (last-touch model claim)
  data/synthetic/_ground_truth/ground_truth.csv   (answer key, do not read from analysis)

Run with:

  python -m src.generator --config config/default.yaml --output data/synthetic

The generator encodes a known causal structure into the data. Each channel
has a deliberately-set true incremental conversion rate, and as conversions
are drawn we record fractional credit to each component of the conversion
hazard. The resulting per-channel sums are the true incremental
conversions, which the geo-lift engine (next session) should rediscover
from the public data alone.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

@dataclass
class ChannelConfig:
    name: str
    label: str
    true_incremental_rate: float
    last_touch_share_target: float
    reach: float


@dataclass
class GeneratorConfig:
    random_seed: int
    n_cities: int
    n_weeks: int
    n_users: int
    baseline_conversion_rate: float
    channels: list[ChannelConfig]
    dark_period_fraction: float
    dark_period_min_weeks: int


def load_config(path: Path) -> GeneratorConfig:
    raw = yaml.safe_load(path.read_text())
    return GeneratorConfig(
        random_seed=raw["random_seed"],
        n_cities=raw["n_cities"],
        n_weeks=raw["n_weeks"],
        n_users=raw["n_users"],
        baseline_conversion_rate=raw["baseline_conversion_rate"],
        channels=[ChannelConfig(**c) for c in raw["channels"]],
        dark_period_fraction=raw["dark_period"]["fraction_of_city_weeks"],
        dark_period_min_weeks=raw["dark_period"]["min_consecutive_weeks"],
    )


def get_city_names(n_cities: int) -> list[str]:
    return [f"city_{i:02d}" for i in range(n_cities)]


# ---------------------------------------------------------------------------
# Step 1: users
# ---------------------------------------------------------------------------

def build_users(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    """
    Build the users table.

    Cities have uneven sizes (5 large, 15 medium, 30 small) so that the
    matching layer in the next session has real variance in observable
    features.

    Each user gets:
      city                     which city they live in
      age_bucket               one of five age groups
      income_tier              low / mid / high
      demographic_multiplier   per-user multiplier on the baseline
                               conversion rate. Higher-income, prime-age
                               (35-49) users have higher propensity to
                               sign up for a premium credit card. Range about 0.30
                               to 2.10, centered near 1.0.
    """
    n_users = cfg.n_users
    city_names = get_city_names(cfg.n_cities)

    # 5 large cities take 50% of users, 15 medium take 35%, 30 small take 15%.
    n_large, n_medium, n_small = 5, 15, 30
    assert n_large + n_medium + n_small == cfg.n_cities, "city bucket counts must sum to n_cities"
    per_large = int(n_users * 0.50 / n_large)
    per_medium = int(n_users * 0.35 / n_medium)
    per_small = int(n_users * 0.15 / n_small)
    targets = [per_large] * n_large + [per_medium] * n_medium + [per_small] * n_small
    targets[0] += n_users - sum(targets)  # absorb rounding into largest city

    city_assignment = np.repeat(city_names, targets)
    assert len(city_assignment) == n_users

    # Demographic distributions, calibrated to look roughly U.S.-adult-ish.
    age_buckets = np.array(["18-24", "25-34", "35-49", "50-64", "65+"])
    age_weights = np.array([0.15, 0.25, 0.30, 0.20, 0.10])
    income_tiers = np.array(["low", "mid", "high"])
    income_weights = np.array([0.30, 0.45, 0.25])

    ages = rng.choice(age_buckets, size=n_users, p=age_weights)
    incomes = rng.choice(income_tiers, size=n_users, p=income_weights)

    age_mult_table = {"18-24": 0.5, "25-34": 1.0, "35-49": 1.5, "50-64": 1.2, "65+": 0.7}
    income_mult_table = {"low": 0.6, "mid": 1.0, "high": 1.4}
    age_mult_arr = np.vectorize(age_mult_table.get)(ages)
    income_mult_arr = np.vectorize(income_mult_table.get)(incomes)
    demo_mult = age_mult_arr * income_mult_arr

    return pd.DataFrame({
        "user_id": np.arange(n_users, dtype=np.int64),
        "city": city_assignment,
        "age_bucket": ages,
        "income_tier": incomes,
        "demographic_multiplier": demo_mult.astype(np.float64),
    })


# ---------------------------------------------------------------------------
# Step 2: dark-period mask (which channels are active in which city-weeks)
# ---------------------------------------------------------------------------

def build_dark_period_mask(cfg: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Build a boolean array of shape (n_cities, n_channels, n_weeks).

    True  = channel is ACTIVE in that city that week (users may be exposed)
    False = dark period (channel held out)

    For each (city, channel) we want roughly cfg.dark_period_fraction of
    weeks dark, in stretches of at least dark_period_min_weeks consecutive
    weeks. Long dark stretches matter because a 1-week dip is just noise.
    Geo-lift needs sustained absence to detect anything.

    We do this by repeatedly picking random start weeks and marking
    [start, start+length) as dark, until enough weeks per (city, channel)
    are dark. Length is a small uniform draw above the minimum.
    """
    n_cities, n_weeks = cfg.n_cities, cfg.n_weeks
    n_channels = len(cfg.channels)
    target_dark = int(round(n_weeks * cfg.dark_period_fraction))
    min_run = cfg.dark_period_min_weeks

    active = np.ones((n_cities, n_channels, n_weeks), dtype=bool)

    for c_idx in range(n_cities):
        for ch_idx in range(n_channels):
            dark_count = 0
            attempts = 0
            while dark_count < target_dark and attempts < 50:
                run_len = int(rng.integers(min_run, min_run + 4))
                latest_start = max(1, n_weeks - run_len)
                start = int(rng.integers(0, latest_start))
                already_dark_in_run = (~active[c_idx, ch_idx, start:start + run_len]).sum()
                newly_dark = run_len - already_dark_in_run
                active[c_idx, ch_idx, start:start + run_len] = False
                dark_count += newly_dark
                attempts += 1

    return active


# ---------------------------------------------------------------------------
# Step 3: channel exposure
# ---------------------------------------------------------------------------

def build_exposure(
    users: pd.DataFrame,
    cfg: GeneratorConfig,
    active_mask: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Build the channel_exposure table.

    For each (user, week, channel):
      If channel is dark in user's city that week, no exposure.
      Else, with probability = channel.reach, the user is exposed.

    Exposures are independent across channels, weeks, and users. This is a
    simplification: real-world exposure is correlated (a user who saw a
    Google search ad is more likely to also see a YouTube ad). For Phase 1
    independence is fine because it does not affect the incrementality
    math, only the realism of cross-channel patterns.

    Vectorized: 100k users x 26 weeks x 6 channels = 15.6M cells, drawn as
    one matrix per channel.
    """
    n_users = len(users)
    n_weeks = cfg.n_weeks
    city_names = get_city_names(cfg.n_cities)
    city_to_idx = {c: i for i, c in enumerate(city_names)}
    user_city_idx = users["city"].map(city_to_idx).to_numpy()

    frames = []
    for ch_idx, ch in enumerate(cfg.channels):
        # active_for_users shape: (n_users, n_weeks). True if channel active in this user's city this week.
        active_for_users = active_mask[user_city_idx, ch_idx, :]
        # Bernoulli draw per (user, week) for whether this user was exposed,
        # conditional on the channel being active in their city.
        draws = rng.random(size=(n_users, n_weeks))
        exposed = active_for_users & (draws < ch.reach)

        u_idx, w_idx = np.where(exposed)
        if len(u_idx) > 0:
            frames.append(pd.DataFrame({
                "user_id": users["user_id"].to_numpy()[u_idx],
                "channel": ch.name,
                "week": w_idx.astype(np.int64),
                "exposure_count": np.ones(len(u_idx), dtype=np.int64),
            }))

    if not frames:
        return pd.DataFrame(columns=["user_id", "channel", "week", "exposure_count"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 4: conversions and ground truth
# ---------------------------------------------------------------------------

def build_conversions_and_ground_truth(
    users: pd.DataFrame,
    exposure: pd.DataFrame,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build conversions and the ground-truth answer key in one pass.

    Each user has at most one conversion event over the n_weeks horizon
    (a user who signs up for a premium credit card does not sign up again next
    week). We process week by week. For each not-yet-converted user, we
    compute their hazard, the probability of converting THIS week:

      hazard = baseline * demographic_multiplier
               + sum over channels they were exposed to this week of
                 channel.true_incremental_rate

    We draw a Bernoulli outcome at that probability. If the user converts,
    we attribute the conversion fractionally to each component of the
    hazard, weighted by that component's share of the total hazard. The
    sum across all conversions of each channel's fractional credit is
    that channel's true incremental conversions: the answer key.

    A "Bernoulli draw" is a single weighted coin flip. With probability
    0.012 it returns 1 (a conversion), otherwise 0.
    """
    n_users = len(users)
    n_weeks = cfg.n_weeks
    n_channels = len(cfg.channels)
    channel_names = [c.name for c in cfg.channels]
    channel_rates = np.array([c.true_incremental_rate for c in cfg.channels], dtype=np.float64)
    name_to_idx = {n: i for i, n in enumerate(channel_names)}

    # exposed_tensor[user_id, week, channel_idx] = was this user exposed?
    # 100k * 26 * 6 = 15.6M booleans = 15.6 MB. Fine.
    exposed_tensor = np.zeros((n_users, n_weeks, n_channels), dtype=bool)
    if len(exposure) > 0:
        ch_idx_arr = exposure["channel"].map(name_to_idx).to_numpy()
        exposed_tensor[
            exposure["user_id"].to_numpy(),
            exposure["week"].to_numpy(),
            ch_idx_arr,
        ] = True

    demo_mult = users["demographic_multiplier"].to_numpy()
    baseline = cfg.baseline_conversion_rate

    not_converted = np.ones(n_users, dtype=bool)
    conversion_week = np.full(n_users, -1, dtype=np.int64)
    truth_credit = np.zeros(n_channels + 1, dtype=np.float64)  # last slot = baseline / direct

    for w in range(n_weeks):
        baseline_term = baseline * demo_mult                          # (n_users,)
        channel_terms = exposed_tensor[:, w, :] * channel_rates        # (n_users, n_channels)
        total_hazard = baseline_term + channel_terms.sum(axis=1)       # (n_users,)

        draws = rng.random(n_users)
        converts_this_week = not_converted & (draws < total_hazard)

        if converts_this_week.any():
            conversion_week[converts_this_week] = w
            # Vectorized fractional credit: only count newly-converted users.
            safe_total = np.where(total_hazard > 0, total_hazard, 1.0)
            mask = converts_this_week
            truth_credit[-1] += np.sum(baseline_term[mask] / safe_total[mask])
            contrib = channel_terms[mask] / safe_total[mask, None]      # (n_converters, n_channels)
            truth_credit[:-1] += contrib.sum(axis=0)
            not_converted[converts_this_week] = False

    converted_idx = np.where(conversion_week >= 0)[0]
    conversions_df = pd.DataFrame({
        "user_id": users["user_id"].to_numpy()[converted_idx],
        "conversion_week": conversion_week[converted_idx],
    })

    total_credit = truth_credit.sum()
    if total_credit <= 0:
        total_credit = 1.0  # avoid divide-by-zero in the rare empty case
    gt_rows = []
    for ch_idx, ch in enumerate(cfg.channels):
        gt_rows.append({
            "channel": ch.name,
            "true_incremental_conversions": float(truth_credit[ch_idx]),
            "true_share": float(truth_credit[ch_idx] / total_credit),
            "label": ch.label,
        })
    gt_rows.append({
        "channel": "_baseline_direct",
        "true_incremental_conversions": float(truth_credit[-1]),
        "true_share": float(truth_credit[-1] / total_credit),
        "label": "BASELINE",
    })
    ground_truth_df = pd.DataFrame(gt_rows)

    return conversions_df, ground_truth_df


# ---------------------------------------------------------------------------
# Step 5: last-touch attribution model (the deliberately-broken model)
# ---------------------------------------------------------------------------

def run_last_touch_model(
    conversions: pd.DataFrame,
    exposure: pd.DataFrame,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Last-touch attribution: for each conversion, find the channel of the
    most recent exposure on or before the conversion week. If no exposure
    in the user's history, label it "direct."

    This model is a strawman. It is deliberately simple and deliberately
    wrong in known ways. The point of the truth-checker is to expose those
    wrong ways using independent measurement. Real production models are
    more sophisticated (multi-touch, time-decay, data-driven Markov), but
    they share the same fundamental limitation: they assign credit based
    on observed touchpoint patterns and have no causal grounding.

    Tiebreak rule when multiple channels touched a user in the same week:
    a uniform random draw, seeded for reproducibility. Alphabetical
    tiebreak would systematically advantage early-alphabet channel names,
    which is a generator artifact unrelated to attribution behavior.
    """
    if conversions.empty:
        rows = [
            {"channel": ch.name, "attributed_conversions": 0, "attributed_share": 0.0}
            for ch in cfg.channels
        ]
        rows.append({"channel": "direct", "attributed_conversions": 0, "attributed_share": 0.0})
        return pd.DataFrame(rows)

    # Filter exposure to converting users only, then merge.
    converted_user_ids = conversions["user_id"]
    exposure_sub = exposure[exposure["user_id"].isin(converted_user_ids)]
    merged = exposure_sub.merge(conversions, on="user_id", how="inner")
    eligible = merged[merged["week"] <= merged["conversion_week"]].copy()

    # Random tiebreak: assign each row a uniform draw, sort by it within ties.
    eligible["_tiebreak"] = rng.random(len(eligible))
    eligible = eligible.sort_values(
        ["user_id", "week", "_tiebreak"], ascending=[True, False, True]
    )
    last_touch = eligible.drop_duplicates("user_id", keep="first")

    direct_users = set(converted_user_ids) - set(last_touch["user_id"])
    direct_count = len(direct_users)

    counts = last_touch["channel"].value_counts()
    total_conv = len(conversions)

    rows = []
    for ch in cfg.channels:
        n = int(counts.get(ch.name, 0))
        rows.append({
            "channel": ch.name,
            "attributed_conversions": n,
            "attributed_share": n / total_conv,
        })
    rows.append({
        "channel": "direct",
        "attributed_conversions": direct_count,
        "attributed_share": direct_count / total_conv,
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic attribution data.")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for CSVs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.random_seed)

    out_public = args.output
    out_truth = args.output / "_ground_truth"
    out_public.mkdir(parents=True, exist_ok=True)
    out_truth.mkdir(parents=True, exist_ok=True)

    print(
        f"Loaded config: {cfg.n_users:,} users, {cfg.n_cities} cities, "
        f"{cfg.n_weeks} weeks, {len(cfg.channels)} channels, seed={cfg.random_seed}"
    )

    print("Building users...")
    users = build_users(cfg, rng)
    print(f"  {len(users):,} users across {users['city'].nunique()} cities")

    print("Building dark-period mask...")
    active_mask = build_dark_period_mask(cfg, rng)
    dark_pct = (1 - active_mask.mean()) * 100
    print(f"  {dark_pct:.1f}% of city-channel-week cells are dark")

    print("Building exposure...")
    exposure = build_exposure(users, cfg, active_mask, rng)
    print(f"  {len(exposure):,} exposure events")

    print("Building conversions and ground truth...")
    conversions, ground_truth = build_conversions_and_ground_truth(users, exposure, cfg, rng)
    print(f"  {len(conversions):,} conversions ({len(conversions) / len(users) * 100:.2f}% of users)")

    print("Running last-touch model...")
    model_attribution = run_last_touch_model(conversions, exposure, cfg, rng)
    print("  done")

    users.to_csv(out_public / "users.csv", index=False)
    exposure.to_csv(out_public / "channel_exposure.csv", index=False)
    conversions.to_csv(out_public / "conversions.csv", index=False)
    model_attribution.to_csv(out_public / "model_attribution.csv", index=False)
    ground_truth.to_csv(out_truth / "ground_truth.csv", index=False)

    print(f"\nWrote 4 public CSVs to {out_public}")
    print(f"Wrote answer key to {out_truth / 'ground_truth.csv'} (do not consume from analysis)")


if __name__ == "__main__":
    main()
