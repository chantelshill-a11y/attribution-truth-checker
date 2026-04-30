"""
City-level feature construction and nearest-neighbor matching.

In this session matching is a sanity-check companion to the TWFE estimator:
pick a treatment city for a given channel, find its closest control city by
observable features, and run a simple 2x2 DiD on a single dark period. The
answer should land in the same ballpark as TWFE; if it does not, the
regression is doing something the data does not support.

The matching infrastructure here will also be the engine for synthetic
control (`src/methods/synthetic_control.py`, future session), which builds
weighted combinations of control cities rather than picking one nearest
neighbor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_city_features(
    users: pd.DataFrame,
    exposure: pd.DataFrame,
    conversions: pd.DataFrame,
    n_weeks: int,
) -> pd.DataFrame:
    """
    Build a per-city feature DataFrame for matching.

    Features:
      n_users                          total users in the city
      pct_age_<bucket>                 share of users in each age group
      pct_income_<tier>                share of users in each income tier
      mean_demographic_multiplier      average user propensity multiplier
      conversion_rate                  conversions / (users * weeks)
      exposure_share_<channel>         exposures per user-week per channel

    All features are derived from the public data (no ground truth read).
    """
    n_users = users.groupby("city").size().rename("n_users")

    age_pct = (
        pd.crosstab(users["city"], users["age_bucket"], normalize="index")
        .add_prefix("pct_age_")
    )
    income_pct = (
        pd.crosstab(users["city"], users["income_tier"], normalize="index")
        .add_prefix("pct_income_")
    )
    mean_demo = (
        users.groupby("city")["demographic_multiplier"]
        .mean()
        .rename("mean_demographic_multiplier")
    )

    conv_with_city = conversions.merge(users[["user_id", "city"]], on="user_id")
    conv_per_city = conv_with_city.groupby("city").size()
    user_weeks = n_users * n_weeks
    conv_rate = (conv_per_city / user_weeks).rename("conversion_rate").fillna(0)

    exp_with_city = exposure.merge(users[["user_id", "city"]], on="user_id")
    exp_per_chan_city = (
        exp_with_city.groupby(["city", "channel"]).size().unstack(fill_value=0)
    )
    exposure_share = exp_per_chan_city.div(user_weeks, axis=0).add_prefix("exposure_share_")

    features = pd.concat(
        [n_users, age_pct, income_pct, mean_demo, conv_rate, exposure_share],
        axis=1,
    ).fillna(0)
    return features


def nearest_neighbor_match(
    features: pd.DataFrame,
    treatment_city: str,
    k: int = 1,
) -> list[str]:
    """
    Find the k cities most similar to `treatment_city` by Euclidean distance
    over standardized features.

    Standardization (z-score: subtract mean, divide by std) is essential
    because features live on very different scales: n_users is in the
    thousands, conversion_rate is around 0.005, exposure shares are in
    [0, 1]. Without standardization the city-size dimension would
    dominate every other dimension and "similar" would just mean
    "similar sized."
    """
    if treatment_city not in features.index:
        raise KeyError(f"treatment_city '{treatment_city}' not in features")

    std = features.std(ddof=0).replace(0, 1)
    standardized = (features - features.mean()) / std
    treatment_vec = standardized.loc[treatment_city].to_numpy()
    others = standardized.drop(treatment_city)
    distances = np.linalg.norm(others.to_numpy() - treatment_vec, axis=1)
    nearest_idx = np.argsort(distances)[:k]
    return others.index[nearest_idx].tolist()
