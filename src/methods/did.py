"""
Two-way fixed-effects (TWFE) difference-in-differences estimator for
per-channel incrementality, with cluster-robust standard errors.

Inputs: users, channel_exposure, conversions (the public synthetic data).
Output: per-channel point estimate, 90% CI, p-value, measured incremental
conversions with CI.

Why this estimator: see the methodology walkthrough in the project. In short,
each channel's beta is the within-city, within-week effect of the channel
being active vs. dark on conversion rate. City and week fixed effects absorb
everything that is constant within a city or constant within a week, leaving
only the dark-period variation as the source of identification.

This module deliberately does NOT depend on statsmodels, both because
statsmodels has no ARM64 wheels on Windows and because hand-rolling the
math makes the methodology visible. Every computation here corresponds to
one line in a textbook OLS derivation.

This module also does NOT read data/synthetic/_ground_truth/. The whole
point of the engine is to recover the truth from the public data alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Normal-distribution helpers (drop-in replacement for scipy.stats.norm.cdf
# and scipy.stats.norm.ppf). With cluster count G >= 30, the t-distribution
# with df = G-1 is close enough to normal that the difference is well below
# the noise floor of the synthetic experiment. This avoids the scipy dep
# entirely, which is blocked by Windows Application Control on this machine.
# ---------------------------------------------------------------------------

def _normal_cdf(z: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """
    Inverse standard normal CDF using Peter Acklam's rational approximation.
    Accurate to ~1e-9, far below any precision we need for confidence intervals.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)


@dataclass
class DiDResult:
    """One row of the geo-lift output, per channel."""
    channel: str
    incremental_rate: float
    standard_error: float
    t_stat: float
    p_value: float
    ci_low_90: float
    ci_high_90: float
    measured_incremental_conversions: float
    incr_conv_ci_low_90: float
    incr_conv_ci_high_90: float
    active_user_weeks: float


# ---------------------------------------------------------------------------
# Step 1: build the (city, week) panel
# ---------------------------------------------------------------------------

def build_panel(
    users: pd.DataFrame,
    exposure: pd.DataFrame,
    conversions: pd.DataFrame,
    n_weeks: int,
    channel_names: list[str],
) -> pd.DataFrame:
    """
    Reshape the user-level synthetic data into a city-week panel.

    Output columns:
      city, week, n_users, conversions, conversion_rate
      <channel>_active for each channel in channel_names (0 or 1)

    The panel has n_cities * n_weeks rows. Conversion rate is the count
    of conversions in that city-week divided by total city users. We use
    total users (not just not-yet-converted users) in the denominator
    for simplicity. The constant-within-city bias this introduces is
    absorbed by the city fixed effect.

    "Channel active" means: at least one user in that city had at least
    one exposure to that channel in that week. This rebuilds the dark-
    period structure from the public exposure data, without consulting
    the mask the generator used internally.
    """
    cities = sorted(users["city"].unique())
    n_users_per_city = users.groupby("city").size().rename("n_users")

    panel = pd.DataFrame(
        [(c, w) for c in cities for w in range(n_weeks)],
        columns=["city", "week"],
    )
    panel = panel.merge(n_users_per_city.reset_index(), on="city")

    # Conversions per (city, week). A user converts at most once.
    conv_with_city = conversions.merge(users[["user_id", "city"]], on="user_id")
    conv_counts = (
        conv_with_city.groupby(["city", "conversion_week"])
        .size()
        .rename("conversions")
        .reset_index()
        .rename(columns={"conversion_week": "week"})
    )
    panel = panel.merge(conv_counts, on=["city", "week"], how="left")
    panel["conversions"] = panel["conversions"].fillna(0).astype(np.int64)
    panel["conversion_rate"] = panel["conversions"] / panel["n_users"]

    # Channel activity per (city, week): 1 if any exposure exists, else 0.
    exp_with_city = exposure.merge(users[["user_id", "city"]], on="user_id")
    activity = (
        exp_with_city.groupby(["city", "week", "channel"])
        .size()
        .unstack(fill_value=0)
    ) > 0
    activity = activity.astype(np.int64).reset_index()

    # Defensive: if a channel never had any exposure in the data, add a 0 column.
    for ch in channel_names:
        if ch not in activity.columns:
            activity[ch] = 0
    activity = activity.rename(columns={ch: f"{ch}_active" for ch in channel_names})

    active_cols = [f"{ch}_active" for ch in channel_names]
    panel = panel.merge(
        activity[["city", "week"] + active_cols],
        on=["city", "week"],
        how="left",
    )
    for col in active_cols:
        panel[col] = panel[col].fillna(0).astype(np.int64)

    return panel


# ---------------------------------------------------------------------------
# Step 2: TWFE OLS with cluster-robust SE
# ---------------------------------------------------------------------------

def _build_design_matrix(
    panel: pd.DataFrame,
    channel_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Construct the OLS design matrix X.

    Columns, in order:
      const                                            1
      <channel>_active for each channel                k_channels
      city dummies, with the first city dropped        n_cities - 1
      week dummies, with the first week dropped        n_weeks - 1

    Dropping one city and one week dummy avoids perfect collinearity with
    the intercept (the standard "drop reference level" treatment for
    fixed effects in regression).
    """
    activity_cols = [f"{ch}_active" for ch in channel_names]
    city_dummies = pd.get_dummies(panel["city"], prefix="city", drop_first=True, dtype=float)
    week_dummies = pd.get_dummies(panel["week"], prefix="week", drop_first=True, dtype=float)

    X_df = pd.concat(
        [
            pd.Series(1.0, index=panel.index, name="const"),
            panel[activity_cols].astype(float),
            city_dummies,
            week_dummies,
        ],
        axis=1,
    )
    return X_df.to_numpy(), X_df.columns.tolist()


def fit_twfe_with_cluster_se(
    panel: pd.DataFrame,
    channel_names: list[str],
    confidence_level: float = 0.90,
) -> list[DiDResult]:
    """
    Two-way fixed-effects OLS with cluster-robust standard errors.

    Model:
        conversion_rate[city, week]
            = alpha
              + beta_search   * search_active
              + beta_social   * social_active
              + ... (one beta per channel)
              + city fixed effects
              + week fixed effects
              + error

    OLS estimator:
        beta_hat = (X'X)^-1 X'y

    Cluster-robust ("CR1", clustering by city) variance:
        V = (X'X)^-1 * M * (X'X)^-1
        M = sum over clusters g of: X_g' u_g u_g' X_g
        M *= (G / (G - 1)) * ((n - 1) / (n - k))     Stata-style small-sample correction
        G = number of clusters, n = observations, k = parameters

    Why clustering: residuals are correlated within a city across weeks
    (the same user pool can convert in any week, so a high-converting
    city's weeks are correlated). Without clustering the standard errors
    would be too small and p-values would look more confident than they
    deserve to be.

    Returns one DiDResult per channel.
    """
    X, columns = _build_design_matrix(panel, channel_names)
    y = panel["conversion_rate"].to_numpy()
    cluster = panel["city"].to_numpy()

    n, k = X.shape

    # OLS via normal equations. np.linalg.solve is more stable than computing
    # the inverse explicitly for this step.
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    residuals = y - X @ beta

    # Need (X'X)^-1 explicitly for the sandwich variance.
    XtX_inv = np.linalg.inv(XtX)

    # Cluster-robust meat matrix. For each city, compute the score X_g' u_g
    # and accumulate its outer product.
    M = np.zeros_like(XtX)
    unique_clusters = np.unique(cluster)
    for g in unique_clusters:
        mask = cluster == g
        X_g = X[mask]
        u_g = residuals[mask]
        score_g = X_g.T @ u_g
        M += np.outer(score_g, score_g)

    G = len(unique_clusters)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    M *= correction

    V = XtX_inv @ M @ XtX_inv
    se = np.sqrt(np.diag(V))

    # For clustered SE the textbook df is G - 1 (clusters minus one). With
    # G >= 30 the t-distribution is close enough to normal that we use the
    # normal approximation for both the critical value and the p-value.
    # Bias is well under 0.5pp at G=50, far below the noise of the experiment.
    alpha = 1 - confidence_level
    t_crit = _normal_ppf(1 - alpha / 2)

    results: list[DiDResult] = []
    for ch in channel_names:
        col = f"{ch}_active"
        idx = columns.index(col)
        b = float(beta[idx])
        s = float(se[idx])
        if s > 0:
            t_stat = b / s
            p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
        else:
            t_stat = 0.0
            p_value = 1.0
        ci_low = b - t_crit * s
        ci_high = b + t_crit * s

        # Convert per-user-week rate into total incremental conversions:
        # incremental conversions = beta_hat * (sum over active city-weeks of n_users)
        active_user_weeks = float((panel[col] * panel["n_users"]).sum())

        results.append(DiDResult(
            channel=ch,
            incremental_rate=b,
            standard_error=s,
            t_stat=float(t_stat),
            p_value=float(p_value),
            ci_low_90=float(ci_low),
            ci_high_90=float(ci_high),
            measured_incremental_conversions=float(b * active_user_weeks),
            incr_conv_ci_low_90=float(ci_low * active_user_weeks),
            incr_conv_ci_high_90=float(ci_high * active_user_weeks),
            active_user_weeks=active_user_weeks,
        ))

    return results


def results_to_dataframe(results: list[DiDResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])
