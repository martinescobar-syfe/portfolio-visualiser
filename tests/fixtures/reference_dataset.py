"""Pinned synthetic reference dataset for regression tests.

Goal: produce a deterministic, hand-traceable mini-portfolio whose every
metric we can compute outside the codebase (or pin once) so that any future
formula change shows up as a numeric diff in CI.

The dataset uses ``np.random.default_rng(seed=42)`` and a fixed business-day
index. **Do not change the seed or generator** without updating the pinned
expectations in ``test_pinned_reference.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEED = 42
N_DAYS = 1000                           # ~4 years of business days
START_DATE = "2021-01-04"
TICKERS = ("ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D")
BENCHMARK_TICKER = "BENCH"


def make_returns(seed: int = SEED) -> pd.DataFrame:
    """Generate a deterministic daily return panel for the test universe.

    Returns a DataFrame indexed by business day with columns for each asset
    and the benchmark. Means and vols differ enough across columns that
    rank-tests can distinguish them, but everything is well-behaved (no NaN,
    no extreme outliers).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=START_DATE, periods=N_DAYS)

    # daily mean / vol per series — chosen so:
    #   ASSET_A: high return, high vol
    #   ASSET_B: medium
    #   ASSET_C: low return, low vol (bond-like)
    #   ASSET_D: negative drift, medium vol (tail tester)
    #   BENCH:   blended profile somewhere in the middle
    spec = {
        "ASSET_A": (0.0008, 0.012),
        "ASSET_B": (0.0005, 0.010),
        "ASSET_C": (0.0002, 0.004),
        "ASSET_D": (-0.0001, 0.011),
        BENCHMARK_TICKER: (0.0004, 0.009),
    }

    data = {}
    for ticker, (mu, sigma) in spec.items():
        data[ticker] = rng.normal(mu, sigma, size=N_DAYS)

    return pd.DataFrame(data, index=idx)


def make_weights(returns: pd.DataFrame, fixed: dict[str, float] | None = None) -> pd.DataFrame:
    """Build a daily weights frame with periodic rebalancing for the
    portfolio assets only (not the benchmark).

    Off-rebalance days drift via ``w_t ∝ w_{t-1} × (1 + r_t)`` then renormalise.
    Rebalance dates land on the first business day of January and July.
    The output mimics the ``_wgt`` suffix convention used elsewhere in the
    code: columns are ``ASSET_A_wgt`` etc., plus a ``reb_flag`` column.
    """
    if fixed is None:
        fixed = {"ASSET_A": 0.4, "ASSET_B": 0.3, "ASSET_C": 0.2, "ASSET_D": 0.1}
    if not np.isclose(sum(fixed.values()), 1.0):
        raise ValueError("Fixed weights must sum to 1.")

    asset_cols = list(fixed.keys())
    ret = returns[asset_cols]

    weights = pd.DataFrame(0.0, index=ret.index, columns=asset_cols)
    reb_flag = pd.Series(False, index=ret.index, name="reb_flag")

    rebal_months = (1, 7)

    for i, dt in enumerate(ret.index):
        if i == 0 or (dt.month in rebal_months and ret.index[i - 1].month != dt.month):
            for ticker, wgt in fixed.items():
                weights.iloc[i, asset_cols.index(ticker)] = wgt
            reb_flag.iloc[i] = True
        else:
            prev = weights.iloc[i - 1]
            drifted = prev * (1 + ret.iloc[i].fillna(0))
            total = drifted.sum()
            if total > 0:
                weights.iloc[i] = drifted / total

    weights.columns = [f"{c}_wgt" for c in weights.columns]
    out = pd.concat([reb_flag, weights], axis=1)
    return out


def make_benchmark_weights(value: float = 1.0) -> pd.Series:
    """Single-ticker benchmark series (sums to ``value``)."""
    return pd.Series({BENCHMARK_TICKER: value})


def make_exposure_table(buckets: tuple[str, ...] = ("Equity", "Bond", "Real Estate")) -> pd.DataFrame:
    """Sector/country-style exposure matrix for the test universe.

    Rows = buckets, columns = tickers. Values are decimals summing to 1
    down each ticker column. We keep ASSET_C as a pure bond and ASSET_D
    as part-equity / part-real-estate so the Brinson tests have variety.
    """
    data = {
        "ASSET_A": [1.0, 0.0, 0.0],
        "ASSET_B": [0.7, 0.0, 0.3],
        "ASSET_C": [0.0, 1.0, 0.0],
        "ASSET_D": [0.5, 0.0, 0.5],
        BENCHMARK_TICKER: [0.6, 0.3, 0.1],
    }
    return pd.DataFrame(data, index=list(buckets))
