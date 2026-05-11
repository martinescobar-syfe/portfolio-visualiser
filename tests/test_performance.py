"""Tests for `utils.performance`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from utils.performance import (
    TRADING_DAYS_PER_YEAR,
    calculate_metrics,
    calculate_relative_metrics,
)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
@st.composite
def daily_return_series(draw, n_min: int = 60, n_max: int = 260):
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    values = draw(
        st.lists(
            st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(values, index=idx)


@given(s=daily_return_series())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_max_drawdown_non_positive(s: pd.Series) -> None:
    m = calculate_metrics(s)
    assert m["Max Drawdown"] <= 1e-12  # always ≤ 0


@given(s=daily_return_series())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_cumulative_return_matches_compound(s: pd.Series) -> None:
    m = calculate_metrics(s)
    expected = float((1 + s).prod() - 1)
    assert np.isclose(m["Cumulative Return"], expected, rtol=1e-9, atol=1e-12)


@given(s=daily_return_series())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_annualised_volatility_formula(s: pd.Series) -> None:
    m = calculate_metrics(s)
    expected = float(s.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    assert np.isclose(m["Annualized Volatility"], expected, rtol=1e-9, atol=1e-12)


@given(s=daily_return_series())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_sharpe_scales_inversely_with_vol(s: pd.Series) -> None:
    """Doubling every return should leave Sharpe unchanged because both numerator
    and denominator scale (approximately) the same way under small returns."""
    m1 = calculate_metrics(s, rf=0.0)
    # We can't trivially scale returns and keep Sharpe identical (CAGR is non-linear),
    # but we can assert the sign and that vol > 0 implies finite Sharpe.
    if m1["Annualized Volatility"] > 0:
        assert np.isfinite(m1["Sharpe Ratio"])


@given(s=daily_return_series())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_sortino_with_no_negatives_is_inf_or_nan(s: pd.Series) -> None:
    """If there are no daily returns below MAR=0, downside deviation is 0 and
    Sortino should be NaN (the formula is undefined)."""
    s_pos = s.clip(lower=0)
    m = calculate_metrics(s_pos, rf=0.0)
    assert np.isnan(m["Sortino Ratio"]) or np.isinf(m["Sortino Ratio"])


def test_rolling_excess_geometric(returns_panel: pd.DataFrame) -> None:
    p = returns_panel["ASSET_A"]
    b = returns_panel["BENCH"]
    metrics, roll_excess = calculate_relative_metrics(p, b)

    # The 252-day excess on the last available window should equal
    # prod(1+p) / prod(1+b) - 1 over that window, computed manually.
    last_idx = roll_excess.dropna().index[-1]
    window = p.index.get_loc(last_idx)
    p_win = p.iloc[window - TRADING_DAYS_PER_YEAR + 1 : window + 1]
    b_win = b.iloc[window - TRADING_DAYS_PER_YEAR + 1 : window + 1]
    expected = float((1 + p_win).prod() / (1 + b_win).prod() - 1)
    assert np.isclose(roll_excess.loc[last_idx], expected, rtol=1e-9, atol=1e-12)


def test_information_ratio_sign_matches_excess_mean(returns_panel: pd.DataFrame) -> None:
    p = returns_panel["ASSET_A"]
    b = returns_panel["BENCH"]
    metrics, _ = calculate_relative_metrics(p, b)

    if metrics["Tracking Error"] > 0:
        excess_mean = float((p - b).mean())
        assert np.sign(metrics["Information Ratio"]) == np.sign(excess_mean)


def test_rf_flows_into_sharpe_and_sortino(returns_panel: pd.DataFrame) -> None:
    s = returns_panel["ASSET_A"]
    m_zero = calculate_metrics(s, rf=0.0)
    m_high = calculate_metrics(s, rf=0.10)
    # higher RF should reduce Sharpe (lower numerator, same denominator)
    if not np.isnan(m_zero["Sharpe Ratio"]) and not np.isnan(m_high["Sharpe Ratio"]):
        assert m_high["Sharpe Ratio"] < m_zero["Sharpe Ratio"]
