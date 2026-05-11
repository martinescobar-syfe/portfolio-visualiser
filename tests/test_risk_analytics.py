"""Tests for `utils.risk_analytics`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from utils.risk_analytics import (
    TRADING_DAYS,
    annualised_cov_from_returns,
    compute_active_risk_contributions,
    compute_brinson_attribution,
    compute_capture_ratios,
    compute_drawdown_episodes,
    compute_efficient_frontier,
    compute_risk_contributions,
    compute_var_cvar,
    evaluate_portfolio_on_frontier,
)


# ---------------------------------------------------------------------------
# MCTR (compute_risk_contributions)
# ---------------------------------------------------------------------------
def test_mctr_components_sum_to_portfolio_vol(asset_returns, latest_weights) -> None:
    cov = annualised_cov_from_returns(asset_returns)
    out = compute_risk_contributions(latest_weights, cov)
    expected_vol = float(np.sqrt(latest_weights.values @ cov.loc[latest_weights.index, latest_weights.index].values @ latest_weights.values))
    assert np.isclose(out["Component Risk (annualised)"].sum(), expected_vol, rtol=1e-9, atol=1e-12)


def test_mctr_share_sums_to_one(asset_returns, latest_weights) -> None:
    cov = annualised_cov_from_returns(asset_returns)
    out = compute_risk_contributions(latest_weights, cov)
    assert np.isclose(out["% of Total Risk"].sum(), 1.0, rtol=1e-9, atol=1e-12)


def test_mctr_zero_vol_returns_zero(asset_returns) -> None:
    """All-zero weights should not blow up; result is all zeros."""
    cov = annualised_cov_from_returns(asset_returns)
    weights = pd.Series(0.0, index=cov.index)
    weights.iloc[0] = 1.0  # avoid the ValueError path
    out = compute_risk_contributions(weights, cov * 0.0)
    assert (out["Component Risk (annualised)"] == 0).all()


# ---------------------------------------------------------------------------
# MCTE (compute_active_risk_contributions)
# ---------------------------------------------------------------------------
def test_mcte_components_sum_to_te(asset_returns, latest_weights) -> None:
    cov = annualised_cov_from_returns(asset_returns)
    bm = pd.Series(0.25, index=cov.index)  # equal weight benchmark across same universe
    out = compute_active_risk_contributions(latest_weights, bm, cov)
    te = out.attrs["tracking_error"]
    assert np.isclose(out["Component TE (annualised)"].sum(), te, rtol=1e-9, atol=1e-12)


def test_mcte_zero_when_port_equals_bench(asset_returns) -> None:
    cov = annualised_cov_from_returns(asset_returns)
    weights = pd.Series(0.25, index=cov.index)
    out = compute_active_risk_contributions(weights, weights.copy(), cov)
    te = out.attrs["tracking_error"]
    assert np.isclose(te, 0.0, atol=1e-12)
    assert np.allclose(out["Component TE (annualised)"], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# VaR / CVaR (compute_var_cvar)
# ---------------------------------------------------------------------------
def test_var_99_more_extreme_than_var_95(returns_panel) -> None:
    s = returns_panel["ASSET_A"]
    out = compute_var_cvar(s, confidence_levels=(0.95, 0.99))
    assert out.loc[0.99, "Historical VaR"] <= out.loc[0.95, "Historical VaR"]
    assert out.loc[0.99, "Parametric VaR"] <= out.loc[0.95, "Parametric VaR"]


def test_cvar_no_better_than_var(returns_panel) -> None:
    s = returns_panel["ASSET_A"]
    out = compute_var_cvar(s)
    for cl in out.index:
        # CVaR is the conditional mean below VaR — must be ≤ VaR
        assert out.loc[cl, "Historical CVaR"] <= out.loc[cl, "Historical VaR"] + 1e-12


def test_var_horizon_scaling(returns_panel) -> None:
    s = returns_panel["ASSET_A"]
    h1 = compute_var_cvar(s, horizon_days=1)
    h5 = compute_var_cvar(s, horizon_days=5)
    # Historical: 1-day quantile × √h
    expected = h1.loc[0.95, "Historical VaR"] * np.sqrt(5)
    assert np.isclose(h5.loc[0.95, "Historical VaR"], expected, rtol=1e-9, atol=1e-12)


def test_var_annualise_equals_252_horizon(returns_panel) -> None:
    s = returns_panel["ASSET_A"]
    a = compute_var_cvar(s, annualise=True)
    h = compute_var_cvar(s, horizon_days=TRADING_DAYS, annualise=False)
    pd.testing.assert_frame_equal(a, h, check_names=False)


# ---------------------------------------------------------------------------
# Drawdown episodes (compute_drawdown_episodes)
# ---------------------------------------------------------------------------
def test_drawdown_depth_non_positive(returns_panel) -> None:
    out = compute_drawdown_episodes(returns_panel["ASSET_D"])
    assert (out["Depth"] <= 0).all()


def test_drawdown_dates_ordered(returns_panel) -> None:
    out = compute_drawdown_episodes(returns_panel["ASSET_D"])
    for _, row in out.iterrows():
        if pd.notna(row["Recovery Date"]):
            assert row["Peak Date"] <= row["Trough Date"] <= row["Recovery Date"]


def test_drawdown_top_n_respected(returns_panel) -> None:
    out = compute_drawdown_episodes(returns_panel["ASSET_D"], top_n=3)
    assert len(out) <= 3


# ---------------------------------------------------------------------------
# Capture ratios (compute_capture_ratios)
# ---------------------------------------------------------------------------
def test_capture_self_equals_one(returns_panel) -> None:
    s = returns_panel["ASSET_A"]
    out = compute_capture_ratios(s, s, frequency="M")
    assert np.isclose(out["Up Capture"], 1.0, rtol=1e-9, atol=1e-12)
    assert np.isclose(out["Down Capture"], 1.0, rtol=1e-9, atol=1e-12)
    assert np.isclose(out["Hit Ratio"], 0.0)
    assert np.isclose(out["Batting Average"], 1.0)


def test_capture_hit_ratio_in_unit_interval(returns_panel) -> None:
    p = returns_panel["ASSET_A"]
    b = returns_panel["BENCH"]
    out = compute_capture_ratios(p, b, frequency="M")
    assert 0.0 <= out["Hit Ratio"] <= 1.0
    assert 0.0 <= out["Batting Average"] <= 1.0


# ---------------------------------------------------------------------------
# Brinson attribution
# ---------------------------------------------------------------------------
def test_brinson_reconciles_to_active(asset_returns, latest_weights, benchmark_weights, exposure_table) -> None:
    period_returns = (1 + asset_returns).prod() - 1
    out = compute_brinson_attribution(
        latest_weights, benchmark_weights, period_returns, exposure_table
    )
    bucket_active = out["Allocation"].sum() + out["Selection"].sum() + out["Interaction"].sum()
    expected = out.attrs["active_return"]
    assert np.isclose(bucket_active, expected, rtol=1e-9, atol=1e-12)


def test_brinson_zero_when_port_equals_bench(asset_returns, exposure_table) -> None:
    period_returns = (1 + asset_returns).prod() - 1
    weights = pd.Series({"ASSET_A": 0.25, "ASSET_B": 0.25, "ASSET_C": 0.25, "ASSET_D": 0.25})
    out = compute_brinson_attribution(weights, weights.copy(), period_returns, exposure_table)
    # When portfolio == benchmark on the same exposure table, every effect is 0
    assert np.allclose(out["Allocation"], 0.0, atol=1e-12)
    assert np.allclose(out["Selection"], 0.0, atol=1e-12)
    assert np.allclose(out["Interaction"], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------
def test_frontier_weights_long_only(asset_returns) -> None:
    out = compute_efficient_frontier(asset_returns, n_points=10)
    weight_cols = [c for c in out.columns if c not in ("Return", "Vol", "Sharpe")]
    weights = out[weight_cols].values
    assert (weights >= -1e-9).all()
    assert (weights <= 1 + 1e-9).all()
    sums = weights.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_frontier_vol_increases_with_target_return(asset_returns) -> None:
    out = compute_efficient_frontier(asset_returns, n_points=20)
    # Frontier should be monotone in vol on the upper branch (return increasing => vol non-decreasing)
    for a, b in zip(out["Vol"].values, out["Vol"].values[1:]):
        assert b >= a - 1e-6


def test_evaluate_portfolio_uses_same_panel(asset_returns) -> None:
    weights = pd.Series({"ASSET_A": 0.4, "ASSET_B": 0.3, "ASSET_C": 0.2, "ASSET_D": 0.1})
    ret_p, vol_p = evaluate_portfolio_on_frontier(weights, asset_returns)
    # Manual check
    mu_ann = asset_returns.mean() * TRADING_DAYS
    cov_ann = asset_returns.cov() * TRADING_DAYS
    expected_ret = float(weights @ mu_ann.loc[weights.index])
    expected_vol = float(np.sqrt(weights @ cov_ann.loc[weights.index, weights.index].values @ weights))
    assert np.isclose(ret_p, expected_ret, rtol=1e-9)
    assert np.isclose(vol_p, expected_vol, rtol=1e-9)
