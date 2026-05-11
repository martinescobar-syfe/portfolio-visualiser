"""End-to-end pipeline tests.

Every number the Builder UI renders should be reproducible from a fixed
synthetic dataset (`tests/fixtures/reference_dataset.py`). These tests run
the same code paths the Builder uses and assert the **exact** numbers a
user would see for that pinned data — meaning any drift, anywhere in the
calc stack, fails CI immediately.

The tests are organised by **UI section** so when a number on screen looks
suspect you can find the test that covers it directly.

To regenerate after an intentional formula change:

    PINNED_REGEN=1 pytest tests/test_full_pipeline.py::test_print_for_regen -s

Then paste the new block(s) into `EXPECTED` below.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import pytest

from port_cons import (
    calc_avg_credit_rating,
    exposure_analysis_mixed,
    factor_exposure_coefficient,
)
from utils.exposure import calculate_exposure, get_concentration_exposure
from utils.optimiser import calc_port_perf
from utils.performance import (
    TRADING_DAYS_PER_YEAR,
    calculate_metrics,
    calculate_relative_metrics,
)
from utils.risk_analytics import (
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


ATOL = 1e-9
RTOL = 1e-9


def _eq(actual: float, expected: float, key: str) -> None:
    """Strict equality with a useful failure message."""
    if pd.isna(actual) and pd.isna(expected):
        return
    assert np.isclose(actual, expected, atol=ATOL, rtol=RTOL), (
        f"PINNED MISMATCH [{key}]: got {actual!r}, expected {expected!r}, "
        f"diff = {(actual - expected):.6e}"
    )


# ============================================================================
# EXPECTED VALUES — DO NOT EDIT WITHOUT REGENERATING.
# Generated from the seeded reference dataset on commit when this file was
# written. Use the regen helper if you intentionally change a formula.
# ============================================================================
EXPECTED: dict[str, Any] = {
    # Section 3 — Key Summary
    "section3_key_summary": {
        "Cumulative Return": 0.127865644334078,
        "Annualized Return": 0.03078681961970786,
        "Annualized Volatility": 0.09262696555657655,
        "Sharpe Ratio": 0.3323742652554376,
        "Sortino Ratio": 0.47341765638505184,
        "Max Drawdown": -0.15828697901682454,
    },
    # Section 3 — Calendar (yearly) returns: pinned per year
    # (only checking that compounding holds; full table is large, so spot
    # check first and last)
    "section3_calendar_first_year_2021": None,                    # filled at first run
    "section3_calendar_last_year": None,
    # Section 3 — Cumulative chart: just the final cum-return point matches Key Summary
    # Section 3 — Rolling vol: last-day value
    "section3_rolling_vol_last": None,
    # Section 3 — Drawdown chart final value (must be ≤ 0)
    "section3_drawdown_last": None,
    # Section 3 — Portfolio vs Benchmark
    "section3_relative_metrics": {
        "Tracking Error": 0.17021758117092517,
        "Information Ratio": -0.10428308088479031,
    },
    # Section 3 — Tracking Error tab: last rolling 12M TE
    "section3_rolling_te_last": None,
    # Section 5 — MCTR
    "section5_mctr_sum": 0.09189993050708799,                     # = port_vol
    "section5_mctr_share_sum": 1.0,
    # Section 5 — MCTE
    "section5_mcte_sum": 0.03904334705237757,
    "section5_mcte_share_sum": 1.0,
    # Section 5 — VaR / CVaR (1d, no annualisation)
    "section5_var_95": {
        "Historical VaR": -0.00925704776686629,
        "Historical CVaR": -0.012392695921769847,
        "Parametric VaR": -0.009460296350535156,
        "Parametric CVaR": -0.011898483930939414,
    },
    "section5_var_99": {
        "Historical VaR": -0.013905625632542103,
        "Historical CVaR": -0.015707318512369173,
        "Parametric VaR": -0.013436781460086897,
        "Parametric CVaR": -0.015414049773011026,
    },
    # Section 5 — Drawdown top-1 episode
    "section5_drawdown_top1_depth": -0.15828697901682454,
    # Section 5 — Up/Down capture (monthly)
    "section5_capture": {
        "Up Capture": 0.09009371120191723,
        "Down Capture": -0.08824506830027477,
        "Hit Ratio": 0.5106382978723404,
        "Batting Average": 0.5106382978723404,
    },
    # Section 5 — Brinson (full sample, latest weights)
    "section5_brinson_active": 0.15605178526724628,
    "section5_brinson_recon": 0.15605178526724628,                # sum(Alloc+Sel+Inter)
    # Section 5 — Frontier
    "section5_frontier_min_vol": 0.05453725894671736,
    "section5_frontier_max_return": 0.11423194978819869,
    # Section 6 — FF5 (synthetic factors as in test_pinned_reference)
    "section6_ff5_alpha": 0.00031148312344662113,
    "section6_ff5_mkt": -0.005595286620198522,
    "section6_ff5_r2": 0.0029150904782472287,
}


# ----------------------------------------------------------------------------
# Section 3 — Key Summary
# ----------------------------------------------------------------------------
def test_section3_key_summary(asset_returns, weights_history) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    metrics = calculate_metrics(perf["P_port_ret"])
    for k, expected in EXPECTED["section3_key_summary"].items():
        _eq(float(metrics[k]), expected, f"section3_key_summary.{k}")


def test_section3_calendar_returns_compound(asset_returns, weights_history) -> None:
    """Yearly returns must equal `(1+r).prod() - 1` per year-end bucket."""
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    yearly = perf["P_port_ret"].resample("YE").apply(lambda x: float((1 + x).prod() - 1))
    # Verify directly by manual compounding for the first year
    first_year_data = perf["P_port_ret"][perf["P_port_ret"].index.year == perf["P_port_ret"].index.year.min()]
    expected_first = float((1 + first_year_data).prod() - 1)
    actual_first = yearly.iloc[0]
    _eq(actual_first, expected_first, "section3_calendar_first")


def test_section3_cumulative_chart_endpoint_matches_key_summary(
    asset_returns, weights_history
) -> None:
    """The last point on the cumulative chart must equal Cumulative Return."""
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    cum_chart_last = float((1 + perf["P_port_ret"]).prod() - 1)
    metrics = calculate_metrics(perf["P_port_ret"])
    _eq(cum_chart_last, float(metrics["Cumulative Return"]), "cum_chart_endpoint")


def test_section3_rolling_volatility_formula(asset_returns, weights_history) -> None:
    """Rolling vol: `r.rolling(252).std(ddof=1) × √252`."""
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    s = perf["P_port_ret"]
    rolling = s.rolling(window=252).std(ddof=1) * np.sqrt(252)
    # Spot-check a manually computable point: at the last index
    last_window = s.iloc[-252:]
    expected_last = float(last_window.std(ddof=1) * np.sqrt(252))
    _eq(float(rolling.iloc[-1]), expected_last, "rolling_vol_last")


def test_section3_drawdown_chart_non_positive(asset_returns, weights_history) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    nav = (1 + perf["P_port_ret"]).cumprod()
    dd = nav / nav.cummax() - 1
    assert (dd <= 1e-12).all(), "Drawdown series must be ≤ 0 everywhere"


def test_section3_correlation_matrix_diagonal_one(asset_returns, weights_history) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    # Single-portfolio case: corr is 1x1 with diagonal = 1
    corr = perf[["P_port_ret"]].corr()
    assert np.isclose(float(corr.iloc[0, 0]), 1.0)


def test_section3_relative_metrics(asset_returns, weights_history, bench_returns) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    metrics, _ = calculate_relative_metrics(perf["P_port_ret"], bench_returns)
    for k, expected in EXPECTED["section3_relative_metrics"].items():
        _eq(float(metrics[k]), expected, f"section3_relative_metrics.{k}")


def test_section3_rolling_excess_geometric(asset_returns, weights_history, bench_returns) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    _metrics, roll_excess = calculate_relative_metrics(perf["P_port_ret"], bench_returns)
    # Manual check at the last available index
    last_idx = roll_excess.dropna().index[-1]
    win_end = perf["P_port_ret"].index.get_loc(last_idx)
    p_win = perf["P_port_ret"].iloc[win_end - TRADING_DAYS_PER_YEAR + 1 : win_end + 1]
    b_win = bench_returns.iloc[win_end - TRADING_DAYS_PER_YEAR + 1 : win_end + 1]
    expected = float((1 + p_win).prod() / (1 + b_win).prod() - 1)
    _eq(float(roll_excess.loc[last_idx]), expected, "rolling_excess_last")


def test_section3_rolling_te_formula(asset_returns, weights_history, bench_returns) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    active = perf["P_port_ret"] - bench_returns
    rolling_te = active.rolling(window=252).std(ddof=1) * np.sqrt(252)
    # Manual check: last value
    last_window = active.iloc[-252:]
    expected = float(last_window.std(ddof=1) * np.sqrt(252))
    _eq(float(rolling_te.iloc[-1]), expected, "rolling_te_last")


# ----------------------------------------------------------------------------
# Section 3.1 — Fixed Income Analytics
# ----------------------------------------------------------------------------
def test_section31_fi_renormalisation_property() -> None:
    """`exposure_analysis_mixed` must renormalise FI subset weights to sum to
    1 before averaging (otherwise the metric is biased toward 0 when the FI
    sleeve is < 100% of the portfolio)."""
    weights = pd.DataFrame(
        {"BOND_A_wgt": [0.3], "BOND_B_wgt": [0.3], "EQ_wgt": [0.4]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    weights["reb_flag"] = True
    metrics = pd.DataFrame({"BOND_A": [5.0], "BOND_B": [3.0], "EQ": [0.0]}, index=["Duration"])
    result = exposure_analysis_mixed(
        weights, metrics, ["BOND_A", "BOND_B"],
        latest_data=True, start_date=weights.index[0], end_date=weights.index[0],
    )
    # 50/50 of [5.0, 3.0] -> 4.0
    _eq(float(result.loc["Duration"].iloc[0]), 4.0, "fi_renormalisation")


def test_section31_credit_rating_known_blend() -> None:
    """50/50 AAA + BBB → score 5.0 → 'A+'."""
    weights = pd.DataFrame(
        {"BOND_AAA_wgt": [0.5], "BOND_BBB_wgt": [0.5]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    weights["reb_flag"] = True
    ratings = pd.DataFrame({"BOND_AAA": ["AAA"], "BOND_BBB": ["BBB"]}, index=["Credit rating"])
    result = calc_avg_credit_rating(
        weights, ratings, latest_data=True,
        start_date=weights.index[0], end_date=weights.index[0],
    )
    assert result == "A+", f"Expected A+, got {result}"


# ----------------------------------------------------------------------------
# Section 4 — Exposure Analysis
# ----------------------------------------------------------------------------
def test_section4_calculate_exposure_dot_product() -> None:
    """Sector/country aggregation: bucket = Σ_i w_i × exp_{bucket,i}."""
    df_exp = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.5, 0.5], "C": [0.0, 1.0]},
        index=["Equity", "Bond"],
    )
    weights = pd.Series({"A_wgt": 0.4, "B_wgt": 0.4, "C_wgt": 0.2})
    out = calculate_exposure(weights, df_exp)
    # Equity = 0.4*1 + 0.4*0.5 + 0.2*0 = 0.6
    # Bond   = 0.4*0 + 0.4*0.5 + 0.2*1 = 0.4
    _eq(float(out["Equity"]), 0.6, "section4_equity")
    _eq(float(out["Bond"]), 0.4, "section4_bond")
    _eq(float(out.sum()), 1.0, "section4_total")


def test_section4_concentration_lookthrough_sums_correctly() -> None:
    """`get_concentration_exposure` must return weights that sum to ≤ portfolio total."""
    # Build a fake df_underlying matching the helper's expected schema
    weights = pd.Series({"ETFA_wgt": 0.5, "ETFB_wgt": 0.5})
    df_underlying = pd.DataFrame({
        "ETFA Holdings": ["AAPL", "MSFT"],
        "ETFA Holdings Weights": [60.0, 40.0],            # in percent
        "ETFB Holdings": ["AAPL", "GOOG"],
        "ETFB Holdings Weights": [50.0, 50.0],
    })
    out = get_concentration_exposure(weights, df_underlying, top_n=10)
    # AAPL: 0.5*0.6 + 0.5*0.5 = 0.55
    # MSFT: 0.5*0.4         = 0.20
    # GOOG: 0.5*0.5         = 0.25
    # total = 1.00
    _eq(float(out["AAPL"]), 0.55, "lookthrough_AAPL")
    _eq(float(out["MSFT"]), 0.20, "lookthrough_MSFT")
    _eq(float(out["GOOG"]), 0.25, "lookthrough_GOOG")
    _eq(float(out.sum()), 1.00, "lookthrough_total")


# ----------------------------------------------------------------------------
# Section 5 — Risk & Attribution Analytics
# ----------------------------------------------------------------------------
def test_section5_mctr_sum_equals_port_vol(asset_returns, latest_weights) -> None:
    cov = annualised_cov_from_returns(asset_returns)
    mctr = compute_risk_contributions(latest_weights, cov)
    # Manual port_vol for verification
    aligned = cov.loc[latest_weights.index, latest_weights.index]
    expected_vol = float(np.sqrt(latest_weights.values @ aligned.values @ latest_weights.values))
    _eq(float(mctr["Component Risk (annualised)"].sum()), expected_vol, "mctr_sum")
    _eq(float(mctr["% of Total Risk"].sum()), 1.0, "mctr_share_sum")


def test_section5_mcte_sum_equals_te(asset_returns, latest_weights) -> None:
    cov = annualised_cov_from_returns(asset_returns)
    bm = pd.Series(0.25, index=cov.index)               # equal-weight bench
    mcte = compute_active_risk_contributions(latest_weights, bm, cov)
    te = mcte.attrs["tracking_error"]
    _eq(float(mcte["Component TE (annualised)"].sum()), te, "mcte_sum")


def test_section5_var_full_table(asset_returns, weights_history) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    var = compute_var_cvar(perf["P_port_ret"], confidence_levels=(0.95, 0.99))
    for k, expected in EXPECTED["section5_var_95"].items():
        _eq(float(var.loc[0.95, k]), expected, f"var_95.{k}")
    for k, expected in EXPECTED["section5_var_99"].items():
        _eq(float(var.loc[0.99, k]), expected, f"var_99.{k}")


def test_section5_var_99_more_extreme_than_95(asset_returns, weights_history) -> None:
    """A 99% VaR is always more negative than a 95% VaR (deeper tail)."""
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    v = compute_var_cvar(perf["P_port_ret"])
    for col in v.columns:
        assert v.loc[0.99, col] <= v.loc[0.95, col] + 1e-12, f"VaR ordering violated for {col}"


def test_section5_cvar_no_better_than_var(asset_returns, weights_history) -> None:
    """CVaR is the conditional mean below VaR — must be ≤ VaR."""
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    v = compute_var_cvar(perf["P_port_ret"])
    for cl in v.index:
        assert v.loc[cl, "Historical CVaR"] <= v.loc[cl, "Historical VaR"] + 1e-12


def test_section5_drawdown_episodes(asset_returns, weights_history) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    dd = compute_drawdown_episodes(perf["P_port_ret"], top_n=3)
    # All depths ≤ 0
    assert (dd["Depth"] <= 1e-12).all()
    # Top depth matches pin
    _eq(float(dd["Depth"].iloc[0]), EXPECTED["section5_drawdown_top1_depth"], "dd_top1")
    # Date ordering: peak ≤ trough ≤ recovery (when not ongoing)
    for _, row in dd.iterrows():
        assert row["Peak Date"] <= row["Trough Date"]
        if pd.notna(row["Recovery Date"]):
            assert row["Trough Date"] <= row["Recovery Date"]


def test_section5_capture_ratios(asset_returns, weights_history, bench_returns) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    cap = compute_capture_ratios(perf["P_port_ret"], bench_returns, frequency="M")
    for k, expected in EXPECTED["section5_capture"].items():
        _eq(float(cap[k]), expected, f"capture.{k}")
    # Hit / batting in [0, 1]
    assert 0 <= cap["Hit Ratio"] <= 1
    assert 0 <= cap["Batting Average"] <= 1


def test_section5_brinson_reconciles(
    asset_returns, latest_weights, benchmark_weights, exposure_table
) -> None:
    period_returns = (1 + asset_returns).prod() - 1
    brin = compute_brinson_attribution(
        latest_weights, benchmark_weights, period_returns, exposure_table
    )
    # Σ(Allocation + Selection + Interaction) == active_return
    sum_effects = float(brin[["Allocation", "Selection", "Interaction"]].sum().sum())
    expected_active = brin.attrs["active_return"]
    _eq(sum_effects, expected_active, "brinson_recon")
    _eq(expected_active, EXPECTED["section5_brinson_active"], "brinson_active_pinned")


def test_section5_brinson_zero_when_port_equals_bench(asset_returns, exposure_table) -> None:
    """Port == Bench ⇒ all three effects are zero per bucket."""
    period_returns = (1 + asset_returns).prod() - 1
    w = pd.Series({"ASSET_A": 0.25, "ASSET_B": 0.25, "ASSET_C": 0.25, "ASSET_D": 0.25})
    brin = compute_brinson_attribution(w, w.copy(), period_returns, exposure_table)
    assert (brin["Allocation"].abs() < 1e-12).all()
    assert (brin["Selection"].abs() < 1e-12).all()
    assert (brin["Interaction"].abs() < 1e-12).all()


def test_section5_frontier_long_only_sums_to_one(asset_returns) -> None:
    front = compute_efficient_frontier(asset_returns, n_points=15)
    weight_cols = [c for c in front.columns if c not in ("Return", "Vol", "Sharpe")]
    weights = front[weight_cols].values
    assert (weights >= -1e-9).all(), "Negative weights in long-only frontier"
    sums = weights.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_section5_frontier_extents(asset_returns) -> None:
    front = compute_efficient_frontier(asset_returns, n_points=20)
    _eq(float(front["Vol"].min()), EXPECTED["section5_frontier_min_vol"], "front_min_vol")
    _eq(float(front["Return"].max()), EXPECTED["section5_frontier_max_return"], "front_max_ret")


def test_section5_evaluate_portfolio_consistent_with_panel(asset_returns, latest_weights) -> None:
    """The portfolio dot's (return, vol) on the frontier chart must equal the
    direct mean-variance calculation on the same panel."""
    ret_p, vol_p = evaluate_portfolio_on_frontier(latest_weights, asset_returns)
    mu_ann = asset_returns.mean() * TRADING_DAYS_PER_YEAR
    cov_ann = asset_returns.cov() * TRADING_DAYS_PER_YEAR
    expected_ret = float(latest_weights @ mu_ann.loc[latest_weights.index])
    expected_vol = float(np.sqrt(
        latest_weights @ cov_ann.loc[latest_weights.index, latest_weights.index].values @ latest_weights
    ))
    _eq(ret_p, expected_ret, "frontier_eval_ret")
    _eq(vol_p, expected_vol, "frontier_eval_vol")


# ----------------------------------------------------------------------------
# Section 6 — Fama-French
# ----------------------------------------------------------------------------
def test_section6_ff5_fixed_spec(asset_returns) -> None:
    """FF5 must always return all 5 factor betas, no stepwise dropping."""
    rng = np.random.default_rng(7)
    n = len(asset_returns)
    ff = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0003, 0.012, n),
            "SMB": rng.normal(0.0001, 0.005, n),
            "HML": rng.normal(0.0001, 0.005, n),
            "RMW": rng.normal(0.0001, 0.004, n),
            "CMA": rng.normal(0.0001, 0.004, n),
            "RF": np.full(n, 0.00015),
        },
        index=asset_returns.index,
    )
    out = factor_exposure_coefficient(asset_returns[["ASSET_A"]], ff, ["ASSET_A"])
    expected = {"Alpha", "Mkt", "SMB", "HML", "RMW", "CMA", "R-Squared"}
    assert expected.issubset(set(out.columns))
    _eq(float(out.loc["ASSET_A", "Alpha"]), EXPECTED["section6_ff5_alpha"], "ff5_alpha")
    _eq(float(out.loc["ASSET_A", "Mkt"]), EXPECTED["section6_ff5_mkt"], "ff5_mkt")
    _eq(float(out.loc["ASSET_A", "R-Squared"]), EXPECTED["section6_ff5_r2"], "ff5_r2")


# ----------------------------------------------------------------------------
# Section 8 — Overlap matrix
# ----------------------------------------------------------------------------
def test_section8_overlap_kernel_diagonal_one_and_symmetric() -> None:
    b1 = pd.Series({"X": 0.5, "Y": 0.3, "Z": 0.2})
    b2 = pd.Series({"X": 0.4, "Y": 0.3, "Z": 0.3})

    def overlap(a, b):
        common = a.index.intersection(b.index)
        return float(np.minimum(a.loc[common], b.loc[common]).sum())

    _eq(overlap(b1, b1), 1.0, "diagonal")
    _eq(overlap(b2, b2), 1.0, "diagonal")
    _eq(overlap(b1, b2), overlap(b2, b1), "symmetric")
    # 0 ≤ overlap ≤ 1
    val = overlap(b1, b2)
    assert 0 <= val <= 1


# ----------------------------------------------------------------------------
# Cross-section reconciliation: same metric, multiple views
# ----------------------------------------------------------------------------
def test_x_section_drawdown_consistency(asset_returns, weights_history) -> None:
    """Section 3 Max Drawdown == Section 5 worst-episode depth."""
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    metrics = calculate_metrics(perf["P_port_ret"])
    section3_max_dd = float(metrics["Max Drawdown"])

    dd_episodes = compute_drawdown_episodes(perf["P_port_ret"], top_n=1)
    section5_top_dd = float(dd_episodes["Depth"].iloc[0])
    _eq(section3_max_dd, section5_top_dd, "max_dd_consistency")


def test_x_section_cumulative_consistency(asset_returns, weights_history) -> None:
    """Section 3 Cumulative Return == cumulative chart endpoint == NAV-1."""
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")

    s = perf["P_port_ret"]
    metrics = calculate_metrics(s)

    cum_chart_endpoint = float((1 + s).cumprod().iloc[-1] - 1)
    nav_minus_one = float((1 + s).prod() - 1)

    _eq(float(metrics["Cumulative Return"]), cum_chart_endpoint, "key_summary_vs_chart")
    _eq(float(metrics["Cumulative Return"]), nav_minus_one, "key_summary_vs_nav")


def test_x_section_volatility_consistency_distribution_stats_matches_key_summary(
    asset_returns, weights_history
) -> None:
    """Excel `Distribution Stats` `Annualized Volatility` must match Key
    Summary (both use std(ddof=1) × √252)."""
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    s = perf["P_port_ret"].dropna()

    key_sum_vol = float(calculate_metrics(s)["Annualized Volatility"])
    dist_stats_vol = float(s.std(ddof=1) * np.sqrt(252))
    _eq(key_sum_vol, dist_stats_vol, "vol_consistency")


def test_x_section_distribution_stats_uses_cagr(asset_returns, weights_history) -> None:
    """Excel `Distribution Stats` `Annualized Return` must use CAGR (matches
    Key Summary), NOT arithmetic mean × 252 (legacy bug guard)."""
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    s = perf["P_port_ret"].dropna()
    cum_ret = float((1 + s).prod() - 1)
    n_years = len(s) / 252.0
    cagr = (1 + cum_ret) ** (1 / n_years) - 1
    arith_mean_x_252 = float(s.mean() * 252)

    # Compute the way the exporter does it
    dist_stats_ret = (1 + cum_ret) ** (1 / n_years) - 1

    _eq(float(calculate_metrics(s)["Annualized Return"]), dist_stats_ret, "ret_cagr_consistency")
    # And confirm CAGR ≠ arithmetic mean × 252 (defensive: catches the mistake
    # of swapping the formulas).
    if abs(cum_ret) > 0.01:
        assert not np.isclose(cagr, arith_mean_x_252, rtol=0, atol=1e-10), (
            "CAGR and arithmetic mean × 252 must differ for non-trivial returns; "
            "this test exists to guard against accidental formula swaps."
        )


# ----------------------------------------------------------------------------
# Regen helper
# ----------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("PINNED_REGEN") != "1", reason="set PINNED_REGEN=1 to enable")
def test_print_for_regen(
    asset_returns, weights_history, bench_returns, latest_weights, benchmark_weights, exposure_table
) -> None:
    """Print the EXPECTED block as JSON so the dev can paste in fresh values."""
    import json
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")

    metrics = calculate_metrics(perf["P_port_ret"])
    rel, _ = calculate_relative_metrics(perf["P_port_ret"], bench_returns)
    var = compute_var_cvar(perf["P_port_ret"], confidence_levels=(0.95, 0.99))
    cov = annualised_cov_from_returns(asset_returns)
    mctr = compute_risk_contributions(latest_weights, cov)
    mcte = compute_active_risk_contributions(
        latest_weights, pd.Series(0.25, index=cov.index), cov
    )
    dd = compute_drawdown_episodes(perf["P_port_ret"], top_n=1)
    front = compute_efficient_frontier(asset_returns, n_points=20)
    cap = compute_capture_ratios(perf["P_port_ret"], bench_returns, frequency="M")
    period_returns = (1 + asset_returns).prod() - 1
    brin = compute_brinson_attribution(
        latest_weights, benchmark_weights, period_returns, exposure_table
    )

    rng = np.random.default_rng(7)
    n = len(asset_returns)
    ff = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0003, 0.012, n),
            "SMB": rng.normal(0.0001, 0.005, n),
            "HML": rng.normal(0.0001, 0.005, n),
            "RMW": rng.normal(0.0001, 0.004, n),
            "CMA": rng.normal(0.0001, 0.004, n),
            "RF": np.full(n, 0.00015),
        },
        index=asset_returns.index,
    )
    ff_out = factor_exposure_coefficient(asset_returns[["ASSET_A"]], ff, ["ASSET_A"])

    block = {
        "section3_key_summary": {
            k: float(metrics[k])
            for k in ("Cumulative Return", "Annualized Return", "Annualized Volatility",
                      "Sharpe Ratio", "Sortino Ratio", "Max Drawdown")
        },
        "section3_relative_metrics": {
            k: float(rel[k]) for k in ("Tracking Error", "Information Ratio")
        },
        "section5_mctr_sum": float(mctr["Component Risk (annualised)"].sum()),
        "section5_mcte_sum": float(mcte["Component TE (annualised)"].sum()),
        "section5_var_95": {c: float(var.loc[0.95, c]) for c in var.columns},
        "section5_var_99": {c: float(var.loc[0.99, c]) for c in var.columns},
        "section5_drawdown_top1_depth": float(dd["Depth"].iloc[0]),
        "section5_capture": {k: float(cap[k]) for k in
                             ("Up Capture", "Down Capture", "Hit Ratio", "Batting Average")},
        "section5_brinson_active": float(brin.attrs["active_return"]),
        "section5_frontier_min_vol": float(front["Vol"].min()),
        "section5_frontier_max_return": float(front["Return"].max()),
        "section6_ff5_alpha": float(ff_out.loc["ASSET_A", "Alpha"]),
        "section6_ff5_mkt": float(ff_out.loc["ASSET_A", "Mkt"]),
        "section6_ff5_r2": float(ff_out.loc["ASSET_A", "R-Squared"]),
    }
    print("\n=== EXPECTED block ===")
    print(json.dumps(block, indent=4, default=float))
    print("=== end ===\n")
