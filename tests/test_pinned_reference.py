"""Pinned regression tests.

These assert *exact* numerical outputs for the synthetic reference dataset.
If any formula changes (intentionally or accidentally), these will fail with
a clear numeric diff.

To regenerate after an intentional change:

    pytest tests/test_pinned_reference.py -v --pinned-regen

(The `--pinned-regen` flag is implemented as an env var below; flip it on,
re-run, copy the printed values into the constants block, and revert the env.)

Update the constants ONLY when you've reviewed the new numbers and confirmed
they match the new formula.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from port_cons import factor_exposure_coefficient
from utils.optimiser import calc_port_perf
from utils.performance import calculate_metrics, calculate_relative_metrics
from utils.risk_analytics import (
    annualised_cov_from_returns,
    compute_active_risk_contributions,
    compute_brinson_attribution,
    compute_capture_ratios,
    compute_drawdown_episodes,
    compute_efficient_frontier,
    compute_risk_contributions,
    compute_var_cvar,
)


# ============================================================================
# PINNED EXPECTATIONS — DO NOT EDIT WITHOUT REGENERATING.
# ============================================================================
# Generated from the reference dataset (seed=42, 1000 business days starting
# 2021-01-04, 4 assets + 1 benchmark) on commit when these tests were written.
# Use os.environ['PINNED_REGEN']='1' + the helper at the bottom of this file
# to regenerate after intentional formula changes.

PINNED = {
    "key_summary": {
        "Cumulative Return": 0.127865644334078,
        "Annualized Return": 0.03078681961970786,
        "Annualized Volatility": 0.09262696555657655,
        "Sharpe Ratio": 0.3323742652554376,
        "Sortino Ratio": 0.47341765638505184,
        "Max Drawdown": -0.15828697901682454,
    },
    "relative_metrics": {
        "Tracking Error": 0.17021758117092517,
        "Information Ratio": -0.10428308088479031,
    },
    "var_cvar_95": {
        "Historical VaR": -0.00925704776686629,
        "Historical CVaR": -0.012392695921769847,
        "Parametric VaR": -0.009460296350535156,
        "Parametric CVaR": -0.011898483930939414,
    },
    "var_cvar_99": {
        "Historical VaR": -0.013905625632542103,
        "Historical CVaR": -0.015707318512369173,
        "Parametric VaR": -0.013436781460086897,
        "Parametric CVaR": -0.015414049773011026,
    },
    "mctr_components_sum": 0.09189993050708799,           # = portfolio vol
    "mcte_components_sum": 0.03904334705237757,           # = TE for equal-weight bench
    "drawdown_top1_depth": -0.15828697901682454,          # equal to Max DD
    "frontier_min_vol": 0.05453725894671736,
    "frontier_max_return": 0.11423194978819869,
    "ff_alpha": 0.00031148312344662113,
    "ff_mkt_beta": -0.005595286620198522,
    "ff_r_squared": 0.0029150904782472287,
    "brinson_active": 0.15605178526724628,                # bucket-aggregated active
    "capture_up": 0.09009371120191723,
    "capture_down": -0.08824506830027477,
    "capture_hit_ratio": 0.5106382978723404,
}

ATOL = 1e-9
RTOL = 1e-9


def _almost(actual: float, expected: float, key: str) -> None:
    assert np.isclose(actual, expected, atol=ATOL, rtol=RTOL), (
        f"PINNED MISMATCH [{key}]: got {actual!r}, expected {expected!r}, "
        f"diff = {actual - expected:.6e}"
    )


# ---------------------------------------------------------------------------
# Tests against the pinned numbers
# ---------------------------------------------------------------------------
def test_pinned_key_summary(asset_returns, weights_history) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    metrics = calculate_metrics(perf["P_port_ret"])
    for k, expected in PINNED["key_summary"].items():
        _almost(metrics[k], expected, f"key_summary.{k}")


def test_pinned_relative_metrics(asset_returns, weights_history, bench_returns) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    metrics, _ = calculate_relative_metrics(perf["P_port_ret"], bench_returns)
    for k, expected in PINNED["relative_metrics"].items():
        _almost(metrics[k], expected, f"relative_metrics.{k}")


def test_pinned_var_cvar(asset_returns, weights_history) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    var = compute_var_cvar(perf["P_port_ret"], confidence_levels=(0.95, 0.99))
    for k, expected in PINNED["var_cvar_95"].items():
        _almost(float(var.loc[0.95, k]), expected, f"var_cvar_95.{k}")
    for k, expected in PINNED["var_cvar_99"].items():
        _almost(float(var.loc[0.99, k]), expected, f"var_cvar_99.{k}")


def test_pinned_mctr_sum(asset_returns, latest_weights) -> None:
    cov = annualised_cov_from_returns(asset_returns)
    out = compute_risk_contributions(latest_weights, cov)
    _almost(
        out["Component Risk (annualised)"].sum(),
        PINNED["mctr_components_sum"],
        "mctr_components_sum",
    )


def test_pinned_mcte_sum(asset_returns, latest_weights) -> None:
    cov = annualised_cov_from_returns(asset_returns)
    bm = pd.Series(0.25, index=cov.index)  # equal-weight bench across asset universe
    out = compute_active_risk_contributions(latest_weights, bm, cov)
    _almost(
        out["Component TE (annualised)"].sum(),
        PINNED["mcte_components_sum"],
        "mcte_components_sum",
    )


def test_pinned_drawdown_top_depth(asset_returns, weights_history) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    out = compute_drawdown_episodes(perf["P_port_ret"], top_n=1)
    _almost(float(out["Depth"].iloc[0]), PINNED["drawdown_top1_depth"], "drawdown_top1_depth")


def test_pinned_frontier_extents(asset_returns) -> None:
    out = compute_efficient_frontier(asset_returns, n_points=20)
    _almost(float(out["Vol"].min()), PINNED["frontier_min_vol"], "frontier_min_vol")
    _almost(float(out["Return"].max()), PINNED["frontier_max_return"], "frontier_max_return")


def test_pinned_factor_alpha_mkt_r2(asset_returns) -> None:
    """Synthetic factor data with the same seed for reproducibility."""
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
    _almost(float(out.loc["ASSET_A", "Alpha"]), PINNED["ff_alpha"], "ff_alpha")
    _almost(float(out.loc["ASSET_A", "Mkt"]), PINNED["ff_mkt_beta"], "ff_mkt_beta")
    _almost(float(out.loc["ASSET_A", "R-Squared"]), PINNED["ff_r_squared"], "ff_r_squared")


def test_pinned_brinson_active(asset_returns, latest_weights, benchmark_weights, exposure_table) -> None:
    period_returns = (1 + asset_returns).prod() - 1
    out = compute_brinson_attribution(
        latest_weights, benchmark_weights, period_returns, exposure_table
    )
    _almost(out.attrs["active_return"], PINNED["brinson_active"], "brinson_active")


def test_pinned_capture(asset_returns, weights_history, bench_returns) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    cap = compute_capture_ratios(perf["P_port_ret"], bench_returns, frequency="M")
    _almost(float(cap["Up Capture"]), PINNED["capture_up"], "capture_up")
    _almost(float(cap["Down Capture"]), PINNED["capture_down"], "capture_down")
    _almost(float(cap["Hit Ratio"]), PINNED["capture_hit_ratio"], "capture_hit_ratio")


# ---------------------------------------------------------------------------
# Helper: print the actual values to stdout so you can paste them into PINNED
# above after an intentional formula change. Run with:
#     PINNED_REGEN=1 pytest tests/test_pinned_reference.py::test_print_for_regen -s
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("PINNED_REGEN") != "1", reason="set PINNED_REGEN=1 to enable")
def test_print_for_regen(
    asset_returns, weights_history, bench_returns, latest_weights, benchmark_weights, exposure_table
) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    metrics = calculate_metrics(perf["P_port_ret"])
    rel, _ = calculate_relative_metrics(perf["P_port_ret"], bench_returns)
    var = compute_var_cvar(perf["P_port_ret"], confidence_levels=(0.95, 0.99))
    cov = annualised_cov_from_returns(asset_returns)
    mctr = compute_risk_contributions(latest_weights, cov)
    bm_eq = pd.Series(0.25, index=cov.index)
    mcte = compute_active_risk_contributions(latest_weights, bm_eq, cov)
    dd = compute_drawdown_episodes(perf["P_port_ret"], top_n=1)
    frontier = compute_efficient_frontier(asset_returns, n_points=20)

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

    period_returns = (1 + asset_returns).prod() - 1
    brin = compute_brinson_attribution(
        latest_weights, benchmark_weights, period_returns, exposure_table
    )
    cap = compute_capture_ratios(perf["P_port_ret"], bench_returns, frequency="M")

    block = {
        "key_summary": {k: float(metrics[k]) for k in
                        ("Cumulative Return", "Annualized Return", "Annualized Volatility",
                         "Sharpe Ratio", "Sortino Ratio", "Max Drawdown")},
        "relative_metrics": {k: float(rel[k]) for k in ("Tracking Error", "Information Ratio")},
        "var_cvar_95": {c: float(var.loc[0.95, c]) for c in
                        ("Historical VaR", "Historical CVaR", "Parametric VaR", "Parametric CVaR")},
        "var_cvar_99": {c: float(var.loc[0.99, c]) for c in
                        ("Historical VaR", "Historical CVaR", "Parametric VaR", "Parametric CVaR")},
        "mctr_components_sum": float(mctr["Component Risk (annualised)"].sum()),
        "mcte_components_sum": float(mcte["Component TE (annualised)"].sum()),
        "drawdown_top1_depth": float(dd["Depth"].iloc[0]),
        "frontier_min_vol": float(frontier["Vol"].min()),
        "frontier_max_return": float(frontier["Return"].max()),
        "ff_alpha": float(ff_out.loc["ASSET_A", "Alpha"]),
        "ff_mkt_beta": float(ff_out.loc["ASSET_A", "Mkt"]),
        "ff_r_squared": float(ff_out.loc["ASSET_A", "R-Squared"]),
        "brinson_active": float(brin.attrs["active_return"]),
        "capture_up": float(cap["Up Capture"]),
        "capture_down": float(cap["Down Capture"]),
        "capture_hit_ratio": float(cap["Hit Ratio"]),
    }

    print("\n=== PINNED block (paste into PINNED dict in this file) ===")
    import json
    print(json.dumps(block, indent=4, default=float))
    print("=== end ===\n")
