"""Tests for `utils.optimiser`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.optimiser import calc_port_perf, create_constraints, min_vol


def test_calc_port_perf_no_lookahead(asset_returns, weights_history) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    # First day must be 0 (no prior weights to use)
    assert perf["P_port_ret"].iloc[0] == 0.0


def test_calc_port_perf_uses_lagged_weights(asset_returns, weights_history) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    # For each day t≥1, port_ret_t = sum_i w_{i,t-1} × r_{i,t}
    weights_only = weights_history[[c for c in weights_history.columns if c.endswith("_wgt")]].copy()
    weights_only.columns = [c[:-4] for c in weights_only.columns]
    expected = (asset_returns * weights_only.shift(1)).sum(axis=1)
    expected.iloc[0] = 0.0
    np.testing.assert_allclose(perf["P_port_ret"].values, expected.values, rtol=1e-12)


def test_calc_port_perf_cumret_is_consistent(asset_returns, weights_history) -> None:
    perf = calc_port_perf(asset_returns, weights_history, name_prefix="P")
    expected_cumret = (1 + perf["P_port_ret"]).cumprod()
    np.testing.assert_allclose(perf["P_cum_ret"].values, expected_cumret.values, rtol=1e-12)


def test_create_constraints_long_only_default() -> None:
    cons = create_constraints(0.0, 1.0)
    # Should have one equality (sum = 1) plus two inequality blocks (≥0 and ≤1)
    eqs = [c for c in cons if c["type"] == "eq"]
    ineqs = [c for c in cons if c["type"] == "ineq"]
    assert len(eqs) == 1
    assert len(ineqs) == 2

    # Check that the equality constraint enforces sum ≈ 1
    w = np.array([0.5, 0.5])
    assert np.isclose(eqs[0]["fun"](w), 0.0)


def test_min_vol_respects_default_bounds() -> None:
    """min_vol should never produce negative weights with default bounds."""
    np.random.seed(7)
    rets = pd.DataFrame(
        np.random.normal(0.0, 0.01, size=(252, 4)),
        columns=["A", "B", "C", "D"],
    )
    cov = rets.cov().values * 252
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    weights = min_vol(rets, cov, constraints)
    # Bounds default to (0, 1) so all weights must be non-negative and bounded above
    assert (weights >= -1e-9).all()
    assert (weights <= 1 + 1e-9).all()
    assert np.isclose(weights.sum(), 1.0, rtol=1e-6)


def test_min_vol_explicit_bounds() -> None:
    np.random.seed(7)
    rets = pd.DataFrame(
        np.random.normal(0.0, 0.01, size=(252, 3)),
        columns=["A", "B", "C"],
    )
    cov = rets.cov().values * 252
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = [(0.1, 0.5)] * 3
    weights = min_vol(rets, cov, constraints, bounds=bounds)
    assert (weights >= 0.1 - 1e-6).all()
    assert (weights <= 0.5 + 1e-6).all()


def test_port_cons_drift_handles_zero_sum() -> None:
    """Drift formula in port_cons should not raise ZeroDivisionError when the
    weighted sum collapses to zero (defensive guard against bad return data)."""
    import port_cons as pc

    # Construct a returns panel where every asset returns -100% on day 1
    # (after the first row). The drift formula `w_{t-1} * (1 + r)` then
    # produces all-zero rows and `weights_sum == 0`.
    idx = pd.bdate_range("2024-01-02", periods=10)
    rets = pd.DataFrame(
        np.zeros((10, 2)),
        index=idx,
        columns=["A", "B"],
    )
    rets.iloc[1] = -1.0  # everyone -100% on day 2 -> drifted weights = 0

    # equal_weight_rebal_wgt rebalances on Jan 1 (first day) only here
    out = pc.equal_weight_rebal_wgt(rets, idx[0], idx[-1], rebal_months=[4])
    # Should complete without raising; subsequent rows can be NaN/0 but the
    # function must not crash on division by zero
    assert "reb_flag" in out.columns
    assert len(out) == len(rets)
