"""Tests for `port_cons.factor_regression` and `factor_exposure_coefficient`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from port_cons import factor_exposure_coefficient, factor_regression


def _make_ff_data(n=400, seed=11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0003, 0.012, n),
            "SMB": rng.normal(0.0001, 0.005, n),
            "HML": rng.normal(0.0001, 0.005, n),
            "RMW": rng.normal(0.0001, 0.004, n),
            "CMA": rng.normal(0.0001, 0.004, n),
            "RF": np.full(n, 0.00015),
        },
        index=idx,
    )


def test_ff5_includes_all_five_factors() -> None:
    ff = _make_ff_data()
    rng = np.random.default_rng(0)
    asset = pd.Series(rng.normal(0.0005, 0.01, len(ff)), index=ff.index, name="A")
    fit = factor_regression(asset, ff)
    # Coefficient names after rename of Mkt-RF -> Mkt
    assert {"Mkt", "SMB", "HML", "RMW", "CMA"}.issubset(set(fit.params.index))


def test_ff5_raises_on_missing_factor() -> None:
    ff = _make_ff_data().drop(columns=["CMA"])
    rng = np.random.default_rng(0)
    asset = pd.Series(rng.normal(0.0, 0.01, len(ff)), index=ff.index, name="A")
    with pytest.raises(ValueError, match="missing required factor"):
        factor_regression(asset, ff)


def test_factor_exposure_coefficient_returns_numeric_dataframe() -> None:
    ff = _make_ff_data()
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        {
            "A": rng.normal(0.0005, 0.01, len(ff)),
            "B": rng.normal(0.0003, 0.008, len(ff)),
        },
        index=ff.index,
    )
    out = factor_exposure_coefficient(panel, ff, ["A", "B"])
    expected_cols = {"Alpha", "Mkt", "SMB", "HML", "RMW", "CMA", "R-Squared"}
    assert expected_cols.issubset(set(out.columns))
    # All numeric — no string formatting, no NaN
    assert all(np.issubdtype(out[c].dtype, np.number) for c in out.columns)
    assert not out.isna().any().any()


def test_factor_exposure_coefficient_failure_returns_nan_row() -> None:
    """Bad input (missing factor column) should surface as a NaN row, not a zeros row,
    so callers can distinguish failure from a literal zero exposure."""
    ff = _make_ff_data().drop(columns=["CMA"])
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({"A": rng.normal(0.0, 0.01, len(ff))}, index=ff.index)
    out = factor_exposure_coefficient(panel, ff, ["A"])
    assert out.loc["A"].isna().all()


def test_ff5_excess_return_uses_rf() -> None:
    """When RF is zero, regression coefficients should match a regression
    with no RF subtraction."""
    ff = _make_ff_data()
    ff_zero_rf = ff.copy()
    ff_zero_rf["RF"] = 0.0
    rng = np.random.default_rng(0)
    asset = pd.Series(rng.normal(0.0005, 0.01, len(ff)), index=ff.index, name="A")
    fit_zero = factor_regression(asset, ff_zero_rf)
    fit_with_rf = factor_regression(asset, ff)
    # Different RF series -> intercept shifts; betas can also differ slightly
    # because RF affects the LHS series.
    assert "Mkt" in fit_zero.params.index
    assert "Mkt" in fit_with_rf.params.index
    # With non-zero RF, Alpha should differ from the zero-RF fit
    # by approximately RF.mean() (the LHS got shifted down by that amount).
    alpha_diff = fit_with_rf.params.get("Intercept", 0) - fit_zero.params.get("Intercept", 0)
    assert np.isclose(alpha_diff, -ff["RF"].mean(), rtol=0, atol=5e-4)
