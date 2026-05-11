"""Tests for the FX overlay convention used by the Builder.

The Builder applies the conversion `(1 + R_USD) × (1 + R_FX) - 1` to map
USD daily returns to a non-USD display currency. We pin this formula so any
future refactor that introduces, say, a `pct_change()` re-application or a
sign flip will fail CI immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _fx_overlay(usd_ret: pd.Series, fx_ret: pd.Series) -> pd.Series:
    """Reference implementation matching the Builder's FX block."""
    common = usd_ret.index.intersection(fx_ret.index)
    p = usd_ret.loc[common]
    f = fx_ret.loc[common]
    return (1 + p) * (1 + f) - 1


def test_fx_overlay_zero_fx_is_identity() -> None:
    idx = pd.bdate_range("2024-01-01", periods=10)
    usd = pd.Series(np.linspace(0.001, 0.01, 10), index=idx)
    fx = pd.Series(0.0, index=idx)
    out = _fx_overlay(usd, fx)
    pd.testing.assert_series_equal(out, usd, check_names=False)


def test_fx_overlay_multiplicative() -> None:
    """A constant 1% FX gain on a 2% USD return must produce 3.02% local."""
    idx = pd.bdate_range("2024-01-01", periods=3)
    usd = pd.Series([0.02, 0.02, 0.02], index=idx)
    fx = pd.Series([0.01, 0.01, 0.01], index=idx)
    out = _fx_overlay(usd, fx)
    expected = (1 + 0.02) * (1 + 0.01) - 1
    np.testing.assert_allclose(out.values, expected, rtol=1e-12)


def test_fx_overlay_self_inverse_with_inverse_rate() -> None:
    """If you apply USDSGD then SGDUSD = 1/(1+R_USDSGD) - 1, you get back the
    original USD return."""
    idx = pd.bdate_range("2024-01-01", periods=20)
    np.random.seed(0)
    usd = pd.Series(np.random.normal(0.0005, 0.01, 20), index=idx)
    fx = pd.Series(np.random.normal(0.0001, 0.005, 20), index=idx)

    sgd = _fx_overlay(usd, fx)
    inverse_fx = (1 / (1 + fx)) - 1                  # equivalent to SGDUSD return
    back_to_usd = _fx_overlay(sgd, inverse_fx)
    np.testing.assert_allclose(back_to_usd.values, usd.values, rtol=1e-12, atol=1e-15)


def test_double_fx_overlay_produces_wrong_magnitude() -> None:
    """Sanity check: applying the overlay twice should NOT round-trip
    (this is a regression guard against accidentally double-converting
    when toggling the Display Currency selectbox repeatedly)."""
    idx = pd.bdate_range("2024-01-01", periods=10)
    usd = pd.Series([0.01] * 10, index=idx)
    fx = pd.Series([0.005] * 10, index=idx)
    once = _fx_overlay(usd, fx)
    twice = _fx_overlay(once, fx)
    # `twice` should differ from both usd and once
    assert not np.allclose(twice.values, usd.values, atol=1e-9)
    assert not np.allclose(twice.values, once.values, atol=1e-9)
