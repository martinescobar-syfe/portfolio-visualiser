"""Tests for `utils.exposure`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.exposure import calculate_exposure, clean_holding_name


def test_calculate_exposure_decimal_input() -> None:
    """Exposure matrix with decimal columns (sum ≈ 1) — output should equal
    the matrix-vector product with the weight series."""
    df_exposure = pd.DataFrame(
        {
            "A": [1.0, 0.0],
            "B": [0.5, 0.5],
            "C": [0.0, 1.0],
        },
        index=["Equity", "Bond"],
    )
    weights = pd.Series({"A_wgt": 0.6, "B_wgt": 0.3, "C_wgt": 0.1})
    out = calculate_exposure(weights, df_exposure)
    # Equity = 0.6*1 + 0.3*0.5 + 0.1*0   = 0.75
    # Bond   = 0.6*0 + 0.3*0.5 + 0.1*1   = 0.25
    assert np.isclose(out["Equity"], 0.75, atol=1e-12)
    assert np.isclose(out["Bond"], 0.25, atol=1e-12)


def test_calculate_exposure_percent_input_auto_normalises() -> None:
    """Exposure matrix where columns sum to ~100 (percentage) should be detected
    by the > 1.5 sum heuristic and divided by 100."""
    df_exposure = pd.DataFrame(
        {
            "A": [100.0, 0.0],
            "B": [50.0, 50.0],
            "C": [0.0, 100.0],
        },
        index=["Equity", "Bond"],
    )
    weights = pd.Series({"A_wgt": 0.6, "B_wgt": 0.3, "C_wgt": 0.1})
    out = calculate_exposure(weights, df_exposure)
    assert np.isclose(out["Equity"], 0.75, atol=1e-12)
    assert np.isclose(out["Bond"], 0.25, atol=1e-12)


def test_calculate_exposure_missing_ticker_falls_back() -> None:
    df_exposure = pd.DataFrame(
        {
            "A": [1.0, 0.0],
        },
        index=["Equity", "Bond"],
    )
    weights = pd.Series({"A_wgt": 0.6, "UNKNOWN_wgt": 0.4})
    out = calculate_exposure(weights, df_exposure)
    # Either 'Others' or some fallback bucket should hold the unknown 0.4
    total = out.sum()
    assert np.isclose(total, 1.0, atol=1e-9)


def test_calculate_exposure_handles_dataframe_input(returns_panel) -> None:
    """When given a DataFrame, calculate_exposure should take the latest row."""
    df_exposure = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.5, 0.5]},
        index=["Equity", "Bond"],
    )
    df_weights = pd.DataFrame({"A_wgt": [0.5, 0.6], "B_wgt": [0.5, 0.4]})
    out = calculate_exposure(df_weights, df_exposure)
    # Should use the last row: A=0.6, B=0.4
    assert np.isclose(out["Equity"], 0.6 + 0.4 * 0.5, atol=1e-12)
    assert np.isclose(out["Bond"], 0.4 * 0.5, atol=1e-12)


def test_overlap_min_kernel_symmetric_diagonal_one() -> None:
    """Standalone implementation check: sum_h min(b_i, b_j) is symmetric and
    diagonal is 1.0 when b_i sums to 1."""
    b1 = pd.Series({"X": 0.6, "Y": 0.3, "Z": 0.1})
    b2 = pd.Series({"X": 0.4, "Y": 0.4, "Z": 0.2})

    def overlap(b_i, b_j):
        common = b_i.index.intersection(b_j.index)
        return float(np.minimum(b_i.loc[common], b_j.loc[common]).sum())

    assert np.isclose(overlap(b1, b2), overlap(b2, b1), atol=1e-12)
    assert np.isclose(overlap(b1, b1), 1.0, atol=1e-12)
    assert np.isclose(overlap(b2, b2), 1.0, atol=1e-12)
    # Off-diagonal overlap is in [0, 1]
    assert 0.0 <= overlap(b1, b2) <= 1.0


def test_clean_holding_name_unifies_share_classes_not_companies() -> None:
    """Share-class unification must not over-merge: stripping only CLASS
    designators keeps GOOG/GOOGL-style merges while keeping unrelated firms that
    merely share a base name (Titan Co Ltd vs Titan SA) distinct."""
    # share classes of the SAME company unify
    assert clean_holding_name("Alphabet Inc Class A") == clean_holding_name("Alphabet Inc Class C")
    # unrelated companies that merely share a base name must NOT merge
    assert clean_holding_name("Titan Co Ltd") != clean_holding_name("Titan SA")
