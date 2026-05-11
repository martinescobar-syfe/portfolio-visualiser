"""Tests for `port_cons.exposure_analysis_mixed` and `calc_avg_credit_rating`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from port_cons import calc_avg_credit_rating, exposure_analysis_mixed


def test_exposure_analysis_mixed_renormalises_fi_subset() -> None:
    """If the FI sleeve is < 100% of the portfolio, the helper renormalises
    so the FI metric averages reflect the FI sleeve only (not diluted)."""
    # Portfolio: 60% FI (split 30/30 across BOND_A / BOND_B), 40% Equity
    weights = pd.DataFrame(
        {"BOND_A_wgt": [0.3], "BOND_B_wgt": [0.3], "EQ_wgt": [0.4]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    weights["reb_flag"] = True

    metrics = pd.DataFrame(
        {"BOND_A": [5.0], "BOND_B": [3.0], "EQ": [0.0]},
        index=["Duration"],
    )

    fi_list = ["BOND_A", "BOND_B"]
    result = exposure_analysis_mixed(weights, metrics, fi_list, latest_data=True,
                                     start_date=weights.index[0],
                                     end_date=weights.index[0])

    # Expected: FI subset weights renormalised to 50/50 → weighted duration = 4.0
    assert np.isclose(result.loc["Duration"].iloc[0], 4.0, atol=1e-9)


def test_calc_avg_credit_rating_known_blend() -> None:
    """A 50/50 blend of AAA (1) and BBB (9) should round to A (6).

    Note: AAA=1, AA+=2, AA=3, AA-=4, A+=5, A=6, A-=7, BBB+=8, BBB=9.
    Weighted avg = 5.0 -> round to 5 -> A+. So actually this should give A+.
    Adjust expectations accordingly.
    """
    weights = pd.DataFrame(
        {"BOND_AAA_wgt": [0.5], "BOND_BBB_wgt": [0.5]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    weights["reb_flag"] = True

    ratings = pd.DataFrame(
        {"BOND_AAA": ["AAA"], "BOND_BBB": ["BBB"]},
        index=["Credit rating"],
    )

    result = calc_avg_credit_rating(
        weights, ratings, latest_data=True,
        start_date=weights.index[0], end_date=weights.index[0]
    )
    # (1 + 9) / 2 = 5.0 → round(5) = 5 → "A+"
    assert result == "A+"


def test_calc_avg_credit_rating_skip_zero_weights() -> None:
    """Tickers with zero weight should not influence the average."""
    weights = pd.DataFrame(
        {"BOND_AAA_wgt": [0.5], "BOND_AA_wgt": [0.5], "BOND_D_wgt": [0.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    weights["reb_flag"] = True

    ratings = pd.DataFrame(
        {"BOND_AAA": ["AAA"], "BOND_AA": ["AA"], "BOND_D": ["D"]},
        index=["Credit rating"],
    )

    result = calc_avg_credit_rating(
        weights, ratings, latest_data=True,
        start_date=weights.index[0], end_date=weights.index[0]
    )
    # (1 + 3) / 2 = 2.0 → "AA+"
    assert result == "AA+"
