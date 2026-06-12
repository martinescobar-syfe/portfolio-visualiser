"""Tests for the weighted-yield cashflow calculator (mcp.cash_flow.get_weighted_yield).

Exercises the underlying calculator directly (no fastmcp), so it runs in CI. The
MCP tool wrappers (weighted_dividend_yield / cash_flow) are thin pass-throughs of
_resolve_product_keys + _resolve_ccy_map and are covered in test_exposure_aliases.py.

Yields come from the FI Metrics sheet of Inv_Data_Hardcoded.xlsx
(T12M dividend yield / YTM rows). Expected percentages below are the sheet values
at the time of writing; update them if the reference data changes.
"""

from __future__ import annotations

import math
import os

import pytest

from mcp._data_loader import get_reference_data, set_reference_data
from mcp.cash_flow import get_weighted_yield, project_monthly_cash_flow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")


@pytest.fixture(scope="module", autouse=True)
def _prime_reference_data():
    """Pin the cache to the repo's workbook for this module, then clear it."""
    set_reference_data(None)
    get_reference_data(data_path=DATA)
    yield
    set_reference_data(None)


def test_single_income_product_uses_t12m() -> None:
    r = get_weighted_yield({"Income Preserve": 100_000}, display_currency="SGD")
    assert r["weighted_yield_pct"] == pytest.approx(5.6, abs=0.02)
    assert r["coverage_pct"] == pytest.approx(1.0)
    assert r["breakdown"][0]["yield_source"] == "T12M"


def test_weighted_average_across_two_sleeves() -> None:
    # 210k @ 5.6% + 90k @ 1.56% over 300k = 4.388%
    r = get_weighted_yield(
        {"Income Preserve": 210_000, "Cash SGD flexi": 90_000}, display_currency="SGD"
    )
    assert r["weighted_yield_pct"] == pytest.approx(4.388, abs=0.02)
    assert r["annual_income"] == pytest.approx(210_000 * 0.056 + 90_000 * 0.0156, rel=1e-4)
    assert r["total_nav"] == pytest.approx(300_000)


def test_equity_dividends_are_included_not_excluded() -> None:
    """Unlike project_monthly_cash_flow, the weighted yield counts equity
    sleeves' (small) dividend yields rather than filtering them out."""
    r = get_weighted_yield({"Core Equity100": 100_000}, display_currency="USD")
    assert r["weighted_yield_pct"] == pytest.approx(0.742, abs=0.02)
    assert r["coverage_pct"] == pytest.approx(1.0)


def test_base_currencies_make_multicurrency_exact() -> None:
    nav = {"Equity Alpha": 1_850_000, "Income Enhance": 1_300_000, "Cash SGD guaranteed": 1_050_000}
    ccy = {"Equity Alpha": "USD", "Income Enhance": "SGD", "Cash SGD guaranteed": "SGD"}
    r = get_weighted_yield(nav, base_currencies=ccy, display_currency="USD", usdsgd=1.345)
    expected_total = 1_850_000 + 1_300_000 / 1.345 + 1_050_000 / 1.345
    assert r["total_nav"] == pytest.approx(expected_total, rel=1e-6)
    # Only Income Enhance yields (5.8% T12M); Equity Alpha / Cash guaranteed are 0.
    expected_yld = (1_300_000 / 1.345 * 0.058) / expected_total * 100
    assert r["weighted_yield_pct"] == pytest.approx(expected_yld, rel=1e-3)


def test_base_currency_overrides_unreliable_name_inference() -> None:
    """'Equity Alpha' has no currency marker -> name inference would call it SGD;
    an explicit base_currency must win."""
    r = get_weighted_yield(
        {"Equity Alpha": 100_000}, base_currencies={"Equity Alpha": "USD"},
        display_currency="USD", usdsgd=1.345,
    )
    assert r["breakdown"][0]["base_currency"] == "USD"
    assert r["total_nav"] == pytest.approx(100_000)


def test_yield_preference_switches_source() -> None:
    t12 = get_weighted_yield({"Income Enhance": 100_000}, yield_preference="T12M")
    ytm = get_weighted_yield({"Income Enhance": 100_000}, yield_preference="YTM")
    assert t12["weighted_yield_pct"] == pytest.approx(5.8, abs=0.02)
    assert ytm["weighted_yield_pct"] == pytest.approx(7.18, abs=0.02)


def test_empty_and_zero_yield_books() -> None:
    assert get_weighted_yield({})["weighted_yield"] == 0.0
    # SRS Equity100 carries no dividend/YTM figure -> 0 yield, 0 coverage.
    r = get_weighted_yield({"SRS Equity100": 100_000}, display_currency="SGD")
    assert r["weighted_yield"] == 0.0
    assert r["coverage_pct"] == 0.0


def test_no_yield_data_flags_gaps() -> None:
    """Holdings with no yield figure are listed in no_yield_data so the advisor
    can see exactly which sleeves dragged the blended yield down."""
    r = get_weighted_yield(
        {"Income Preserve": 100_000, "Cash SGD guaranteed": 100_000}, display_currency="SGD"
    )
    # Cash SGD guaranteed carries no yield in the sheet; Income Preserve does.
    assert "Cash SGD guaranteed" in r["no_yield_data"]
    assert "Income Preserve" not in r["no_yield_data"]
    assert r["coverage_pct"] == pytest.approx(0.5)


def test_high_yield_alt_funds_resolve() -> None:
    """The Private Wealth alt funds carry high T12M yields and are USD."""
    r = get_weighted_yield(
        {"Monroe Income Plus Fund": 500_000, "Blackrock Private Credit Fund": 500_000},
        display_currency="USD",
    )
    # Monroe 9.2% + Blackrock 9.8% equal-weighted = 9.5%
    assert r["weighted_yield_pct"] == pytest.approx(9.5, abs=0.05)
    assert all(b["base_currency"] == "USD" for b in r["breakdown"])


def test_negative_or_zero_fx_falls_back_to_reference() -> None:
    """A non-positive FX override must NOT yield negative NAV/income — it falls
    back to the reference rate (stress-test Bug B)."""
    r = get_weighted_yield(
        {"Monroe Income Plus Fund": 10_000, "Income Preserve": 10_000},
        base_currencies={"Monroe Income Plus Fund": "USD", "Income Preserve": "SGD"},
        display_currency="SGD", usdsgd=-1.35,
    )
    assert r["total_nav"] > 0
    assert r["annual_income"] >= 0
    assert r["weighted_yield"] >= 0
    assert all(b["nav"] >= 0 for b in r["breakdown"])
    # zero is also handled (no div-by-zero, falls back)
    r0 = get_weighted_yield({"Income Preserve": 10_000},
                            base_currencies={"Income Preserve": "SGD"},
                            display_currency="SGD", usdsgd=0)
    assert r0["total_nav"] > 0


def test_nonfinite_nav_is_dropped() -> None:
    """inf / NaN NAVs must not crash or leak non-finite into the output
    (stress-test Bug A defense)."""
    r = get_weighted_yield({"Income Preserve": float("inf"), "Income Enhance": 100_000})
    assert math.isfinite(r["total_nav"])
    assert math.isfinite(r["weighted_yield"])
    assert r["total_nav"] == pytest.approx(100_000, rel=1e-6)  # only the finite sleeve counts


def test_absurd_fx_rate_does_not_overflow() -> None:
    """A tiny-but-positive FX divisor + a large NAV must NOT overflow to Inf/null
    — an out-of-range FX rate falls back to the reference rate (stress-test Bug C)."""
    r = get_weighted_yield(
        {"Income Enhance": 1e15, "Income Preserve": 1e15},
        base_currencies={"Income Enhance": "SGD", "Income Preserve": "SGD"},
        display_currency="USD", usdsgd=5e-309,
    )
    assert math.isfinite(r["total_nav"]) and r["total_nav"] > 0
    assert 0 <= r["coverage_pct"] <= 1
    assert math.isfinite(r["weighted_yield"])
    c = project_monthly_cash_flow(
        {"Income Enhance": 1e15, "Income Preserve": 1e15},
        display_currency="USD", usdsgd=5e-309,
    )
    assert math.isfinite(c["annual_total"])
    assert all(math.isfinite(v) for v in c["monthly"].values())
