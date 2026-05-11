"""End-to-end smoke + content tests for `create_builder_excel`.

Confirms that:
  1. The function builds a valid xlsx workbook for a typical input.
  2. Every sheet that should be present *is* present.
  3. Per-sheet pinned numbers match the source-of-truth formulas (so any
     drift between the UI and the Excel export shows up in CI).
"""

from __future__ import annotations

import io

import numpy as np
import openpyxl
import pandas as pd
import pytest


def _build_args(asset_returns, bench_returns, weights_history, latest_weights, exposure_table):
    """Assemble the argument bundle the Builder passes to `create_builder_excel`."""
    from utils.optimiser import calc_port_perf

    perf = calc_port_perf(asset_returns, weights_history, name_prefix="Builder")
    all_port_ret = perf[[c for c in perf.columns if c.endswith("_port_ret")]].copy()
    all_port_ret.columns = ["Builder Portfolio"]
    ret_bm_df = bench_returns.to_frame(name="BENCH")

    results = {
        "Builder Portfolio": {
            "weights": weights_history,
            "perf": perf,
            "tickers": list(asset_returns.columns),
        }
    }

    return {
        "results": results,
        "all_port_ret": all_port_ret,
        "ret_bm": ret_bm_df,
        "currency": "USD",
        "config": {
            "Allocation Method": "Fixed Weight",
            "Rebalance Frequency": "Semi-Annual (Apr/Oct)",
            "Start Date": "2021-01-04",
            "End Date": "2024-11-01",
            "Optimisation Currency": "USD",
            "Display Currency": "USD",
            "Universe (count)": len(asset_returns.columns),
            "Universe Tickers": ", ".join(asset_returns.columns),
            "Benchmark Label": "BENCH",
            "# Benchmark Components": 1,
        },
        "constraints_log": [
            {"Scope": "Per-ETF", "Target": "ASSET_A", "Operator": "Max (<)", "Value": 0.5}
        ],
        "benchmark_info": [{"Component": 1, "Ticker": "BENCH", "Weight": 1.0}],
        "ret_all_for_risk": asset_returns,
        "bm_weights_series": pd.Series({"BENCH": 1.0}),
        "df_sector_for_brinson": exposure_table,
        "df_country_for_brinson": exposure_table,
    }


def test_excel_exporter_builds_without_error(
    asset_returns, bench_returns, weights_history, latest_weights, exposure_table
) -> None:
    from pages.Builder import create_builder_excel
    args = _build_args(asset_returns, bench_returns, weights_history, latest_weights, exposure_table)
    excel_bytes = create_builder_excel(**args)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 1000               # non-trivial workbook


def test_excel_exporter_contains_expected_sheets(
    asset_returns, bench_returns, weights_history, latest_weights, exposure_table
) -> None:
    from pages.Builder import create_builder_excel
    args = _build_args(asset_returns, bench_returns, weights_history, latest_weights, exposure_table)
    excel_bytes = create_builder_excel(**args)

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes), read_only=True)
    sheets = set(wb.sheetnames)

    must_have = {
        "Configuration",
        "Constraints",
        "Benchmark Composition",
        "Key Summary",
        "Monthly Returns",
        "Calendar Returns",
        "Rolling Volatility",
        "Drawdowns",
        "Correlation Matrix",
        "Distribution Stats",
        "Concentration HHI",
        "VaR CVaR",
        "Drawdown Top5",
        "Efficient Frontier",
        "Frontier Overlay",
    }
    missing = must_have - sheets
    assert not missing, f"Excel workbook missing sheets: {missing}"


def test_excel_key_summary_matches_source(
    asset_returns, bench_returns, weights_history, latest_weights, exposure_table
) -> None:
    """The 'Key Summary' sheet in Excel must equal `get_performance_summary(all_port_ret)`."""
    from pages.Builder import create_builder_excel
    from utils.performance import get_performance_summary

    args = _build_args(asset_returns, bench_returns, weights_history, latest_weights, exposure_table)
    excel_bytes = create_builder_excel(**args)

    sheet = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Key Summary", index_col=0)
    expected = get_performance_summary(args["all_port_ret"])

    np.testing.assert_allclose(
        sheet.values.astype(float), expected.values.astype(float), rtol=1e-9, atol=1e-12,
        err_msg="Excel 'Key Summary' sheet diverges from in-memory calculation",
    )


def test_excel_distribution_stats_uses_cagr_not_arithmetic(
    asset_returns, bench_returns, weights_history, latest_weights, exposure_table
) -> None:
    """Regression test: 'Annualized Return' in Distribution Stats must be CAGR
    so the workbook is internally consistent with Key Summary."""
    from pages.Builder import create_builder_excel
    args = _build_args(asset_returns, bench_returns, weights_history, latest_weights, exposure_table)
    excel_bytes = create_builder_excel(**args)

    dist = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Distribution Stats", index_col=0)
    s = args["all_port_ret"]["Builder Portfolio"].dropna()

    cagr_expected = (1 + s).prod() ** (252 / len(s)) - 1
    arith_expected = s.mean() * 252

    actual = float(dist.loc["Annualized Return", "Builder Portfolio"])
    assert np.isclose(actual, cagr_expected, rtol=1e-9, atol=1e-12), (
        f"Distribution Stats Annualized Return = {actual:.6f}; expected CAGR = {cagr_expected:.6f}"
    )
    # And that it is NOT the arithmetic-mean form (defensive guard)
    if abs(cagr_expected - arith_expected) > 1e-6:
        assert not np.isclose(actual, arith_expected, rtol=1e-9, atol=1e-12), (
            "Distribution Stats Annualized Return appears to be arithmetic mean × 252; should be CAGR"
        )


def test_excel_rolling_excess_is_geometric(
    asset_returns, bench_returns, weights_history, latest_weights, exposure_table
) -> None:
    """The Rolling Excess Return 12M sheet must be the geometric form
    (1+p_window).prod() / (1+b_window).prod() - 1, not arithmetic sum."""
    from pages.Builder import create_builder_excel
    args = _build_args(asset_returns, bench_returns, weights_history, latest_weights, exposure_table)
    excel_bytes = create_builder_excel(**args)

    df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Rolling Excess Return 12M", index_col=0)
    df.index = pd.to_datetime(df.index)

    p = args["all_port_ret"]["Builder Portfolio"]
    b = args["ret_bm"].iloc[:, 0]
    common = p.index.intersection(b.index)
    p = p.loc[common]
    b = b.loc[common]

    last_idx = df.dropna().index[-1]
    win_end = p.index.get_loc(last_idx)
    p_win = p.iloc[win_end - 252 + 1 : win_end + 1]
    b_win = b.iloc[win_end - 252 + 1 : win_end + 1]
    expected_geo = float((1 + p_win).prod() / (1 + b_win).prod() - 1)
    arith_alt = float((p_win - b_win).sum())

    actual = float(df.loc[last_idx, "Builder Portfolio"])
    assert np.isclose(actual, expected_geo, rtol=1e-9, atol=1e-12), (
        f"Rolling Excess 12M = {actual:.6f}; expected geometric = {expected_geo:.6f}"
    )
    if abs(expected_geo - arith_alt) > 1e-6:
        assert not np.isclose(actual, arith_alt, rtol=1e-9, atol=1e-12), (
            "Rolling Excess appears to be arithmetic sum; should be geometric"
        )


def test_excel_brinson_uses_time_averaged_weights(
    asset_returns, bench_returns, weights_history, latest_weights, exposure_table
) -> None:
    """The Brinson sheet must show 'Avg Port Weight' columns (renamed from
    'Port Weight' to make the time-averaging explicit)."""
    from pages.Builder import create_builder_excel
    args = _build_args(asset_returns, bench_returns, weights_history, latest_weights, exposure_table)
    excel_bytes = create_builder_excel(**args)

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes), read_only=True)
    brin_sheets = [s for s in wb.sheetnames if s.startswith("Brinson Sector")]
    assert brin_sheets, "Brinson Sector sheet missing from Excel export"

    df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=brin_sheets[0], index_col=0)
    assert "Avg Port Weight" in df.columns, (
        "Brinson sheet column should be labelled 'Avg Port Weight' to indicate time-averaging"
    )
    assert "Avg Bench Weight" in df.columns
    assert "Avg Active Weight" in df.columns
