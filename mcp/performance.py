"""Performance — backtested metrics from daily return history.

Two flavours of tools here:

1. :func:`get_portfolio_metrics` — given a portfolio mix, build the
   weighted daily-return series from the reference data and return the
   standard summary (CAGR, vol, Sharpe, Sortino, max drawdown).
2. :func:`get_period_return` — same daily series but truncated to a
   named window (YTD, 1Y, 3Y, 5Y, ITD).
3. :func:`get_relative_metrics` — vs a chosen benchmark (tracking error,
   information ratio, latest 12M geometric excess).

All series use the convention enforced everywhere in the codebase:
**decimal returns**, **252 trading days/year**, **lagged weights are
not relevant** here because the daily-return series is already realised.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.performance import calculate_metrics, calculate_relative_metrics  # noqa: E402

from mcp._data_loader import get_reference_data


def _build_weighted_returns(
    weights: dict[str, float],
    daily_ret: pd.DataFrame,
) -> Optional[pd.Series]:
    """Build the weighted daily return series r_C(t) = Σ wᵢ · rᵢ(t).

    Mirrors the construction in ``pages/Client_PV.py`` (Performance section).
    Returns ``None`` when no constituent has usable history.
    """
    if not weights or daily_ret is None or daily_ret.empty:
        return None

    available = [p for p in weights if p in daily_ret.columns]
    if not available:
        return None

    port_rets = daily_ret[available].dropna(how='all').copy()
    port_rets = port_rets.apply(pd.to_numeric, errors='coerce')
    drop_cols = [c for c in port_rets.columns if port_rets[c].isna().all()]
    if drop_cols:
        port_rets = port_rets.drop(columns=drop_cols)
        available = [p for p in available if p not in drop_cols]
    if port_rets.empty or not available:
        return None

    # Defensive percent → decimal (a true daily return can't exceed 1 in decimal).
    if not port_rets.empty:
        col_max = port_rets.abs().max()
        pct_cols = col_max[col_max > 1].index
        if len(pct_cols) > 0:
            port_rets[pct_cols] = port_rets[pct_cols] / 100

    if port_rets.index.duplicated().any():
        port_rets = port_rets.groupby(level=0).first()

    common_start = port_rets.dropna().index.min()
    port_rets = port_rets.loc[common_start:]

    series = pd.Series(0.0, index=port_rets.index)
    for p in available:
        series += port_rets[p].fillna(0) * float(weights.get(p, 0))
    series = series.dropna()
    if series.index.duplicated().any():
        series = series.groupby(level=0).first()
    return series if not series.empty else None


def _metrics_dict(series: pd.Series, rf: float = 0.0) -> dict:
    m = calculate_metrics(series, rf=rf)
    if m.empty:
        return {}
    return {
        "cumulative_return": float(m.get('Cumulative Return', np.nan)),
        "annualized_return": float(m.get('Annualized Return', np.nan)),
        "annualized_volatility": float(m.get('Annualized Volatility', np.nan)),
        "sharpe_ratio": float(m.get('Sharpe Ratio', np.nan)),
        "sortino_ratio": float(m.get('Sortino Ratio', np.nan)),
        "max_drawdown": float(m.get('Max Drawdown', np.nan)),
    }


def get_portfolio_metrics(
    weights: dict[str, float],
    risk_free_rate: float = 0.0,
) -> dict:
    """Return headline performance metrics for a portfolio mix.

    Use when the user asks "how has this portfolio performed?", what
    the Sharpe / Sortino ratio is, or how big the worst drawdown was.

    Computed from backtested daily returns (Bloomberg-sourced) anchored
    to the earliest date every constituent has data.

    Args:
        weights: ``{portfolio_name: weight}`` summing to ~1.
        risk_free_rate: annualised decimal (e.g. 0.04 for 4%). Default 0.

    Returns:
        ``{
            "cumulative_return":     float,
            "annualized_return":     float,  # CAGR
            "annualized_volatility": float,
            "sharpe_ratio":          float,
            "sortino_ratio":         float,
            "max_drawdown":          float,  # ≤ 0
            "data_start":            "YYYY-MM-DD",
            "data_end":              "YYYY-MM-DD",
            "n_days":                int,
        }``

        Empty dict (with the warning key) if no constituent has data.
    """
    ref = get_reference_data()
    series = _build_weighted_returns(weights, ref.daily_ret)
    if series is None:
        return {"error": "no_return_history_for_constituents"}

    out = _metrics_dict(series, rf=risk_free_rate)
    out.update({
        "data_start": series.index.min().strftime("%Y-%m-%d"),
        "data_end": series.index.max().strftime("%Y-%m-%d"),
        "n_days": int(len(series)),
    })
    return out


_PERIOD_WINDOWS = {
    "YTD": None,        # special-cased below
    "1Y": 252,
    "3Y": 252 * 3,
    "5Y": 252 * 5,
    "ITD": None,        # full history
}


def get_period_return(weights: dict[str, float], period: str = "YTD") -> dict:
    """Return cumulative return over a named period.

    Use when the user asks "what's my YTD return?" or "how have I done
    over the last year?". Windowed off the same backtested daily-return
    series that :func:`get_portfolio_metrics` uses.

    Args:
        weights: ``{portfolio_name: weight}``.
        period: one of ``"YTD"``, ``"1Y"``, ``"3Y"``, ``"5Y"``, ``"ITD"``.

    Returns:
        ``{
            "period":            str,
            "cumulative_return": float,
            "annualized_return": float,
            "start":             "YYYY-MM-DD",
            "end":               "YYYY-MM-DD",
            "n_days":            int,
        }``
    """
    if period not in _PERIOD_WINDOWS:
        return {"error": f"unknown_period: {period!r}", "valid_periods": list(_PERIOD_WINDOWS)}

    ref = get_reference_data()
    series = _build_weighted_returns(weights, ref.daily_ret)
    if series is None:
        return {"error": "no_return_history_for_constituents"}

    if period == "YTD":
        ystart = datetime(series.index.max().year, 1, 1)
        windowed = series[series.index >= ystart]
    elif period == "ITD":
        windowed = series
    else:
        n = _PERIOD_WINDOWS[period]
        windowed = series.tail(n)

    if windowed.empty:
        return {"error": "empty_window"}

    cum = float((1 + windowed).prod() - 1)
    n_years = len(windowed) / 252.0 if len(windowed) >= 1 else 1.0
    cagr = (1 + cum) ** (1 / n_years) - 1 if n_years > 0 else float('nan')

    return {
        "period": period,
        "cumulative_return": cum,
        "annualized_return": cagr,
        "start": windowed.index.min().strftime("%Y-%m-%d"),
        "end": windowed.index.max().strftime("%Y-%m-%d"),
        "n_days": int(len(windowed)),
    }


def get_relative_metrics(weights: dict[str, float], benchmark_ticker: str) -> dict:
    """Return tracking error, information ratio and latest 12M excess return.

    Use when the user asks "how am I doing vs ACWI?" or wants to
    compare their book to a benchmark like SPY / URTH.

    Args:
        weights: ``{portfolio_name: weight}``.
        benchmark_ticker: a column name available in the Daily Return
            sheet (e.g. ``"ACWI"``, ``"URTH"``, ``"SPY"``, ``"M1WO Index"``).

    Returns:
        ``{
            "benchmark":                  str,
            "tracking_error":             float,   # annualised
            "information_ratio":          float,
            "latest_12m_excess_return":   float,   # geometric, most recent value
            "data_start":                 "YYYY-MM-DD",
            "data_end":                   "YYYY-MM-DD",
        }``
    """
    ref = get_reference_data()
    series = _build_weighted_returns(weights, ref.daily_ret)
    if series is None:
        return {"error": "no_return_history_for_constituents"}

    if benchmark_ticker not in ref.daily_ret.columns:
        return {"error": f"benchmark_not_found: {benchmark_ticker!r}"}

    bm = ref.daily_ret[benchmark_ticker]
    if isinstance(bm, pd.DataFrame):
        bm = bm.iloc[:, 0]
    if bm.index.duplicated().any():
        bm = bm.groupby(level=0).first()
    bm = pd.to_numeric(bm, errors='coerce')
    bm = (bm / 100) if bm.abs().max() > 1 else bm
    bm = bm.reindex(series.index).fillna(0)

    rel, rolling = calculate_relative_metrics(series, bm)
    latest_excess = float(rolling.dropna().iloc[-1]) if not rolling.dropna().empty else float('nan')

    return {
        "benchmark": benchmark_ticker,
        "tracking_error": float(rel.get('Tracking Error', np.nan)),
        "information_ratio": float(rel.get('Information Ratio', np.nan)),
        "latest_12m_excess_return": latest_excess,
        "data_start": series.index.min().strftime("%Y-%m-%d"),
        "data_end": series.index.max().strftime("%Y-%m-%d"),
    }
