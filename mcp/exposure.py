"""Exposure / slicing — asset-class, sector, country and underlying-holding breakdowns.

Thin wrappers around ``utils.exposure``. Inputs are plain ``{portfolio: weight}``
dicts; outputs are sorted ``{bucket: weight}`` dicts ready to be rendered or
passed to an LLM.

All tools handle CUSTOM-portfolio expansion when you supply ``custom_etf_nav``
(produced by :func:`mcp.holdings.aggregate_client_holdings`).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.exposure import (  # noqa: E402
    calculate_exposure,
    expand_custom_in_weights_dict,
    get_concentration_exposure,
)

from mcp._data_loader import get_reference_data


def _weights_to_wgt_series(
    weights: dict[str, float],
    custom_etf_nav: Optional[dict[str, float]] = None,
) -> pd.Series:
    """Convert ``{portfolio: weight}`` into the ``_wgt``-suffixed Series shape
    that ``utils.exposure`` functions expect, expanding CUSTOM if provided."""
    if custom_etf_nav:
        weights = expand_custom_in_weights_dict(weights, custom_etf_nav)
    return pd.Series({f"{k}_wgt": v for k, v in weights.items()})


def _series_to_sorted_dict(series: pd.Series) -> dict:
    """Convert an exposure Series → JSON-friendly dict, sorted by weight desc."""
    if series is None or len(series) == 0:
        return {}
    out = series.sort_values(ascending=False)
    return {str(k): float(v) for k, v in out.items()}


def get_asset_class_exposure(
    weights: dict[str, float],
    custom_etf_nav: Optional[dict[str, float]] = None,
) -> dict:
    """Return the asset-class breakdown of a portfolio.

    Use when the user asks how their portfolio is split between equity,
    fixed income, cash, etc.

    Args:
        weights: ``{portfolio_name: weight}`` summing to ~1.
        custom_etf_nav: optional ``{etf: nav}`` for expanding CUSTOM
            portfolios into their constituents. Pass the value from
            ``aggregate_client_holdings()`` if the book contains any.

    Returns:
        ``{asset_class: weight_0_to_1}`` sorted by weight descending.
        Weights sum to ~1 across known asset classes (small gap possible
        for unmapped tickers).
    """
    ref = get_reference_data()
    s = _weights_to_wgt_series(weights, custom_etf_nav)
    exposure = calculate_exposure(s, ref.asset_class)
    return _series_to_sorted_dict(exposure)


def get_sector_exposure(
    weights: dict[str, float],
    custom_etf_nav: Optional[dict[str, float]] = None,
) -> dict:
    """Return the sector breakdown of a portfolio (e.g. Technology, Financials).

    Use when the user asks which sectors they're exposed to or whether
    they're over/underweight a particular sector.

    Args:
        weights: ``{portfolio_name: weight}``.
        custom_etf_nav: optional CUSTOM expansion map.

    Returns:
        ``{sector: weight_0_to_1}`` sorted by weight descending.
    """
    ref = get_reference_data()
    s = _weights_to_wgt_series(weights, custom_etf_nav)
    exposure = calculate_exposure(s, ref.sector)
    return _series_to_sorted_dict(exposure)


def get_country_exposure(
    weights: dict[str, float],
    custom_etf_nav: Optional[dict[str, float]] = None,
) -> dict:
    """Return the country / region breakdown of a portfolio.

    Use when the user asks about geographic exposure ("how much US?",
    "what's my emerging-markets allocation?").

    Args:
        weights: ``{portfolio_name: weight}``.
        custom_etf_nav: optional CUSTOM expansion map.

    Returns:
        ``{country: weight_0_to_1}`` sorted by weight descending.
    """
    ref = get_reference_data()
    s = _weights_to_wgt_series(weights, custom_etf_nav)
    exposure = calculate_exposure(s, ref.country)
    return _series_to_sorted_dict(exposure)


def get_top_holdings(
    weights: dict[str, float],
    top_n: int = 10,
    custom_etf_nav: Optional[dict[str, float]] = None,
) -> dict:
    """Return the top-N underlying holdings (look-through) across the portfolio.

    Use when the user asks "what stocks am I actually holding?" or
    "what's my biggest single-name exposure?". Equity ETF holdings are
    aggregated across portfolios so that AAPL inside SPY + AAPL inside
    QQQ shows up as one combined exposure.

    Share-class variants (e.g. GOOG / GOOGL) are unified.

    Args:
        weights: ``{portfolio_name: weight}``.
        top_n: how many holdings to return (default 10).
        custom_etf_nav: optional CUSTOM expansion map.

    Returns:
        ``{
            "holdings": {holding_name: weight_0_to_1},  # top N, sorted desc
            "top_n":    int,
            "total_weight_in_top_n": float,             # sum of the returned weights
        }``
    """
    ref = get_reference_data()
    s = _weights_to_wgt_series(weights, custom_etf_nav)
    conc = get_concentration_exposure(s, ref.underlying, top_n=top_n)
    holdings = _series_to_sorted_dict(conc.head(top_n))
    return {
        "holdings": holdings,
        "top_n": top_n,
        "total_weight_in_top_n": float(sum(holdings.values())),
    }
