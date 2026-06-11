"""Exposure / slicing — asset-class, sector, country and underlying-holding breakdowns.

Thin wrappers around ``utils.exposure``. Inputs are plain ``{portfolio: weight}``
dicts; outputs are sorted ``{bucket: weight}`` dicts ready to be rendered or
passed to an LLM.

All tools handle CUSTOM-portfolio expansion when you supply ``custom_etf_nav``
(produced by :func:`mcp.holdings.aggregate_client_holdings`).

NAV-friendly + normalised outputs
---------------------------------
The slice functions (``get_asset_class_exposure``, ``get_sector_exposure``,
``get_country_exposure``) are deliberately tolerant of how you express the input
``weights`` and always return a clean, comparable distribution:

  * **Input may be NAVs or weights.** ``{"Core Equity100": 60000,
    "Equity Alpha": 40000}`` and ``{"Core Equity100": 0.6, "Equity Alpha": 0.4}``
    produce the *same* breakdown — the output is rescaled by the sum of the
    inputs. This is what lets the advisor bot pull a client's raw managed-
    portfolio NAVs straight from the playground and feed them in unchanged.
  * **No negative numbers.** Any negative bucket weight is clipped to 0 before
    normalisation.
  * **Sums to 1, remainder → "Others".** Whatever weight does not map to a known
    bucket (products with no column in the relevant sheet, unmapped tickers) is
    collected into an ``"Others"`` bucket so the distribution always sums to 1.0.
    A large ``"Others"`` is the signal of a data-coverage gap, not a real bucket.

``get_top_holdings`` is likewise NAV-friendly (rescaled by the input total and
clipped non-negative) but is intentionally *not* forced to sum to 1 — it is a
top-N slice of the full holdings list, so the returned weights sum to whatever
that top-N captures (reported as ``total_weight_in_top_n``).
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

_OTHERS_LABEL = "Others"


def _weights_to_wgt_series(
    weights: dict[str, float],
    custom_etf_nav: Optional[dict[str, float]] = None,
) -> pd.Series:
    """Convert ``{portfolio: weight}`` into the ``_wgt``-suffixed Series shape
    that ``utils.exposure`` functions expect, expanding CUSTOM if provided."""
    if custom_etf_nav:
        weights = expand_custom_in_weights_dict(weights, custom_etf_nav)
    return pd.Series({f"{k}_wgt": v for k, v in weights.items()})


def _input_total(series: pd.Series) -> float:
    """Sum of the raw input weights/NAVs (the scale factor for normalisation).

    Works whether the caller passed weights that sum to 1, percentages that sum
    to 100, or raw dollar NAVs. Negative inputs are clipped to 0 first so a
    short/erroneous position can't inflate the denominator.
    """
    if series is None or len(series) == 0:
        return 0.0
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    return float(vals.sum())


def _scale_and_clip(series: pd.Series, input_total: float) -> pd.Series:
    """Clip negatives to 0 and rescale a bucket Series by the input total.

    After this the Series is expressed as fractions of the *whole* input book
    (so NAV inputs and weight inputs collapse to the same numbers).
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    if input_total and input_total > 0:
        s = s / float(input_total)
    return s


def _normalise_breakdown(
    series: pd.Series,
    input_total: float,
    others_label: str = _OTHERS_LABEL,
) -> pd.Series:
    """Return a non-negative breakdown that sums to exactly 1.0.

    Steps: clip negatives → rescale by the input total → if the mapped weight
    overshoots 1 (e.g. percent-style columns slipped through), renormalise down;
    if it undershoots 1 (unmapped products/tickers), park the remainder in
    ``others_label`` so the distribution always closes to 1.0.
    """
    s = _scale_and_clip(series, input_total)
    if len(s) == 0:
        return s
    mapped = float(s.sum())
    if mapped > 1.0 + 1e-9:
        s = s / mapped
    elif mapped < 1.0 - 1e-9:
        s[others_label] = float(s.get(others_label, 0.0)) + (1.0 - mapped)
    return s


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
        weights: ``{portfolio_name: weight_or_nav}``. May be normalised weights,
            percentages, or raw NAVs — the result is rescaled by their sum.
        custom_etf_nav: optional ``{etf: nav}`` for expanding CUSTOM
            portfolios into their constituents. Pass the value from
            ``aggregate_client_holdings()`` if the book contains any.

    Returns:
        ``{asset_class: weight_0_to_1}`` sorted by weight descending, clipped
        non-negative and summing to 1.0 (any unmapped weight is collected under
        ``"Others"``).
    """
    ref = get_reference_data()
    s = _weights_to_wgt_series(weights, custom_etf_nav)
    exposure = calculate_exposure(s, ref.asset_class)
    exposure = _normalise_breakdown(exposure, _input_total(s))
    return _series_to_sorted_dict(exposure)


def get_sector_exposure(
    weights: dict[str, float],
    custom_etf_nav: Optional[dict[str, float]] = None,
) -> dict:
    """Return the sector breakdown of a portfolio (e.g. Technology, Financials).

    Use when the user asks which sectors they're exposed to or whether
    they're over/underweight a particular sector.

    Args:
        weights: ``{portfolio_name: weight_or_nav}``. May be normalised weights,
            percentages, or raw NAVs — the result is rescaled by their sum.
        custom_etf_nav: optional CUSTOM expansion map.

    Returns:
        ``{sector: weight_0_to_1}`` sorted by weight descending, clipped
        non-negative and summing to 1.0 (any unmapped weight is collected under
        ``"Others"``).
    """
    ref = get_reference_data()
    s = _weights_to_wgt_series(weights, custom_etf_nav)
    exposure = calculate_exposure(s, ref.sector)
    exposure = _normalise_breakdown(exposure, _input_total(s))
    return _series_to_sorted_dict(exposure)


def get_country_exposure(
    weights: dict[str, float],
    custom_etf_nav: Optional[dict[str, float]] = None,
) -> dict:
    """Return the country / region breakdown of a portfolio.

    Use when the user asks about geographic exposure ("how much US?",
    "what's my emerging-markets allocation?").

    Args:
        weights: ``{portfolio_name: weight_or_nav}``. May be normalised weights,
            percentages, or raw NAVs — the result is rescaled by their sum.
        custom_etf_nav: optional CUSTOM expansion map.

    Returns:
        ``{country: weight_0_to_1}`` sorted by weight descending, clipped
        non-negative and summing to 1.0 (any unmapped weight is collected under
        ``"Others"``).
    """
    ref = get_reference_data()
    s = _weights_to_wgt_series(weights, custom_etf_nav)
    exposure = calculate_exposure(s, ref.country)
    exposure = _normalise_breakdown(exposure, _input_total(s))
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

    NAV-friendly: ``weights`` may be normalised weights, percentages, or raw
    NAVs — holdings are rescaled by the input total and clipped non-negative.
    Unlike the slice functions this is a top-N slice, so the returned weights
    sum to whatever that slice captures (see ``total_weight_in_top_n``), not
    forced to 1.0.

    Args:
        weights: ``{portfolio_name: weight_or_nav}``.
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
    conc = _scale_and_clip(conc, _input_total(s)).sort_values(ascending=False)
    holdings = _series_to_sorted_dict(conc.head(top_n))
    return {
        "holdings": holdings,
        "top_n": top_n,
        "total_weight_in_top_n": float(sum(holdings.values())),
    }
