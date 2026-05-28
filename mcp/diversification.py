"""Diversification score — single MCP tool that wraps the composite scorer.

The score is the same 0-10 number rendered in the donut on Client_PV.
Sub-scores cover asset class (40%), geography (30%), sector (20%),
concentration (10%).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.diversification_score import calculate_diversification_score  # noqa: E402

from mcp._data_loader import get_reference_data


def _round_score(s):
    """Round a sub-score to 1 dp, preserving None."""
    return None if s is None else round(float(s), 1)


def get_diversification_score(
    weights: dict[str, float],
    custom_etf_nav: Optional[dict[str, float]] = None,
) -> dict:
    """Return the 0-10 diversification score for a portfolio mix.

    Use when the user asks "how diversified am I?" or "what's wrong
    with my portfolio?". The composite is a weighted average of four
    sub-scores: asset class (40%), geography (30%), sector (20%) and
    concentration (10%).

    Args:
        weights: ``{portfolio_name: weight}`` summing to ~1.
        custom_etf_nav: optional ``{etf: nav}`` for expanding CUSTOM
            portfolios into constituents before scoring. Without this,
            CUSTOM positions are ignored (they have no exposure data).

    Returns:
        ``{
            "composite_score": float,      # 0-10
            "sub_scores": {
                "asset_class":   float | None,
                "geography":     float | None,
                "sector":        float | None,
                "concentration": float | None,
            },
            "rating": "Excellent" | "Good" | "Fair" | "Needs Work" | "N/A",
            "data_warnings": [str, ...],   # missing data caveats, if any
        }``

        ``None`` for a sub-score means it could not be computed (e.g.
        concentration when the portfolio has no equity sleeves).
    """
    ref = get_reference_data()
    result = calculate_diversification_score(
        client_weights=weights,
        df_asset_class=ref.asset_class,
        df_sector=ref.sector,
        df_country=ref.country,
        df_underlying=ref.underlying,
        custom_breakdown=custom_etf_nav,
    )

    composite = result.get('composite_score')
    sub_keys = ['asset_class', 'geography', 'sector', 'concentration']
    sub_scores = {k: _round_score(result.get(k, {}).get('score')) for k in sub_keys}

    rating = "N/A"
    if composite is not None:
        if composite >= 8:
            rating = "Excellent"
        elif composite >= 6:
            rating = "Good"
        elif composite >= 4:
            rating = "Fair"
        else:
            rating = "Needs Work"

    return {
        "composite_score": _round_score(composite),
        "sub_scores": sub_scores,
        "rating": rating,
        "data_warnings": list(result.get('data_warnings', []) or []),
    }
