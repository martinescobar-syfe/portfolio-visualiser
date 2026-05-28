"""Cash flow — projected monthly income from FI / yield-bearing portfolios.

Wraps ``utils.cash_flow.compute_portfolio_cash_flows`` and
``utils.cash_flow.build_fi_metrics_table``. Inputs are ``{portfolio: nav}``;
outputs are dicts of 12-month projections + a per-portfolio breakdown.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.cash_flow import (  # noqa: E402
    build_fi_metrics_table,
    compute_portfolio_cash_flows,
    convert_currency,
    get_reference_fx,
    get_ticker_base_currency,
)

from mcp._data_loader import get_reference_data


def project_monthly_cash_flow(
    portfolio_navs: dict[str, float],
    yield_preference: str = "YTM",
    display_currency: str = "SGD",
    usdsgd: Optional[float] = None,
    usdhkd: Optional[float] = None,
) -> dict:
    """Project the next 12 months of cash flow from income-bearing portfolios.

    Use when the user asks "how much income will I get?", "what's my
    expected monthly payout?", or for any retirement-style income view.

    Non-income portfolios (cash, growth, thematic, CUSTOM) are filtered
    out automatically. REIT-style portfolios pay quarterly (Mar / Jun /
    Sep / Dec); everything else is amortised evenly across 12 months.

    Args:
        portfolio_navs: ``{portfolio_name: nav_in_base_ccy}``. Currency
            of each NAV is inferred from the ticker name.
        yield_preference: ``"YTM"`` (yield-to-maturity, primary) or
            ``"T12M"`` (trailing 12-month dividend yield, fallback).
        display_currency: ``"SGD"``, ``"USD"`` or ``"HKD"``.
        usdsgd: override FX rate. Defaults to latest from reference data.
        usdhkd: override FX rate. Defaults to latest from reference data.

    Returns:
        ``{
            "annual_total":     float,
            "monthly_average":  float,
            "effective_yield":  float,  # annual_total / total_nav (display ccy)
            "currency":         str,
            "monthly":          {"Jan": float, ..., "Dec": float},  # totals
            "breakdown": [
                {
                    "portfolio":        str,
                    "nav":              float,
                    "base_currency":    str,
                    "yield_source":     str,   # "YTM" or "T12M"
                    "yield":            float, # decimal, e.g. 0.045 = 4.5%
                    "frequency":        str,   # "monthly" or "quarterly"
                    "annual_cash_flow": float, # in display currency
                },
                ...
            ],
        }``

        Empty monthly + breakdown if no income portfolios are present.
    """
    ref = get_reference_data()
    ref_usdsgd, ref_usdhkd = get_reference_fx(ref.fx)
    fx_usdsgd = ref_usdsgd if usdsgd is None else float(usdsgd)
    fx_usdhkd = ref_usdhkd if usdhkd is None else float(usdhkd)

    monthly_df, breakdown_df = compute_portfolio_cash_flows(
        portfolio_navs,
        ref.fi_metrics,
        yield_preference=yield_preference,
        display_currency=display_currency,
        usdsgd=fx_usdsgd,
        usdhkd=fx_usdhkd,
    )

    if monthly_df.empty:
        return {
            "annual_total": 0.0,
            "monthly_average": 0.0,
            "effective_yield": 0.0,
            "currency": display_currency,
            "monthly": {m: 0.0 for m in [
                'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
            ]},
            "breakdown": [],
        }

    annual_total = float(monthly_df.sum().sum())
    monthly_average = annual_total / 12.0

    # Effective yield = annual income (display ccy) / total NAV in display ccy.
    total_nav_disp = 0.0
    for ticker, nav in (portfolio_navs or {}).items():
        try:
            total_nav_disp += convert_currency(
                float(nav),
                get_ticker_base_currency(ticker),
                display_currency,
                fx_usdsgd,
                fx_usdhkd,
            )
        except (ValueError, TypeError):
            continue
    eff_yld = annual_total / total_nav_disp if total_nav_disp > 0 else 0.0

    monthly_totals = monthly_df.sum(axis=0).to_dict()
    breakdown = [
        {
            "portfolio": str(r["Portfolio"]),
            "nav": float(r["NAV"]),
            "base_currency": str(r["Base Currency"]),
            "yield_source": str(r["Yield Source"]),
            "yield": float(r["Yield"]),
            "frequency": str(r["Frequency"]),
            "annual_cash_flow": float(r["Annual Cash Flow"]),
        }
        for r in breakdown_df.to_dict('records')
    ]

    return {
        "annual_total": annual_total,
        "monthly_average": monthly_average,
        "effective_yield": eff_yld,
        "currency": display_currency,
        "monthly": {k: float(v) for k, v in monthly_totals.items()},
        "breakdown": breakdown,
    }


def get_fixed_income_metrics(
    portfolio_names: list[str],
    yield_preference: str = "YTM",
) -> dict:
    """Return duration / yield / credit metrics for FI-eligible portfolios.

    Use when the user asks "what's the duration of my bond sleeve?" or
    "what's the average credit quality?".

    Args:
        portfolio_names: list of portfolio names (e.g. the keys from
            ``aggregate_client_holdings()['holdings']``).
        yield_preference: ``"YTM"`` (primary) or ``"T12M"`` (fallback).

    Returns:
        ``{
            "portfolios": [
                {
                    "portfolio":      str,
                    "duration_years": float | None,
                    "yield":          float | None,  # decimal
                    "credit_rating":  str | None,
                },
                ...
            ],
            "n":  int,
        }``

        Empty list when none of the portfolios are FI-eligible or have
        yield data.
    """
    ref = get_reference_data()
    df = build_fi_metrics_table(portfolio_names, ref.fi_metrics, yield_preference=yield_preference)

    if df.empty:
        return {"portfolios": [], "n": 0}

    portfolios = []
    for row in df.to_dict('records'):
        portfolios.append({
            "portfolio": str(row.get('Portfolio', row.get('portfolio', ''))),
            "duration_years": _opt_float(row.get('Duration (yrs)')),
            "yield": _opt_float(row.get('Yield')),
            "credit_rating": (
                str(row.get('Credit Rating')) if row.get('Credit Rating') is not None else None
            ),
        })
    return {"portfolios": portfolios, "n": len(portfolios)}


def _opt_float(v):
    try:
        if v is None:
            return None
        f = float(v)
        return None if f != f else f  # NaN check
    except (ValueError, TypeError):
        return None
