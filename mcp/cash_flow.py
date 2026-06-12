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
    get_combined_yield,
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


def get_weighted_yield(
    portfolio_navs: dict[str, float],
    base_currencies: Optional[dict[str, str]] = None,
    yield_preference: str = "T12M",
    display_currency: str = "SGD",
    usdsgd: Optional[float] = None,
    usdhkd: Optional[float] = None,
) -> dict:
    """NAV-weighted trailing-12-month dividend yield across the WHOLE book.

    Unlike :func:`project_monthly_cash_flow` (which models *income* sleeves and
    their payout schedule, filtering out growth / cash / thematic), this is the
    pure portfolio dividend yield: every holding's yield weighted by its NAV, so
    a growth-heavy book correctly shows a low blended yield instead of excluding
    those sleeves entirely::

        weighted_yield = Σ(nav_i × yield_i) / Σ(nav_i)

    Per-holding yield is the same PV ``get_combined_yield`` lookup the app uses,
    with ``yield_preference`` ("T12M" = trailing 12-month dividend yield, the
    default here; it falls back to YTM only when a holding has no T12M figure).
    NAVs are FX-converted to ``display_currency`` first (base currency inferred
    from the portfolio name), so a multi-currency book is weighted correctly.

    Args:
        portfolio_navs: ``{portfolio_name: nav_in_base_ccy}``.
        base_currencies: optional ``{portfolio_name: "USD"|"SGD"|"HKD"}`` — the
            actual base currency per sleeve (pass it straight from
            ``get_portfolios``). When given it is authoritative; otherwise the
            currency is inferred from the portfolio name, which is unreliable for
            USD equity sleeves (e.g. "Equity Alpha"). Supplying it makes the
            multi-currency weighting exact.
        yield_preference: ``"T12M"`` (default) or ``"YTM"``.
        display_currency: ``"SGD"``, ``"USD"`` or ``"HKD"``.
        usdsgd: override FX rate. Defaults to latest from reference data.
        usdhkd: override FX rate. Defaults to latest from reference data.

    Returns:
        ``{
            "weighted_yield":     float,  # decimal, e.g. 0.0412 = 4.12%
            "weighted_yield_pct": float,  # convenience = weighted_yield * 100
            "annual_income":      float,  # display ccy
            "total_nav":          float,  # display ccy (sum of positive NAVs)
            "coverage_pct":       float,  # share of NAV carrying a non-zero yield
            "no_yield_data":      [str],  # holdings with no yield figure — they
                                          # drag the average down; a yield-bearing
                                          # product here is a reference-data gap
            "currency":           str,
            "yield_preference":   str,
            "breakdown": [
                {"portfolio", "nav", "base_currency", "yield", "yield_pct",
                 "yield_source", "annual_income"}, ...   # sorted by income desc
            ],
        }``
    """
    ref = get_reference_data()
    ref_usdsgd, ref_usdhkd = get_reference_fx(ref.fx)
    fx_usdsgd = ref_usdsgd if usdsgd is None else float(usdsgd)
    fx_usdhkd = ref_usdhkd if usdhkd is None else float(usdhkd)

    rows = []
    total_nav = 0.0
    weighted_income = 0.0
    covered_nav = 0.0
    for name, nav in (portfolio_navs or {}).items():
        try:
            nav_f = float(nav)
        except (ValueError, TypeError):
            continue
        if nav_f <= 0:
            continue
        base = (base_currencies or {}).get(name) or get_ticker_base_currency(name)
        nav_disp = convert_currency(nav_f, base, display_currency, fx_usdsgd, fx_usdhkd)
        y, src = get_combined_yield(name, ref.fi_metrics, yield_preference)
        total_nav += nav_disp
        weighted_income += nav_disp * y
        if y > 0:
            covered_nav += nav_disp
        rows.append({
            "portfolio": str(name),
            "nav": nav_disp,
            "base_currency": base,
            "yield": y,
            "yield_pct": y * 100.0,
            "yield_source": src,
            "annual_income": nav_disp * y,
        })

    weighted_yield = (weighted_income / total_nav) if total_nav > 0 else 0.0
    rows.sort(key=lambda r: r["annual_income"], reverse=True)
    no_yield_data = [r["portfolio"] for r in rows if r["yield"] <= 0]
    return {
        "weighted_yield": weighted_yield,
        "weighted_yield_pct": weighted_yield * 100.0,
        "annual_income": weighted_income,
        "total_nav": total_nav,
        "coverage_pct": (covered_nav / total_nav) if total_nav > 0 else 0.0,
        "no_yield_data": no_yield_data,
        "currency": display_currency,
        "yield_preference": yield_preference,
        "breakdown": rows,
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
