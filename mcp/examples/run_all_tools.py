"""End-to-end smoke test for every MCP tool wrapper.

Run from the repo root::

    cd portfolio-visualiser
    python -m mcp.examples.run_all_tools

This script:
  1. Loads the dummy client CSV via :func:`mcp.holdings.parse_holdings_csv`.
  2. Runs every other tool against the resulting holdings.
  3. Pretty-prints the result for each tool.
  4. Exits non-zero if any wrapper throws.

It does NOT touch Metabase, Supabase, Gemini, or yfinance — pure local
reference data + dummy CSV. Use it as the "is this folder broken?"
check before you wire an MCP server.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pprint import pprint

# Make the project root importable when this script is run directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mcp.cash_flow import get_fixed_income_metrics, project_monthly_cash_flow
from mcp.diversification import get_diversification_score
from mcp.exposure import (
    get_asset_class_exposure,
    get_country_exposure,
    get_sector_exposure,
    get_top_holdings,
)
from mcp.forecast import (
    project_income_payout,
    project_portfolio_growth,
    suggested_topup_for_target,
)
from mcp.holdings import parse_holdings_csv
from mcp.performance import get_period_return, get_portfolio_metrics
from mcp.recommender import (
    get_eligible_portfolios_for_risk,
    recommend_portfolio,
    score_client_risk,
)


DUMMY_CSV = os.path.join(_THIS_DIR, "..", "dummy_data", "sample_client_holdings.csv")

# ── reused for the sample profile ──
SAMPLE_PROFILE = {
    "risk_willingness": 2,        # higher = more risk-tolerant per the engine
    "time_horizon_years": 20,
    "annual_income": 120_000,
    "liquid_assets": 80_000,
    "savings_rate": 0.25,
    "investment_knowledge": "Average",
}


def _section(title: str) -> None:
    bar = "─" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def _ok(name: str, value):
    """Pretty-print a successful tool call. Truncate huge structures."""
    serialised = json.dumps(value, indent=2, default=str)
    if len(serialised) > 1800:
        serialised = serialised[:1800] + "\n  …(truncated)"
    print(f"\n[OK] {name}\n{serialised}")


def _fail(name: str, exc: BaseException) -> None:
    print(f"\n[FAIL] {name}: {type(exc).__name__}: {exc}")
    traceback.print_exc()


def main() -> int:
    failures = 0

    # ── 1. Holdings ingest ────────────────────────────────────────────
    _section("1. Holdings (CSV → canonical holdings)")
    try:
        holdings = parse_holdings_csv(DUMMY_CSV)
        weights = holdings["weights"]
        nav_total = holdings["total_nav"]
        custom_etf_nav = holdings["custom_etf_nav"]
        _ok("parse_holdings_csv", {
            "total_nav": nav_total,
            "n_portfolios": len(weights),
            "weights": weights,
            "custom_etf_nav": custom_etf_nav,
        })
    except Exception as e:  # noqa: BLE001
        _fail("parse_holdings_csv", e)
        print("\nCannot continue without holdings. Aborting.")
        return 1

    # ── 2. Exposure tools ────────────────────────────────────────────
    _section("2. Exposure (asset class / sector / country / top holdings)")
    for fn, label in [
        (get_asset_class_exposure, "get_asset_class_exposure"),
        (get_sector_exposure, "get_sector_exposure"),
        (get_country_exposure, "get_country_exposure"),
    ]:
        try:
            _ok(label, fn(weights, custom_etf_nav))
        except Exception as e:  # noqa: BLE001
            _fail(label, e); failures += 1
    try:
        _ok("get_top_holdings (top_n=8)", get_top_holdings(weights, top_n=8, custom_etf_nav=custom_etf_nav))
    except Exception as e:  # noqa: BLE001
        _fail("get_top_holdings", e); failures += 1

    # ── 3. Diversification score ─────────────────────────────────────
    _section("3. Diversification score")
    try:
        _ok("get_diversification_score", get_diversification_score(weights, custom_etf_nav))
    except Exception as e:  # noqa: BLE001
        _fail("get_diversification_score", e); failures += 1

    # ── 4. Performance ───────────────────────────────────────────────
    _section("4. Performance (backtested daily-returns series)")
    try:
        _ok("get_portfolio_metrics", get_portfolio_metrics(weights, risk_free_rate=0.0))
    except Exception as e:  # noqa: BLE001
        _fail("get_portfolio_metrics", e); failures += 1
    for period in ("YTD", "1Y", "3Y", "ITD"):
        try:
            _ok(f"get_period_return ({period})", get_period_return(weights, period=period))
        except Exception as e:  # noqa: BLE001
            _fail(f"get_period_return {period}", e); failures += 1

    # ── 5. Cash flow ─────────────────────────────────────────────────
    _section("5. Cash flow (12-month income projection)")
    try:
        navs = holdings["holdings"]
        _ok("project_monthly_cash_flow", project_monthly_cash_flow(
            portfolio_navs=navs,
            yield_preference="YTM",
            display_currency="SGD",
        ))
        _ok("get_fixed_income_metrics", get_fixed_income_metrics(
            list(navs.keys()), yield_preference="YTM",
        ))
    except Exception as e:  # noqa: BLE001
        _fail("cash_flow tools", e); failures += 1

    # ── 6. Forecast ──────────────────────────────────────────────────
    _section("6. Forecast (Monte-Carlo growth + income projection)")
    mix = {"Core Equity100": 0.5, "Core Growth": 0.3, "Income Enhance": 0.2}
    try:
        _ok("project_portfolio_growth", project_portfolio_growth(
            initial=100_000, monthly_topup=1_000, years=20, mix=mix, target_value=500_000,
        ))
    except Exception as e:  # noqa: BLE001
        _fail("project_portfolio_growth", e); failures += 1
    try:
        _ok("project_income_payout", project_income_payout(
            initial=100_000, monthly_topup=500, years=20, mix=mix, target_monthly_payout=2_500,
        ))
    except Exception as e:  # noqa: BLE001
        _fail("project_income_payout", e); failures += 1
    try:
        _ok("suggested_topup_for_target", suggested_topup_for_target(
            initial=100_000, current_monthly_topup=500, years=20, mix=mix, target_monthly_payout=2_500,
        ))
    except Exception as e:  # noqa: BLE001
        _fail("suggested_topup_for_target", e); failures += 1

    # ── 7. Recommender ───────────────────────────────────────────────
    _section("7. Recommender (deterministic, no LLM)")
    try:
        score = score_client_risk(SAMPLE_PROFILE)
        _ok("score_client_risk", score)
        rs = score["risk_score"]
        _ok("get_eligible_portfolios_for_risk", get_eligible_portfolios_for_risk(rs, "Wealth Growth"))
        _ok("recommend_portfolio (Wealth Growth)", recommend_portfolio(
            risk_score=rs, objective="Wealth Growth", time_horizon=20, interest_in_ai=True,
        ))
    except Exception as e:  # noqa: BLE001
        _fail("recommender tools", e); failures += 1

    # ── Summary ──────────────────────────────────────────────────────
    _section("Summary")
    if failures == 0:
        print("All tools ran successfully ✓")
        return 0
    print(f"{failures} tool(s) failed — see [FAIL] entries above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
