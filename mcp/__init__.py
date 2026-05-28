"""mcp/ — calculator functions ready to be wrapped as MCP tools.

Each submodule exposes plain-Python functions that take primitive args
and return JSON-serialisable dicts. Underlying logic is delegated to
`utils/` and `pages/` — no math is re-implemented here.

See README.md for the philosophy and how to add a new tool.

Top-level re-exports for convenience (so an MCP server file can do
``from mcp import project_portfolio_growth`` instead of digging through
submodules).
"""

# Holdings ingestion
from mcp.holdings import (
    parse_holdings_csv,
    aggregate_client_holdings,
    get_client_holdings_metabase,
)

# Exposure / slicing
from mcp.exposure import (
    get_asset_class_exposure,
    get_sector_exposure,
    get_country_exposure,
    get_top_holdings,
)

# Diversification scoring
from mcp.diversification import get_diversification_score

# Performance (backtested)
from mcp.performance import (
    get_portfolio_metrics,
    get_period_return,
    get_relative_metrics,
)

# Cash flow / FI analytics
from mcp.cash_flow import (
    project_monthly_cash_flow,
    get_fixed_income_metrics,
)

# Forecasts (Monte-Carlo style projections)
from mcp.forecast import (
    project_portfolio_growth,
    project_income_payout,
    suggested_topup_for_target,
)

# Recommender (deterministic, no LLM)
from mcp.recommender import (
    score_client_risk,
    recommend_portfolio,
    get_eligible_portfolios_for_risk,
)

__all__ = [
    # holdings
    "parse_holdings_csv",
    "aggregate_client_holdings",
    "get_client_holdings_metabase",
    # exposure
    "get_asset_class_exposure",
    "get_sector_exposure",
    "get_country_exposure",
    "get_top_holdings",
    # diversification
    "get_diversification_score",
    # performance
    "get_portfolio_metrics",
    "get_period_return",
    "get_relative_metrics",
    # cash flow
    "project_monthly_cash_flow",
    "get_fixed_income_metrics",
    # forecast
    "project_portfolio_growth",
    "project_income_payout",
    "suggested_topup_for_target",
    # recommender
    "score_client_risk",
    "recommend_portfolio",
    "get_eligible_portfolios_for_risk",
]
