"""Recommender — deterministic risk scoring + product eligibility.

Thin wrappers around ``utils.risk_engine``. No LLM involved — these are
the same rule-based functions the Advisory Tool uses for the underlying
recommendation (the LLM in ``utils.llm_helper`` then writes the prose
around them).
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.risk_engine import (  # noqa: E402
    calculate_risk_score,
    get_eligible_portfolios,
    get_recommendation_text,
    get_smart_recommendation,
)


def score_client_risk(profile: dict) -> dict:
    """Compute a client's risk score (1-5) from their questionnaire profile.

    Use when the user is onboarding a new client or sanity-checking
    the risk score on an existing one. Deterministic — same inputs
    always give the same score. No LLM.

    Args:
        profile: dict of client attributes. Recognised keys:
            - ``risk_willingness`` (1-4, lower = more risk averse)
            - ``time_horizon_years`` (int)
            - ``annual_income`` (number)
            - ``liquid_assets`` (number)
            - ``savings_rate`` (decimal, e.g. 0.20 for 20%)
            - ``short_term_need`` (bool)            — hard brake to score 1
            - ``concentration_high`` (bool)         — caps at 3
            - ``investment_knowledge`` (str: ``"None"``, ``"Minimal"``,
              ``"Average"``, ``"Strong"``)         — None/Minimal caps at 2

    Returns:
        ``{
            "risk_score": int,           # 1 (most conservative) to 5 (most aggressive)
            "overrides": [str, ...],     # any hard-brake reasons that applied
            "profile":   dict,           # echo of the input profile
        }``
    """
    score, overrides = calculate_risk_score(profile or {})
    return {
        "risk_score": int(score),
        "overrides": list(overrides),
        "profile": dict(profile or {}),
    }


def get_eligible_portfolios_for_risk(
    risk_score: int,
    objective_category: str,
    is_accredited: bool = False,
) -> dict:
    """Return the catalogue of portfolios the client is eligible for.

    Use when the user asks "what can I invest in?" or wants to know
    every portfolio that fits their risk score + objective.

    Args:
        risk_score: 1-5.
        objective_category: ``"Wealth Growth"``, ``"Income Generation"``,
            or ``"Capital Preservation"``.
        is_accredited: accredited-investor flag (currently no-op in
            the rule engine; kept for forward compatibility).

    Returns:
        ``{
            "risk_score":         int,
            "objective":          str,
            "eligible_portfolios": [
                {"name": str, "min_score": int, "category": str},
                ...
            ],
            "n":                  int,
        }``
    """
    portfolios = get_eligible_portfolios(
        int(risk_score), str(objective_category), bool(is_accredited)
    )
    return {
        "risk_score": int(risk_score),
        "objective": str(objective_category),
        "eligible_portfolios": list(portfolios),
        "n": len(portfolios),
    }


def recommend_portfolio(
    risk_score: int,
    objective: str,
    time_horizon: int,
    interest_in_ai: bool = False,
) -> dict:
    """Recommend a specific portfolio allocation for a client.

    Use when the user asks "what should this client invest in?" or
    "what's the recommended portfolio?". Returns both a deterministic
    allocation (``{portfolio: weight}``) and a one-line markdown summary.

    Args:
        risk_score: 1-5.
        objective: ``"Wealth Growth"`` | ``"Income Generation"`` | ``"Capital Preservation"``.
        time_horizon: years (used by the recommendation-text helper).
        interest_in_ai: if True, satellite 30% into Disruptive Technology
            for high-risk growth clients.

    Returns:
        ``{
            "allocation":          {portfolio_name: weight},  # sums to ~1
            "summary_markdown":    str,
            "risk_score":          int,
            "objective":           str,
            "time_horizon_years":  int,
        }``
    """
    allocation = get_smart_recommendation(
        int(risk_score),
        str(objective),
        int(time_horizon),
        bool(interest_in_ai),
    )
    summary = get_recommendation_text(int(risk_score), str(objective), int(time_horizon))
    return {
        "allocation": {str(k): float(v) for k, v in (allocation or {}).items()},
        "summary_markdown": str(summary),
        "risk_score": int(risk_score),
        "objective": str(objective),
        "time_horizon_years": int(time_horizon),
    }
