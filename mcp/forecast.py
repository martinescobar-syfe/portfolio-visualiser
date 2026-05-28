"""Forecast — Monte-Carlo style growth projection + yield-based income projection.

Extracted from the "Growth Forecast" and "Income Forecast" blocks of
``pages/Client_PV.py``. The math is identical to what the Streamlit
page displays — same percentile bands, same binary-search for the
suggested top-up, same blended capital-market assumptions.

Convention: blend per-portfolio CMAs at the *parameter* level first
(``μ_C = Σ wᵢ μᵢ``), then run ONE projection. **Not** project each
sleeve separately and then blend the forecasts — those are not equal.
Volatility blend assumes zero pairwise correlation across sleeves
(``σ_C = √(Σ (wᵢ σᵢ)²)``), which underestimates true σ for correlated
sleeves but is captured by the conservative-band fallback.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.stats import norm as scipy_norm


# ── Capital-market assumptions, mirror PORTFOLIO_ESTIMATES in Client_PV.py ──
# If you change these here, change them in pages/Client_PV.py too (and vice
# versa). They are deliberately kept in both places so the MCP layer has no
# runtime dependency on a Streamlit page.
PORTFOLIO_ESTIMATES: dict[str, dict[str, float]] = {
    'Core Equity100':                {'ret': 0.10,  'vol': 0.15,  'yield': 0.015},
    'SRS Equity100':                 {'ret': 0.10,  'vol': 0.15,  'yield': 0.015},
    'Core Growth':                   {'ret': 0.08,  'vol': 0.12,  'yield': 0.020},
    'Core Balanced':                 {'ret': 0.06,  'vol': 0.09,  'yield': 0.025},
    'Core Defensive':                {'ret': 0.04,  'vol': 0.06,  'yield': 0.030},
    'Downside protected':            {'ret': 0.05,  'vol': 0.08,  'yield': 0.025},
    'Disruptive Technology':         {'ret': 0.12,  'vol': 0.22,  'yield': 0.005},
    'ESG & Clean Energy':            {'ret': 0.09,  'vol': 0.18,  'yield': 0.010},
    'Healthcare Innovation':         {'ret': 0.10,  'vol': 0.17,  'yield': 0.010},
    'China Growth':                  {'ret': 0.08,  'vol': 0.25,  'yield': 0.015},
    'Income Enhance':                {'ret': 0.05,  'vol': 0.08,  'yield': 0.045},
    'Income Preserve':               {'ret': 0.04,  'vol': 0.05,  'yield': 0.035},
    'SRS Income Enhance':            {'ret': 0.05,  'vol': 0.08,  'yield': 0.045},
    'SRS Income Preserve':           {'ret': 0.04,  'vol': 0.05,  'yield': 0.035},
    'REIT 100':                      {'ret': 0.06,  'vol': 0.16,  'yield': 0.040},
    'REIT Risk Managed':             {'ret': 0.05,  'vol': 0.12,  'yield': 0.035},
    'Blackrock Private Credit Fund': {'ret': 0.07,  'vol': 0.05,  'yield': 0.070},
    'Monroe Income Plus Fund':       {'ret': 0.08,  'vol': 0.05,  'yield': 0.080},
    'Cash SGD flexi':                {'ret': 0.035, 'vol': 0.005, 'yield': 0.035},
    'Cash USD flexi':                {'ret': 0.040, 'vol': 0.005, 'yield': 0.040},
    'Cash SGD guaranteed':           {'ret': 0.030, 'vol': 0.001, 'yield': 0.030},
    'SRS Cash SGD flexi':            {'ret': 0.035, 'vol': 0.005, 'yield': 0.035},
}
_DEFAULT_EST = {'ret': 0.06, 'vol': 0.12, 'yield': 0.03}


def _get_est(name: str) -> dict[str, float]:
    """Look up CMA for a portfolio name with Silverdale + balanced fallback."""
    if name in PORTFOLIO_ESTIMATES:
        return PORTFOLIO_ESTIMATES[name]
    if 'silverdale' in name.lower():
        return {'ret': 0.065, 'vol': 0.04, 'yield': 0.065}
    return _DEFAULT_EST


def _weighted_params(mix: dict[str, float]) -> tuple[float, float, float]:
    """Blend per-portfolio CMAs by weight. Returns (ret, vol, yield).

    Vol uses the zero-correlation aggregation ``√(Σ (wᵢ σᵢ)²)``.
    """
    total = sum(max(0.0, float(v)) for v in mix.values())
    if total <= 0:
        return _DEFAULT_EST['ret'], _DEFAULT_EST['vol'], _DEFAULT_EST['yield']
    w_ret, w_vol2, w_yld = 0.0, 0.0, 0.0
    for p, raw in mix.items():
        w = max(0.0, float(raw)) / total
        e = _get_est(p)
        w_ret += w * e['ret']
        w_vol2 += (w * e['vol']) ** 2
        w_yld += w * e['yield']
    return w_ret, math.sqrt(w_vol2), w_yld


def _build_projection(index_path: np.ndarray, initial: float, monthly_c: float) -> np.ndarray:
    """Apply a log-normal index growth path + monthly contributions to NAV."""
    vals = [initial]
    for i in range(1, len(index_path)):
        growth = index_path[i] / index_path[i - 1]
        vals.append(vals[-1] * growth + monthly_c)
    return np.asarray(vals)


def project_portfolio_growth(
    initial: float,
    monthly_topup: float,
    years: int,
    mix: dict[str, float],
    target_value: Optional[float] = None,
) -> dict:
    """Project a portfolio's future value over a horizon under log-normal returns.

    Use when the user asks "how much will I have in N years?",
    "will I hit my retirement target?", or wants a growth-scenario
    forecast for a mix of Syfe portfolios.

    Produces three NAV trajectories — the 5th, 50th and 95th percentile
    bands of a geometric-Brownian-motion process parameterised by the
    *blended* expected return and volatility of ``mix``. Monthly top-ups
    are added on top of the geometric growth at each step.

    Args:
        initial: starting NAV (in any single currency).
        monthly_topup: recurring monthly contribution (same currency).
        years: investment horizon in years (≥ 1).
        mix: ``{portfolio_name: weight}`` summing to ~1. Names should
            match keys in :data:`PORTFOLIO_ESTIMATES`; unknown names
            fall back to a balanced-profile CMA.
        target_value: optional target value used to compute the
            achievability score (0-10).

    Returns:
        ``{
            "weighted_return":            float,
            "weighted_volatility":        float,
            "weighted_yield":             float,
            "years":                      int,
            "months":                     int,
            "initial":                    float,
            "monthly_topup":              float,
            "final_value": {
                "conservative_p5":  float,
                "median_p50":       float,
                "optimistic_p95":   float,
            },
            "trajectories": {                       # one entry per year (0..N)
                "year":             [0, 1, ..., N],
                "conservative_p5":  [float, ...],
                "median_p50":       [float, ...],
                "optimistic_p95":   [float, ...],
            },
            "target_value":   float | None,
            "achievability":  {                     # only if target_value given
                "median_vs_target":  float,         # ratio: median / target
                "score_0_to_10":     float,
                "rating":            str,
                "suggested_monthly_topup": float,
            } | None,
        }``
    """
    ann_ret, ann_vol, ann_yld = _weighted_params(mix)
    log_ret = np.log(1 + ann_ret)
    months_total = max(1, int(years)) * 12
    months_arr = np.arange(0, months_total + 1)

    # Log-normal percentile bands of an index starting at 100.
    conservative = np.ones(len(months_arr)) * 100.0
    likely = np.ones(len(months_arr)) * 100.0
    optimistic = np.ones(len(months_arr)) * 100.0
    m_vec = months_arr[1:]
    drift = (m_vec / 12) * log_ret
    vol_arr = np.sqrt(m_vec / 12) * ann_vol
    conservative[1:] = np.exp(drift + vol_arr * scipy_norm.ppf(0.05)) * 100
    likely[1:] = np.exp(drift + vol_arr * scipy_norm.ppf(0.50)) * 100
    optimistic[1:] = np.exp(drift + vol_arr * scipy_norm.ppf(0.95)) * 100

    proj_p5 = _build_projection(conservative, float(initial), float(monthly_topup))
    proj_p50 = _build_projection(likely, float(initial), float(monthly_topup))
    proj_p95 = _build_projection(optimistic, float(initial), float(monthly_topup))

    # Down-sample to yearly points so the LLM response stays compact (1 + N entries).
    yearly_idx = np.arange(0, months_total + 1, 12).astype(int)
    years_arr = (yearly_idx / 12.0).astype(int).tolist()

    fin_med = float(proj_p50[-1])
    out: dict = {
        "weighted_return": ann_ret,
        "weighted_volatility": ann_vol,
        "weighted_yield": ann_yld,
        "years": int(years),
        "months": int(months_total),
        "initial": float(initial),
        "monthly_topup": float(monthly_topup),
        "final_value": {
            "conservative_p5": float(proj_p5[-1]),
            "median_p50": fin_med,
            "optimistic_p95": float(proj_p95[-1]),
        },
        "trajectories": {
            "year": years_arr,
            "conservative_p5": [float(proj_p5[i]) for i in yearly_idx],
            "median_p50": [float(proj_p50[i]) for i in yearly_idx],
            "optimistic_p95": [float(proj_p95[i]) for i in yearly_idx],
        },
        "target_value": float(target_value) if target_value is not None else None,
        "achievability": None,
    }

    if target_value and float(target_value) > 0:
        ratio = fin_med / float(target_value)
        if ratio >= 1.0:
            score = min(10.0, 7.0 + ((ratio - 1.0) * 100 / 10.0) * 0.3)
        else:
            score = max(3.0, 7.0 - ((1.0 - ratio) / 0.4) * 4.0)

        # Suggested monthly top-up to close the gap (mirrors Client_PV logic).
        if fin_med < float(target_value) and ann_ret > 0:
            fv_initial = float(initial) * (likely[-1] / 100.0)
            gap = float(target_value) - fv_initial
            ann_factor = (((1 + ann_ret) ** int(years) - 1) / ann_ret) * (1 + ann_ret)
            suggested = max(0.0, (gap / ann_factor) / 12) if ann_factor > 0 else 0.0
        else:
            suggested = 0.0

        out["achievability"] = {
            "median_vs_target": float(ratio),
            "score_0_to_10": float(round(score, 1)),
            "rating": _rating(score),
            "suggested_monthly_topup": float(round(suggested, 2)),
        }

    return out


def project_income_payout(
    initial: float,
    monthly_topup: float,
    years: int,
    mix: dict[str, float],
    target_monthly_payout: Optional[float] = None,
) -> dict:
    """Project monthly income payout over a horizon.

    Use when the user asks "how much income will I get per month?",
    "what monthly payout can I expect at retirement?", or wants an
    income-scenario forecast.

    The model splits total return into capital growth (compounds NAV)
    and yield (pays out as income). NAV accretes at the capital-growth
    rate plus monthly top-ups; forecast income at step t = NAV(t) ×
    monthly yield. If ``target_monthly_payout`` is given, also binary-
    searches the monthly top-up that would land the final monthly
    income on that target.

    Args:
        initial: starting NAV.
        monthly_topup: recurring monthly contribution.
        years: investment horizon in years.
        mix: ``{portfolio_name: weight}``.
        target_monthly_payout: optional target monthly income at the
            end of the horizon.

    Returns:
        ``{
            "weighted_return":      float,
            "weighted_yield":       float,
            "years":                int,
            "initial":              float,
            "monthly_topup":        float,
            "final_monthly_income": float,  # at end of horizon
            "trajectory": {                  # yearly points
                "year":            [0, 1, ..., N],
                "monthly_income":  [float, ...],
                "nav":             [float, ...],
            },
            "target_monthly_payout":  float | None,
            "achievability": {                # only if target given
                "score_0_to_10":           float,
                "rating":                  str,
                "suggested_monthly_topup": float,
            } | None,
        }``
    """
    ann_ret, _ann_vol, ann_yld = _weighted_params(mix)
    months_total = max(1, int(years)) * 12

    ann_cap_growth = max(0.0, ann_ret - ann_yld)
    monthly_cap = (1 + ann_cap_growth) ** (1 / 12) - 1
    monthly_yld = ann_yld / 12

    nav = np.zeros(months_total + 1)
    nav[0] = float(initial)
    for m in range(1, months_total + 1):
        nav[m] = nav[m - 1] * (1 + monthly_cap) + float(monthly_topup)
    income = nav * monthly_yld

    yearly_idx = np.arange(0, months_total + 1, 12).astype(int)
    years_arr = (yearly_idx / 12.0).astype(int).tolist()

    out: dict = {
        "weighted_return": ann_ret,
        "weighted_yield": ann_yld,
        "years": int(years),
        "initial": float(initial),
        "monthly_topup": float(monthly_topup),
        "final_monthly_income": float(income[-1]),
        "trajectory": {
            "year": years_arr,
            "monthly_income": [float(income[i]) for i in yearly_idx],
            "nav": [float(nav[i]) for i in yearly_idx],
        },
        "target_monthly_payout": (
            float(target_monthly_payout) if target_monthly_payout is not None else None
        ),
        "achievability": None,
    }

    if target_monthly_payout and float(target_monthly_payout) > 0:
        suggested = suggested_topup_for_target(
            initial=float(initial),
            current_monthly_topup=float(monthly_topup),
            years=int(years),
            mix=mix,
            target_monthly_payout=float(target_monthly_payout),
        )["suggested_monthly_topup"]

        ratio = float(income[-1]) / float(target_monthly_payout)
        if ratio >= 1.0:
            score = min(10.0, 7.0 + ((ratio - 1.0) * 100 / 10.0) * 0.3)
        else:
            score = max(3.0, 7.0 - ((1.0 - ratio) / 0.4) * 4.0)

        out["achievability"] = {
            "score_0_to_10": float(round(score, 1)),
            "rating": _rating(score),
            "suggested_monthly_topup": float(round(suggested, 2)),
        }

    return out


def suggested_topup_for_target(
    initial: float,
    current_monthly_topup: float,
    years: int,
    mix: dict[str, float],
    target_monthly_payout: float,
) -> dict:
    """Binary-search the monthly top-up needed to hit a target monthly payout.

    Use when the user asks "what would I need to save per month to get
    $X/month at retirement?". Same algorithm as the Income forecast in
    ``pages/Client_PV.py`` — 50 iterations across the bracket
    ``[current_topup, max(current_topup*5, target*20)]``.

    Args:
        initial: starting NAV.
        current_monthly_topup: current monthly contribution (lower bracket).
        years: horizon in years.
        mix: ``{portfolio_name: weight}``.
        target_monthly_payout: desired monthly income at end of horizon.

    Returns:
        ``{
            "suggested_monthly_topup":     float,
            "target_monthly_payout":       float,
            "horizon_years":               int,
            "weighted_return":             float,
            "weighted_yield":              float,
        }``
    """
    ann_ret, _ann_vol, ann_yld = _weighted_params(mix)
    months_total = max(1, int(years)) * 12

    ann_cap_growth = max(0.0, ann_ret - ann_yld)
    monthly_cap = (1 + ann_cap_growth) ** (1 / 12) - 1
    monthly_yld = ann_yld / 12

    lo = float(current_monthly_topup)
    hi = max(float(current_monthly_topup) * 5, float(target_monthly_payout) * 20)
    for _ in range(50):
        mid = (lo + hi) / 2
        nav_t = float(initial)
        for _m in range(1, months_total + 1):
            nav_t = nav_t * (1 + monthly_cap) + mid
        if nav_t * monthly_yld >= float(target_monthly_payout):
            hi = mid
        else:
            lo = mid

    return {
        "suggested_monthly_topup": float(round(hi, 2)),
        "target_monthly_payout": float(target_monthly_payout),
        "horizon_years": int(years),
        "weighted_return": ann_ret,
        "weighted_yield": ann_yld,
    }


def _rating(score: float) -> str:
    if score >= 8:
        return "Excellent"
    if score >= 6:
        return "Good"
    if score >= 4:
        return "Fair"
    return "Needs Work"
