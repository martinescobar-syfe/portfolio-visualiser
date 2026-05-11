"""Performance metric helpers for the portfolio analytics app.

All metrics are computed from a daily decimal return Series / DataFrame.
Annualisation uses ``TRADING_DAYS_PER_YEAR = 252`` everywhere by default,
which matches the convention used throughout the Builder, the optimiser
configurable via the ``annualized`` argument), and the new
``utils.risk_analytics`` helpers. A single constant is exported so callers
can reuse it without re-introducing 252/261 drift.

Risk-adjusted ratios (Sharpe, Sortino, Information) accept a daily
risk-free rate ``rf`` so callers can pass ``rf=0`` (legacy default) or a
real series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def calculate_metrics(df_ret, annualized: int = TRADING_DAYS_PER_YEAR, rf: float = 0.0):
    """Calculate the headline metrics for a daily-return Series.

    Parameters
    ----------
    df_ret : pd.Series of daily decimal returns.
    annualized : trading days per year (default 252).
    rf : annualised risk-free rate as a decimal (e.g. ``0.04`` for 4%).
         Converted to a daily rate internally for Sharpe / Sortino
         numerators.
    """
    metrics = {}

    df = df_ret.dropna()
    if df.empty:
        return pd.Series({})

    rf_daily = (1.0 + rf) ** (1.0 / annualized) - 1.0 if rf else 0.0

    metrics['Cumulative Return'] = float((1 + df).prod() - 1)

    n_years = len(df) / annualized if annualized else 1.0
    if n_years > 0:
        metrics['Annualized Return'] = (1 + metrics['Cumulative Return']) ** (1 / n_years) - 1
    else:
        metrics['Annualized Return'] = float('nan')

    metrics['Annualized Volatility'] = float(df.std(ddof=1) * np.sqrt(annualized))

    sharpe_num = metrics['Annualized Return'] - rf
    metrics['Sharpe Ratio'] = (
        sharpe_num / metrics['Annualized Volatility']
        if metrics['Annualized Volatility'] not in (0.0, None) and not np.isnan(metrics['Annualized Volatility'])
        else float('nan')
    )

    # Sortino: downside deviation against the (daily) MAR == rf. Returns
    # *equal* to or *above* MAR contribute zero — this is the formal
    # definition (Sortino & Price, 1994) and avoids the asymmetry of
    # `df.std()` on a truncated subset.
    excess_daily = df - rf_daily
    downside = np.minimum(excess_daily, 0.0)
    downside_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(annualized))
    if downside_dev > 0 and not np.isnan(metrics['Annualized Return']):
        metrics['Sortino Ratio'] = (metrics['Annualized Return'] - rf) / downside_dev
    else:
        metrics['Sortino Ratio'] = float('nan')

    cum_ret = (1 + df).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    metrics['Max Drawdown'] = float(drawdown.min())

    return pd.Series(metrics)


PERCENTAGE_METRICS = [
    'Cumulative Return', 'Annualized Return', 'Annualized Volatility',
    'Max Drawdown', 'Tracking Error',
]
RATIO_METRICS = ['Sharpe Ratio', 'Sortino Ratio', 'Information Ratio']


def format_metric_value(value, metric_name):
    """Format metric value for display."""
    if pd.isna(value):
        return "N/A"
    if metric_name in PERCENTAGE_METRICS:
        return f"{value:.2%}"
    elif metric_name in RATIO_METRICS:
        return f"{value:.2f}"
    else:
        return f"{value:.2f}"


def calculate_relative_metrics(port_ret, bm_ret, annualized: int = TRADING_DAYS_PER_YEAR):
    """Tracking Error, Information Ratio and a *compounded* 12M rolling
    excess return.

    The previous implementation returned ``excess_ret.rolling(N).sum()``
    which is an arithmetic approximation. We now compute the geometric
    excess `(1+r_p)/(1+r_b) - 1` rolled over the chosen window so the
    series can be interpreted directly as a 12-month excess return number.
    """
    metrics = {}
    common_idx = port_ret.index.intersection(bm_ret.index)
    port_ret = port_ret.loc[common_idx]
    bm_ret = bm_ret.loc[common_idx]

    excess_ret = port_ret - bm_ret
    te = float(excess_ret.std(ddof=1) * np.sqrt(annualized))
    metrics['Tracking Error'] = te
    metrics['Information Ratio'] = (
        (excess_ret.mean() * annualized) / te if te > 0 else float('nan')
    )

    # geometric 12M rolling excess: prod(1+p_t)/prod(1+b_t) - 1 over the window
    log_p = np.log1p(port_ret)
    log_b = np.log1p(bm_ret)
    rolling_log_p = log_p.rolling(window=annualized).sum()
    rolling_log_b = log_b.rolling(window=annualized).sum()
    rolling_excess = np.exp(rolling_log_p) / np.exp(rolling_log_b) - 1.0
    rolling_excess.name = 'Rolling 12M Excess Return'

    return pd.Series(metrics), rolling_excess


def get_performance_summary(all_port_ret, bm_ret=None, annualized: int = TRADING_DAYS_PER_YEAR, rf: float = 0.0):
    """DataFrame of headline metrics across portfolios."""
    summary_list = []

    if all_port_ret.empty or len(all_port_ret.columns) == 0:
        return pd.DataFrame()

    for col in all_port_ret.columns:
        metrics = calculate_metrics(all_port_ret[col], annualized=annualized, rf=rf)
        summary_list.append(metrics)

    summary_df = pd.concat(summary_list, axis=1)
    summary_df.columns = all_port_ret.columns

    if bm_ret is not None:
        # Optionally append benchmark metrics here in the future.
        pass

    return summary_df
