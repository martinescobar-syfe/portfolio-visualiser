"""Cash flow projection + FI metrics helpers used by both the Advisory Tool
and the Client PV income-goal detail view. Logic mirrors the existing Advisory
Tool 4.1 / 4.4 sections but without the backtesting dependencies so it can be
called from any context that has a {portfolio_name: nav} dict.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


# Tickers that don't generate cash flow and should be excluded from FI/cashflow
# calcs (cash/MMF, equity/growth/thematic portfolios, custom DIY baskets).
FI_EXCLUDED_PATTERNS = [
    'CASH', 'MMF', 'FLEXI', 'GUARANTEED',
    'CORE', 'EQUITY', 'GROWTH',
    'CHINA', 'ESG', 'HEALTHCARE', 'DISRUPTIVE', 'TECHNOLOGY', 'THEMATIC',
    'CUSTOM', 'STRATEGIC',
]


def is_cash_flow_eligible(ticker):
    """True if the ticker generates recurring cash flow (not cash/growth/custom)."""
    t = str(ticker).upper()
    return not any(p in t for p in FI_EXCLUDED_PATTERNS)


def get_combined_yield(ticker, df_fi_metrics, preference='YTM'):
    """Return (yield_decimal, source_label). Primary source from ``preference``;
    falls back to the other source when the primary is missing or zero.

    Yields in ``df_fi_metrics`` are stored as percentages (e.g. 7.18 = 7.18%)
    so we divide by 100 on the way out.
    """
    if df_fi_metrics is None or ticker not in df_fi_metrics.columns:
        return (0.0, None)

    def _safe(row_name):
        if row_name not in df_fi_metrics.index:
            return 0.0
        v = df_fi_metrics.loc[row_name, ticker]
        try:
            return float(v) if pd.notna(v) else 0.0
        except (ValueError, TypeError):
            return 0.0

    ytm_pct = _safe('YTM / est. yield')
    t12m_pct = _safe('T12M dividend yield')
    ytm = ytm_pct / 100.0
    t12m = t12m_pct / 100.0

    if preference == 'YTM':
        if ytm > 0:
            return (ytm, 'YTM')
        if t12m > 0:
            return (t12m, 'T12M')
    else:
        if t12m > 0:
            return (t12m, 'T12M')
        if ytm > 0:
            return (ytm, 'YTM')
    return (0.0, None)


def get_payment_frequency(ticker):
    t = str(ticker).upper()
    # Singapore REITs and REIT ETFs pay quarterly
    if 'REIT' in t or 'VNQ' in t or 'VNQI' in t:
        return 'quarterly'
    return 'monthly'


def get_ticker_base_currency(ticker):
    t = str(ticker).upper()
    if 'BLACKROCK' in t or 'MONROE' in t or 'SILVERDALE' in t:
        return 'USD'
    if 'USD' in t:
        return 'USD'
    if 'REIT' in t or 'INCOME' in t or 'SGD' in t or 'SRS' in t:
        return 'SGD'
    return 'SGD'


def convert_currency(amount, from_ccy, to_ccy, usdsgd=1.27, usdhkd=7.8):
    """Convert between USD/SGD/HKD via USD bridge."""
    if from_ccy == to_ccy:
        return amount
    if from_ccy == 'SGD':
        usd = amount / usdsgd if usdsgd > 0 else amount
    elif from_ccy == 'HKD':
        usd = amount / usdhkd if usdhkd > 0 else amount
    else:
        usd = amount
    if to_ccy == 'USD':
        return usd
    if to_ccy == 'SGD':
        return usd * usdsgd
    if to_ccy == 'HKD':
        return usd * usdhkd
    return usd


def get_reference_fx(df_fx):
    """Latest USDSGD / USDHKD from df_fx, with sensible defaults."""
    usdsgd, usdhkd = 1.27, 7.80
    if df_fx is None or df_fx.empty:
        return usdsgd, usdhkd
    if 'USDSGD' in df_fx.columns:
        s = df_fx['USDSGD'].dropna()
        if not s.empty:
            try:
                v = float(s.iloc[-1])
                if v > 1:
                    usdsgd = v
            except (ValueError, TypeError):
                pass
    if 'USDHKD' in df_fx.columns:
        s = df_fx['USDHKD'].dropna()
        if not s.empty:
            try:
                v = float(s.iloc[-1])
                if v > 1:
                    usdhkd = v
            except (ValueError, TypeError):
                pass
    return usdsgd, usdhkd


def compute_portfolio_cash_flows(
    portfolio_navs,
    df_fi_metrics,
    yield_preference='YTM',
    display_currency='SGD',
    usdsgd=1.27,
    usdhkd=7.8,
):
    """Return (monthly_df, breakdown_df) for a dict of {ticker: nav_in_base_ccy}.

    - ``monthly_df`` is indexed by ticker with 12 columns (Jan…Dec) in display ccy
    - ``breakdown_df`` lists per-ticker yield source, freq, annual cash flow
    """
    rows = []
    breakdown = []
    for ticker, nav in (portfolio_navs or {}).items():
        try:
            nav_f = float(nav)
        except (ValueError, TypeError):
            continue
        if nav_f <= 0 or not is_cash_flow_eligible(ticker):
            continue

        yld, src = get_combined_yield(ticker, df_fi_metrics, preference=yield_preference)
        if yld <= 0:
            continue

        base_ccy = get_ticker_base_currency(ticker)
        freq = get_payment_frequency(ticker)

        annual_cf_base = nav_f * yld
        annual_cf = convert_currency(annual_cf_base, base_ccy, display_currency, usdsgd, usdhkd)

        if freq == 'quarterly':
            monthly = [0.0] * 12
            qpay = annual_cf / 4.0
            for mi in (2, 5, 8, 11):  # Mar, Jun, Sep, Dec
                monthly[mi] = qpay
        else:
            monthly = [annual_cf / 12.0] * 12

        rows.append([ticker] + monthly)
        breakdown.append({
            'Portfolio': ticker,
            'NAV': nav_f,
            'Base Currency': base_ccy,
            'Yield Source': src,
            'Yield': yld,
            'Frequency': freq,
            'Annual Cash Flow': annual_cf,
        })

    month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    if not rows:
        return pd.DataFrame(columns=month_cols), pd.DataFrame()
    df = pd.DataFrame(rows, columns=['Portfolio'] + month_cols).set_index('Portfolio')
    bd = pd.DataFrame(breakdown)
    return df, bd


def plot_monthly_cash_flow(monthly_df, currency_symbol='$', title='Projected Monthly Cash Flows'):
    """Stacked bar chart: months on x-axis, portfolios stacked as colored bars."""
    if monthly_df is None or monthly_df.empty:
        return None
    # Ordered months for display
    months = list(monthly_df.columns)
    fig = go.Figure()
    for ticker in monthly_df.index:
        fig.add_trace(go.Bar(
            name=ticker,
            x=months,
            y=monthly_df.loc[ticker].values,
            hovertemplate='<b>' + ticker + '</b><br>%{x}: ' + currency_symbol + '%{y:,.0f}<extra></extra>',
        ))
    total_by_month = monthly_df.sum(axis=0)
    annual_total = total_by_month.sum()
    fig.update_layout(
        title={'text': f'<b>{title}</b> — Annual: {currency_symbol}{annual_total:,.0f}', 'font': {'size': 18}},
        barmode='stack',
        height=440,
        margin=dict(l=30, r=20, t=60, b=60),
        xaxis={'title': 'Month'},
        yaxis={'title': f'Cash Flow ({currency_symbol})', 'tickprefix': currency_symbol, 'tickformat': ',.0f'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=11)),
    )
    return fig


def build_fi_metrics_table(portfolio_names, df_fi_metrics, yield_preference='YTM'):
    """Return a DataFrame with Duration / Yield / Credit rating columns for
    FI-eligible portfolios in ``portfolio_names``.
    """
    if df_fi_metrics is None:
        return pd.DataFrame()
    eligible = [p for p in portfolio_names if is_cash_flow_eligible(p) and p in df_fi_metrics.columns]
    if not eligible:
        return pd.DataFrame()
    rows = []
    for p in eligible:
        duration = None
        if 'Duration' in df_fi_metrics.index:
            try:
                v = df_fi_metrics.loc['Duration', p]
                duration = float(v) if pd.notna(v) else None
            except (ValueError, TypeError):
                duration = None
        yld, src = get_combined_yield(p, df_fi_metrics, preference=yield_preference)
        credit = None
        if 'Credit rating' in df_fi_metrics.index:
            v = df_fi_metrics.loc['Credit rating', p]
            credit = v if pd.notna(v) else None
        rows.append({
            'Portfolio': p,
            'Duration (yrs)': duration,
            'Yield': yld,
            'Yield Source': src,
            'Credit Rating': credit,
        })
    return pd.DataFrame(rows).set_index('Portfolio')
