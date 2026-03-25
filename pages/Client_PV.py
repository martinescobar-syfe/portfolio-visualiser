import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import uuid
import math
import re
from scipy.stats import norm as scipy_norm

from utils.data_loader import load_static_data, get_data_path
from utils.ui_functions import (
    get_portfolio_objective,
    _get_asset_brief, _get_geo_brief, _get_sector_brief, _get_conc_brief,
)
from utils.exposure import get_exposure_charts, get_concentration_exposure, calculate_exposure
from utils.diversification_score import (
    calculate_diversification_score, WEIGHTS, IGNORED_PORTFOLIOS,
    calculate_geography_score, calculate_sector_score,
    calculate_concentration_score, _classify_portfolio_type,
)
from styles.syfe_colors import (
    PRIMARY_BLUE, SECONDARY_BLUE, TERTIARY_BLUE, LIGHT_BLUE, DARK_BLUE,
    GREEN, YELLOW, RED, DARK_GREY, LIGHT_GREY,
    OBJECTIVE_COLORS, get_portfolio_color,
)
from utils.metabase_helper import get_client_info_by_id, get_active_portfolios
from utils.plotting import plot_horizontal_bar_chart

# ── page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Client Portfolio Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
.stApp {background-color:#f8f9fa;font-family:'Helvetica Neue',sans-serif;}
h1,h2,h3 {color:#172B4D;font-weight:700;}
.stButton>button {background-color:#0052CC;color:white;border-radius:8px;border:none;padding:10px 20px;}
.stButton>button:hover {background-color:#0747A6;}
.goal-card {background:white;border-radius:14px;padding:22px;box-shadow:0 2px 10px rgba(0,0,0,0.08);
  border:2px solid transparent;transition:all .2s;cursor:pointer;min-height:200px;}
.goal-card:hover {box-shadow:0 4px 18px rgba(0,0,0,0.14);border-color:#0052CC;}
.goal-badge-growth {background:#E3F2FD;color:#0052CC;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;}
.goal-badge-income {background:#E8F5E9;color:#2E7D32;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;}
.metric-card {background:white;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:20px;}
</style>
""", unsafe_allow_html=True)

# ── constants ────────────────────────────────────────────────
API_KEY = "AIzaSyCZThunnMaO5Ni6fJDLnWsN24JX0yCN3p0"

CLIENT_PORTFOLIO_MAPPING = {
    'GLOBAL_EQUITY_100': 'Core Equity100', 'GLOBAL_EQUITY': 'Core Equity100',
    'CORE_EQUITY100': 'Core Equity100', 'CORE_EQUITY_100': 'Core Equity100',
    'EQUITY100': 'Core Equity100',
    'CORE_GROWTH': 'Core Growth', 'CORE_BALANCED': 'Core Balanced',
    'CORE_DEFENSIVE': 'Core Defensive', 'DOWNSIDE_PROTECTED': 'Downside protected',
    'CHINA_GROWTH': 'China Growth',
    'DISRUPTIVE_TECHNOLOGY': 'Disruptive Technology', 'DISRUPTIVE_TECH': 'Disruptive Technology',
    'ESG_AND_CLEAN_ENERGY': 'ESG & Clean Energy', 'ESG_CLEAN_ENERGY': 'ESG & Clean Energy',
    'HEALTHCARE_INNOVATION': 'Healthcare Innovation',
    'INCOME_ENHANCE': 'Income Enhance', 'INCOME_ENHANCED': 'Income Enhance',
    'INCOME_PRESERVE': 'Income Preserve',
    'REIT': 'REIT 100', 'REIT_100': 'REIT 100',
    'REIT_RISK_MANAGED': 'REIT Risk Managed', 'REIT_RM': 'REIT Risk Managed',
    'BLACKROCK': 'Blackrock Private Credit Fund',
    'BLACKROCK_PRIVATE_CREDIT': 'Blackrock Private Credit Fund',
    'BLACKROCK_PCF': 'Blackrock Private Credit Fund',
    'MONROE': 'Monroe Income Plus Fund',
    'MONROE_INCOME_PLUS': 'Monroe Income Plus Fund',
    'MONROE_CAPITAL': 'Monroe Income Plus Fund',
    'SILVERDALE': 'Silverdale Fund (Nov 2027)',
    'CASH_PLUS_GUARANTEED_SGD': 'Cash SGD guaranteed',
    'CASH_PLUS_GUARANTEED': 'Cash SGD guaranteed', 'CASH_GUARANTEED': 'Cash SGD guaranteed',
    'CASH_PLUS_USD': 'Cash USD flexi', 'CASH_USD': 'Cash USD flexi',
    'CASH_PLUS': 'Cash SGD flexi', 'CASH_SGD': 'Cash SGD flexi',
    'CASH_PLUS_SGD': 'Cash SGD flexi',
    'SRS_INCOME_ENHANCE': 'SRS Income Enhance', 'SRS_INCOME_ENHANCED': 'SRS Income Enhance',
    'SRS_INCOME_PRESERVE': 'SRS Income Preserve',
    'SRS_CASH_PLUS': 'SRS Cash SGD flexi', 'SRS_CASH': 'SRS Cash SGD flexi',
    'SRS_GLOBAL_EQUITY_100': 'SRS Equity100', 'SRS_EQUITY100': 'SRS Equity100',
    'SRS_EQUITY_100': 'SRS Equity100',
}

ALL_MANAGED_PORTFOLIOS = [
    'Core Equity100', 'Core Growth', 'Core Balanced', 'Core Defensive',
    'Disruptive Technology', 'ESG & Clean Energy', 'Healthcare Innovation', 'China Growth',
    'Income Enhance', 'Income Preserve',
    'REIT 100', 'REIT Risk Managed',
    'Blackrock Private Credit Fund', 'Monroe Income Plus Fund',
    'Downside protected',
    'Cash SGD flexi', 'Cash USD flexi', 'Cash SGD guaranteed',
    'SRS Equity100', 'SRS Income Enhance', 'SRS Income Preserve', 'SRS Cash SGD flexi',
]

PORTFOLIO_ESTIMATES = {
    'Core Equity100':              {'ret': 0.10, 'vol': 0.15, 'yield': 0.015},
    'SRS Equity100':               {'ret': 0.10, 'vol': 0.15, 'yield': 0.015},
    'Core Growth':                 {'ret': 0.08, 'vol': 0.12, 'yield': 0.020},
    'Core Balanced':               {'ret': 0.06, 'vol': 0.09, 'yield': 0.025},
    'Core Defensive':              {'ret': 0.04, 'vol': 0.06, 'yield': 0.030},
    'Downside protected':          {'ret': 0.05, 'vol': 0.08, 'yield': 0.025},
    'Disruptive Technology':       {'ret': 0.12, 'vol': 0.22, 'yield': 0.005},
    'ESG & Clean Energy':          {'ret': 0.09, 'vol': 0.18, 'yield': 0.010},
    'Healthcare Innovation':       {'ret': 0.10, 'vol': 0.17, 'yield': 0.010},
    'China Growth':                {'ret': 0.08, 'vol': 0.25, 'yield': 0.015},
    'Income Enhance':              {'ret': 0.05, 'vol': 0.08, 'yield': 0.045},
    'Income Preserve':             {'ret': 0.04, 'vol': 0.05, 'yield': 0.035},

    'SRS Income Enhance':          {'ret': 0.05, 'vol': 0.08, 'yield': 0.045},
    'SRS Income Preserve':         {'ret': 0.04, 'vol': 0.05, 'yield': 0.035},
    'REIT 100':                    {'ret': 0.06, 'vol': 0.16, 'yield': 0.040},
    'REIT Risk Managed':           {'ret': 0.05, 'vol': 0.12, 'yield': 0.035},
    'Blackrock Private Credit Fund': {'ret': 0.07, 'vol': 0.05, 'yield': 0.070},
    'Monroe Income Plus Fund':     {'ret': 0.08, 'vol': 0.05, 'yield': 0.080},
    'Cash SGD flexi':              {'ret': 0.035,'vol': 0.005,'yield': 0.035},
    'Cash USD flexi':              {'ret': 0.040,'vol': 0.005,'yield': 0.040},
    'Cash SGD guaranteed':         {'ret': 0.030,'vol': 0.001,'yield': 0.030},
    'SRS Cash SGD flexi':          {'ret': 0.035,'vol': 0.005,'yield': 0.035},
}
_DEFAULT_EST = {'ret': 0.06, 'vol': 0.12, 'yield': 0.03}


# ── helpers ──────────────────────────────────────────────────
def _norm_type(p_type):
    if not isinstance(p_type, str):
        return None
    n = p_type.upper().strip()
    for sfx in ('_ACTIVE_NEW', '_ACTIVE', '_INACTIVE', '_CLOSED', '_PENDING'):
        if n.endswith(sfx):
            n = n[:-len(sfx)]
    return n


def _safe_float(val, default=0.0):
    if pd.isna(val):
        return default
    try:
        return float(str(val).replace(',', ''))
    except (ValueError, TypeError):
        return default


def _get_est(name):
    if name in PORTFOLIO_ESTIMATES:
        return PORTFOLIO_ESTIMATES[name]
    if 'silverdale' in name.lower():
        return {'ret': 0.065, 'vol': 0.04, 'yield': 0.065}
    return _DEFAULT_EST


def process_client_data(df_client):
    """Return (client_weights, client_holdings, total_nav)."""
    holdings, total = {}, 0.0
    for _, row in df_client.iterrows():
        p_type = row.get('type', None)
        currency = str(row.get('currency', 'SGD'))
        nav = _safe_float(row.get('nav_in_sgd', 0))
        if nav < 50:
            continue
        nt = _norm_type(p_type)
        if nt is None:
            continue
        ticker = None
        if 'CASH' in nt and 'PLUS' in nt:
            if currency == 'USD' or 'USD' in nt:
                ticker = 'Cash USD flexi'
            elif 'GUARANTEED' in nt:
                ticker = 'Cash SGD guaranteed'
            else:
                ticker = 'Cash SGD flexi'
        elif nt in CLIENT_PORTFOLIO_MAPPING:
            ticker = CLIENT_PORTFOLIO_MAPPING[nt]
        elif 'BLACKROCK' in nt:
            ticker = 'Blackrock Private Credit Fund'
        elif 'MONROE' in nt:
            ticker = 'Monroe Income Plus Fund'
        elif 'CUSTOM' in nt:
            ticker = 'Custom Usd'
        elif 'DOWNSIDE' in nt:
            ticker = 'Downside protected'
        elif 'SILVER' in nt or 'SILVERDALE' in nt:
            m = re.search(r'(NOV|DEC|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT)\s*(\d{4})', nt)
            ticker = f'Silverdale Fund {m.group(1).title()}{m.group(2)}' if m else 'Silverdale Fund Feb2029'
        if ticker:
            holdings[ticker] = holdings.get(ticker, 0.0) + nav
            total += nav
    if total <= 0:
        return None, None, 0
    weights = {k: v / total for k, v in holdings.items()}
    return weights, holdings, total


def _goal_weighted_params(goal, client_holdings):
    """Compute weighted return, vol, yield for a goal's portfolio mix.
    Returns (total_return, vol, yield) where total_return includes yield."""
    ports = goal['portfolios']
    is_custom = goal.get('source', 'existing') != 'existing'

    if is_custom:
        allocs = goal.get('allocations', {})
        if allocs:
            navs = {p: allocs.get(p, 0) for p in ports}
        else:
            navs = {p: 1.0 for p in ports}
    else:
        navs = {p: client_holdings.get(p, 0) for p in ports}

    total = sum(navs.values())
    if total <= 0:
        return _DEFAULT_EST['ret'], _DEFAULT_EST['vol'], _DEFAULT_EST['yield']
    w_ret, w_vol2, w_yld = 0.0, 0.0, 0.0
    for p, nav in navs.items():
        w = nav / total
        e = _get_est(p)
        w_ret += w * e['ret']
        w_vol2 += (w * e['vol']) ** 2
        w_yld += w * e['yield']
    return w_ret, math.sqrt(w_vol2), w_yld


def _build_projection(index_path, initial, monthly_c):
    vals = [initial]
    for i in range(1, len(index_path)):
        growth = index_path[i] / index_path[i - 1]
        vals.append(vals[-1] * growth + monthly_c)
    return np.array(vals)


def _score_rating(s):
    if s is None:
        return 'N/A'
    if s >= 8:
        return 'Excellent'
    if s >= 6:
        return 'Good'
    if s >= 4:
        return 'Fair'
    return 'Needs Work'


# ── load static data ─────────────────────────────────────────
data_path = get_data_path()
df_sec_list, df_asset_class, df_sector, df_country, df_fi_metrics, df_fx, df_daily_ret, df_risk_rating, df_underlying = load_static_data(data_path)

# ── session state ────────────────────────────────────────────
for key, default in [
    ('cpv_goals', []),
    ('cpv_active_goal_id', None),
    ('cpv_card_page', 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── page header ──────────────────────────────────────────────
st.title("Portfolio Visualizer")
st.markdown("Upload your portfolio data to analyze holdings, diversification, and plan financial goals.")
st.divider()

# ═══════════════════════════════════════════════════════════════
# DATA SOURCE
# ═══════════════════════════════════════════════════════════════
tab_csv, tab_meta = st.tabs(["📂 CSV Upload", "🔍 Metabase Lookup"])
df_client = None

with tab_csv:
    uploaded = st.file_uploader("Upload Client Portfolio CSV", type=['csv'], key="cpv_csv_upload")
    if uploaded:
        try:
            df_client = pd.read_csv(uploaded)
            df_client.columns = df_client.columns.str.lower().str.strip()
            if 'type' in df_client.columns:
                df_client['type'] = df_client['type'].str.replace('BlackRock', 'Blackrock', case=False)
                df_client['type'] = df_client['type'].str.replace('Monroe', 'Monroe', case=False)
            if 'status' in df_client.columns:
                df_client = df_client[df_client['status'].str.upper().isin(['ACTIVE', 'ACTIVE_NEW'])]
            st.session_state['cpv_df_client'] = df_client
            st.success(f"Loaded {len(df_client)} portfolios from CSV")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

with tab_meta:
    c_id, c_btn = st.columns([3, 1])
    with c_id:
        client_id_input = st.text_input("Client ID", key="cpv_meta_id")
    with c_btn:
        st.write("")
        st.write("")
        lookup = st.button("Fetch Client", key="cpv_meta_btn")
    if lookup and client_id_input:
        with st.spinner("Fetching client data from Metabase..."):
            try:
                info = get_client_info_by_id(int(client_id_input.strip()))
                if info.get('client_id'):
                    fetched = get_active_portfolios(info['client_id'])
                    st.session_state['cpv_client_info'] = info
                    st.session_state['cpv_df_client'] = fetched
                    st.success(f"Found client: {info.get('name')} (ID: {info.get('client_id')})")
                else:
                    st.error("Client not found.")
            except Exception as e:
                st.error(f"Error fetching data: {e}")

    if st.session_state.get('cpv_client_info') and st.session_state.get('cpv_df_client') is not None:
        ci = st.session_state['cpv_client_info']
        c1, c2, c3 = st.columns(3)
        c1.metric("Name", ci.get('name'))
        c2.metric("Age", ci.get('age'))
        c3.metric("Risk Score", ci.get('risk_score'))
        dfm = st.session_state['cpv_df_client']
        if not dfm.empty:
            cols_show = [c for c in ['name', 'type', 'nav_in_sgd', 'pnl_inception'] if c in dfm.columns]
            st.dataframe(dfm[cols_show] if cols_show else dfm, use_container_width=True)
            df_client = dfm

if df_client is None and 'cpv_df_client' in st.session_state:
    df_client = st.session_state['cpv_df_client']

# ═══════════════════════════════════════════════════════════════
# PROCESS DATA
# ═══════════════════════════════════════════════════════════════
if df_client is not None and not df_client.empty:
    client_weights, client_holdings, total_nav = process_client_data(df_client)
else:
    client_weights, client_holdings, total_nav = None, None, 0

if client_weights and total_nav > 0:
    # store for goal calculations
    st.session_state['cpv_client_weights'] = client_weights
    st.session_state['cpv_client_holdings'] = client_holdings
    st.session_state['cpv_total_nav'] = total_nav

    # ───────────────────────────────────────────────────────────
    # GOAL DETAIL VIEW (if a goal is selected)
    # ───────────────────────────────────────────────────────────
    if st.session_state.cpv_active_goal_id:
        goal = next((g for g in st.session_state.cpv_goals if g['id'] == st.session_state.cpv_active_goal_id), None)
        if goal is None:
            st.session_state.cpv_active_goal_id = None
            st.rerun()

        if st.button("← Back to Dashboard", key="cpv_back"):
            st.session_state.cpv_active_goal_id = None
            st.rerun()

        st.header(f"{goal['name']}")
        if goal.get('description'):
            st.caption(goal['description'])

        # ── compute goal NAV, weights, and return series upfront ──
        is_existing = goal.get('source', 'existing') == 'existing'
        if is_existing:
            goal_navs = {p: client_holdings.get(p, 0) for p in goal['portfolios']}
            goal_nav = sum(goal_navs.values())
            goal_total = goal_nav if goal_nav > 0 else 1
            goal_weights = {p: v / goal_total for p, v in goal_navs.items() if v > 0}
        else:
            allocs = goal.get('allocations', {})
            if allocs:
                goal_nav = sum(allocs.values())
                goal_total = goal_nav if goal_nav > 0 else 1
                goal_weights = {p: v / goal_total for p, v in allocs.items() if v > 0}
            else:
                goal_nav = goal.get('initial_value', 0) or 0
                n_ports = len(goal['portfolios'])
                goal_weights = {p: 1.0 / n_ports for p in goal['portfolios']} if n_ports > 0 else {}
        goal_obj = "Wealth Growth" if goal['type'] == 'Growth' else "Income Generation"
        kp = f"cpv_goal_{goal['id']}"

        # build return series (used by both forecast and performance)
        from utils.performance import calculate_metrics
        goal_daily = None
        returns_label = "Backtested Returns"
        available_ports = [p for p in goal['portfolios'] if p in df_daily_ret.columns] if df_daily_ret is not None else []
        if available_ports and goal_weights:
            port_rets = df_daily_ret[available_ports].dropna(how='all') / 100  # convert from % to decimal
            common_start = port_rets.dropna().index.min()
            port_rets = port_rets.loc[common_start:]
            goal_daily = pd.Series(0.0, index=port_rets.index)
            for p in available_ports:
                goal_daily += port_rets[p].fillna(0) * goal_weights.get(p, 0)
            goal_daily = goal_daily.dropna()
            if is_existing:
                returns_label = "Actual Returns"

        # ══════════════════════════════════════════════════════════
        # SECTION 1: FORECAST + FACTUAL DATA
        # ══════════════════════════════════════════════════════════
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Current Value", f"${goal_nav:,.2f}")
        mc2.metric("Objective", goal_obj)
        mc3.metric("Time Horizon", f"{goal.get('time_horizon', 20)} years")
        if goal['type'] == 'Growth':
            mc4.metric("Target Value", f"${goal.get('target_value', 0):,.0f}")
        else:
            mc4.metric("Target Payout", f"${goal.get('target_payout', 0):,.0f}/mo")

        st.divider()

        if goal['type'] == 'Growth':
            st.subheader("Growth Forecast")
            ann_ret, ann_vol, _ = _goal_weighted_params(goal, client_holdings)
            log_ret = np.log(1 + ann_ret)
            initial = goal.get('initial_value', goal_nav) or goal_nav
            mc_val = goal.get('monthly_topup', 0) or 0
            t_horizon = goal.get('time_horizon', 20)
            target = goal.get('target_value', 500000) or 500000
            months_total = t_horizon * 12
            months_arr = np.arange(0, months_total + 1)

            conservative = np.ones(len(months_arr)) * 100.0
            likely = np.ones(len(months_arr)) * 100.0
            optimistic = np.ones(len(months_arr)) * 100.0
            m_vec = months_arr[1:]
            drift = (m_vec / 12) * log_ret
            vol_arr = np.sqrt(m_vec / 12) * ann_vol
            conservative[1:] = np.exp(drift + vol_arr * scipy_norm.ppf(0.05)) * 100
            likely[1:] = np.exp(drift + vol_arr * scipy_norm.ppf(0.50)) * 100
            optimistic[1:] = np.exp(drift + vol_arr * scipy_norm.ppf(0.95)) * 100
            percentiles = {5: conservative, 50: likely, 95: optimistic}
            projected = {p: _build_projection(path, initial, mc_val) for p, path in percentiles.items()}
            years_arr = months_arr / 12.0
            fin_med = projected[50][-1]

            if target > 0:
                ratio = fin_med / target
            else:
                ratio = 1.0
            if ratio >= 1.0:
                goal_score = min(10.0, 7.0 + ((ratio - 1.0) * 100 / 10.0) * 0.3)
            else:
                goal_score = max(3.0, 7.0 - ((1.0 - ratio) / 0.4) * 4.0)

            gc1, gc2 = st.columns([1, 2])
            with gc1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=goal_score,
                    number={'font': {'size': 52, 'color': DARK_BLUE}, 'suffix': '/10'},
                    gauge={'axis': {'range': [0, 10], 'tickwidth': 2, 'dtick': 2, 'tickcolor': DARK_BLUE, 'tickfont': {'size': 12, 'color': '#888'}},
                           'bar': {'color': DARK_BLUE, 'thickness': 0.08}, 'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                           'steps': [{'range': [0, 3], 'color': '#FF5630'}, {'range': [3, 5], 'color': '#FFAB00'}, {'range': [5, 7], 'color': TERTIARY_BLUE}, {'range': [7, 10], 'color': PRIMARY_BLUE}],
                           'threshold': {'line': {'color': DARK_BLUE, 'width': 5}, 'thickness': 0.85, 'value': goal_score}},
                ))
                fig_gauge.update_layout(height=240, margin=dict(l=25, r=25, t=40, b=0), paper_bgcolor='rgba(0,0,0,0)', font={'color': DARK_BLUE})
                st.plotly_chart(fig_gauge, use_container_width=True, key=f"{kp}_gauge")
                if goal_score >= 7:
                    st.success("You are set to meet or exceed your goals!")
                elif goal_score >= 5:
                    st.warning("Needs more work, consider increasing Monthly Top-ups!")
                else:
                    st.error(f"At risk of not being able to meet {goal['name']}!")
                m1, m2 = st.columns(2)
                m1.metric("Current NAV", f"${initial:,.0f}")
                m2.metric("Projected Value (Median)", f"${fin_med:,.0f}")
                m3, m4 = st.columns(2)
                if fin_med < target and ann_ret > 0:
                    fv_initial = initial * (likely[-1] / 100.0)
                    gap = target - fv_initial
                    ann_factor = (((1 + ann_ret) ** t_horizon - 1) / ann_ret) * (1 + ann_ret)
                    suggested = max(0, (gap / ann_factor) / 12) if ann_factor > 0 else 0
                else:
                    suggested = 0
                m3.metric("Suggested Monthly", f"${suggested:,.0f}")
                m4.metric("Current Monthly", f"${mc_val:,.0f}")

            with gc2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=years_arr, y=projected[95], mode='lines', name='Optimistic (95th)', line=dict(color=GREEN, width=2, dash='dot')))
                fig.add_trace(go.Scatter(x=years_arr, y=projected[50], mode='lines', name='Median', line=dict(color='#000000', width=3)))
                fig.add_trace(go.Scatter(x=years_arr, y=projected[5], mode='lines', name='Conservative (5th)', line=dict(color=RED, width=2, dash='dot')))
                fig.add_trace(go.Scatter(x=years_arr, y=[target] * len(years_arr), mode='lines', name='Goal', line=dict(color='#999', dash='dash', width=1)))
                fig.update_layout(title=f"Growth Forecast — {goal['name']}", xaxis_title="Years", yaxis_title="Portfolio Value ($)",
                                  hovermode='x unified', yaxis=dict(tickformat='$,.0f'), font=dict(family='Helvetica Neue'), height=450,
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True, key=f"{kp}_growth_fvc")

        else:
            st.subheader("Income Forecast")
            ann_ret, ann_vol, ann_yld = _goal_weighted_params(goal, client_holdings)
            initial = goal.get('initial_value', goal_nav) or goal_nav
            mc_val = goal.get('monthly_topup', 0) or 0
            t_horizon = goal.get('time_horizon', 20)
            target_payout = goal.get('target_payout', 2500) or 2500
            months_total = t_horizon * 12
            ann_cap_growth = max(0, ann_ret - ann_yld)
            monthly_cap = (1 + ann_cap_growth) ** (1 / 12) - 1
            monthly_yld = ann_yld / 12

            nav_current = np.zeros(months_total + 1)
            nav_current[0] = initial
            for m in range(1, months_total + 1):
                nav_current[m] = nav_current[m - 1] * (1 + monthly_cap) + mc_val
            income_current = nav_current * monthly_yld

            lo, hi = mc_val, max(mc_val * 5, target_payout * 20)
            for _ in range(50):
                mid = (lo + hi) / 2
                nav_t = initial
                for m in range(1, months_total + 1):
                    nav_t = nav_t * (1 + monthly_cap) + mid
                if nav_t * monthly_yld >= target_payout:
                    hi = mid
                else:
                    lo = mid
            suggested_topup = hi

            nav_suggested = np.zeros(months_total + 1)
            nav_suggested[0] = initial
            for m in range(1, months_total + 1):
                nav_suggested[m] = nav_suggested[m - 1] * (1 + monthly_cap) + suggested_topup
            income_suggested = nav_suggested * monthly_yld
            years_arr = np.arange(0, months_total + 1) / 12.0
            forecasted_payout = income_current[-1] if len(income_current) > 0 else 0

            if target_payout > 0:
                inc_ratio = forecasted_payout / target_payout
            else:
                inc_ratio = 1.0
            inc_score = min(10.0, 7.0 + ((inc_ratio - 1.0) * 100 / 10.0) * 0.3) if inc_ratio >= 1.0 else max(3.0, 7.0 - ((1.0 - inc_ratio) / 0.4) * 4.0)

            kp_i = f"{kp}_inc"
            ic1, ic2 = st.columns([1, 2])
            with ic1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=inc_score,
                    number={'font': {'size': 52, 'color': DARK_BLUE}, 'suffix': '/10'},
                    gauge={'axis': {'range': [0, 10], 'tickwidth': 2, 'dtick': 2, 'tickcolor': DARK_BLUE, 'tickfont': {'size': 12, 'color': '#888'}},
                           'bar': {'color': DARK_BLUE, 'thickness': 0.08}, 'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                           'steps': [{'range': [0, 3], 'color': '#FF5630'}, {'range': [3, 5], 'color': '#FFAB00'}, {'range': [5, 7], 'color': TERTIARY_BLUE}, {'range': [7, 10], 'color': PRIMARY_BLUE}],
                           'threshold': {'line': {'color': DARK_BLUE, 'width': 5}, 'thickness': 0.85, 'value': inc_score}},
                ))
                fig_gauge.update_layout(height=240, margin=dict(l=25, r=25, t=40, b=0), paper_bgcolor='rgba(0,0,0,0)', font={'color': DARK_BLUE})
                st.plotly_chart(fig_gauge, use_container_width=True, key=f"{kp_i}_gauge")
                st.markdown(f'<div style="text-align:center;margin-top:-8px"><span style="font-size:28px;font-weight:700;color:{DARK_BLUE}">${forecasted_payout:,.0f}</span>'
                            f'<span style="font-size:14px;color:#888"> /month forecasted</span></div>', unsafe_allow_html=True)
                if inc_score >= 7:
                    st.success("You are set to meet or exceed your income goals!")
                elif inc_score >= 5:
                    st.warning("Needs more work, consider increasing Monthly Top-ups!")
                else:
                    st.error(f"At risk of not meeting {goal['name']}!")
                g1, g2 = st.columns(2)
                g1.metric("Target Payout", f"${target_payout:,.0f}/mo")
                g2.metric("Current NAV", f"${initial:,.0f}")
                g3, g4 = st.columns(2)
                g3.metric("Current Monthly", f"${mc_val:,.0f}")
                g4.metric("Suggested Top Up", f"${suggested_topup:,.0f}")

            with ic2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=years_arr, y=income_suggested, mode='lines', name='Suggested Top Up', line=dict(color=GREEN, width=2, dash='dash')))
                fig.add_trace(go.Scatter(x=years_arr, y=income_current, mode='lines', name='Current Top Up', line=dict(color=PRIMARY_BLUE, width=3)))
                fig.add_trace(go.Scatter(x=years_arr, y=[target_payout] * len(years_arr), mode='lines', name='Goal', line=dict(color='#999', dash='dash', width=1)))
                fig.add_trace(go.Scatter(x=[years_arr[-1]], y=[target_payout], mode='markers', name='Goal Target', marker=dict(color=GREEN, size=10), showlegend=False))
                fig.update_layout(title="Projected Income Growth", xaxis_title="Years", yaxis_title="Monthly Income ($)",
                                  hovermode='x unified', yaxis=dict(tickformat='$,.0f'), font=dict(family='Helvetica Neue'), height=450,
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True, key=f"{kp_i}_chart")

        st.divider()

        # ══════════════════════════════════════════════════════════
        # SECTION 2: PERFORMANCE (returns charts)
        # ══════════════════════════════════════════════════════════
        st.subheader(f"Performance — {returns_label}")
        if goal_daily is not None and len(goal_daily) > 20:
            metrics = calculate_metrics(goal_daily)
            km1, km2, km3, km4 = st.columns(4)
            km1.metric("Annualized Return", f"{metrics['Annualized Return']:.2%}")
            km2.metric("Annualized Volatility", f"{metrics['Annualized Volatility']:.2%}")
            km3.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
            km4.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")

            kp_perf = f"{kp}_perf"
            cum_ret = (1 + goal_daily).cumprod()
            running_max = cum_ret.cummax()
            drawdown_series = (cum_ret - running_max) / running_max * 100

            from utils.plotting import plot_monthly_heatmap, plot_cumulative_returns, plot_drawdowns

            perf_tab_cum, perf_tab_cal, perf_tab_dd = st.tabs(["Cumulative Returns", "Calendar Returns", "Drawdowns"])

            with perf_tab_cum:
                cum_pct = (cum_ret - 1).to_frame(goal['name'])
                fig_cum = plot_cumulative_returns(cum_pct)
                st.plotly_chart(fig_cum, use_container_width=True, key=f"{kp_perf}_cum")

            with perf_tab_cal:
                fig_cal = plot_monthly_heatmap(goal_daily.to_frame(goal['name']))
                st.plotly_chart(fig_cal, use_container_width=True, key=f"{kp_perf}_cal")

            with perf_tab_dd:
                fig_dd = plot_drawdowns(goal_daily.to_frame(goal['name']))
                st.plotly_chart(fig_dd, use_container_width=True, key=f"{kp_perf}_dd")

            st.caption(f"Data: {goal_daily.index.min().strftime('%b %Y')} – {goal_daily.index.max().strftime('%b %Y')}")
        else:
            st.info("Insufficient return data for the selected portfolios.")

        st.divider()

        # ══════════════════════════════════════════════════════════
        # SECTION 3: PORTFOLIO CARDS AND COMPOSITION
        # ══════════════════════════════════════════════════════════
        st.subheader("Portfolio Composition")
        if goal_weights:
            bk_pills = []
            for p, w in sorted(goal_weights.items(), key=lambda x: x[1], reverse=True):
                nav_val = goal_nav * w
                bk_pills.append(
                    f'<span style="display:inline-block;background:#EAF0FB;color:#0052CC;'
                    f'font-size:12px;padding:4px 12px;border-radius:10px;margin:3px 4px 3px 0;font-weight:500">'
                    f'{p}: {w:.1%} (${nav_val:,.0f})</span>'
                )
            st.markdown('<div>' + " ".join(bk_pills) + '</div>', unsafe_allow_html=True)

        st.divider()

        # ══════════════════════════════════════════════════════════
        # SECTION 4: EXPOSURE / DIVERSIFICATION SCORE
        # ══════════════════════════════════════════════════════════
        st.subheader("Diversification Score")
        if goal_weights:
            gw_filtered = {k: v for k, v in goal_weights.items() if k not in IGNORED_PORTFOLIOS}
            if gw_filtered:
                gw_total = sum(gw_filtered.values())
                gw_norm = {k: v / gw_total for k, v in gw_filtered.items()} if gw_total > 0 else gw_filtered
                gw_series = pd.Series({f"{k}_wgt": v for k, v in gw_norm.items()})

                div_result = calculate_diversification_score(gw_norm, df_asset_class, df_sector, df_country, df_underlying)
                composite = div_result['composite_score']
                sub_labels = ['Asset Class', 'Geography', 'Sector', 'Concentration']
                sub_keys = ['asset_class', 'geography', 'sector', 'concentration']
                weight_keys = ['asset', 'geo', 'sector', 'conc']
                sub_scores = [div_result[k].get('score') for k in sub_keys]
                sub_w = [WEIGHTS[k] for k in weight_keys]
                s_colors = [DARK_BLUE, PRIMARY_BLUE, SECONDARY_BLUE, TERTIARY_BLUE]

                charts = get_exposure_charts(gw_series, df_sector, df_country, df_asset_class, df_underlying, concentration_top_n=20)

                col_l, col_r = st.columns([2, 3])
                with col_l:
                    d_labels = [f'{l}  ({s:.1f})' if s is not None else f'{l}  (N/A)' for l, s in zip(sub_labels, sub_scores)]
                    d_values = [w * (s or 0) for w, s in zip(sub_w, sub_scores)]
                    fig_d = go.Figure(data=[go.Pie(
                        labels=d_labels, values=d_values, hole=0.7,
                        marker=dict(colors=s_colors, line=dict(color='white', width=2)),
                        textinfo='none', hovertemplate='%{label}<extra></extra>',
                        sort=False, direction='clockwise',
                    )])
                    fig_d.update_layout(
                        showlegend=True,
                        legend=dict(orientation='h', yanchor='top', y=-0.02, xanchor='center', x=0.5, font=dict(size=11, color='#555')),
                        height=340, margin=dict(l=5, r=5, t=5, b=60),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        annotations=[dict(text=f'<b style="font-size:40px">{composite:.1f}</b><b style="font-size:16px;color:#172B4D"> /10</b>',
                                          x=0.5, y=0.5, font_size=16, showarrow=False, font_color='#172B4D')],
                    )
                    st.plotly_chart(fig_d, use_container_width=True, key=f"{kp}_div_donut")

                    ins_key = f"{kp}_div_ins"
                    if ins_key not in st.session_state:
                        try:
                            from utils.llm_helper import generate_diversification_insights
                            with st.spinner("Generating insights..."):
                                st.session_state[ins_key] = generate_diversification_insights(API_KEY, div_result, gw_norm)
                        except Exception:
                            st.session_state[ins_key] = []
                    for ins in (st.session_state.get(ins_key) or []):
                        cat = ins.get('category', 'observation')
                        styles = {'strength': ('#E8F5E9', '#2E7D32'), 'improvement': ('#FFF3E0', '#E65100'), 'observation': ('#E3F2FD', '#0052CC')}
                        bg, accent = styles.get(cat, styles['observation'])
                        st.markdown(
                            f'<div style="background:{bg};border-left:4px solid {accent};border-radius:8px;padding:12px 14px;margin-bottom:8px">'
                            f'<div style="font-weight:700;color:#172B4D;font-size:13px;margin-bottom:3px">{ins.get("title","")}</div>'
                            f'<div style="color:#333;font-size:12px;line-height:1.4">{ins.get("body","")}</div></div>',
                            unsafe_allow_html=True)

                with col_r:
                    tab_aa, tab_geo, tab_sec, tab_conc = st.tabs(["Asset Class", "Geography", "Sector", "Concentration"])
                    def _tab_score(score, color):
                        if score is not None:
                            st.markdown(f'<div style="text-align:center;padding:8px 0"><span style="font-size:28px;font-weight:800;color:{color}">{score:.1f}</span>'
                                        f'<span style="font-size:14px;color:#999"> / 10</span>'
                                        f'<span style="font-size:12px;color:#888;margin-left:8px">{_score_rating(score)}</span></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div style="text-align:center;padding:8px 0"><span style="font-size:28px;font-weight:800;color:#B0B0B0">N/A</span></div>', unsafe_allow_html=True)

                    with tab_aa:
                        _tab_score(sub_scores[0], s_colors[0])
                        if 'asset_pie' in charts:
                            st.plotly_chart(charts['asset_pie'], use_container_width=True, key=f"{kp}_aa_pie")
                            with st.expander("View Data"):
                                st.dataframe(charts['asset_table'].style.format("{:.2%}"), use_container_width=True)
                        st.caption(_get_asset_brief(div_result['asset_class']))
                    with tab_geo:
                        _tab_score(sub_scores[1], s_colors[1])
                        if 'country_pie' in charts:
                            st.plotly_chart(charts['country_pie'], use_container_width=True, key=f"{kp}_geo_pie")
                            with st.expander("View Data"):
                                st.dataframe(charts['country_table'].style.format("{:.2%}"), use_container_width=True)
                        st.caption(_get_geo_brief(div_result['geography']))
                    with tab_sec:
                        _tab_score(sub_scores[2], s_colors[2])
                        if 'sector_pie' in charts:
                            st.plotly_chart(charts['sector_pie'], use_container_width=True, key=f"{kp}_sec_pie")
                            with st.expander("View Data"):
                                st.dataframe(charts['sector_table'].style.format("{:.2%}"), use_container_width=True)
                        st.caption(_get_sector_brief(div_result['sector']))
                    with tab_conc:
                        _tab_score(sub_scores[3], s_colors[3])
                        if df_underlying is not None:
                            eq_w = {}
                            for col_name in gw_series.index:
                                tk = col_name.replace('_wgt', '')
                                if _classify_portfolio_type(tk, df_asset_class) in ('equity', 'reit'):
                                    eq_w[col_name] = gw_series[col_name]
                            eq_s = pd.Series(eq_w) if eq_w else pd.Series(dtype=float)
                            if not eq_s.empty:
                                eq_s = eq_s / eq_s.sum()
                            conc_raw = get_concentration_exposure(eq_s, df_underlying, top_n=10) if not eq_s.empty else pd.Series(dtype=float)
                            if not conc_raw.empty:
                                fig_bar = plot_horizontal_bar_chart(conc_raw.head(10), title=f'Top 10 Holdings ({conc_raw.head(10).sum():.1%} Total)')
                                st.plotly_chart(fig_bar, use_container_width=True, key=f"{kp}_conc_bar")
                        st.caption(_get_conc_brief(div_result['concentration']))

    # ───────────────────────────────────────────────────────────
    # MAIN DASHBOARD (no goal selected)
    # ───────────────────────────────────────────────────────────
    else:
        # objective allocation
        objective_alloc = {}
        for ticker, w in client_weights.items():
            obj = get_portfolio_objective(ticker)
            objective_alloc[obj] = objective_alloc.get(obj, 0) + w

        # ═══════════════════════════════════════════════════════
        # 1. PORTFOLIO SUMMARY
        # ═══════════════════════════════════════════════════════
        st.markdown("### Portfolio Summary")
        sorted_ports = sorted(client_weights.items(), key=lambda x: x[1], reverse=True)
        top_port = sorted_ports[0][0] if sorted_ports else "N/A"
        sorted_objs = sorted(objective_alloc.items(), key=lambda x: x[1], reverse=True)
        top_obj = sorted_objs[0][0] if sorted_objs else "N/A"

        top_str = ""
        if df_underlying is not None and not df_underlying.empty:
            w_s = pd.Series({f"{k}_wgt": v for k, v in client_weights.items()})
            conc = get_concentration_exposure(w_s, df_underlying, top_n=3)
            if not conc.empty:
                top_str = ", ".join([f"{n} ({w:.1%})" for n, w in conc.head(3).items()])

        summary = f"Your current portfolio is primarily allocated to **{top_port}**, aligning with a **{top_obj}** objective."
        if top_str:
            summary += f" Across your holdings, your top underlying exposures are **{top_str}**."
        st.info(summary)

        # ═══════════════════════════════════════════════════════
        # FILTER EXPOSURES
        # ═══════════════════════════════════════════════════════
        st.markdown("### 1. Summary")
        st.markdown("#### Filter Exposures")
        fc, _ = st.columns([2, 1])
        with fc:
            filt_mode = st.radio("Filter Mode", ["Show All", "By Objective", "By Portfolio Type"], horizontal=True, key="cpv_filt_mode")

        filtered = {}
        if filt_mode == "Show All":
            filtered = {f"{k}_wgt": v for k, v in client_weights.items()}
        elif filt_mode == "By Objective":
            all_obj = sorted(set(get_portfolio_objective(t) for t in client_weights))
            sel_obj = st.multiselect("Select Objective(s)", all_obj, default=all_obj, key="cpv_filt_obj")
            for t, w in client_weights.items():
                if get_portfolio_objective(t) in sel_obj:
                    filtered[f"{t}_wgt"] = w
        else:
            all_pt = sorted(client_weights.keys())
            sel_pt = st.multiselect("Select Portfolio Type(s)", all_pt, default=all_pt, key="cpv_filt_pt")
            for t, w in client_weights.items():
                if t in sel_pt:
                    filtered[f"{t}_wgt"] = w

        filtered = {k: v for k, v in filtered.items() if k.replace('_wgt', '') not in IGNORED_PORTFOLIOS}
        ws = pd.Series(filtered)
        ws_total = ws.sum()
        if ws_total > 0:
            ws = ws / ws_total

        if not ws.empty:
            charts = get_exposure_charts(ws, df_sector, df_country, df_asset_class, df_underlying, concentration_top_n=20)
            filt_pw = {k.replace('_wgt', ''): v for k, v in filtered.items()}
            filt_total = sum(filt_pw.values())
            filt_pw_n = {k: v / filt_total for k, v in filt_pw.items()} if filt_total > 0 else filt_pw
            filt_nav = sum(client_holdings.get(t, 0) for t in filt_pw)

            filt_obj = {}
            for t, w in filt_pw_n.items():
                o = get_portfolio_objective(t)
                filt_obj[o] = filt_obj.get(o, 0) + w

            col_nav, col_obj = st.columns(2)
            with col_nav:
                st.markdown("**By Portfolio Type**")
                p_colors = [get_portfolio_color(n) for n in filt_pw_n]
                fig_nav = go.Figure(data=[go.Pie(
                    labels=list(filt_pw_n.keys()), values=list(filt_pw_n.values()),
                    hole=0.4, marker=dict(colors=p_colors),
                    textinfo='none', hovertemplate='%{label}: %{value:.4f}<extra></extra>',
                )])
                fig_nav.update_layout(
                    title={'text': f'<b>NAV: ${filt_nav:,.2f}</b>', 'font': {'size': 20}},
                    font={'family': 'Helvetica Neue'}, height=500,
                    margin=dict(l=40, r=40, t=80, b=40),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_nav, use_container_width=True, key="cpv_pie_nav")
                with st.expander("View Data"):
                    pdf = pd.DataFrame({'Portfolio': list(filt_pw_n.keys()), 'Weight': list(filt_pw_n.values()),
                                        'Est. Value ($)': [filt_pw_n[k] * filt_nav for k in filt_pw_n]}).set_index('Portfolio')
                    st.dataframe(pdf.style.format({'Weight': '{:.2%}', 'Est. Value ($)': '${:,.2f}'}), use_container_width=True)

            with col_obj:
                st.markdown("**By Objective**")
                o_colors = [OBJECTIVE_COLORS.get(n, '#666') for n in filt_obj]
                fig_obj = go.Figure(data=[go.Pie(
                    labels=list(filt_obj.keys()), values=list(filt_obj.values()),
                    hole=0.4, marker=dict(colors=o_colors),
                    textinfo='none', hovertemplate='%{label}: %{value:.4f}<extra></extra>',
                )])
                fig_obj.update_layout(
                    title={'text': '<b>Objective Allocation</b>', 'font': {'size': 20}},
                    font={'family': 'Helvetica Neue'}, height=500,
                    margin=dict(l=40, r=40, t=80, b=40),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_obj, use_container_width=True, key="cpv_pie_obj")
                with st.expander("View Data"):
                    odf = pd.DataFrame({'Objective': list(filt_obj.keys()), 'Weight': list(filt_obj.values()),
                                        'Est. Value ($)': [filt_obj[k] * filt_nav for k in filt_obj]}).set_index('Objective')
                    st.dataframe(odf.style.format({'Weight': '{:.2%}', 'Est. Value ($)': '${:,.2f}'}), use_container_width=True)

            # ═══════════════════════════════════════════════════
            # 2. DIVERSIFICATION SCORE
            # ═══════════════════════════════════════════════════
            st.markdown("### 2. Client Exposures")
            div_w = {k.replace('_wgt', ''): v for k, v in ws.items()}
            div_result = calculate_diversification_score(div_w, df_asset_class, df_sector, df_country, df_underlying)
            composite = div_result['composite_score']
            sub_labels = ['Asset Class', 'Geography', 'Sector', 'Concentration']
            sub_keys = ['asset_class', 'geography', 'sector', 'concentration']
            weight_keys = ['asset', 'geo', 'sector', 'conc']
            sub_scores = [div_result[k].get('score') for k in sub_keys]
            sub_w = [WEIGHTS[k] for k in weight_keys]
            s_colors = [DARK_BLUE, PRIMARY_BLUE, SECONDARY_BLUE, TERTIARY_BLUE]

            col_left, col_right = st.columns([2, 3])
            with col_left:
                d_labels = [f'{l}  ({s:.1f})' if s is not None else f'{l}  (N/A)' for l, s in zip(sub_labels, sub_scores)]
                d_values = [w * (s or 0) for w, s in zip(sub_w, sub_scores)]
                fig_donut = go.Figure(data=[go.Pie(
                    labels=d_labels, values=d_values, hole=0.7,
                    marker=dict(colors=s_colors, line=dict(color='white', width=2)),
                    textinfo='none', hovertemplate='%{label}<extra></extra>',
                    sort=False, direction='clockwise',
                )])
                fig_donut.update_layout(
                    showlegend=True,
                    legend=dict(orientation='h', yanchor='top', y=-0.02, xanchor='center', x=0.5, font=dict(size=11, color='#555'),
                                itemsizing='constant', itemwidth=30),
                    height=340, margin=dict(l=5, r=5, t=5, b=60),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    annotations=[dict(
                        text=f'<b style="font-size:40px">{composite:.1f}</b><b style="font-size:16px;color:#172B4D"> /10</b>',
                        x=0.5, y=0.5, font_size=16, showarrow=False, font_color='#172B4D',
                    )],
                )
                st.plotly_chart(fig_donut, use_container_width=True, key="cpv_div_donut")

                ins_key = "cpv_div_insights"
                if ins_key not in st.session_state:
                    try:
                        from utils.llm_helper import generate_diversification_insights
                        with st.spinner("Generating insights..."):
                            st.session_state[ins_key] = generate_diversification_insights(API_KEY, div_result, client_weights)
                    except Exception:
                        st.session_state[ins_key] = []
                CATS = {'strength': ('#E8F5E9', '#2E7D32', '#172B4D'), 'improvement': ('#FFF3E0', '#E65100', '#172B4D'), 'observation': ('#E3F2FD', '#0052CC', '#172B4D')}
                for ins in (st.session_state.get(ins_key) or []):
                    cat = ins.get('category', 'observation')
                    bg, accent, tc = CATS.get(cat, CATS['observation'])
                    st.markdown(
                        f'<div style="background:{bg};border-left:4px solid {accent};border-radius:8px;padding:12px 14px;margin-bottom:8px">'
                        f'<div style="font-weight:700;color:{tc};font-size:13px;margin-bottom:3px">{ins.get("title","")}</div>'
                        f'<div style="color:#333;font-size:12px;line-height:1.4">{ins.get("body","")}</div></div>',
                        unsafe_allow_html=True)

                warnings = div_result.get('data_warnings', [])
                if warnings:
                    items = "".join(f"<li>{w}</li>" for w in warnings)
                    st.markdown(
                        f'<div style="background:#FFF8E1;border-left:4px solid #FFC107;border-radius:8px;padding:12px 14px;margin:8px 0">'
                        f'<div style="font-weight:700;color:#7B6100;font-size:13px;margin-bottom:4px">Data Completeness Issues</div>'
                        f'<ul style="color:#333;font-size:12px;line-height:1.5;margin:0;padding-left:16px">{items}</ul></div>',
                        unsafe_allow_html=True)

            with col_right:
                tab_aa, tab_geo, tab_sec, tab_conc = st.tabs(["Asset Class", "Geography", "Sector", "Concentration"])
                def _render_tab(score, color):
                    if score is not None:
                        st.markdown(f'<div style="text-align:center;padding:8px 0"><span style="font-size:28px;font-weight:800;color:{color}">{score:.1f}</span>'
                                    f'<span style="font-size:14px;color:#999"> / 10</span>'
                                    f'<span style="font-size:12px;color:#888;margin-left:8px">{_score_rating(score)}</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="text-align:center;padding:8px 0"><span style="font-size:28px;font-weight:800;color:#B0B0B0">N/A</span></div>', unsafe_allow_html=True)

                with tab_aa:
                    _render_tab(sub_scores[0], s_colors[0])
                    if 'asset_pie' in charts:
                        st.plotly_chart(charts['asset_pie'], use_container_width=True, key="cpv_tab_aa")
                        with st.expander("View Data"):
                            st.dataframe(charts['asset_table'].style.format("{:.2%}"), use_container_width=True)
                    st.caption(_get_asset_brief(div_result['asset_class']))

                with tab_geo:
                    _render_tab(sub_scores[1], s_colors[1])
                    if 'country_pie' in charts:
                        st.plotly_chart(charts['country_pie'], use_container_width=True, key="cpv_tab_geo")
                        with st.expander("View Data"):
                            st.dataframe(charts['country_table'].style.format("{:.2%}"), use_container_width=True)
                    st.caption(_get_geo_brief(div_result['geography']))

                with tab_sec:
                    _render_tab(sub_scores[2], s_colors[2])
                    if 'sector_pie' in charts:
                        st.plotly_chart(charts['sector_pie'], use_container_width=True, key="cpv_tab_sec")
                        with st.expander("View Data"):
                            st.dataframe(charts['sector_table'].style.format("{:.2%}"), use_container_width=True)
                    st.caption(_get_sector_brief(div_result['sector']))

                with tab_conc:
                    _render_tab(sub_scores[3], s_colors[3])
                    topn = st.selectbox("Show Top", [10, 15, 20], key="cpv_topn")
                    if df_underlying is not None:
                        eq_w = {}
                        for col_name in ws.index:
                            tk = col_name.replace('_wgt', '')
                            if _classify_portfolio_type(tk, df_asset_class) in ('equity', 'reit'):
                                eq_w[col_name] = ws[col_name]
                        eq_s = pd.Series(eq_w) if eq_w else pd.Series(dtype=float)
                        if not eq_s.empty:
                            eq_s = eq_s / eq_s.sum()
                        conc_raw = get_concentration_exposure(eq_s, df_underlying, top_n=topn) if not eq_s.empty else pd.Series(dtype=float)
                        if not conc_raw.empty:
                            t_conc = conc_raw.head(topn).sum()
                            fig_bar = plot_horizontal_bar_chart(conc_raw.head(topn), title=f'Top {topn} Holdings ({t_conc:.1%} Total)')
                            st.plotly_chart(fig_bar, use_container_width=True, key="cpv_tab_conc_bar")
                            with st.expander("View Data"):
                                cdf = conc_raw.head(topn).to_frame(name='Weight')
                                cdf['Est. Value ($)'] = cdf['Weight'] * total_nav
                                st.dataframe(cdf.style.format({'Weight': '{:.2%}', 'Est. Value ($)': '${:,.2f}'}), use_container_width=True)
                    st.caption(_get_conc_brief(div_result['concentration']))

        else:
            st.warning("No data matches the selected filters.")

        st.divider()

        # ═══════════════════════════════════════════════════════
        # 3. GOALS
        # ═══════════════════════════════════════════════════════
        st.markdown("### 3. Goals")

        managed_ports = [p for p in client_weights.keys() if p not in IGNORED_PORTFOLIOS]

        # portfolios already claimed by existing-source goals (custom goals don't lock portfolios)
        _assigned = set()
        for g in st.session_state.cpv_goals:
            if g.get('source', 'existing') == 'existing':
                _assigned.update(g.get('portfolios', []))
        available_ports = [p for p in managed_ports if p not in _assigned]

        # ── existing goal cards ──
        goals = st.session_state.cpv_goals
        if goals:
            CARDS_PER_PAGE = 3
            n_pages = max(1, math.ceil(len(goals) / CARDS_PER_PAGE))
            page = st.session_state.cpv_card_page

            nav_l, nav_title, nav_r = st.columns([1, 6, 1])
            with nav_l:
                if st.button("◀", key="cpv_prev", disabled=page <= 0):
                    st.session_state.cpv_card_page = max(0, page - 1)
                    st.rerun()
            with nav_title:
                st.markdown(f"<div style='text-align:center;color:#888;font-size:13px'>Goals {page * CARDS_PER_PAGE + 1}–{min(len(goals), (page + 1) * CARDS_PER_PAGE)} of {len(goals)}</div>", unsafe_allow_html=True)
            with nav_r:
                if st.button("▶", key="cpv_next", disabled=page >= n_pages - 1):
                    st.session_state.cpv_card_page = min(n_pages - 1, page + 1)
                    st.rerun()

            page_goals = goals[page * CARDS_PER_PAGE:(page + 1) * CARDS_PER_PAGE]
            cols = st.columns(CARDS_PER_PAGE)
            for i, g in enumerate(page_goals):
                with cols[i]:
                    if g.get('source', 'existing') == 'existing':
                        g_nav = sum(client_holdings.get(p, 0) for p in g['portfolios'])
                    else:
                        g_nav = g.get('initial_value', 0) or 0
                    badge_cls = 'goal-badge-growth' if g['type'] == 'Growth' else 'goal-badge-income'
                    source_label = "Existing" if g.get('source', 'existing') == 'existing' else "Planned"
                    source_color = "#666" if source_label == "Existing" else "#6554C0"
                    target_label = f"${g.get('target_value', 0):,.0f}" if g['type'] == 'Growth' else f"${g.get('target_payout', 0):,.0f}/mo"
                    port_pills = "".join(
                        f'<span style="display:inline-block;background:#EAF0FB;color:#0052CC;font-size:11px;'
                        f'padding:2px 8px;border-radius:10px;margin:2px 4px 2px 0;font-weight:500">{p}</span>'
                        for p in g['portfolios']
                    )
                    st.markdown(
                        f'<div class="goal-card">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">'
                        f'<span style="font-weight:700;font-size:16px;color:#172B4D">{g["name"]}</span>'
                        f'<span class="{badge_cls}">{g["type"]}</span></div>'
                        f'<div style="font-size:11px;color:{source_color};margin-bottom:6px;font-weight:500">{source_label}</div>'
                        f'<div style="font-size:13px;color:#666;margin-bottom:12px">{g.get("description","")[:60]}</div>'
                        f'<div style="margin-bottom:12px">{port_pills}</div>'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:8px">'
                        f'<div><div style="font-size:11px;color:#999">Current NAV</div><div style="font-size:18px;font-weight:700;color:#172B4D">${g_nav:,.0f}</div></div>'
                        f'<div style="text-align:right"><div style="font-size:11px;color:#999">Target</div><div style="font-size:18px;font-weight:700;color:#0052CC">{target_label}</div></div></div>'
                        f'<div style="font-size:11px;color:#999;margin-bottom:4px">{len(g["portfolios"])} portfolio(s) · {g.get("time_horizon",20)}y horizon</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    c_open, c_del = st.columns(2)
                    with c_open:
                        if st.button("View Details", key=f"cpv_open_{g['id']}", use_container_width=True):
                            st.session_state.cpv_active_goal_id = g['id']
                            st.rerun()
                    with c_del:
                        if st.button("Delete", key=f"cpv_del_{g['id']}", use_container_width=True):
                            st.session_state.cpv_goals = [x for x in st.session_state.cpv_goals if x['id'] != g['id']]
                            st.rerun()

        st.divider()

        # ── goal creation (two tabs) ──
        st.markdown("#### Create New Goal")
        tab_existing, tab_custom = st.tabs(["From Existing Portfolios", "Plan a New Goal"])

        with tab_existing:
            st.caption("Build a goal from portfolios the client currently holds. Each portfolio can belong to one goal only.")

            # portfolio selection lives OUTSIDE the form so NAV updates reactively
            ex_sel_col, ex_nav_col = st.columns([3, 1])
            with ex_sel_col:
                if available_ports:
                    g_ports = st.multiselect("Select Portfolios", available_ports, key="cpv_ex_ports")
                else:
                    st.warning("All portfolios are already assigned to goals.")
                    g_ports = []
            auto_nav = sum(client_holdings.get(p, 0) for p in g_ports) if g_ports else 0
            with ex_nav_col:
                st.metric("Current Portfolio Value", f"${auto_nav:,.2f}")

            if g_ports:
                pill_spans = []
                for p in g_ports:
                    nav_val = client_holdings.get(p, 0)
                    wgt = (nav_val / auto_nav * 100) if auto_nav > 0 else 0
                    pill_spans.append(
                        f'<span style="display:inline-block;background:#EAF0FB;color:#0052CC;'
                        f'font-size:12px;padding:3px 10px;border-radius:10px;margin:2px 4px 2px 0">'
                        f'{p}: ${nav_val:,.0f} ({wgt:.1f}%)</span>'
                    )
                st.markdown(
                    '<div style="margin-bottom:8px">' + " ".join(pill_spans) + '</div>',
                    unsafe_allow_html=True,
                )

            with st.form("cpv_goal_form_existing", clear_on_submit=True):
                fc1, fc2 = st.columns(2)
                with fc1:
                    g_name = st.text_input("Goal Name", placeholder="e.g. Retirement Fund", key="cpv_ex_name")
                with fc2:
                    g_type = st.selectbox("Goal Type", ["Growth", "Income"], key="cpv_ex_type")
                g_desc = st.text_area("Goal Description", placeholder="Describe your goal...", height=68, key="cpv_ex_desc")

                fc3, fc4, fc5, fc6 = st.columns(4)
                with fc3:
                    g_horizon = st.slider("Time Horizon (Years)", 1, 30, 20, key="cpv_ex_horizon")
                with fc4:
                    if g_type == "Growth":
                        g_target = st.number_input("Target Value ($)", min_value=0, value=500000, step=10000, format="%d", key="cpv_ex_target")
                    else:
                        g_target = st.number_input("Target Payout ($/month)", min_value=0, value=2500, step=100, format="%d", key="cpv_ex_target")
                with fc5:
                    g_lump = st.number_input("Additional Lump Sum ($)", min_value=0, value=0, step=1000, format="%d", key="cpv_ex_lump")
                with fc6:
                    g_monthly = st.number_input("Monthly Top-Up ($)", min_value=0, value=0, step=100, format="%d", key="cpv_ex_monthly")

                submitted = st.form_submit_button("Confirm", type="primary", use_container_width=True)
                if submitted:
                    if not g_name:
                        st.error("Please provide a goal name.")
                    elif not g_ports:
                        st.error("Please select at least one portfolio.")
                    else:
                        new_goal = {
                            'id': str(uuid.uuid4()),
                            'name': g_name,
                            'description': g_desc,
                            'type': g_type,
                            'source': 'existing',
                            'portfolios': g_ports,
                            'time_horizon': g_horizon,
                            'target_value': g_target if g_type == 'Growth' else None,
                            'target_payout': g_target if g_type == 'Income' else None,
                            'initial_value': auto_nav + g_lump,
                            'monthly_topup': g_monthly,
                            'created_at': datetime.now().isoformat(),
                        }
                        st.session_state.cpv_goals.append(new_goal)
                        st.success(f"Goal **{g_name}** created!")
                        st.rerun()

        with tab_custom:
            st.caption("Plan a hypothetical goal from any Syfe managed portfolio, not limited to current holdings.")

            c_ports = st.multiselect("Select Portfolios", ALL_MANAGED_PORTFOLIOS, key="cpv_cu_ports")

            # per-portfolio allocation inputs (outside form for reactivity)
            cu_allocs = {}
            if c_ports:
                st.markdown("**Allocation per Portfolio**")
                alloc_cols = st.columns(min(len(c_ports), 4))
                for idx, p in enumerate(c_ports):
                    with alloc_cols[idx % min(len(c_ports), 4)]:
                        cu_allocs[p] = st.number_input(
                            p, min_value=0, value=10000, step=1000, format="%d",
                            key=f"cpv_cu_alloc_{p}",
                        )
                cu_total = sum(cu_allocs.values())

                # show weight pills
                pill_spans = []
                for p in c_ports:
                    amt = cu_allocs.get(p, 0)
                    wgt = (amt / cu_total * 100) if cu_total > 0 else 0
                    pill_spans.append(
                        f'<span style="display:inline-block;background:#EAF0FB;color:#0052CC;'
                        f'font-size:12px;padding:3px 10px;border-radius:10px;margin:2px 4px 2px 0">'
                        f'{p}: ${amt:,.0f} ({wgt:.1f}%)</span>'
                    )
                st.markdown(
                    '<div style="margin-bottom:4px">' + " ".join(pill_spans) + '</div>'
                    + f'<div style="font-size:13px;font-weight:600;color:#172B4D">Total: ${cu_total:,.0f}</div>',
                    unsafe_allow_html=True,
                )
            else:
                cu_total = 0

            with st.form("cpv_goal_form_custom", clear_on_submit=True):
                cc1, cc2 = st.columns(2)
                with cc1:
                    c_name = st.text_input("Goal Name", placeholder="e.g. Future Income Plan", key="cpv_cu_name")
                with cc2:
                    c_type = st.selectbox("Goal Type", ["Growth", "Income"], key="cpv_cu_type")
                c_desc = st.text_area("Goal Description", placeholder="Describe your goal...", height=68, key="cpv_cu_desc")

                cc3, cc4, cc5 = st.columns(3)
                with cc3:
                    c_horizon = st.slider("Time Horizon (Years)", 1, 30, 20, key="cpv_cu_horizon")
                with cc4:
                    if c_type == "Growth":
                        c_target = st.number_input("Target Value ($)", min_value=0, value=500000, step=10000, format="%d", key="cpv_cu_target")
                    else:
                        c_target = st.number_input("Target Payout ($/month)", min_value=0, value=2500, step=100, format="%d", key="cpv_cu_target")
                with cc5:
                    c_monthly = st.number_input("Monthly Top-Up ($)", min_value=0, value=0, step=100, format="%d", key="cpv_cu_monthly")

                c_submitted = st.form_submit_button("Confirm", type="primary", use_container_width=True)
                if c_submitted:
                    if not c_name:
                        st.error("Please provide a goal name.")
                    elif not c_ports:
                        st.error("Please select at least one portfolio.")
                    elif cu_total <= 0:
                        st.error("Total allocation must be greater than zero.")
                    else:
                        new_goal = {
                            'id': str(uuid.uuid4()),
                            'name': c_name,
                            'description': c_desc,
                            'type': c_type,
                            'source': 'custom',
                            'portfolios': c_ports,
                            'allocations': cu_allocs,
                            'time_horizon': c_horizon,
                            'target_value': c_target if c_type == 'Growth' else None,
                            'target_payout': c_target if c_type == 'Income' else None,
                            'initial_value': cu_total,
                            'monthly_topup': c_monthly,
                            'created_at': datetime.now().isoformat(),
                        }
                        st.session_state.cpv_goals.append(new_goal)
                        st.success(f"Goal **{c_name}** created!")
                        st.rerun()

# ── footer ───────────────────────────────────────────────────
st.divider()
st.caption("""
**Disclaimer:** The insights provided are for informational purposes only and do not constitute financial advice.
Investment decisions should be made based on your individual circumstances and in consultation with a qualified financial advisor.
""")
