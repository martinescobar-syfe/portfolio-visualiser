
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import sys
import os

# local imports
from utils.data_loader import load_static_data, get_data_path, download_data
import port_cons as pc
from utils.optimiser import create_constraints, optimize_portfolio, calc_port_perf
import plotly.express as px
import plotly.graph_objects as go
import zipfile
import io
import requests
from utils.performance import get_performance_summary, calculate_relative_metrics
from utils.exposure import get_exposure_charts, get_concentration_exposure
from utils.plotting import (
    plot_cumulative_returns, plot_drawdowns, plot_correlation_heatmap, 
    plot_scatter_risk_return, plot_allocation_pie, apply_syfe_layout, plot_bar_chart, 
    plot_rolling_volatility, plot_monthly_heatmap
)
from styles.syfe_colors import PALETTE

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from utils.portfolio_storage import save_portfolio, load_portfolios, delete_portfolio

# page config
st.set_page_config(
    page_title="Portfolio Builder",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# password gate
pwd = st.sidebar.text_input("Access Password", type="password")
if pwd != "algo1":
    st.warning("Please enter the correct password to access the Builder.")
    st.stop()

# css (same as app.py)
st.markdown("""
<style>
    /* hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* global */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* headers */
    h1, h2, h3 {
        color: #172B4D;
        font-weight: 700;
    }
    
    /* buttons */
    .stButton>button {
        background-color: #0052CC;
        color: white;
        border-radius: 4px;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2684FF;
        color: white;
        border-radius: 4px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# init
data_path = get_data_path()
df_sec_list, df_asset_class, df_sector, df_country, df_fi_metrics, df_fx, df_daily_ret, df_risk_rating, df_underlying = load_static_data(data_path)

if df_sec_list is None:
    st.stop()

# session state
if 'builder_portfolio' not in st.session_state:
    st.session_state.builder_portfolio = {
        'tickers': [],
        'weights': {},
        'results': None
    }

if 'custom_constraints' not in st.session_state:
    st.session_state.custom_constraints = [
        {'type': 'Country', 'target': 'United States', 'operator': 'Max (<)', 'value': 0.65},
        {'type': 'Country', 'target': 'China', 'operator': 'Equal (=)', 'value': 0.15}
    ]

st.title("Portfolio Builder (WIP)")

col1, col2, col3 = st.columns(3)

# 1. data selection
st.header("1. Universe Selection")

# need this early for load logic
available_tickers = sorted(df_sec_list['Ticker'].tolist())

# load saved settings
with st.expander("Load Saved Template", expanded=False):
    saved_portfolios = load_portfolios()
    if saved_portfolios:
        saved_names = list(saved_portfolios.keys())
        c_buttons = st.columns([1, 1, 6])
        with c_buttons[0]:
            if st.button("Load", key="builder_load_btn"):
                st.session_state.trigger_load = True
        
        with c_buttons[1]:
            if st.button("Delete", key="builder_del_btn"):
                st.session_state.trigger_delete_conf = True
                
        selected_load = st.selectbox("Select Saved Template", saved_names, key="builder_load_select")
        
        if st.session_state.get('trigger_load'):
            st.session_state.trigger_load = False
            data = saved_portfolios[selected_load]
            # restore state
            if 'tickers' in data:
                valid_t = [t for t in data['tickers'] if t in available_tickers]
                st.session_state.builder_universe = valid_t
            
            if 'bm_tickers' in data and 'bm_weights' in data:
                valid_bm = [t for t in data['bm_tickers'] if t in available_tickers]
                if valid_bm:
                    st.session_state.builder_bm_tickers = valid_bm
                    st.session_state.builder_bm_weights = data['bm_weights'][:len(valid_bm)]
                    st.session_state.builder_bm_count = data.get('bm_count', len(valid_bm))
            elif 'benchmark' in data and data['benchmark'] in available_tickers:
                # backwards compat: single benchmark from old templates
                st.session_state.builder_bm_tickers = [data['benchmark']]
                st.session_state.builder_bm_weights = [1.0]
                st.session_state.builder_bm_count = 1

            if 'allocation_type' in data:
                mapping = {
                    'Max sharpe': 'Max Sharpe', 'Max Sharpe': 'Max Sharpe',
                    'Min vol': 'Min Volatility', 'Min Volatility': 'Min Volatility',
                    'Equal weight': 'Equal Weight', 'Equal Weight': 'Equal Weight',
                    'Fixed Weight': 'Fixed Weight'
                }
                mapped_alloc = mapping.get(data['allocation_type'], 'Max Sharpe')
                if mapped_alloc in ["Fixed Weight", "Equal Weight", "Max Sharpe", "Min Volatility"]:
                    st.session_state.builder_alloc_method = mapped_alloc
            
            if 'rebal_freq' in data:
                rebal_map = {
                    'Monthly': 'Monthly', 'Quarterly': 'Quarterly',
                    'Semi-annually': 'Semi-Annual (Apr/Oct)', 'Annually': 'Annual (Apr)',
                    'Semi-Annual (Apr/Oct)': 'Semi-Annual (Apr/Oct)', 'Annual (Apr)': 'Annual (Apr)'
                }
                mapped_freq = rebal_map.get(data['rebal_freq'], 'Annual (Apr)')
                if mapped_freq in ["Semi-Annual (Apr/Oct)", "Annual (Apr)", "Monthly", "Quarterly"]:
                    st.session_state.builder_rebal_freq = mapped_freq
            
            if 'fixed_weights' in data:
                st.session_state.builder_portfolio['fixed_weights'] = data['fixed_weights']
            
            if 'custom_constraints' in data:
                st.session_state.custom_constraints = data['custom_constraints']
            if 'granular_constraints' in data:
                st.session_state.builder_portfolio['granular_constraints'] = data['granular_constraints']

            st.success(f"Loaded '{selected_load}'")
            st.rerun()

        if st.session_state.get('trigger_delete_conf'):
            st.session_state.trigger_delete_conf = False
            if delete_portfolio(selected_load):
                st.success("Deleted!")
                st.rerun()
    else:
        st.info("No saved templates found.")

col1, col2, col3 = st.columns(3)

# filter by asset class (optional)
unique_asset_classes = sorted(df_asset_class.index.unique().tolist()) if df_asset_class is not None else []
available_tickers = sorted(df_sec_list['Ticker'].tolist())

with col1:
    if df_daily_ret is not None and not df_daily_ret.empty:
        latest_data_date = df_daily_ret.index.max().date()
    else:
        latest_data_date = date.today()
        
    start_date = st.date_input("Start Date", value=date(2015, 1, 1))
    end_date = st.date_input("End Date", value=latest_data_date)

with col2:
    if 'builder_universe' not in st.session_state:
        st.session_state.builder_universe = ['CSPX', 'DUHP', 'MOAT', 'EFA', 'KWEB', 'MCHI', 'EIMI', 'DFAT', 'QQQ', 'XDEW']
    
    selected_tickers = st.multiselect(
        "Select Assets for Universe",
        options=available_tickers,
        key="builder_universe"
    )

with col3:
    st.markdown("**Benchmark Configuration**")
    
    # number of benchmark components
    if 'builder_bm_count' not in st.session_state:
        st.session_state.builder_bm_count = 1
    
    num_bm = st.selectbox(
        "Number of Benchmark Components",
        options=[1, 2, 3, 4, 5],
        key="builder_bm_count",
        help="Select 1 for a single benchmark, or 2-5 for a blended benchmark."
    )
    
    # init benchmark tickers/weights in session state
    if 'builder_bm_tickers' not in st.session_state:
        st.session_state.builder_bm_tickers = ['ACWI' if 'ACWI' in available_tickers else available_tickers[0]]
    if 'builder_bm_weights' not in st.session_state:
        st.session_state.builder_bm_weights = [1.0]
    
    # adjust lists to match num_bm
    while len(st.session_state.builder_bm_tickers) < num_bm:
        st.session_state.builder_bm_tickers.append(available_tickers[0])
    while len(st.session_state.builder_bm_weights) < num_bm:
        st.session_state.builder_bm_weights.append(0.0)
    st.session_state.builder_bm_tickers = st.session_state.builder_bm_tickers[:num_bm]
    st.session_state.builder_bm_weights = st.session_state.builder_bm_weights[:num_bm]
    
    bm_tickers_selected = []
    bm_weights_selected = []
    total_bm_w = 0.0
    
    for idx in range(num_bm):
        if num_bm == 1:
            # simple single benchmark — just a selectbox
            default_idx = available_tickers.index(st.session_state.builder_bm_tickers[idx]) if st.session_state.builder_bm_tickers[idx] in available_tickers else 0
            t = st.selectbox(
                "Select Benchmark",
                options=available_tickers,
                index=default_idx,
                key=f"builder_bm_tick_{idx}"
            )
            bm_tickers_selected.append(t)
            bm_weights_selected.append(1.0)
            total_bm_w = 100.0
        else:
            bm_c1, bm_c2 = st.columns([3, 2])
            with bm_c1:
                default_idx = available_tickers.index(st.session_state.builder_bm_tickers[idx]) if st.session_state.builder_bm_tickers[idx] in available_tickers else 0
                t = st.selectbox(
                    f"Benchmark {idx+1}",
                    options=available_tickers,
                    index=default_idx,
                    key=f"builder_bm_tick_{idx}"
                )
            with bm_c2:
                w = st.number_input(
                    f"Weight {idx+1} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.builder_bm_weights[idx] * 100,
                    step=5.0,
                    key=f"builder_bm_wgt_{idx}",
                    help="Fixed weight for this benchmark component. All benchmark weights must sum to 100%. These are fixed weights that remain constant throughout the backtest."
                )
            bm_tickers_selected.append(t)
            bm_weights_selected.append(w / 100.0)
            total_bm_w += w
    
    # update session state
    st.session_state.builder_bm_tickers = bm_tickers_selected
    st.session_state.builder_bm_weights = bm_weights_selected
    
    if num_bm > 1:
        if abs(total_bm_w - 100.0) > 0.01:
            st.warning(f"Benchmark weights: {total_bm_w:.1f}% (must sum to 100%)")
        else:
            st.success(f"Benchmark weights: {total_bm_w:.1f}%")
    
    # single string label for display throughout
    if num_bm == 1:
        benchmark_label = bm_tickers_selected[0]
    else:
        parts = [f"{t} {w:.0%}" for t, w in zip(bm_tickers_selected, bm_weights_selected)]
        benchmark_label = "Blended (" + " / ".join(parts) + ")"

# 2. construction settings
st.header("2. Portfolio Construction")

c_strat, c_rebal = st.columns(2)

with c_strat:
    allocation_method = st.selectbox(
        "Allocation Method",
        ["Fixed Weight", "Equal Weight", "Max Sharpe", "Min Volatility"],
        key="builder_alloc_method"
    )

with c_rebal:
    rebal_freq = st.selectbox(
        "Rebalancing Frequency",
        ["Semi-Annual (Apr/Oct)", "Annual (Apr)", "Monthly", "Quarterly"],
        key="builder_rebal_freq"
    )

if allocation_method == "Fixed Weight":
    st.subheader("Fixed Weights Definition")
    if not selected_tickers:
        st.info("Select assets in 'Universe Selection' first.")
    else:
        if 'fixed_weights' not in st.session_state.builder_portfolio:
            st.session_state.builder_portfolio['fixed_weights'] = {t: 1.0/len(selected_tickers) for t in selected_tickers}
            
        # add any new tickers
        for t in selected_tickers:
            if t not in st.session_state.builder_portfolio['fixed_weights']:
                 st.session_state.builder_portfolio['fixed_weights'][t] = 0.0
        fw_cols = st.columns(3)
        total_w = 0.0
        new_weights = {}
        for i, t in enumerate(selected_tickers):
            with fw_cols[i % 3]:
                val = st.number_input(
                    f"{t} Weight (%)", 
                    0.0, 100.0, 
                    st.session_state.builder_portfolio['fixed_weights'].get(t, 0.0) * 100, 
                    step=1.0,
                    key=f"fw_{t}"
                )
                new_weights[t] = val / 100.0
                total_w += val
        
        st.session_state.builder_portfolio['fixed_weights'] = new_weights
        
        if abs(total_w - 100.0) > 0.01:
            st.warning(f"Total Weight: {total_w:.2f}%. Should sum to 100%.")
        else:
            st.success(f"Total Weight: {total_w:.2f}%")

# 3. constraints
st.subheader("Constraints")

with st.expander("Define Constraints", expanded=True):
    st.markdown("#### per-ETF Constraints")
    
    if not selected_tickers:
        st.info("Select assets in 'Universe Selection' first.")
    else:
        # init granular constraints if needed
        if 'granular_constraints' not in st.session_state.builder_portfolio or not st.session_state.builder_portfolio['granular_constraints']:
            defaults = {
                'CSPX': {'op': 'Equal (=)', 'val': 0.20}
            }
            st.session_state.builder_portfolio['granular_constraints'] = defaults
        grp_constraints = {}
        h1, h2, h3 = st.columns([1, 1, 1])
        h1.markdown("**Ticker**")
        h2.markdown("**Operator**")
        h3.markdown("**Value (%)**")
        
        for t in selected_tickers:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.markdown(f"**{t}**")
            current_c = st.session_state.builder_portfolio['granular_constraints'].get(t, {'op': 'None', 'val': 0.0})
            
            with c2:
                op = st.selectbox(
                    f"Operator ({t})", 
                    ["None", "Max (<)", "Min (>)", "Equal (=)"],
                    index=["None", "Max (<)", "Min (>)", "Equal (=)"].index(current_c['op']),
                    label_visibility="collapsed"
                )
            with c3:
                val = st.number_input(
                    f"Val ({t})", 
                    0.0, 100.0, 
                    current_c['val'] * 100, 
                    step=1.0,
                    label_visibility="collapsed"
                ) / 100.0
            
            if op != "None":
                grp_constraints[t] = {'op': op, 'val': val}
        
        st.session_state.builder_portfolio['granular_constraints'] = grp_constraints

    st.markdown("#### Group Constraints (Sector / Country / Tracking Error)")
    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 1])
    
    with cc1:
        const_type = st.selectbox("Type", ["Sector", "Country", "Tracking Error"])
        
    with cc2:
        if const_type == "Sector":
            options = sorted([str(x) for x in df_sector.index.unique().tolist() if pd.notna(x)])
            const_target = st.selectbox("Target", options)
        elif const_type == "Country":
            options = sorted([str(x) for x in df_country.index.unique().tolist() if pd.notna(x)])
            const_target = st.selectbox("Target", options)
        else:
            st.selectbox(
                "Target", 
                ["vs Selected Benchmark"], 
                disabled=True,
                help="Tracking Error is calculated relative to the benchmark selected in Universe Selection."
            )
            const_target = "Benchmark"
        
    with cc3:
        if const_type == "Tracking Error":
            const_operator = "Max (<)"
            st.selectbox("Operator", ["Max (<)"], disabled=True)
            const_val = st.number_input(
                "Max TE (%)", 0.0, 50.0, 5.0, step=1.0,
                help="TE constraint uses: TE = √(Var(Portfolio - Benchmark) × 252). Calculated from historical covariance at each rebalance."
            ) / 100.0
        else:
            const_operator = st.selectbox("Operator", ["Max (<)", "Min (>)", "Equal (=)"]) 
            const_val = st.number_input("Value (%)", 0.0, 100.0, 20.0, step=5.0) / 100.0
        
    with cc4:
        st.write("")
        st.write("")
        if st.button("Add"):
            st.session_state.custom_constraints.append({
                "type": const_type,
                "target": const_target,
                "operator": const_operator,
                "value": const_val
            })
    if st.session_state.custom_constraints:
        st.markdown("**Active Group Constraints:**")
        for i, c in enumerate(st.session_state.custom_constraints):
            col_txt, col_del = st.columns([8, 1])
            col_txt.text(f"{c['type']} - {c['target']} {c['operator']} {c['value']:.1%}")
            if col_del.button("❌", key=f"del_const_{i}"):
                st.session_state.custom_constraints.pop(i)
                st.rerun()

# 4. optimisation
st.markdown("---")
has_constraints = (
    st.session_state.custom_constraints or 
    (st.session_state.builder_portfolio.get('granular_constraints') and 
     any(c['op'] != 'None' for c in st.session_state.builder_portfolio['granular_constraints'].values()))
)

if has_constraints:
    if allocation_method not in ["Max Sharpe", "Min Volatility"]:
        st.warning(f"Constraints are ignored when using '{allocation_method}'. Switch to 'Max Sharpe' or 'Min Volatility' to optimise with constraints.")

if st.button("🚀 Build & Optimise Portfolio", type="primary", use_container_width=True):
    if not selected_tickers:
        st.error("Please select at least one asset.")
        st.stop()
        
    with st.spinner("Optimising..."):
        # 1. prep data
        currency = "USD"
        rebal_map = {
            "Semi-Annual (Apr/Oct)": [4, 10],
            "Annual (Apr)": [4],
            "Quarterly": [1, 4, 7, 10],
            "Monthly": list(range(1, 13))
        }
        rebal_months = rebal_map.get(rebal_freq, [4, 10])
        all_tickers_req = list(set(selected_tickers + bm_tickers_selected))
        
        ret_all, ret_bm_raw = download_data(
            data_source="Bloomberg",
            start_date=start_date,
            end_date=end_date,
            all_tickers=all_tickers_req,
            bm_tickers=bm_tickers_selected,
            currency=currency,
            df_sec_list=df_sec_list,
            df_fx=df_fx,
            data_path=data_path,
            df_daily_ret=df_daily_ret
        )
        
        # compute blended benchmark returns
        if num_bm == 1:
            ret_bm = ret_bm_raw[[bm_tickers_selected[0]]].copy()
            ret_bm.columns = [benchmark_label]
        else:
            bm_weights_arr = np.array(bm_weights_selected)
            valid_bm_cols = [t for t in bm_tickers_selected if t in ret_bm_raw.columns]
            if len(valid_bm_cols) == len(bm_tickers_selected):
                blended_ret = (ret_bm_raw[bm_tickers_selected] * bm_weights_arr).sum(axis=1)
                ret_bm = blended_ret.to_frame(name=benchmark_label)
            else:
                st.error(f"Missing benchmark tickers in data: {set(bm_tickers_selected) - set(valid_bm_cols)}")
                st.stop()
        
        if ret_all.empty:
            st.error("No data found.")
            st.stop()
            
        # 2. build constraints
        valid_tickers = [t for t in selected_tickers if t in ret_all.columns]
        extra_constraints = []
        
        def make_etf_constraint(idx, op, val):
            if op == "Max (<)": # w[i] <= val -> val - w[i] >= 0
                return {'type': 'ineq', 'fun': lambda w, i=idx, v=val: v - w[i]}
            elif op == "Min (>)": # w[i] >= val -> w[i] - val >= 0
                return {'type': 'ineq', 'fun': lambda w, i=idx, v=val: w[i] - v}
            elif op == "Equal (=)":
                return {'type': 'eq', 'fun': lambda w, i=idx, v=val: w[i] - v}
        if 'granular_constraints' in st.session_state.builder_portfolio:
            for t, c in st.session_state.builder_portfolio['granular_constraints'].items():
                if t in valid_tickers and c['op'] != "None":
                    t_idx = valid_tickers.index(t)
                    extra_constraints.append(make_etf_constraint(t_idx, c['op'], c['val']))
        
        # group constraints (sector/country) - weighted exposure dot product
        def make_group_constraint(exposure_vec, op, limit):
            """weighted exposure constraint"""
            e = np.array(exposure_vec)
            if op == "Max (<)":  # Σ(w[i] * exp[i]) <= limit → limit - dot(w, exp) >= 0
                return {'type': 'ineq', 'fun': lambda w, exp=e, lim=limit: lim - np.dot(w, exp)}
            elif op == "Min (>)":  # Σ(w[i] * exp[i]) >= limit → dot(w, exp) - limit >= 0
                return {'type': 'ineq', 'fun': lambda w, exp=e, lim=limit: np.dot(w, exp) - lim}
            elif op == "Equal (=)":  # Σ(w[i] * exp[i]) == limit
                return {'type': 'eq', 'fun': lambda w, exp=e, lim=limit: np.dot(w, exp) - lim}

        st.session_state['constraint_debug'] = []
        st.session_state['constraint_details'] = []
        
        for c in st.session_state.custom_constraints:
            exposure_row = None
            constraint_info = {
                'type': c['type'],
                'target': c['target'],
                'operator': c['operator'],
                'limit': c['value'],
                'status': 'PENDING',
                'exposures': {},
                'feasible_range': None,
                'warning': None
            }
            
            if c['type'] == 'Sector':
                if not df_sector.empty and c['target'] in df_sector.index:
                    exposure_row = df_sector.loc[c['target']]
                else:
                    constraint_info['status'] = 'ERROR'
                    constraint_info['warning'] = f"Sector '{c['target']}' not found in data"

            elif c['type'] == 'Country':
                if not df_country.empty and c['target'] in df_country.index:
                    exposure_row = df_country.loc[c['target']]
                else:
                    constraint_info['status'] = 'ERROR'
                    constraint_info['warning'] = f"Country '{c['target']}' not found in data"
            
            elif c['type'] == 'Tracking Error':
                try:
                    ret_subset = ret_all[valid_tickers].dropna()
                    cov_matrix = ret_subset.cov().values
                    bm_ret = ret_bm.reindex(ret_subset.index).iloc[:, 0].dropna()
                    common_idx = ret_subset.index.intersection(bm_ret.index)
                    ret_subset = ret_subset.loc[common_idx]
                    bm_ret = bm_ret.loc[common_idx]
                    
                    # covar of each asset with benchmark
                    asset_bm_cov = np.array([ret_subset[t].cov(bm_ret) for t in valid_tickers])
                    bm_var = bm_ret.var()
                    
                    limit = c['value']
                    
                    def te_constraint(w, cov=cov_matrix, asset_bm_c=asset_bm_cov, bm_v=bm_var, lim=limit):
                        """te constraint"""
                        port_var = w @ cov @ w
                        cross_term = 2 * np.dot(w, asset_bm_c)
                        active_var = max(port_var - cross_term + bm_v, 0)
                        te = np.sqrt(active_var * 252)
                        return lim - te
                    
                    extra_constraints.append({'type': 'ineq', 'fun': te_constraint})
                    constraint_info['status'] = 'OK'
                    constraint_info['exposures'] = {'Tracking Error Target': limit}
                    
                except Exception as e:
                    constraint_info['status'] = 'ERROR'
                    constraint_info['warning'] = f"Failed to build TE constraint: {str(e)}"
            
            if exposure_row is not None:
                exposure_vec = np.array([float(exposure_row.get(t, 0)) for t in valid_tickers])
                
                # normalize % to decimal if needed
                if exposure_vec.max() > 1:
                    exposure_vec = exposure_vec / 100
                
                for i, t in enumerate(valid_tickers):
                    constraint_info['exposures'][t] = exposure_vec[i]
                
                # feasibility check
                min_possible = exposure_vec.min()
                max_possible = exposure_vec.max()
                constraint_info['feasible_range'] = (min_possible, max_possible)
                target_val = c['value']
                if c['operator'] == "Max (<)" and target_val < min_possible:
                    constraint_info['warning'] = f"INFEASIBLE: All ETFs have >{target_val:.1%} exposure (min={min_possible:.1%})"
                    constraint_info['status'] = 'WARNING'
                elif c['operator'] == "Min (>)" and target_val > max_possible:
                    constraint_info['warning'] = f"INFEASIBLE: No ETF has >{target_val:.1%} exposure (max={max_possible:.1%})"
                    constraint_info['status'] = 'WARNING'
                elif c['operator'] == "Equal (=)" and (target_val < min_possible or target_val > max_possible):
                    constraint_info['warning'] = f"INFEASIBLE: Target {target_val:.1%} outside range [{min_possible:.1%}, {max_possible:.1%}]"
                    constraint_info['status'] = 'WARNING'
                else:
                    constraint_info['status'] = 'OK'
                
                extra_constraints.append(make_group_constraint(exposure_vec, c['operator'], c['value']))
            
            st.session_state['constraint_details'].append(constraint_info)
        
        warnings_list = [ci for ci in st.session_state['constraint_details'] if ci.get('warning')]
        if warnings_list:
            warning_msgs = [f"{ci['type']} {ci['target']}: {ci['warning']}" for ci in warnings_list]
            st.warning("Constraint Feasibility Issues:\n" + "\n".join(warning_msgs))


        base_constraints = create_constraints(0.0, 1.0) 
        full_constraints = base_constraints + extra_constraints
        
        # 3. optimise
        offset_days = 90
        start_opt_date = ret_all.index.min() + timedelta(days=offset_days)
        start_opt_date = max(start_opt_date, pd.Timestamp(start_date))

        try:
            if allocation_method == "Max Sharpe":
                weights_df = pc.max_sharpe_rebal_wgt(ret_all[valid_tickers], start_date=start_opt_date, end_date=end_date, constraints_dict=full_constraints, rebal_months=rebal_months)
            elif allocation_method == "Min Volatility":
                weights_df = pc.min_vol_rebal_wgt(ret_all[valid_tickers], start_date=start_opt_date, end_date=end_date, constraints_dict=full_constraints, rebal_months=rebal_months)
            elif allocation_method == "Equal Weight":
                 weights_df = pc.equal_weight_rebal_wgt(ret_all[valid_tickers], start_date=start_opt_date, end_date=end_date, rebal_months=rebal_months)
            elif allocation_method == "Fixed Weight":
                user_weights = st.session_state.builder_portfolio.get('fixed_weights', {})
                user_weights = {k: v for k, v in user_weights.items() if k in valid_tickers}
                if abs(sum(user_weights.values()) - 1.0) > 0.001:
                     total = sum(user_weights.values())
                     if total > 0: user_weights = {k: v/total for k, v in user_weights.items()}
                weights_df = pc.fixed_weight_rebal_wgt(ret_all[valid_tickers], start_date=start_opt_date, end_date=end_date, fixed_weights=user_weights, rebal_months=rebal_months)

            # validate constraints against first rebal weights
            if allocation_method in ["Max Sharpe", "Min Volatility"] and extra_constraints:
                first_rebal = weights_df[weights_df['reb_flag'] == True].iloc[0]
                w_vec = np.array([first_rebal[f"{t}_wgt"] for t in valid_tickers])
                
                violations = []
                tol = 0.02
                if 'granular_constraints' in st.session_state.builder_portfolio:
                    for t, c in st.session_state.builder_portfolio['granular_constraints'].items():
                        if t in valid_tickers and c['op'] != "None":
                            t_idx = valid_tickers.index(t)
                            actual = w_vec[t_idx]
                            target = c['val']
                            if c['op'] == "Equal (=)" and abs(actual - target) > tol:
                                violations.append(f"ETF {t}: Expected {target:.1%}, Got {actual:.1%}")
                            elif c['op'] == "Max (<)" and actual > target + tol:
                                violations.append(f"ETF {t}: Expected <{target:.1%}, Got {actual:.1%}")
                            elif c['op'] == "Min (>)" and actual < target - tol:
                                violations.append(f"ETF {t}: Expected >{target:.1%}, Got {actual:.1%}")
                
                for c in st.session_state.custom_constraints:
                    exposure_row = None
                    if c['type'] == 'Sector' and not df_sector.empty and c['target'] in df_sector.index:
                        exposure_row = df_sector.loc[c['target']]
                    elif c['type'] == 'Country' and not df_country.empty and c['target'] in df_country.index:
                        exposure_row = df_country.loc[c['target']]
                    elif c['type'] == 'Tracking Error':
                        try:
                            ret_subset = ret_all[valid_tickers].dropna()
                            cov_matrix = ret_subset.cov().values
                            bm_ret = ret_bm.reindex(ret_subset.index).iloc[:, 0].dropna()
                            common_idx = ret_subset.index.intersection(bm_ret.index)
                            ret_subset = ret_subset.loc[common_idx]
                            bm_ret = bm_ret.loc[common_idx]
                            
                            asset_bm_cov = np.array([ret_subset[t].cov(bm_ret) for t in valid_tickers])
                            bm_var = bm_ret.var()
                            
                            port_var = w_vec @ cov_matrix @ w_vec
                            cross_term = 2 * np.dot(w_vec, asset_bm_cov)
                            active_var = max(port_var - cross_term + bm_var, 0)
                            actual_te = np.sqrt(active_var * 252)
                            target_te = c['value']
                            
                            st.session_state['constraint_debug'].append(
                                f"VALIDATION Tracking Error: Target=Max {target_te:.1%}, Actual={actual_te:.1%}"
                            )
                            
                            if actual_te > target_te + tol:
                                violations.append(f"Tracking Error: Expected <{target_te:.1%}, Got {actual_te:.1%}")
                        except Exception as e:
                            violations.append(f"Tracking Error validation failed: {str(e)}")
                        continue
                    
                    if exposure_row is not None:
                        exp_vec = np.array([float(exposure_row.get(t, 0)) for t in valid_tickers])
                        if exp_vec.max() > 1:
                            exp_vec = exp_vec / 100
                        
                        actual_exposure = np.dot(w_vec, exp_vec)
                        target = c['value']
                        
                        st.session_state['constraint_debug'].append(
                            f"VALIDATION {c['type']} {c['target']}: Target={c['operator']}{target:.1%}, Actual={actual_exposure:.1%}"
                        )
                        
                        if c['operator'] == "Equal (=)" and abs(actual_exposure - target) > tol:
                            violations.append(f" {c['type']} {c['target']}: Expected {target:.1%}, Got {actual_exposure:.1%}")
                        elif c['operator'] == "Max (<)" and actual_exposure > target + tol:
                            violations.append(f"{c['type']} {c['target']}: Expected <{target:.1%}, Got {actual_exposure:.1%}")
                        elif c['operator'] == "Min (>)" and actual_exposure < target - tol:
                            violations.append(f"{c['type']} {c['target']}: Expected >{target:.1%}, Got {actual_exposure:.1%}")
                
                if violations:
                    st.warning("Constraint Warnings:\n" + "\n".join(violations))
            
            perf_df = calc_port_perf(ret_all[valid_tickers], weights_df, name_prefix="Builder")
            
            res_struct = {
                'results': {
                    "Builder Portfolio": {
                        'weights': weights_df,
                        'perf': perf_df,
                        'tickers': valid_tickers
                    }
                },
                'ret_all': ret_all,
                'ret_bm': ret_bm,
                'currency': currency,                  # base currency used for the optimisation
            }
            
            st.session_state.builder_portfolio['results'] = res_struct
            st.success("Optimisation Complete!")
            
        except Exception as e:
            st.error(f"Optimisation failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# save config
st.markdown("---")
with st.expander("💾 Save Configuration as Template", expanded=False):
    save_name = st.text_input("Template Name", value="My Strategy", key="builder_save_name")
    if st.button("Save Template", key="builder_save_btn"):
        save_data = {
            'tickers': selected_tickers,
            'benchmark': bm_tickers_selected[0],  # backwards compat
            'bm_tickers': bm_tickers_selected,
            'bm_weights': bm_weights_selected,
            'bm_count': num_bm,
            'allocation_type': allocation_method,
            'rebal_freq': rebal_freq,
            'fixed_weights': st.session_state.builder_portfolio.get('fixed_weights', {}),
            'custom_constraints': st.session_state.custom_constraints,
            'granular_constraints': st.session_state.builder_portfolio.get('granular_constraints', {})
        }
        
        if save_portfolio(save_name, save_data):
            st.success(f"Saved '{save_name}'!")
            st.rerun()
        else:
            st.error("Failed to save.")


# helpers
@st.cache_data
def load_ff_data():
    try:
        ff_file_path = os.path.join(os.getcwd(), 'Data', 'F-F_Research_Data_5_Factors_2x3_daily.csv')
        if os.path.exists(ff_file_path):
            df_ff = pd.read_csv(ff_file_path, skiprows=3, index_col=0)
            df_ff = df_ff[pd.to_numeric(df_ff.index, errors='coerce').notna()]
            df_ff = df_ff.astype(float) / 100
            df_ff.index = pd.to_datetime(df_ff.index, format='%Y%m%d')
            return df_ff
        else:
            return None
    except:
        return None

# helper function for excel export
def _safe_sheet_name(name, used_names=None):
    """Excel sheet names: max 31 chars, no []:*?/\\ characters.

    If ``used_names`` (a mutable set) is provided, the result is also
    de-duplicated by appending ``_1``, ``_2``, ... when a clash would occur.
    The chosen name is added to ``used_names`` for the next call.
    """
    bad = ['[', ']', ':', '*', '?', '/', '\\']
    cleaned = name
    for b in bad:
        cleaned = cleaned.replace(b, '_')
    cleaned = cleaned[:31]

    if used_names is None:
        return cleaned

    base = cleaned
    suffix_n = 1
    while cleaned in used_names:
        suffix = f"_{suffix_n}"
        cleaned = base[: 31 - len(suffix)] + suffix
        suffix_n += 1
    used_names.add(cleaned)
    return cleaned

def create_builder_excel(
    results,
    all_port_ret,
    ret_bm,
    currency,
    top_holdings_df=None,
    sector_df=None,
    country_df=None,
    overlap_matrix=None,
    config=None,
    constraints_log=None,
    benchmark_info=None,
    universe_df=None,
    fi_analytics_df=None,
    asset_class_df=None,
    factor_df=None,
    df_underlying=None,
    ret_all_for_risk=None,
    bm_weights_series=None,
    df_sector_for_brinson=None,
    df_country_for_brinson=None,
):
    """Create a comprehensive Excel report covering optimisation inputs, weights, and analytics.

    Sheets are grouped roughly into:
        Inputs   -> Configuration, Universe, Constraints, Benchmark Composition
        Weights  -> per-portfolio Daily Weights, Rebalance Weights, Turnover
        Returns  -> NAV, Daily/Monthly/Calendar/Cumulative Returns
        Risk     -> Vol, Drawdowns, Drawdown Summary, Correlation, Distribution Stats
        Vs Bench -> Relative Metrics, Rolling Excess Return, Rolling TE, Up/Down Capture
        Exposure -> Top Holdings, Sector, Country, Asset Class, HHI, Overlap, FI, Factor
    """
    output = io.BytesIO()

    # local de-duplicating sheet-name helper -- shadows the module function so
    # every call within this writer auto-tracks names used so far
    _used_sheet_names = set()
    def _sheet(name):
        return _safe_sheet_name(name, _used_sheet_names)

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # ---------- INPUTS ----------
        # configuration sheet
        if config is not None:
            cfg_rows = []
            for k, v in config.items():
                cfg_rows.append({'Parameter': k, 'Value': str(v)})
            pd.DataFrame(cfg_rows).to_excel(writer, sheet_name='Configuration', index=False)

        # universe
        if universe_df is not None and not universe_df.empty:
            universe_df.to_excel(writer, sheet_name='Universe', index=False)

        # constraints log
        if constraints_log is not None and len(constraints_log) > 0:
            pd.DataFrame(constraints_log).to_excel(writer, sheet_name='Constraints', index=False)

        # benchmark composition
        if benchmark_info is not None:
            pd.DataFrame(benchmark_info).to_excel(writer, sheet_name='Benchmark Composition', index=False)

        # ---------- WEIGHTS (per portfolio) ----------
        for port_name, port_data in results.items():
            if 'weights' not in port_data:
                continue
            w_df = port_data['weights'].copy()
            has_flag = 'reb_flag' in w_df.columns

            # daily weights (drift between rebals)
            w_daily = w_df.copy()
            w_daily.columns = [c[:-4] if isinstance(c, str) and c.endswith('_wgt') else c for c in w_daily.columns]
            w_daily.index.name = 'Date'
            w_daily.to_excel(writer, sheet_name=_sheet(f'Weights Daily - {port_name}'))

            # rebalance weights only
            if has_flag:
                w_rebal = w_df[w_df['reb_flag'] == True].drop(columns=['reb_flag'])
                w_rebal.columns = [c[:-4] if isinstance(c, str) and c.endswith('_wgt') else c for c in w_rebal.columns]
                w_rebal.index.name = 'Date'
                w_rebal.to_excel(writer, sheet_name=_sheet(f'Weights Rebal - {port_name}'))

                # turnover at each rebalance: 0.5 * sum(|w_new - w_drifted_prev|)
                # we approximate w_drifted_prev as the daily row immediately before each rebalance
                turnover_rows = []
                reb_idx = w_df.index[w_df['reb_flag'] == True]
                for i, rd in enumerate(reb_idx):
                    if i == 0:
                        turnover_rows.append({'Date': rd, 'Two-Sided Turnover': np.nan})
                        continue
                    prev_loc = w_df.index.get_loc(rd) - 1
                    if prev_loc < 0:
                        turnover_rows.append({'Date': rd, 'Two-Sided Turnover': np.nan})
                        continue
                    w_new = w_df.loc[rd].drop('reb_flag', errors='ignore')
                    w_prev = w_df.iloc[prev_loc].drop('reb_flag', errors='ignore')
                    common = w_new.index.intersection(w_prev.index)
                    to_val = 0.5 * np.abs(w_new[common].values - w_prev[common].values).sum()
                    turnover_rows.append({'Date': rd, 'Two-Sided Turnover': float(to_val)})
                if turnover_rows:
                    pd.DataFrame(turnover_rows).to_excel(
                        writer, sheet_name=_sheet(f'Turnover - {port_name}'), index=False
                    )

        # ---------- RETURNS ----------
        nav_df = (1 + all_port_ret).cumprod()
        nav_df.index.name = 'Date'
        nav_df.to_excel(writer, sheet_name=_sheet(f'NAV ({currency})'))

        daily_ret = all_port_ret.copy()
        daily_ret.index.name = 'Date'
        daily_ret.to_excel(writer, sheet_name=_sheet(f'Daily Returns ({currency})'))

        cum_ret = (1 + all_port_ret).cumprod() - 1
        cum_ret.index.name = 'Date'
        cum_ret.to_excel(writer, sheet_name=_sheet(f'Cumulative Returns ({currency})'))

        monthly_ret = all_port_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_ret.index.name = 'Date'
        monthly_ret.to_excel(writer, sheet_name='Monthly Returns')

        yearly_ret = all_port_ret.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        yearly_ret.index = yearly_ret.index.year
        yearly_ret.index.name = 'Year'
        yearly_ret.to_excel(writer, sheet_name='Calendar Returns')

        # ---------- RISK ----------
        summary = get_performance_summary(all_port_ret)
        summary.to_excel(writer, sheet_name='Key Summary')

        rolling_vol = all_port_ret.rolling(window=252).std(ddof=1) * np.sqrt(252)
        rolling_vol.index.name = 'Date'
        rolling_vol.to_excel(writer, sheet_name='Rolling Volatility')

        cum_ret_nav = (1 + all_port_ret).cumprod()
        running_max = cum_ret_nav.cummax()
        drawdowns = (cum_ret_nav - running_max) / running_max
        drawdowns.index.name = 'Date'
        drawdowns.to_excel(writer, sheet_name='Drawdowns')

        # NOTE: Top-N drawdown episodes are exported via the helper-based
        # `Drawdown Top5` sheet further below (see ADVANCED RISK section),
        # which uses `compute_drawdown_episodes` from utils.risk_analytics
        # and is consistent with the Risk tab in the UI.

        corr_matrix = all_port_ret.corr()
        corr_matrix.to_excel(writer, sheet_name='Correlation Matrix')

        # distribution stats: CAGR (matches Key Summary), vol, skew, kurt, VaR/CVaR (95% and 99%, historical)
        dist_rows = {}
        for col in all_port_ret.columns:
            s = all_port_ret[col].dropna()
            if s.empty:
                continue
            cum_ret = float((1 + s).prod() - 1)
            n_years = len(s) / 252.0
            cagr = (1 + cum_ret) ** (1 / n_years) - 1 if n_years > 0 else float('nan')
            ann_vol = float(s.std(ddof=1) * np.sqrt(252))
            dist_rows[col] = {
                'Annualized Return': float(cagr),
                'Annualized Volatility': ann_vol,
                'Sharpe Ratio': (cagr / ann_vol) if ann_vol > 0 else float('nan'),
                'Skewness': float(s.skew()),
                'Excess Kurtosis': float(s.kurt()),
                'Daily VaR 95%': float(s.quantile(0.05)),
                'Daily CVaR 95%': float(s[s <= s.quantile(0.05)].mean()) if (s <= s.quantile(0.05)).any() else np.nan,
                'Daily VaR 99%': float(s.quantile(0.01)),
                'Daily CVaR 99%': float(s[s <= s.quantile(0.01)].mean()) if (s <= s.quantile(0.01)).any() else np.nan,
                'Best Day': float(s.max()),
                'Worst Day': float(s.min()),
            }
        if dist_rows:
            pd.DataFrame(dist_rows).to_excel(writer, sheet_name='Distribution Stats')

        # ---------- VS BENCHMARK ----------
        if ret_bm is not None and not ret_bm.empty:
            bm_nav = (1 + ret_bm).cumprod()
            bm_nav.index.name = 'Date'
            bm_nav.to_excel(writer, sheet_name=_sheet(f'Benchmark NAV ({currency})'))

            common_idx_bm = all_port_ret.index.intersection(ret_bm.index)
            if len(common_idx_bm) > 0 and ret_bm.shape[1] >= 1:
                bm_series = ret_bm.iloc[:, 0].loc[common_idx_bm]

                # relative metrics per portfolio
                rel_rows = {}
                for col in all_port_ret.columns:
                    p_series = all_port_ret[col].loc[common_idx_bm]
                    try:
                        rel_metrics, _roll = calculate_relative_metrics(p_series, bm_series)
                        rel_rows[col] = rel_metrics
                    except Exception:
                        pass
                if rel_rows:
                    pd.DataFrame(rel_rows).to_excel(writer, sheet_name='Relative Metrics')

                # rolling excess return (geometric, compound 12M) + rolling tracking error
                roll_excess = pd.DataFrame(index=common_idx_bm)
                roll_te = pd.DataFrame(index=common_idx_bm)
                bm_log = np.log1p(bm_series)
                bm_rolling_log = bm_log.rolling(252).sum()
                for col in all_port_ret.columns:
                    p_series = all_port_ret[col].loc[common_idx_bm]
                    p_log = np.log1p(p_series)
                    # geometric 12M excess: (∏(1+p))/(∏(1+b)) - 1
                    roll_excess[col] = np.exp(p_log.rolling(252).sum()) / np.exp(bm_rolling_log) - 1.0
                    active = p_series - bm_series
                    roll_te[col] = active.rolling(252).std(ddof=1) * np.sqrt(252)
                roll_excess.index.name = 'Date'
                roll_te.index.name = 'Date'
                roll_excess.to_excel(writer, sheet_name='Rolling Excess Return 12M')
                roll_te.to_excel(writer, sheet_name='Rolling Tracking Error 12M')

                # NOTE: Up/Down capture is exported via the geometric helper-based
                # `Capture & Hit Ratios` sheet further below (see ADVANCED RISK section).
                # The legacy arithmetic-mean implementation here is removed to keep the
                # workbook internally consistent with the UI, which uses geometric capture.

        # ---------- EXPOSURE / ANALYTICS ----------
        if top_holdings_df is not None and not top_holdings_df.empty:
            top_holdings_df.to_excel(writer, sheet_name='Top Holdings')

        if sector_df is not None and not sector_df.empty:
            sector_df.to_excel(writer, sheet_name='Sector Allocation')

        if country_df is not None and not country_df.empty:
            country_df.to_excel(writer, sheet_name='Country Allocation')

        if asset_class_df is not None and not asset_class_df.empty:
            asset_class_df.to_excel(writer, sheet_name='Asset Class Allocation')

        # HHI / effective N per portfolio (using latest rebalance weights)
        hhi_rows = {}
        for port_name, port_data in results.items():
            if 'weights' not in port_data:
                continue
            w_df = port_data['weights']
            if 'reb_flag' in w_df.columns:
                latest_rebal = w_df[w_df['reb_flag'] == True]
                latest_w = latest_rebal.iloc[-1].drop('reb_flag') if not latest_rebal.empty else w_df.iloc[-1].drop('reb_flag', errors='ignore')
            else:
                latest_w = w_df.iloc[-1]
            latest_w = latest_w[latest_w > 0]
            if latest_w.empty:
                continue
            hhi = float((latest_w ** 2).sum())
            hhi_rows[port_name] = {
                'HHI (sum of w^2)': hhi,
                'Effective N (1/HHI)': 1.0 / hhi if hhi > 0 else np.nan,
                'Max Weight': float(latest_w.max()),
                'Min Weight': float(latest_w[latest_w > 0].min()),
                '# Holdings': int((latest_w > 0).sum()),
            }
        if hhi_rows:
            pd.DataFrame(hhi_rows).to_excel(writer, sheet_name='Concentration HHI')

        if overlap_matrix is not None and not overlap_matrix.empty:
            overlap_matrix.to_excel(writer, sheet_name='Overlap Matrix')

        if fi_analytics_df is not None and not fi_analytics_df.empty:
            fi_analytics_df.to_excel(writer, sheet_name='Fixed Income Analytics')

        if factor_df is not None and not factor_df.empty:
            factor_df.to_excel(writer, sheet_name='Factor Exposures (FF5)')

        # ---------- ADVANCED RISK & ATTRIBUTION ----------
        from utils.risk_analytics import (
            compute_risk_contributions as _ra_mctr,
            compute_active_risk_contributions as _ra_mcte,
            compute_var_cvar as _ra_var,
            compute_drawdown_episodes as _ra_dd,
            compute_capture_ratios as _ra_cap,
            compute_brinson_attribution as _ra_brinson,
            compute_efficient_frontier as _ra_frontier,
            evaluate_portfolio_on_frontier as _ra_eval,
            annualised_cov_from_returns as _ra_cov,
        )

        # Helper to extract latest non-zero weights from a portfolio's weight df
        def _latest_weights(w_df):
            if 'reb_flag' in w_df.columns:
                last_rebal = w_df[w_df['reb_flag'] == True]
                lw = (last_rebal.iloc[-1] if not last_rebal.empty else w_df.iloc[-1]).drop('reb_flag', errors='ignore')
            else:
                lw = w_df.iloc[-1]
            lw = lw[lw != 0].copy()
            # suffix-safe: only strip a trailing `_wgt`
            lw.index = [c[:-4] if isinstance(c, str) and c.endswith('_wgt') else c for c in lw.index]
            return lw

        # Build covariance once if asset returns are provided
        ann_cov_for_export = None
        if ret_all_for_risk is not None and not ret_all_for_risk.empty:
            try:
                ann_cov_for_export = _ra_cov(ret_all_for_risk)
            except Exception:
                ann_cov_for_export = None

        # MCTR per portfolio
        if ann_cov_for_export is not None:
            for port_name, port_data in results.items():
                if 'weights' not in port_data:
                    continue
                lw = _latest_weights(port_data['weights'])
                try:
                    mctr_df = _ra_mctr(lw, ann_cov_for_export)
                    mctr_df.to_excel(writer, sheet_name=_sheet(f'MCTR - {port_name}'))
                except Exception:
                    pass

            # MCTE per portfolio (needs benchmark weights)
            if bm_weights_series is not None and not bm_weights_series.empty:
                for port_name, port_data in results.items():
                    if 'weights' not in port_data:
                        continue
                    lw = _latest_weights(port_data['weights'])
                    try:
                        mcte_df = _ra_mcte(lw, bm_weights_series, ann_cov_for_export)
                        mcte_df.to_excel(writer, sheet_name=_sheet(f'MCTE - {port_name}'))
                    except Exception:
                        pass

        # VaR/CVaR per portfolio (1d horizon, 95% & 99%, both historical & parametric)
        var_combined = {}
        for port_name in all_port_ret.columns:
            try:
                v = _ra_var(all_port_ret[port_name], confidence_levels=(0.95, 0.99), horizon_days=1, annualise=False)
                if not v.empty:
                    v.index = [f"{int(x*100)}%" for x in v.index]
                    v.columns = [f"{port_name} {c}" for c in v.columns]
                    var_combined[port_name] = v
            except Exception:
                pass
        if var_combined:
            var_all = pd.concat(var_combined.values(), axis=1)
            var_all.index.name = 'Confidence'
            var_all.to_excel(writer, sheet_name='VaR CVaR')

        # Top drawdown episodes per portfolio (already had a separate Drawdown Summary;
        # this version uses the helper for consistency with the UI)
        dd_rows_helper = []
        for port_name in all_port_ret.columns:
            try:
                dd_top = _ra_dd(all_port_ret[port_name], top_n=5)
                if not dd_top.empty:
                    dd_top.insert(0, 'Portfolio', port_name)
                    dd_rows_helper.append(dd_top)
            except Exception:
                pass
        if dd_rows_helper:
            dd_all = pd.concat(dd_rows_helper, axis=0)
            dd_all.to_excel(writer, sheet_name='Drawdown Top5')

        # Capture ratios (monthly) using helper
        if ret_bm is not None and not ret_bm.empty and ret_bm.shape[1] > 0:
            cap_rows_helper = {}
            for port_name in all_port_ret.columns:
                try:
                    cap = _ra_cap(all_port_ret[port_name], ret_bm.iloc[:, 0], frequency='M')
                    if not cap.empty:
                        cap_rows_helper[port_name] = cap
                except Exception:
                    pass
            if cap_rows_helper:
                pd.DataFrame(cap_rows_helper).to_excel(writer, sheet_name='Capture & Hit Ratios')

        # Brinson attribution (full sample) for sector and country.
        # Use TIME-AVERAGED daily weights (matches the UI tab) so the
        # decomposition reflects the average exposure actually held
        # across the backtest, not just the latest rebalance snapshot.
        if bm_weights_series is not None and ret_all_for_risk is not None:
            full_period_returns = (1 + ret_all_for_risk).prod() - 1
            for bucket_name, bucket_df, sheet_label in [
                ('Sector', df_sector_for_brinson, 'Brinson Sector'),
                ('Country', df_country_for_brinson, 'Brinson Country'),
            ]:
                if bucket_df is None or bucket_df.empty:
                    continue
                for port_name, port_data in results.items():
                    if 'weights' not in port_data:
                        continue
                    w_df_p = port_data['weights'].copy()
                    if 'reb_flag' in w_df_p.columns:
                        w_df_p = w_df_p.drop(columns=['reb_flag'])
                    w_df_p.columns = [
                        c[:-4] if isinstance(c, str) and c.endswith('_wgt') else c
                        for c in w_df_p.columns
                    ]
                    avg_w_p_export = w_df_p.mean()
                    avg_w_p_export = avg_w_p_export[avg_w_p_export > 0]
                    if avg_w_p_export.empty:
                        avg_w_p_export = _latest_weights(port_data['weights'])
                    try:
                        brin = _ra_brinson(avg_w_p_export, bm_weights_series, full_period_returns, bucket_df)
                        if not brin.empty:
                            # rename columns for clarity (avg vs snapshot)
                            brin = brin.rename(columns={
                                'Port Weight': 'Avg Port Weight',
                                'Bench Weight': 'Avg Bench Weight',
                                'Active Weight': 'Avg Active Weight',
                            })
                            brin.to_excel(writer, sheet_name=_sheet(f'{sheet_label} - {port_name}'))
                    except Exception:
                        pass

        # Efficient frontier (one sheet, latest covariance, full universe)
        if ret_all_for_risk is not None and ret_all_for_risk.shape[1] >= 2:
            try:
                frontier_df = _ra_frontier(ret_all_for_risk, n_points=30)
                if not frontier_df.empty:
                    frontier_df.to_excel(writer, sheet_name='Efficient Frontier')
                    # plus a row for each actual portfolio + benchmark
                    overlay_rows = []
                    for port_name, port_data in results.items():
                        if 'weights' not in port_data:
                            continue
                        lw = _latest_weights(port_data['weights'])
                        ret_p, vol_p = _ra_eval(lw, ret_all_for_risk)
                        overlay_rows.append({'Label': port_name, 'Return': ret_p, 'Vol': vol_p})
                    if bm_weights_series is not None and not bm_weights_series.empty:
                        ret_b, vol_b = _ra_eval(bm_weights_series, ret_all_for_risk)
                        overlay_rows.append({'Label': 'Benchmark', 'Return': ret_b, 'Vol': vol_b})
                    if overlay_rows:
                        pd.DataFrame(overlay_rows).to_excel(writer, sheet_name='Frontier Overlay', index=False)
            except Exception:
                pass

    output.seek(0)
    return output.getvalue()

# 5. analysis & visualization
if st.session_state.builder_portfolio['results']:
    res_data = st.session_state.builder_portfolio['results']
    results = res_data['results']
    
    ret_all = res_data['ret_all']
    ret_bm = res_data['ret_bm']
    
    st.header("3. Analysis Results")
    
    # fx conversion and export options
    col_fx, col_export = st.columns([1, 1])
    
    with col_fx:
        fx_currency = st.selectbox(
            "Display Currency",
            ["USD", "SGD"],
            index=0,
            key="builder_fx_currency",
            help="Convert returns to selected currency. Base currency is USD."
        )
    
    with col_export:
        st.write("")  # spacing
    
    # combined returns
    all_port_ret = pd.DataFrame()
    for p_name, p_res in results.items():
        p_ret = p_res['perf'][[c for c in p_res['perf'].columns if c.endswith('_port_ret')]].copy()
        p_ret.columns = [p_name]
        all_port_ret = pd.concat([all_port_ret, p_ret], axis=1)

    # align with benchmark
    common_idx = all_port_ret.index.intersection(ret_bm.index)
    all_port_ret = all_port_ret.loc[common_idx].copy()                  # defensive copy
    ret_bm = ret_bm.loc[common_idx].copy()

    # ---- preserve USD-base copies for currency-sensitive analytics ----
    # The FX overlay below mutates `all_port_ret` and `ret_bm` to the
    # display currency. Some downstream blocks need the *base-currency*
    # series (e.g. Fama-French regression: factors are USD-denominated, so
    # the LHS must also be in USD). Keep a snapshot here.
    all_port_ret_base_ccy = all_port_ret.copy()
    ret_bm_base_ccy = ret_bm.copy()

    # apply fx conversion if SGD selected
    if fx_currency == "SGD" and df_fx is not None and 'USDSGD' in df_fx.columns:
        fx_returns = df_fx['USDSGD'].copy()
        # convert levels to returns if needed
        if fx_returns.abs().max() > 1:
            fx_returns = fx_returns.pct_change().fillna(0)

        common_fx_idx = all_port_ret.index.intersection(fx_returns.index)
        if len(common_fx_idx) > 0:
            fx_ret_aligned = fx_returns.loc[common_fx_idx]
            # convert usd returns to sgd: (1+r_usd) * (1+r_fx) - 1
            for col in all_port_ret.columns:
                all_port_ret.loc[common_fx_idx, col] = (1 + all_port_ret.loc[common_fx_idx, col]) * (1 + fx_ret_aligned) - 1
            for col in ret_bm.columns:
                ret_bm.loc[common_fx_idx, col] = (1 + ret_bm.loc[common_fx_idx, col]) * (1 + fx_ret_aligned) - 1
    
    # tabs
    tabs = st.tabs([
        "Key Summary", "Calendar Returns", "Monthly Returns", "Cumulative Returns", 
        "Return/Risk Scatter", "Returns Distribution", "Rolling Volatility", 
        "Drawdown", "Correlation Heatmap", "Portfolio vs Benchmark", "Tracking Error"
    ])

    with tabs[0]:
        summary = get_performance_summary(all_port_ret)
        # format
        def format_summary_value(val, metric_name):
            if pd.isna(val): return "N/A"
            if metric_name in ['Sharpe Ratio', 'Sortino Ratio', 'Information Ratio']: return f"{val:.2f}"
            else: return f"{val:.2%}"
        
        # Build the formatted view on object dtype so string values can be
        # written into what was previously a float64 column without tripping
        # pandas 2.0's strict dtype guard.
        summary_formatted = summary.astype(object).copy()
        for idx in summary.index:
            for col in summary.columns:
                summary_formatted.loc[idx, col] = format_summary_value(summary.loc[idx, col], idx)
            
        st.dataframe(summary_formatted, use_container_width=True)
        st.markdown("---")
        
        st.subheader("Historical Weights (Rebalance Dates)")
        p_select_w = st.selectbox("Select Portfolio", list(results.keys()), key="builder_w_sel")
        w_df = results[p_select_w]['weights']
        # w_df has 'reb_flag' column, but sometimes just 1 row.
        if 'reb_flag' in w_df.columns:
            w_rebal = w_df[w_df['reb_flag'] == True].drop(columns=['reb_flag'])
        else:
            w_rebal = w_df # fallback
            
        w_rebal.columns = [c[:-4] if isinstance(c, str) and c.endswith('_wgt') else c for c in w_rebal.columns]
        w_rebal_display = w_rebal.copy()
        try:
             w_rebal_display.index = w_rebal_display.index.strftime('%Y-%m-%d')
        except: pass
        
        st.dataframe(w_rebal_display.style.background_gradient(cmap='Blues', axis=None).format("{:.2%}"), use_container_width=True)

        # drift weights between rebalance dates
        show_drift = st.checkbox(
            "Show Drift Between Rebalance Dates",
            key="builder_show_drift",
            help="Show how weights drift between rebalance dates based on returns. Weights are renormalised daily and sum to 1."
        )

        if show_drift and 'reb_flag' in w_df.columns:
            reb_dates = w_df.index[w_df['reb_flag'] == True].tolist()

            if len(reb_dates) == 0:
                st.info("No rebalance dates found.")
            else:
                # build labels for each drift period (rebalance date -> next rebalance date or end)
                period_options = {}
                for i, rd in enumerate(reb_dates):
                    period_end = reb_dates[i + 1] if i + 1 < len(reb_dates) else w_df.index[-1]
                    label = f"{rd.strftime('%Y-%m-%d')} → {period_end.strftime('%Y-%m-%d')}"
                    period_options[label] = (rd, period_end, i + 1 < len(reb_dates))

                selected_period = st.selectbox(
                    "Select Rebalance Period",
                    list(period_options.keys()),
                    key="builder_drift_period_sel"
                )

                period_start, period_end, has_next_reb = period_options[selected_period]
                # exclude the next rebalance row so we only show drift within the period
                if has_next_reb:
                    drift_slice = w_df.loc[period_start:period_end].iloc[:-1]
                else:
                    drift_slice = w_df.loc[period_start:period_end]

                drift_display = drift_slice.drop(columns=['reb_flag']).copy()
                drift_display.columns = [c[:-4] if isinstance(c, str) and c.endswith('_wgt') else c for c in drift_display.columns]
                # drop tickers that are zero throughout this drift window for readability
                drift_display = drift_display.loc[:, (drift_display != 0).any(axis=0)]

                st.caption(
                    f"Showing daily drift weights from {period_start.strftime('%Y-%m-%d')} "
                    f"to {period_end.strftime('%Y-%m-%d')} "
                    f"({len(drift_display)} trading days). Weights drift with returns and renormalise to 1 daily."
                )

                drift_view = drift_display.copy()
                try:
                    drift_view.index = drift_view.index.strftime('%Y-%m-%d')
                except Exception:
                    pass

                st.dataframe(
                    drift_view.style.background_gradient(cmap='Blues', axis=None).format("{:.2%}"),
                    use_container_width=True
                )

                # quick visual of the drift over the period
                fig_drift = px.area(
                    drift_display,
                    title=f"Weight Drift: {period_start.strftime('%Y-%m-%d')} → {period_end.strftime('%Y-%m-%d')}",
                    labels={'value': 'Weight', 'index': 'Date', 'variable': 'Ticker'}
                )
                fig_drift.update_layout(yaxis_tickformat='.0%', yaxis_range=[0, 1])
                st.plotly_chart(fig_drift, use_container_width=True, key="builder_drift_chart")

    with tabs[1]:
        yearly_ret = all_port_ret.resample('YE').apply(lambda x: (1+x).prod()-1)
        yearly_ret.index = yearly_ret.index.year
        st.dataframe(yearly_ret.style.background_gradient(cmap='Blues', axis=None).format("{:.2%}"), use_container_width=True)

    with tabs[2]:
        fig = plot_monthly_heatmap(all_port_ret)
        st.plotly_chart(fig, use_container_width=True, key="builder_monthly")
    
    with tabs[3]:
        # calculate cumulative returns as percentage (0.94 = 94%, not 1.94)
        cum_ret = (1 + all_port_ret).cumprod() - 1
        fig = plot_cumulative_returns(cum_ret, benchmark_ret=ret_bm)
        st.plotly_chart(fig, use_container_width=True, key="builder_cum")
        
    with tabs[4]:
        fig = plot_scatter_risk_return(all_port_ret, benchmark_ret=ret_bm)
        st.plotly_chart(fig, use_container_width=True, key="builder_scatter")
        
    with tabs[5]:
        import plotly.figure_factory as ff
        hist_data = [all_port_ret[c].dropna() for c in all_port_ret.columns]
        group_labels = all_port_ret.columns
        try:
            fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=PALETTE[:len(group_labels)])
        except:
            fig = go.Figure()
        fig.update_layout(title_text='Returns Distribution', plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True, key="builder_dist")
        
    with tabs[6]:
        fig = plot_rolling_volatility(all_port_ret)
        st.plotly_chart(fig, use_container_width=True, key="builder_vol")

    with tabs[7]:
        fig = plot_drawdowns(all_port_ret)
        st.plotly_chart(fig, use_container_width=True, key="builder_dd")
        
    with tabs[8]:
        fig = plot_correlation_heatmap(all_port_ret)
        st.plotly_chart(fig, use_container_width=True, key="builder_corr")
        
    with tabs[9]:
        rel_metrics, roll_excess = calculate_relative_metrics(all_port_ret.iloc[:, 0], ret_bm.iloc[:, 0])
        st.markdown("#### Relative Performance Metrics")
        rel_df = rel_metrics.to_frame().T
        rel_format = {col: "{:.2f}" if 'Ratio' in col else "{:.2%}" for col in rel_df.columns}
        st.dataframe(rel_df.style.format(rel_format), use_container_width=True)
        st.markdown("#### Rolling 12M Excess Return")
        fig_roll = plot_cumulative_returns(roll_excess.to_frame(name="Rolling Excess Return"))
        st.plotly_chart(fig_roll, use_container_width=True, key="builder_excess")
        
    with tabs[10]:
        col_te_h, col_te_help = st.columns([10, 1])
        with col_te_h:
            st.markdown("#### Rolling 12M Tracking Error")
        with col_te_help:
            st.markdown("####")
            st.markdown(
                ":grey_question:", 
                help="This chart shows **realized** TE based on actual returns. TE constraints are applied at rebalance using *expected* TE from historical covariance. Realized TE may differ due to changing market conditions."
            )
        active_ret = all_port_ret.iloc[:, 0] - ret_bm.iloc[:, 0]
        roll_te = active_ret.rolling(window=252).std(ddof=1) * np.sqrt(252)
        
        fig_te = px.line(
            roll_te, 
            title="Rolling 12-Month Tracking Error (Annualized)", 
            labels={'value': 'Tracking Error', 'index': 'Date'}
        )
        fig_te.update_layout(yaxis_tickformat=".2%")
        st.plotly_chart(fig_te, use_container_width=True, key="builder_te_roll")

    # 3.1 fixed income analytics (mirrors Advisory Tool section 4.1)
    st.markdown("---")
    st.subheader("3.1 Fixed Income Analytics")

    # offset matches the optimisation default in this page (90d post earliest data)
    fi_offset_days = 90
    fi_start_opt_date = ret_all.index.min() + timedelta(days=fi_offset_days)
    fi_start_opt_date = max(fi_start_opt_date, pd.Timestamp(start_date))

    try:
        has_ytm = 'YTM / est. yield' in df_fi_metrics.index
        has_t12m = 'T12M dividend yield' in df_fi_metrics.index
        has_dur = 'Duration' in df_fi_metrics.index

        if has_ytm or has_t12m:
            # build two yield rows with symmetric per-ticker fallback:
            #   - Weighted Yield to Maturity: YTM first, T12M fallback
            #   - Weighted 12M Dividend Yield: T12M first, YTM fallback
            ytm_yield = pd.Series(0.0, index=df_fi_metrics.columns)
            t12m_yield = pd.Series(0.0, index=df_fi_metrics.columns)

            for col in df_fi_metrics.columns:
                ytm_val = 0.0
                t12m_val = 0.0
                if has_ytm:
                    v = df_fi_metrics.loc['YTM / est. yield', col]
                    if pd.notna(v) and v != 0:
                        ytm_val = float(v)
                if has_t12m:
                    v = df_fi_metrics.loc['T12M dividend yield', col]
                    if pd.notna(v) and v != 0:
                        t12m_val = float(v)

                ytm_yield[col] = ytm_val if ytm_val != 0 else t12m_val
                t12m_yield[col] = t12m_val if t12m_val != 0 else ytm_val

            # FI universe = any ticker that has either yield reported
            fi_tickers = [
                c for c in df_fi_metrics.columns
                if ytm_yield[c] != 0 or t12m_yield[c] != 0
            ]

            # exclude tickers that don't generate cash flow (aligned with cash flow exclusions)
            FI_EXCLUDED_PATTERNS = [
                'CASH', 'MMF', 'FLEXI', 'GUARANTEED',
                'CORE', 'EQUITY', 'GROWTH',
                'CHINA', 'ESG', 'HEALTHCARE', 'DISRUPTIVE', 'TECHNOLOGY', 'THEMATIC',
                'CUSTOM', 'STRATEGIC',
            ]
            fi_tickers = [t for t in fi_tickers if not any(p in t.upper() for p in FI_EXCLUDED_PATTERNS)]

            if fi_tickers:
                fi_metrics_for_calc = pd.DataFrame(
                    index=['Weighted Duration',
                           'Weighted Yield to Maturity',
                           'Weighted 12M Dividend Yield'],
                    columns=df_fi_metrics.columns,
                    dtype=float,
                )
                if has_dur:
                    fi_metrics_for_calc.loc['Weighted Duration'] = df_fi_metrics.loc['Duration'].astype(float)
                else:
                    fi_metrics_for_calc.loc['Weighted Duration'] = 0.0
                fi_metrics_for_calc.loc['Weighted Yield to Maturity'] = ytm_yield
                fi_metrics_for_calc.loc['Weighted 12M Dividend Yield'] = t12m_yield

                port_fi_metrics = pd.DataFrame()
                has_fi = False

                for p_name_fi in results.keys():
                    w_df_fi = results[p_name_fi]['weights']
                    p_tickers = [c[:-4] for c in w_df_fi.columns if isinstance(c, str) and c.endswith('_wgt')]
                    p_tickers = [t for t in p_tickers if not any(p in t.upper() for p in FI_EXCLUDED_PATTERNS)]
                    common_fi = list(set(p_tickers).intersection(fi_tickers))

                    if common_fi:
                        has_fi = True
                        metrics_df = pc.exposure_analysis_mixed(
                            w_df_fi,
                            fi_metrics_for_calc,
                            fi_tickers,
                            latest_data=True,
                            start_date=fi_start_opt_date,
                            end_date=end_date,
                        )
                        metrics_df.columns = [p_name_fi]

                        try:
                            if 'Credit rating' in df_fi_metrics.index:
                                cr = pc.calc_avg_credit_rating(
                                    w_df_fi,
                                    df_fi_metrics.loc[['Credit rating']],
                                    latest_data=True,
                                    start_date=fi_start_opt_date,
                                    end_date=end_date,
                                )
                                metrics_df.loc['Credit rating', p_name_fi] = cr
                        except Exception:
                            metrics_df.loc['Credit rating', p_name_fi] = 'N/A'

                        port_fi_metrics = pd.concat([port_fi_metrics, metrics_df], axis=1)

                if has_fi:
                    st.dataframe(port_fi_metrics.T, use_container_width=True)
                else:
                    st.info("No Fixed Income data available.")
            else:
                st.info("No Fixed Income data available.")
        else:
            st.info("Fixed Income metrics data missing.")
    except Exception as e:
        st.error(f"Error calculating FI metrics: {e}")

    # 4. exposure analysis
    st.header("4. Exposure Analysis")
    
    p_name = list(results.keys())[0]
    w_df = results[p_name]['weights']
    
    # get weights at last rebal date
    if 'reb_flag' in w_df.columns:
        w_rebal_only = w_df[w_df['reb_flag'] == True]
        if not w_rebal_only.empty:
            last_w = w_rebal_only.iloc[-1].drop(['reb_flag'])
        else:
            last_w = w_df.iloc[-1].drop(['reb_flag'], errors='ignore')
    else:
        last_w = w_df.iloc[-1].drop(['reb_flag'], errors='ignore')
        
    last_w = last_w[last_w > 0]
    current_holdings = {(k[:-4] if isinstance(k, str) and k.endswith('_wgt') else k): v for k, v in last_w.items()}
    
    import plotly.express as px
    
    charts = get_exposure_charts(last_w, df_sector, df_country, df_asset_class=None)
    
    # row 1: top holdings
    st.markdown("### Top Holdings (Underlying)")
    t_topn = st.selectbox("Show Top", [10, 15, 20], key="builder_topn")

    conc_raw = pd.DataFrame() 
    if df_underlying is not None:
         if 'reb_flag' in w_df.columns:
             last_w_full = w_df[w_df['reb_flag'] == True].iloc[-1]
         else:
             last_w_full = w_df.iloc[-1]
             
         conc_raw = get_concentration_exposure(last_w_full, df_underlying, top_n=t_topn)

    if not conc_raw.empty:
        conc_raw = conc_raw.sort_values(ascending=False)
        conc_df = conc_raw.head(t_topn).to_frame(name='Weight')
        top_sum = conc_df['Weight'].sum()
        
        st.caption(f"Top {t_topn} holdings represent **{top_sum:.1%}** of the overall portfolio.")
        
        conc_df_chart = conc_df.sort_values('Weight', ascending=True)
        
        fig_top = px.bar(
            conc_df_chart, 
            x='Weight', 
            y=conc_df_chart.index, 
            orientation='h',
            text_auto='.2%'
        )
        fig_top.update_layout(showlegend=False, xaxis_tickformat='.1%', yaxis_title=None)
        st.plotly_chart(fig_top, use_container_width=True, key="builder_top_bar")
        
        with st.expander("View Data"):
           st.dataframe(conc_df.style.format({'Weight': '{:.2%}'}), use_container_width=True)
    elif df_underlying is not None and conc_raw.empty:
         st.warning("No underlying data found or weights are zero.")
    else:
         st.info("Underlying data not available.")

    st.markdown("### Allocation Analysis")
    show_bm_diff = st.checkbox("Show vs Benchmark", key="show_bm_diff")
        
    bm_charts = None
    
    if show_bm_diff:
        try:
            bm_w = pd.Series({f"{t}_wgt": w for t, w in zip(bm_tickers_selected, bm_weights_selected)})
            bm_charts = get_exposure_charts(bm_w, df_sector, df_country, df_asset_class=None)
            if bm_charts.get('sector_table') is None or bm_charts['sector_table'].empty:
                st.warning(f"Benchmark '{benchmark_label}' sector/country data not found. Comparison disabled.")
                bm_charts = None
        except Exception as e:
            st.warning(f"Could not load benchmark '{benchmark_label}' data: {str(e)}")
            bm_charts = None

    c_sec, c_ctry = st.columns(2)
    
    def plot_alloc_comparison(port_df, bm_df, title):
        comb = pd.DataFrame({'Portfolio': port_df['Weight'], 'Benchmark': bm_df['Weight'] if bm_df is not None else 0})
        comb = comb.fillna(0)
        comb = comb.sort_values('Portfolio', ascending=True)
        comb_melt = comb.reset_index().melt(id_vars='index', var_name='Type', value_name='Weight')
        fig = px.bar(comb_melt, y='index', x='Weight', color='Type', barmode='group',
                     orientation='h', title=title, text_auto='.1%')
        fig.update_layout(yaxis_title=None, xaxis_tickformat='.0%')
        return fig

    with c_sec:
        st.markdown("**Sector Allocation**")
        if 'sector_table' in charts:
            if show_bm_diff and bm_charts:
                fig_s = plot_alloc_comparison(charts['sector_table'], bm_charts.get('sector_table'), "Sector vs Benchmark")
                st.plotly_chart(fig_s, use_container_width=True, key="builder_sec_comp")
            else:
                fig_s = charts['sector_pie']
                fig_s.update_traces(textinfo='none')
                st.plotly_chart(fig_s, use_container_width=True, key="builder_sec_pie")
            
            with st.expander("View Data"):
                st.dataframe(charts['sector_table'].style.format("{:.2%}"), use_container_width=True)
                
    with c_ctry:
        st.markdown("**Country Allocation**")
        if 'country_table' in charts:
             if show_bm_diff and bm_charts:
                 fig_c = plot_alloc_comparison(charts['country_table'], bm_charts.get('country_table'), "Country vs Benchmark")
                 st.plotly_chart(fig_c, use_container_width=True, key="builder_ctry_comp")
             else:
                  fig_c = charts['country_pie']
                  fig_c.update_traces(textinfo='none')
                  st.plotly_chart(fig_c, use_container_width=True, key="builder_ctry_pie")
                  
             with st.expander("View Data"):
                st.dataframe(charts['country_table'].style.format("{:.2%}"), use_container_width=True)

    # 5. risk & attribution analytics
    from utils.risk_analytics import (
        compute_risk_contributions,
        compute_active_risk_contributions,
        compute_var_cvar,
        compute_drawdown_episodes,
        compute_capture_ratios,
        compute_brinson_attribution,
        compute_efficient_frontier,
        evaluate_portfolio_on_frontier,
        annualised_cov_from_returns,
    )

    st.header("5. Risk & Attribution Analytics")

    # which portfolio to inspect (most analytics are single-portfolio)
    if len(results) > 1:
        port_for_risk = st.selectbox(
            "Select Portfolio",
            list(results.keys()),
            key="builder_risk_port_sel",
        )
    else:
        port_for_risk = list(results.keys())[0]

    # latest weights for the selected portfolio (last rebalance row, ex reb_flag)
    w_df_risk = results[port_for_risk]['weights']
    if 'reb_flag' in w_df_risk.columns:
        last_rebal = w_df_risk[w_df_risk['reb_flag'] == True]
        last_w_risk = (last_rebal.iloc[-1] if not last_rebal.empty else w_df_risk.iloc[-1]).drop('reb_flag', errors='ignore')
    else:
        last_w_risk = w_df_risk.iloc[-1]
    last_w_risk = last_w_risk[last_w_risk != 0]
    # suffix-safe strip: only remove '_wgt' when it appears as the trailing suffix
    last_w_risk.index = [c[:-4] if isinstance(c, str) and c.endswith('_wgt') else c for c in last_w_risk.index]

    # benchmark weights as a Series (may be blended). Normalise so weights sum
    # to 1 — the universe selection UI only warns when the blend is off-100%.
    bm_w_series = pd.Series({t: float(w) for t, w in zip(bm_tickers_selected, bm_weights_selected)})
    bm_total = float(bm_w_series.sum())
    if bm_total > 0 and abs(bm_total - 1.0) > 1e-6:
        bm_w_series = bm_w_series / bm_total

    # asset returns used for cov estimation (restrict to tickers we actually need)
    risk_universe = sorted(set(last_w_risk.index) | set(bm_w_series.index))
    cov_returns = ret_all.reindex(columns=[c for c in risk_universe if c in ret_all.columns]).dropna(how='all')

    # apply the same FX overlay used by the Section 3 metrics so MCTR/MCTE
    # vol numbers reconcile with the Annualized Volatility shown in Key Summary.
    # Without this, MCTR is in USD while Key Summary is in display currency.
    if fx_currency != "USD" and df_fx is not None and not cov_returns.empty:
        _fx_col_risk = 'USDSGD' if fx_currency == 'SGD' else ('USDHKD' if fx_currency == 'HKD' else None)
        if _fx_col_risk and _fx_col_risk in df_fx.columns:
            _fx_ret_risk = df_fx[_fx_col_risk].copy()
            if _fx_ret_risk.abs().max() > 1:
                _fx_ret_risk = _fx_ret_risk.pct_change().fillna(0)
            _common_idx_risk = cov_returns.index.intersection(_fx_ret_risk.index)
            if len(_common_idx_risk) > 0:
                _fx_aligned_risk = _fx_ret_risk.loc[_common_idx_risk]
                cov_returns = cov_returns.copy()
                for _col in cov_returns.columns:
                    cov_returns.loc[_common_idx_risk, _col] = (
                        (1 + cov_returns.loc[_common_idx_risk, _col]) * (1 + _fx_aligned_risk) - 1
                    )

    ann_cov = annualised_cov_from_returns(cov_returns) if not cov_returns.empty else pd.DataFrame()

    # surface tickers that were silently dropped because they have no return history
    missing_from_cov = [t for t in risk_universe if t not in cov_returns.columns]
    if missing_from_cov:
        st.caption(
            f"⚠️ Excluded from risk decomposition (no return history): "
            f"{', '.join(missing_from_cov)}"
        )

    risk_tabs = st.tabs([
        "Risk Contribution (MCTR)",
        "Active Risk (MCTE)",
        "VaR / CVaR",
        "Drawdown Episodes",
        "Up/Down Capture",
        "Brinson Attribution",
        "Efficient Frontier",
    ])

    # ---- Tab 1: MCTR ----
    with risk_tabs[0]:
        st.markdown(
            "Risk contribution per asset. Splits the total annualised portfolio "
            "volatility into per-asset pieces, so you can see which holdings drive "
            "the overall risk. The split uses a method called the Euler "
            "decomposition: for each asset we compute a marginal contribution "
            "to risk (how much a small increase in that asset's weight would push "
            "portfolio volatility up or down), then multiply that by the asset's "
            "current weight to get its component contribution to risk. The component "
            "contributions add up to total portfolio volatility, and the % share "
            "column adds up to 100%. The numbers are computed from your latest "
            "rebalance weights and the historical covariance matrix, so they "
            "represent the current snapshot of risk and will not match the "
            "realised backtest volatility shown in Section 3."
        )
        if ann_cov.empty:
            st.info("Insufficient return history to estimate covariance.")
        else:
            try:
                mctr_df = compute_risk_contributions(last_w_risk, ann_cov)
                port_vol = mctr_df.attrs.get('portfolio_vol', np.nan)
                st.caption(
                    f"Estimated annualised portfolio volatility: **{port_vol:.2%}** "
                    f"(sum of component risks)."
                )
                fmt = {
                    'Weight': '{:.2%}',
                    'MCTR (annualised)': '{:.4f}',
                    'Component Risk (annualised)': '{:.4f}',
                    '% of Total Risk': '{:.2%}',
                }
                st.dataframe(
                    mctr_df.sort_values('% of Total Risk', ascending=False).style.format(fmt),
                    use_container_width=True,
                )
                fig_mctr = px.bar(
                    mctr_df.sort_values('% of Total Risk', ascending=True),
                    x='% of Total Risk',
                    y=mctr_df.sort_values('% of Total Risk', ascending=True).index,
                    orientation='h',
                    text_auto='.1%',
                    title="Risk Contribution by Asset",
                )
                fig_mctr.update_layout(yaxis_title=None, xaxis_tickformat='.1%')
                st.plotly_chart(fig_mctr, use_container_width=True, key="builder_mctr_chart")
            except Exception as e:
                st.error(f"Could not compute risk contributions: {e}")

    # ---- Tab 2: MCTE (active risk) ----
    with risk_tabs[1]:
        st.markdown(
            "Active risk contribution per asset. Same idea as the MCTR table, "
            "but applied to the active weights (your weight minus the benchmark's "
            "weight in each asset). Holdings you over-weight relative to the "
            "benchmark add to tracking error; holdings you under-weight reduce it. "
            "The numbers are computed from your latest rebalance weights and the "
            "historical covariance matrix, so they represent the current-snapshot "
            "expected tracking error, not the realised tracking error from the "
            "backtest."
        )
        if ann_cov.empty:
            st.info("Insufficient return history to estimate covariance.")
        else:
            try:
                mcte_df = compute_active_risk_contributions(last_w_risk, bm_w_series, ann_cov)
                te = mcte_df.attrs.get('tracking_error', np.nan)
                st.caption(f"Estimated annualised tracking error: **{te:.2%}**.")
                fmt = {
                    'Portfolio Weight': '{:.2%}',
                    'Benchmark Weight': '{:.2%}',
                    'Active Weight': '{:+.2%}',
                    'MCTE (annualised)': '{:.4f}',
                    'Component TE (annualised)': '{:.4f}',
                    '% of Total TE': '{:+.2%}',
                }
                st.dataframe(
                    mcte_df.sort_values('% of Total TE', ascending=False).style.format(fmt),
                    use_container_width=True,
                )
                # Plot only the assets that actually contribute (skip near-zero rows)
                plot_df = mcte_df.copy()
                plot_df = plot_df[plot_df['% of Total TE'].abs() > 1e-4]
                if not plot_df.empty:
                    plot_df = plot_df.sort_values('% of Total TE')
                    fig_mcte = px.bar(
                        plot_df,
                        x='% of Total TE',
                        y=plot_df.index,
                        orientation='h',
                        text_auto='.1%',
                        color='% of Total TE',
                        color_continuous_scale='RdBu_r',
                        color_continuous_midpoint=0,
                        title="Active Risk Contribution by Asset",
                    )
                    fig_mcte.update_layout(yaxis_title=None, xaxis_tickformat='.1%')
                    st.plotly_chart(fig_mcte, use_container_width=True, key="builder_mcte_chart")
            except Exception as e:
                st.error(f"Could not compute active risk contributions: {e}")

    # ---- Tab 3: VaR / CVaR ----
    with risk_tabs[2]:
        st.markdown(
            "Value at Risk (VaR) is the loss threshold for a given confidence "
            "level. At a 95% confidence level, the daily VaR is the loss you "
            "would expect to exceed only 5% of the time. CVaR (also called "
            "Expected Shortfall) is the average loss on the days when VaR is "
            "breached, so it tells you how bad things look in the tail beyond "
            "VaR. Both are reported as negative numbers (losses)."
        )
        c_h, c_a = st.columns([1, 1])
        with c_h:
            horizon = st.selectbox(
                "Horizon",
                options=[1, 5, 10, 21],
                format_func=lambda d: f"{d} day{'s' if d > 1 else ''}",
                index=0,
                key="builder_var_horizon",
            )
        with c_a:
            annualise_var = st.checkbox(
                "Annualise (252-day horizon)",
                value=False,
                key="builder_var_annualise",
                help=(
                    "When ticked, the calculation is reported on a one-year (252 trading day) "
                    "horizon instead of one day. Historical numbers are scaled using the standard "
                    "square-root-of-time rule, which assumes daily returns are independent. "
                    "Parametric numbers use the daily mean multiplied by 252, plus the daily "
                    "volatility multiplied by the square root of 252, applied to the Gaussian tail."
                ),
            )

        if port_for_risk not in all_port_ret.columns:
            st.error(f"Returns column for '{port_for_risk}' not found in computed performance data.")
            st.stop()
        port_ret_for_var = all_port_ret[port_for_risk]
        try:
            var_df = compute_var_cvar(
                port_ret_for_var,
                confidence_levels=(0.95, 0.99),
                horizon_days=horizon,
                annualise=annualise_var,
            )
            if var_df.empty:
                st.info("No return data available to compute VaR.")
            else:
                var_display = var_df.copy()
                var_display.index = [f"{int(x*100)}%" for x in var_display.index]
                var_display.index.name = 'Confidence'
                st.dataframe(
                    var_display.style.format({c: '{:.2%}' for c in var_display.columns}),
                    use_container_width=True,
                )
                # also include benchmark VaR for reference
                if ret_bm is not None and not ret_bm.empty and ret_bm.shape[1] > 0:
                    bm_var_df = compute_var_cvar(
                        ret_bm.iloc[:, 0],
                        confidence_levels=(0.95, 0.99),
                        horizon_days=horizon,
                        annualise=annualise_var,
                    )
                    if not bm_var_df.empty:
                        bm_var_df.index = [f"{int(x*100)}%" for x in bm_var_df.index]
                        bm_var_df.index.name = 'Confidence'
                        with st.expander("Benchmark VaR / CVaR"):
                            st.dataframe(
                                bm_var_df.style.format({c: '{:.2%}' for c in bm_var_df.columns}),
                                use_container_width=True,
                            )
        except Exception as e:
            st.error(f"VaR / CVaR computation failed: {e}")

    # ---- Tab 4: Drawdown episodes ----
    with risk_tabs[3]:
        st.markdown(
            "Top drawdown episodes ranked by depth. The peak date, trough date, "
            "and recovery date are based on the realised NAV path. A drawdown "
            "that has not recovered to the prior peak by the end of the data "
            "window is flagged as ongoing."
        )
        if port_for_risk not in all_port_ret.columns:
            st.error(f"Returns column for '{port_for_risk}' not found in computed performance data.")
            st.stop()
        port_ret_for_dd = all_port_ret[port_for_risk]
        top_n_dd = st.slider("Number of episodes", 3, 10, 5, key="builder_dd_topn")
        try:
            dd_df = compute_drawdown_episodes(port_ret_for_dd, top_n=top_n_dd)
            if dd_df.empty:
                st.info("No drawdown episodes detected.")
            else:
                dd_view = dd_df.copy()
                for col in ('Peak Date', 'Trough Date', 'Recovery Date'):
                    dd_view[col] = pd.to_datetime(dd_view[col]).dt.strftime('%Y-%m-%d').fillna('Ongoing')
                st.dataframe(
                    dd_view.style.format({'Depth': '{:.2%}'}),
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Drawdown analysis failed: {e}")

    # ---- Tab 5: Up/Down capture + hit / batting ----
    with risk_tabs[4]:
        st.markdown(
            "How much of the benchmark's up-period and down-period performance "
            "the portfolio captures, plus how often it beats the benchmark."
        )
        if ret_bm is None or ret_bm.empty:
            st.info("No benchmark returns available.")
        else:
            freq_label = st.selectbox(
                "Period",
                options=[("Monthly", "M"), ("Quarterly", "Q"), ("Annual", "Y")],
                format_func=lambda x: x[0],
                index=0,
                key="builder_capture_freq",
            )
            if port_for_risk not in all_port_ret.columns:
                st.error(f"Returns column for '{port_for_risk}' not found in computed performance data.")
                st.stop()
            port_ret_for_cap = all_port_ret[port_for_risk]
            bm_series_cap = ret_bm.iloc[:, 0]
            try:
                cap = compute_capture_ratios(port_ret_for_cap, bm_series_cap, frequency=freq_label[1])
                if cap.empty:
                    st.info("Not enough overlapping data.")
                else:
                    c_a, c_b, c_c, c_d = st.columns(4)
                    with c_a:
                        st.metric(
                            "Up Capture",
                            f"{cap['Up Capture']:.1%}" if pd.notna(cap['Up Capture']) else "N/A",
                            help=(
                                "On periods when the benchmark was positive, this is the average "
                                "portfolio return divided by the average benchmark return. A value "
                                "of 100% means the portfolio captured all of the benchmark's "
                                "up-period gains; above 100% means it outperformed in up periods, "
                                "below 100% means it lagged."
                            ),
                        )
                    with c_b:
                        st.metric(
                            "Down Capture",
                            f"{cap['Down Capture']:.1%}" if pd.notna(cap['Down Capture']) else "N/A",
                            help=(
                                "On periods when the benchmark was negative, this is the average "
                                "portfolio return divided by the average benchmark return. Lower is "
                                "better: 100% means the portfolio fell as much as the benchmark, "
                                "below 100% means it fell less, above 100% means it fell more."
                            ),
                        )
                    with c_c:
                        st.metric(
                            "Hit Ratio",
                            f"{cap['Hit Ratio']:.1%}",
                            help=(
                                "The fraction of periods where the portfolio strictly beat the "
                                "benchmark. Ties are not counted. 50% means the portfolio outperformed "
                                "in half of all periods."
                            ),
                        )
                    with c_d:
                        st.metric(
                            "Batting Avg",
                            f"{cap['Batting Average']:.1%}",
                            help=(
                                "Like the hit ratio but ties count as a win. The fraction of periods "
                                "where the portfolio matched or beat the benchmark."
                            ),
                        )

                    cap_df = cap.to_frame(name='Value')
                    pct_rows = ['Up Capture', 'Down Capture', 'Up/Down Capture Ratio', 'Hit Ratio', 'Batting Average', 'Avg Excess Return (period)']
                    cap_view = cap_df.copy().astype(object)
                    for idx in cap_view.index:
                        v = cap_df.loc[idx, 'Value']
                        if idx in pct_rows and isinstance(v, float) and pd.notna(v):
                            cap_view.loc[idx, 'Value'] = f"{v:.2%}" if idx != 'Up/Down Capture Ratio' else f"{v:.2f}"
                    st.dataframe(cap_view, use_container_width=True)
            except Exception as e:
                st.error(f"Capture analysis failed: {e}")

    # ---- Tab 6: Brinson attribution ----
    with risk_tabs[5]:
        st.markdown(
            "Brinson-Fachler attribution splits the portfolio's active return "
            "(portfolio return minus benchmark return) into three sources: "
            "Allocation, which is the effect of over- or under-weighting each "
            "bucket relative to the benchmark; Selection, which is the effect "
            "of picking different securities inside each bucket; and Interaction, "
            "which is the cross-effect of doing both at once. The decomposition "
            "is a single-period model that uses time-averaged portfolio weights "
            "and the cumulative ETF return for the chosen period. Because the "
            "actual backtest rebalances and weights drift, the decomposition is "
            "an approximation of the realised active return shown above. Any "
            "unexplained gap is reported as a Residual."
        )

        bucket_choice = st.radio(
            "Bucket Type",
            options=['Sector', 'Country'],
            index=0,
            horizontal=True,
            key="builder_brinson_bucket",
        )
        exposure_table = df_sector if bucket_choice == 'Sector' else df_country
        if exposure_table is None or exposure_table.empty:
            st.info(f"{bucket_choice} exposure data not available.")
        else:
            # period selector
            period_label = st.selectbox(
                "Attribution Period",
                options=['Full Sample', 'Last 1Y', 'Last 3Y', 'Last 5Y', 'YTD'],
                index=0,
                key="builder_brinson_period",
            )

            # restrict returns to the chosen period
            rets_period = ret_all.copy()
            if period_label != 'Full Sample':
                last_dt = rets_period.index.max()
                if period_label == 'YTD':
                    cutoff = pd.Timestamp(year=last_dt.year, month=1, day=1)
                else:
                    yrs = {'Last 1Y': 1, 'Last 3Y': 3, 'Last 5Y': 5}[period_label]
                    cutoff = last_dt - pd.DateOffset(years=yrs)
                rets_period = rets_period.loc[rets_period.index >= cutoff]

            # apply the same FX conversion used by the realised port/bench cards
            # so the Brinson decomposition is reported in the **display currency**.
            # Without this step, the bucket math is in USD while the realised
            # cards above are in SGD/HKD -> they cannot reconcile.
            if fx_currency != "USD" and df_fx is not None:
                fx_col_brin = 'USDSGD' if fx_currency == 'SGD' else ('USDHKD' if fx_currency == 'HKD' else None)
                if fx_col_brin and fx_col_brin in df_fx.columns:
                    fx_ret_brin = df_fx[fx_col_brin].copy()
                    if fx_ret_brin.abs().max() > 1:
                        fx_ret_brin = fx_ret_brin.pct_change().fillna(0)
                    common_fx_brin = rets_period.index.intersection(fx_ret_brin.index)
                    if len(common_fx_brin) > 0:
                        fx_aligned_brin = fx_ret_brin.loc[common_fx_brin]
                        for col in rets_period.columns:
                            rets_period.loc[common_fx_brin, col] = (
                                (1 + rets_period.loc[common_fx_brin, col]) * (1 + fx_aligned_brin) - 1
                            )

            if rets_period.empty or len(rets_period) < 2:
                st.info("Not enough data in the selected period.")
            else:
                period_returns = (1 + rets_period).prod() - 1                  # per-ETF total period return

                # ---- Realised port / bench / active over the same window ----
                # this is the ground-truth comparator from Section 3
                period_idx = rets_period.index
                try:
                    realised_port_period = float(
                        (1 + all_port_ret[port_for_risk].reindex(period_idx).dropna()).prod() - 1
                    )
                except Exception:
                    realised_port_period = float('nan')
                try:
                    realised_bench_period = float(
                        (1 + ret_bm.iloc[:, 0].reindex(period_idx).dropna()).prod() - 1
                    )
                except Exception:
                    realised_bench_period = float('nan')
                realised_active = (
                    realised_port_period - realised_bench_period
                    if pd.notna(realised_port_period) and pd.notna(realised_bench_period)
                    else float('nan')
                )

                # ---- Time-averaged portfolio weights for the period ----
                # weighted by trading days. This is a much closer proxy for
                # "what the portfolio actually held" than the latest snapshot.
                w_df_period = w_df_risk.copy()
                if 'reb_flag' in w_df_period.columns:
                    w_df_period = w_df_period.drop(columns=['reb_flag'])
                w_df_period.columns = [
                    c[:-4] if isinstance(c, str) and c.endswith('_wgt') else c
                    for c in w_df_period.columns
                ]
                w_df_period = w_df_period.reindex(period_idx).dropna(how='all')
                if w_df_period.empty:
                    avg_w_p = last_w_risk
                else:
                    avg_w_p = w_df_period.mean()
                    avg_w_p = avg_w_p[avg_w_p > 0]

                try:
                    brinson_df = compute_brinson_attribution(
                        avg_w_p,
                        bm_w_series,
                        period_returns,
                        exposure_table,
                    )
                    # rename the weight columns so it's obvious the portfolio
                    # weights are time-averaged daily weights, not a snapshot
                    brinson_df = brinson_df.rename(columns={
                        'Port Weight': 'Avg Port Weight',
                        'Bench Weight': 'Avg Bench Weight',
                        'Active Weight': 'Avg Active Weight',
                    })
                    port_ret_total = brinson_df.attrs.get('port_total_return', float('nan'))
                    bm_ret_total = brinson_df.attrs.get('bench_total_return', float('nan'))
                    brinson_active = brinson_df.attrs.get('active_return', float('nan'))

                    # tell the user how the avg weights were built and what window they cover
                    n_days_avg = len(w_df_period) if not w_df_period.empty else 0
                    avg_window_caption = (
                        f"Portfolio weights are the mean of daily weights across "
                        f"{n_days_avg} trading days in the selected period "
                        f"({period_idx.min().strftime('%Y-%m-%d')} to "
                        f"{period_idx.max().strftime('%Y-%m-%d')}). "
                        f"Benchmark weights are the configured static blend."
                    )

                    # ---- Realised metric cards (these match Section 3 numbers) ----
                    st.markdown("##### Realised over the selected period")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(
                            "Port Return",
                            f"{realised_port_period:.2%}" if pd.notna(realised_port_period) else "N/A",
                            help=(
                                "The portfolio's actual compounded daily return over the chosen "
                                "window. This is exactly the same number you would see in Section 3 "
                                "if you restricted that section to this period."
                            ),
                        )
                    with c2:
                        st.metric(
                            "Bench Return",
                            f"{realised_bench_period:.2%}" if pd.notna(realised_bench_period) else "N/A",
                            help=(
                                "The benchmark's actual compounded daily return over the same "
                                "window. Computed by chaining each daily benchmark return."
                            ),
                        )
                    with c3:
                        st.metric(
                            "Active Return",
                            f"{realised_active:+.2%}" if pd.notna(realised_active) else "N/A",
                            help=(
                                "Portfolio return minus benchmark return over the period. "
                                "Positive means the portfolio beat the benchmark."
                            ),
                        )

                    # ---- Brinson-implied numbers + residual ----
                    st.markdown("##### Brinson decomposition (time-averaged weights × period return)")
                    st.caption(avg_window_caption)
                    b1, b2, b3, b4 = st.columns(4)
                    with b1:
                        st.metric(
                            "Brinson Port",
                            f"{port_ret_total:.2%}",
                            help=(
                                "Portfolio return implied by the single-period Brinson model. "
                                "For each bucket, take the portfolio's average weight in that "
                                "bucket times the bucket's return over the period, then sum "
                                "across buckets. This is an approximation: it ignores the "
                                "rebalancing and drift that happen during the period, so it "
                                "will not exactly equal the Realised Port Return above."
                            ),
                        )
                    with b2:
                        st.metric(
                            "Brinson Bench",
                            f"{bm_ret_total:.2%}",
                            help=(
                                "Benchmark return implied by the single-period Brinson model. "
                                "For each bucket, take the benchmark's weight in that bucket "
                                "times the bucket's return over the period, then sum across "
                                "buckets. Again this is an approximation that ignores any "
                                "rebalancing inside the period."
                            ),
                        )
                    with b3:
                        st.metric(
                            "Brinson Active",
                            f"{brinson_active:+.2%}",
                            help=(
                                "Brinson Port minus Brinson Bench. This is the active return "
                                "the model attributes via the Allocation, Selection and "
                                "Interaction effects shown below, which sum to this number."
                            ),
                        )
                    with b4:
                        residual = (
                            realised_active - brinson_active
                            if pd.notna(realised_active) and pd.notna(brinson_active)
                            else float('nan')
                        )
                        st.metric(
                            "Residual",
                            f"{residual:+.2%}" if pd.notna(residual) else "N/A",
                            help=(
                                "Realised Active Return minus Brinson Active Return. The gap "
                                "comes from rebalancing and weight drift over the period, "
                                "real-world effects that a single-period Brinson model cannot "
                                "capture. A small residual means the decomposition is a good "
                                "approximation; a large one means most of the active return came "
                                "from path-dependent effects."
                            ),
                        )

                    st.caption(
                        f"Allocation total: {brinson_df['Allocation'].sum():+.2%}, "
                        f"Selection total: {brinson_df['Selection'].sum():+.2%}, "
                        f"Interaction total: {brinson_df['Interaction'].sum():+.2%}."
                    )
                    fmt_b = {
                        'Avg Port Weight': '{:.2%}',
                        'Avg Bench Weight': '{:.2%}',
                        'Avg Active Weight': '{:+.2%}',
                        'Port Return': '{:.2%}', 'Bench Return': '{:.2%}',
                        'Allocation': '{:+.2%}', 'Selection': '{:+.2%}', 'Interaction': '{:+.2%}', 'Total': '{:+.2%}',
                    }
                    st.dataframe(
                        brinson_df.style.format(fmt_b),
                        use_container_width=True,
                    )

                    # stacked bar of effects per bucket
                    plot_b = brinson_df[['Allocation', 'Selection', 'Interaction']].copy()
                    plot_b = plot_b.loc[plot_b.abs().sum(axis=1) > 1e-6]
                    if not plot_b.empty:
                        plot_long = plot_b.reset_index().melt(id_vars=plot_b.index.name or 'index', var_name='Effect', value_name='Value')
                        idx_col = plot_b.index.name or 'index'
                        fig_brin = px.bar(
                            plot_long,
                            x=idx_col,
                            y='Value',
                            color='Effect',
                            barmode='relative',
                            title=f"Brinson Attribution by {bucket_choice}",
                            text_auto='.2%',
                        )
                        fig_brin.update_layout(yaxis_tickformat='.1%', xaxis_title=None)
                        st.plotly_chart(fig_brin, use_container_width=True, key="builder_brinson_chart")

                    st.markdown(
                        """
**Note on bucket-level returns:** Because we don't have direct returns for
each *bucket* (sector/country) of each ETF, the per-bucket return is
approximated as the contribution-weighted average of ETF returns whose
exposure to that bucket is non-zero. For Brinson on ETF portfolios this
is a standard look-through approximation; precision improves with more
single-bucket ETFs.
                        """,
                    )
                except Exception as e:
                    st.error(f"Brinson attribution failed: {e}")

    # ---- Tab 7: Efficient frontier ----
    with risk_tabs[6]:
        st.markdown(
            "Efficient frontier built from the historical mean and covariance "
            "of asset returns. Long-only with no leverage: every weight is "
            "constrained between 0 and 1 and weights must sum to 1. "
            "The portfolio and benchmark dots show what the expected "
            "annualised return and volatility would be if you had held those "
            "exact current weights over the whole historical sample. They are "
            "not the same as the realised backtest return shown in Section 3, "
            "because the actual backtest rebalances periodically and weights "
            "drift between rebalances."
        )
        ef_universe_default = list(set(last_w_risk.index) | set(bm_w_series.index) | set(selected_tickers))
        ef_universe_default = [t for t in ef_universe_default if t in ret_all.columns]
        if len(ef_universe_default) < 2:
            st.info("Need at least two assets in the universe to estimate a frontier.")
        else:
            n_pts = st.slider("Frontier resolution", 10, 60, 30, key="builder_ef_points")
            # build the panel once and use it for the curve, asset dots AND overlay
            # so all three views share identical mean / cov estimates
            ef_returns = ret_all[ef_universe_default].dropna(how='all').copy()

            # apply the same FX overlay so frontier axes (Annualised Return / Vol)
            # are in the chosen display currency and reconcile with Section 3.
            if fx_currency != "USD" and df_fx is not None and not ef_returns.empty:
                _fx_col_ef = 'USDSGD' if fx_currency == 'SGD' else ('USDHKD' if fx_currency == 'HKD' else None)
                if _fx_col_ef and _fx_col_ef in df_fx.columns:
                    _fx_ret_ef = df_fx[_fx_col_ef].copy()
                    if _fx_ret_ef.abs().max() > 1:
                        _fx_ret_ef = _fx_ret_ef.pct_change().fillna(0)
                    _common_idx_ef = ef_returns.index.intersection(_fx_ret_ef.index)
                    if len(_common_idx_ef) > 0:
                        _fx_aligned_ef = _fx_ret_ef.loc[_common_idx_ef]
                        for _col in ef_returns.columns:
                            ef_returns.loc[_common_idx_ef, _col] = (
                                (1 + ef_returns.loc[_common_idx_ef, _col]) * (1 + _fx_aligned_ef) - 1
                            )
            try:
                frontier_df = compute_efficient_frontier(ef_returns, n_points=n_pts)
                if frontier_df.empty:
                    st.warning("Frontier optimisation failed (check covariance / data window).")
                else:
                    # current portfolio + benchmark dots, computed off the SAME panel
                    ret_p, vol_p = evaluate_portfolio_on_frontier(last_w_risk, ef_returns)
                    ret_b, vol_b = evaluate_portfolio_on_frontier(bm_w_series, ef_returns)
                    # individual asset dots — also off the same panel
                    asset_mu = ef_returns.mean() * 252
                    asset_vol = ef_returns.std() * np.sqrt(252)
                    asset_dots = pd.DataFrame({'Vol': asset_vol, 'Return': asset_mu})

                    fig_ef = go.Figure()
                    fig_ef.add_trace(go.Scatter(
                        x=frontier_df['Vol'], y=frontier_df['Return'],
                        mode='lines+markers',
                        name='Efficient Frontier',
                        line=dict(color='#0052CC', width=3),
                        marker=dict(size=4),
                    ))
                    fig_ef.add_trace(go.Scatter(
                        x=asset_dots['Vol'], y=asset_dots['Return'],
                        mode='markers+text',
                        name='Individual Assets',
                        text=asset_dots.index,
                        textposition='top center',
                        marker=dict(size=8, color='#999'),
                    ))
                    if not np.isnan(ret_p):
                        fig_ef.add_trace(go.Scatter(
                            x=[vol_p], y=[ret_p],
                            mode='markers+text',
                            name=f"{port_for_risk}",
                            text=[port_for_risk],
                            textposition='top center',
                            marker=dict(size=14, color='#FF5630', symbol='star'),
                        ))
                    if not np.isnan(ret_b):
                        fig_ef.add_trace(go.Scatter(
                            x=[vol_b], y=[ret_b],
                            mode='markers+text',
                            name='Benchmark',
                            text=['Benchmark'],
                            textposition='bottom center',
                            marker=dict(size=12, color='#36B37E', symbol='diamond'),
                        ))
                    fig_ef.update_layout(
                        title=f"Efficient Frontier (long-only, latest covariance over {len(ef_returns)} days)",
                        xaxis_title="Annualised Volatility",
                        yaxis_title="Annualised Return",
                        xaxis_tickformat='.1%',
                        yaxis_tickformat='.1%',
                        plot_bgcolor='white',
                    )
                    st.plotly_chart(fig_ef, use_container_width=True, key="builder_frontier")

                    # Summary metrics for portfolio vs nearest frontier point
                    if not np.isnan(ret_p) and not np.isnan(vol_p):
                        # find frontier point with same return -> risk gap
                        same_ret = frontier_df.iloc[(frontier_df['Return'] - ret_p).abs().argsort()].iloc[0]
                        # find frontier point with same vol -> return gap
                        same_vol = frontier_df.iloc[(frontier_df['Vol'] - vol_p).abs().argsort()].iloc[0]
                        c_g1, c_g2, c_g3 = st.columns(3)
                        with c_g1:
                            st.metric("Portfolio Vol", f"{vol_p:.2%}")
                        with c_g2:
                            st.metric("Vol Gap (same return)", f"{vol_p - same_ret['Vol']:+.2%}",
                                      help="Excess volatility vs the frontier point matching your portfolio return.")
                        with c_g3:
                            st.metric("Return Gap (same vol)", f"{ret_p - same_vol['Return']:+.2%}",
                                      help="Return left on the table vs the frontier point matching your portfolio volatility.")

                    with st.expander("View frontier data"):
                        st.dataframe(
                            frontier_df.style.format({
                                'Return': '{:.2%}', 'Vol': '{:.2%}', 'Sharpe': '{:.2f}',
                                **{c: '{:.2%}' for c in frontier_df.columns if c not in ('Return', 'Vol', 'Sharpe')}
                            }),
                            use_container_width=True,
                        )
            except Exception as e:
                st.error(f"Frontier computation failed: {e}")

    # 6. factor analysis
    st.header("6. Factor Analysis (Fama-French)")
    
    st.markdown("""
    **Fama-French 5-Factor Exposure Coefficients**:
    - **Mkt**: Market risk premium
    - **SMB**: Size (Small Minus Big)
    - **HML**: Value (High Minus Low)
    - **RMW**: Profitability (Robust Minus Weak)
    - **CMA**: Investment (Conservative Minus Aggressive)
    """)
    
    try:
         df_ff_factor = load_ff_data()

         if df_ff_factor is not None:
             # FF factors are USD-denominated, so the LHS of the regression
             # MUST also be in USD. Use the pre-FX-overlay base-currency series.
             common_index = all_port_ret_base_ccy.index.intersection(df_ff_factor.index)

             if len(common_index) > 0:
                 port_factors = pc.factor_exposure_coefficient(
                     all_port_ret_base_ccy.loc[common_index],
                     df_ff_factor.loc[common_index],
                     list(results.keys())
                 )

                 bm_factors = pd.DataFrame()
                 if ret_bm_base_ccy is not None and not ret_bm_base_ccy.empty:
                     common_bm = ret_bm_base_ccy.index.intersection(common_index)
                     if len(common_bm) > 0:
                         bm_cols = ret_bm_base_ccy.columns.tolist()
                         bm_factors = pc.factor_exposure_coefficient(
                             ret_bm_base_ccy.loc[common_bm],
                             df_ff_factor.loc[common_bm],
                             bm_cols
                         )

                 all_factors = pd.concat([port_factors, bm_factors])
                 all_factors_numeric = all_factors.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0.0)

                 # show full numeric table (alpha is daily; R² is goodness-of-fit)
                 table_view = all_factors_numeric.copy()
                 fmt_map = {col: '{:.4f}' if col == 'Alpha' else '{:.2f}' for col in table_view.columns}
                 if 'R-Squared' in table_view.columns:
                     fmt_map['R-Squared'] = '{:.2%}'
                 st.dataframe(table_view.style.format(fmt_map), use_container_width=True)

                 # plot only the five betas (exclude Alpha and R-Squared from the chart)
                 beta_cols = [c for c in ['Mkt', 'SMB', 'HML', 'RMW', 'CMA'] if c in all_factors_numeric.columns]
                 betas_only = all_factors_numeric[beta_cols]

                 fig_f = go.Figure()
                 for col in betas_only.index:
                     fig_f.add_trace(go.Bar(
                         name=col,
                         x=betas_only.columns,                                 # factors on x-axis
                         y=betas_only.loc[col],
                         text=betas_only.loc[col].map('{:,.2f}'.format),
                         textposition='auto'
                     ))

                 fig_f.update_layout(
                    barmode='group',
                    title='Factor Exposures Comparison',
                    xaxis_title='Factors',
                    yaxis_title='Coefficient'
                 )
                 st.plotly_chart(fig_f, use_container_width=True, key="builder_factor_chart")
                 st.caption(
                     "Each portfolio's daily excess return (return minus the risk-free rate) "
                     "is regressed against all five Fama-French factors at once. The spec "
                     "is fixed, so no factor is ever dropped. Alpha is the daily intercept; "
                     "to annualise it, multiply by 252."
                 )
                 
             else:
                 st.warning("No overlapping dates with Fama-French data.")
         else:
             st.warning("Failed to load Fama-French data.")
    except Exception as e:
        st.error(f"Error in Factor Analysis: {e}")

    # 6. individual ETF study
    st.header("7. Individual Asset Factor Analysis")
    st.markdown(
        "Pick any single asset and run a factor regression on its daily excess "
        "returns to see how its risk and return are explained by the Fama-French "
        "factors. Useful for diagnosing whether a fund behaves like its label "
        "(e.g. is a 'value' ETF actually loading on HML?)."
    )

    if df_daily_ret is not None:
        avail_assets = sorted(df_daily_ret.columns.tolist())

        c_etf, c_bench, c_opt = st.columns([1, 1, 1])
        with c_etf:
            def_idx = avail_assets.index('DUHP') if 'DUHP' in avail_assets else 0
            study_etf = st.selectbox("Select Asset", avail_assets, index=def_idx, key="study_etf")

        with c_bench:
            def_b_idx = avail_assets.index('CSPX') if 'CSPX' in avail_assets else (avail_assets.index('SPY') if 'SPY' in avail_assets else 0)
            study_bench = st.selectbox("Select Benchmark", avail_assets, index=def_b_idx, key="study_bench")

        with c_opt:
            st.write("")
            st.write("")
            use_ff_mkt = st.checkbox(
                "Use Fama-French Market Factor",
                value=False,
                help=(
                    "Ticked: regress the asset's excess return against the standard "
                    "Fama-French market factor (Mkt-RF). Unticked: use the excess "
                    "return of the benchmark you selected as the market proxy "
                    "instead. The benchmark version is usually more interpretable "
                    "when benchmarking against a specific index ETF."
                ),
            )

        if study_etf and study_bench and df_ff_factor is not None:
             # Heads-up about the FF data window so users don't think the page
             # is broken when their portfolio data extends beyond it.
             ff_min, ff_max = df_ff_factor.index.min().date(), df_ff_factor.index.max().date()
             dr_min, dr_max = df_daily_ret.index.min().date(), df_daily_ret.index.max().date()
             if ff_max < dr_max:
                 st.caption(
                     f"📅 Fama-French factors run from {ff_min} to {ff_max}. Your daily "
                     f"return data runs from {dr_min} to {dr_max}. The regression below "
                     f"is restricted to the overlapping window (capped at {ff_max} "
                     f"because Kenneth French's website releases factor data with a lag). "
                     f"Replace Data/F-F_Research_Data_5_Factors_2x3_daily.csv with a "
                     f"newer download to extend the window."
                 )

             common_dates = df_daily_ret.index.intersection(df_ff_factor.index)

             if len(common_dates) > 60:
                 study_data = pd.DataFrame({
                     'Asset': df_daily_ret.loc[common_dates, study_etf],
                     'Benchmark': df_daily_ret.loc[common_dates, study_bench]
                 })
                 ff_sub = df_ff_factor.loc[common_dates]

                 reg_df = pd.concat([study_data, ff_sub], axis=1).dropna()

                 if reg_df.empty:
                     st.warning("No valid overlapping data after removing NaNs.")
                 else:
                     # `df_daily_ret` is already in decimal units (load_static_data
                     # divides by 100). The legacy scale_factor heuristic that used
                     # to live here was effectively dead and could mis-scale the
                     # benchmark vs the FF factors if it ever fired. Removed.
                     y_asset = reg_df['Asset'] - reg_df['RF']
                     X_factors = reg_df[['SMB', 'HML', 'RMW', 'CMA']].copy()

                     if use_ff_mkt:
                         X_factors.insert(0, 'Mkt-RF', reg_df['Mkt-RF'])
                     else:
                         bench_excess = reg_df['Benchmark'] - reg_df['RF']
                         X_factors.insert(0, 'Mkt (Bench)', bench_excess)
                         
                     X = sm.add_constant(X_factors)
                     model = sm.OLS(y_asset, X).fit()
                     
                     # vif
                     from statsmodels.stats.outliers_influence import variance_inflation_factor
                     vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                     vif_data = pd.Series(vifs, index=X.columns)
                     if 'const' in vif_data:
                         vif_data = vif_data.drop('const')
                     max_vif = vif_data.max() if not vif_data.empty else 0
                     
                     col_res1, col_res2, col_res3 = st.columns(3)
                     
                     with col_res1:
                         st.markdown("**Regression Diagnostics**")
                         r2 = model.rsquared_adj
                         alpha = model.params.get('const', 0) * 252 # annualized alpha
                         alpha_t = model.tvalues.get('const', 0)
                         
                         model_desc = "Standard FF5" if use_ff_mkt else "Benchmark + FF Factors"
                         st.metric("Adjusted R²", f"{r2:.2%}", help=f"Percentage of the asset's variance explained by the model ({model_desc}). High R² (>90%) indicates the fund is well-captured by these factors.")
                         st.metric("Annualized Alpha", f"{alpha:.2%}", help="Excess return not explained by the model (Intercept).", delta_color="normal")
                         
                         sig_label = "Significant" if abs(alpha_t) > 1.96 else "Insignificant"
                         st.write(f"Alpha t-stat: **{alpha_t:.2f}** ({sig_label})")
                         st.write(f"Max VIF: **{max_vif:.2f}**")
                         
                     with col_res2:
                         st.markdown("**Tracking Error**")

                         active_ret_user = reg_df['Asset'] - reg_df['Benchmark']
                         te = active_ret_user.std(ddof=1) * np.sqrt(252)
                         ir = (active_ret_user.mean() * 252) / te if te != 0 else 0

                         st.metric(
                             "Tracking Error (vs Bench)",
                             f"{te:.2%}",
                             help="Annualised standard deviation of the asset's daily return minus the benchmark's daily return.",
                         )
                         st.metric("Information Ratio", f"{ir:.2f}")

                     with col_res3:
                         start_d = reg_df.index[0].strftime('%Y-%m')
                         end_d = reg_df.index[-1].strftime('%Y-%m')
                         st.markdown(
                             f"**Factor Betas (Full Period: {start_d} to {end_d})**",
                             help=(
                                 "Coefficients from regressing the asset's daily excess return "
                                 "(asset return minus risk-free rate) against the chosen market "
                                 "proxy plus the four Fama-French style factors."
                             ),
                         )
                         
                         betas_df = pd.DataFrame({
                             'Factor': model.params.index,
                             'Beta': model.params.values,
                             't-stat': model.tvalues.values,
                             'p-value': model.pvalues.values
                         }).set_index('Factor').drop('const', errors='ignore')
                         
                         def color_significance(val):
                             color = '#d4edda' if abs(val) > 1.96 else 'transparent' # light green
                             return f'background-color: {color}; color: black'
                         
                         st.dataframe(betas_df.style.map(color_significance, subset=['t-stat']).format("{:.4f}"), use_container_width=True)

                     st.subheader("Dynamic Stability (Rolling 12-Month Betas)")
                     
                     window = 252 # 1 year
                     if len(reg_df) > window + 10:
                         rols = RollingOLS(y_asset, X, window=window)
                         rres = rols.fit()
                         rolling_params = rres.params.copy()
                         rolling_params.dropna(inplace=True)
                         
                         if not rolling_params.empty:
                             rolling_betas = rolling_params.drop(columns=['const'], errors='ignore')
                             fig_roll_b = px.line(rolling_betas, title=f"Rolling {window}-Day Factor Betas (vs {study_bench})")
                             fig_roll_b.add_hline(y=0, line_dash="dash", line_color="gray")
                             st.plotly_chart(fig_roll_b, use_container_width=True, key="builder_roll_beta")
                         else:
                             st.warning("Rolling regression produced no valid results.")
                     else:
                         st.info(f"Not enough data for rolling regression (Need > {window} days).")
                         
                     with st.expander("Debug Data View"):
                         st.write("Regression Data Head:", reg_df.head())
                         st.write("Y (Asset Excess) Stats:", y_asset.describe())
                         
             else:
                 st.warning("Insufficient overlapping data between Asset and Fama-French factors (Check dates).")
        else:
            st.info("Loading data...")
            
    # 7. overlap matrix
    st.header("8. Overlap Matrix")
    
    if df_underlying is not None and len(current_holdings) > 1:
        st.markdown(f"Pairwise holding overlap (%) between top assets in the portfolio.")
        
        active_tickers = list(current_holdings.keys())
        
        @st.cache_data
        def calculate_overlap_matrix(tickers, _df_underlying):
            from utils.exposure import get_concentration_exposure
            breakdowns = {}
            for t in tickers:
                dummy_w = pd.Series({f"{t}_wgt": 1.0})
                bd = get_concentration_exposure(dummy_w, _df_underlying, top_n=5000)
                if bd.sum() > 0:
                    bd = bd / bd.sum()
                else:
                    bd = pd.Series({t: 1.0}) 
                breakdowns[t] = bd
            
            n = len(tickers)
            mat = pd.DataFrame(index=tickers, columns=tickers, dtype=float)
            
            for i in range(n):
                for j in range(n):
                    t1 = tickers[i]
                    t2 = tickers[j]
                    if i == j:
                        mat.iloc[i, j] = 1.0
                    else:
                        bd1 = breakdowns[t1]
                        bd2 = breakdowns[t2]
                        common = bd1.index.intersection(bd2.index)
                        ov = 0.0
                        for k in common:
                            ov += min(bd1[k], bd2[k])
                        mat.iloc[i, j] = ov
            return mat

        overlap_mat = calculate_overlap_matrix(active_tickers, df_underlying)
        
        # mask upper triangle (keep lower + diagonal)
        mask = np.triu(np.ones_like(overlap_mat, dtype=bool), k=1) 
        overlap_mat_masked = overlap_mat.where(~mask)
        
        fig_ov = px.imshow(
            overlap_mat_masked, 
            text_auto='.1%', 
            color_continuous_scale='Blues',
            zmin=0, zmax=1,
            title="ETF Overlap Matrix",
            height=800, # increased height
            aspect="auto" # allow non-square aspect if needed, but 'equal' is better for matrix
        )
        fig_ov.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_showgrid=False, 
            yaxis_showgrid=False
        )
        st.plotly_chart(fig_ov, use_container_width=True, key="builder_overlap")

    elif df_underlying is None:
        st.info("Underlying data required for Overlap Matrix.")
    else:
        st.info("Need at least 2 assets to calculate overlap.")
    
    # 8. Excel Export Section (at the very end)
    st.markdown("---")
    st.header("9. Download Excel Report")
    st.markdown(f"Download all analysis data as an Excel file with multiple sheets. **Currency: {fx_currency}**")
    
    # prepare data for excel export
    # get top holdings
    top_holdings_for_export = None
    if df_underlying is not None:
        p_name_export = list(results.keys())[0]
        w_df_export = results[p_name_export]['weights']
        if 'reb_flag' in w_df_export.columns:
            last_w_export = w_df_export[w_df_export['reb_flag'] == True].iloc[-1] if not w_df_export[w_df_export['reb_flag'] == True].empty else w_df_export.iloc[-1]
        else:
            last_w_export = w_df_export.iloc[-1]
        conc_export = get_concentration_exposure(last_w_export, df_underlying, top_n=10)
        if not conc_export.empty:
            top_holdings_for_export = conc_export.to_frame(name='Weight')
    
    # get sector/country allocation
    sector_for_export = None
    country_for_export = None
    if 'sector_table' in charts and charts['sector_table'] is not None:
        sector_for_export = charts['sector_table']
    if 'country_table' in charts and charts['country_table'] is not None:
        country_for_export = charts['country_table']
    
    # get overlap matrix if calculated
    overlap_for_export = None
    if df_underlying is not None and len(current_holdings) > 1:
        try:
            overlap_for_export = overlap_mat
        except:
            pass

    # gather optimisation inputs / configuration
    config_for_export = {
        'Allocation Method': allocation_method,
        'Rebalance Frequency': rebal_freq,
        'Start Date': str(start_date),
        'End Date': str(end_date),
        'Optimisation Currency': res_data.get('currency', 'USD') if isinstance(res_data, dict) else 'USD',
        'Display Currency': fx_currency,
        'Universe (count)': len(selected_tickers),
        'Universe Tickers': ', '.join(selected_tickers) if selected_tickers else '',
        'Benchmark Label': benchmark_label,
        '# Benchmark Components': num_bm,
    }

    # universe sheet (with asset class metadata if available)
    universe_for_export = None
    if selected_tickers:
        uni_rows = []
        for t in selected_tickers:
            row = {'Ticker': t}
            try:
                if df_sec_list is not None and 'Ticker' in df_sec_list.columns:
                    sec_match = df_sec_list[df_sec_list['Ticker'] == t]
                    if not sec_match.empty:
                        for col_meta in df_sec_list.columns:
                            if col_meta != 'Ticker':
                                row[col_meta] = sec_match.iloc[0].get(col_meta, '')
            except Exception:
                pass
            try:
                if df_asset_class is not None and t in df_asset_class.columns:
                    ac_col = df_asset_class[t]
                    nz = ac_col[ac_col != 0]
                    if not nz.empty:
                        row['Asset Class'] = nz.idxmax()
                        row['Asset Class Weight'] = float(nz.max())
            except Exception:
                pass
            # fixed-weight target if applicable
            if allocation_method == 'Fixed Weight':
                fw = st.session_state.builder_portfolio.get('fixed_weights', {}).get(t)
                if fw is not None:
                    row['Fixed Target Weight'] = fw
            # granular constraint if applicable
            gc = st.session_state.builder_portfolio.get('granular_constraints', {}).get(t)
            if gc and gc.get('op') != 'None':
                row['Constraint'] = f"{gc.get('op')} {gc.get('val', 0):.2%}"
            uni_rows.append(row)
        universe_for_export = pd.DataFrame(uni_rows)

    # constraints log: granular + custom + validation results
    constraints_for_export = []
    for t, cfg in (st.session_state.builder_portfolio.get('granular_constraints', {}) or {}).items():
        if cfg.get('op') and cfg['op'] != 'None':
            constraints_for_export.append({
                'Scope': 'Per-ETF',
                'Target': t,
                'Operator': cfg['op'],
                'Value': cfg.get('val'),
            })
    for c in (st.session_state.get('custom_constraints', []) or []):
        constraints_for_export.append({
            'Scope': c.get('type', ''),
            'Target': c.get('target', ''),
            'Operator': c.get('operator', ''),
            'Value': c.get('value'),
        })
    # append validation log if available
    for ci in (st.session_state.get('constraint_details', []) or []):
        constraints_for_export.append({
            'Scope': f"Validation: {ci.get('type', '')}",
            'Target': ci.get('target', ''),
            'Operator': ci.get('operator', ''),
            'Value': ci.get('value'),
            'Achieved': ci.get('achieved'),
            'Warning': ci.get('warning', ''),
        })

    # benchmark composition
    benchmark_for_export = [
        {'Component': i + 1, 'Ticker': t, 'Weight': w}
        for i, (t, w) in enumerate(zip(bm_tickers_selected, bm_weights_selected))
    ]

    # asset class allocation table from charts (if present)
    asset_class_for_export = None
    if isinstance(charts, dict) and 'asset_class_table' in charts and charts['asset_class_table'] is not None:
        asset_class_for_export = charts['asset_class_table']

    # fixed income analytics (recompute once for export)
    fi_for_export = None
    try:
        if df_fi_metrics is not None and not df_fi_metrics.empty:
            has_ytm_x = 'YTM / est. yield' in df_fi_metrics.index
            has_t12m_x = 'T12M dividend yield' in df_fi_metrics.index
            has_dur_x = 'Duration' in df_fi_metrics.index
            if has_ytm_x or has_t12m_x:
                ytm_x = pd.Series(0.0, index=df_fi_metrics.columns)
                t12m_x = pd.Series(0.0, index=df_fi_metrics.columns)
                for col in df_fi_metrics.columns:
                    yv = float(df_fi_metrics.loc['YTM / est. yield', col]) if has_ytm_x and pd.notna(df_fi_metrics.loc['YTM / est. yield', col]) else 0.0
                    tv = float(df_fi_metrics.loc['T12M dividend yield', col]) if has_t12m_x and pd.notna(df_fi_metrics.loc['T12M dividend yield', col]) else 0.0
                    ytm_x[col] = yv if yv != 0 else tv
                    t12m_x[col] = tv if tv != 0 else yv
                FI_EXCL = ['CASH', 'MMF', 'FLEXI', 'GUARANTEED', 'CORE', 'EQUITY', 'GROWTH',
                           'CHINA', 'ESG', 'HEALTHCARE', 'DISRUPTIVE', 'TECHNOLOGY', 'THEMATIC',
                           'CUSTOM', 'STRATEGIC']
                fi_t = [c for c in df_fi_metrics.columns if (ytm_x[c] != 0 or t12m_x[c] != 0)
                        and not any(p in c.upper() for p in FI_EXCL)]
                if fi_t:
                    fi_calc = pd.DataFrame(index=['Weighted Duration', 'Weighted Yield to Maturity', 'Weighted 12M Dividend Yield'],
                                           columns=df_fi_metrics.columns, dtype=float)
                    fi_calc.loc['Weighted Duration'] = df_fi_metrics.loc['Duration'].astype(float) if has_dur_x else 0.0
                    fi_calc.loc['Weighted Yield to Maturity'] = ytm_x
                    fi_calc.loc['Weighted 12M Dividend Yield'] = t12m_x
                    fi_offset = ret_all.index.min() + timedelta(days=90)
                    fi_offset = max(fi_offset, pd.Timestamp(start_date))
                    fi_combined = pd.DataFrame()
                    for p_n in results.keys():
                        w_p = results[p_n]['weights']
                        p_t = [c[:-4] for c in w_p.columns if isinstance(c, str) and c.endswith('_wgt')]
                        p_t = [t for t in p_t if not any(p in t.upper() for p in FI_EXCL)]
                        if set(p_t).intersection(fi_t):
                            m_df = pc.exposure_analysis_mixed(w_p, fi_calc, fi_t, latest_data=True,
                                                              start_date=fi_offset, end_date=end_date)
                            m_df.columns = [p_n]
                            try:
                                if 'Credit rating' in df_fi_metrics.index:
                                    m_df.loc['Credit rating', p_n] = pc.calc_avg_credit_rating(
                                        w_p, df_fi_metrics.loc[['Credit rating']],
                                        latest_data=True, start_date=fi_offset, end_date=end_date)
                            except Exception:
                                m_df.loc['Credit rating', p_n] = 'N/A'
                            fi_combined = pd.concat([fi_combined, m_df], axis=1)
                    if not fi_combined.empty:
                        fi_for_export = fi_combined.T
    except Exception:
        fi_for_export = None

    # factor exposures (recompute once for export). FF factors are USD-
    # denominated so the regression LHS must be base-currency too — use the
    # pre-FX-overlay copies, not the display-currency `all_port_ret`.
    factor_for_export = None
    try:
        df_ff_export = load_ff_data()
        if df_ff_export is not None:
            common_ff = all_port_ret_base_ccy.index.intersection(df_ff_export.index)
            if len(common_ff) > 0:
                pf = pc.factor_exposure_coefficient(
                    all_port_ret_base_ccy.loc[common_ff],
                    df_ff_export.loc[common_ff],
                    list(results.keys()),
                )
                bf = pd.DataFrame()
                if ret_bm_base_ccy is not None and not ret_bm_base_ccy.empty:
                    common_ff_bm = ret_bm_base_ccy.index.intersection(common_ff)
                    if len(common_ff_bm) > 0:
                        bf = pc.factor_exposure_coefficient(
                            ret_bm_base_ccy.loc[common_ff_bm],
                            df_ff_export.loc[common_ff_bm],
                            ret_bm_base_ccy.columns.tolist(),
                        )
                factor_for_export = pd.concat([pf, bf]) if not bf.empty else pf
    except Exception:
        factor_for_export = None

    # benchmark weights as a Series (for risk analytics in export). Normalise
    # the same way the UI does so MCTE / Brinson reconcile across views.
    bm_weights_series_export = pd.Series({t: float(w) for t, w in zip(bm_tickers_selected, bm_weights_selected)})
    _bm_total_export = float(bm_weights_series_export.sum())
    if _bm_total_export > 0 and abs(_bm_total_export - 1.0) > 1e-6:
        bm_weights_series_export = bm_weights_series_export / _bm_total_export

    # restrict ret_all to the relevant universe for cov/frontier (portfolio assets + benchmark assets)
    ret_universe_export = sorted(set(selected_tickers) | set(bm_tickers_selected))
    ret_universe_export = [t for t in ret_universe_export if t in ret_all.columns]
    ret_all_for_risk_export = ret_all[ret_universe_export].dropna(how='all') if ret_universe_export else None

    # apply the same FX overlay that was applied to all_port_ret / ret_bm earlier,
    # so Excel sheets that use this panel (Brinson, MCTR/MCTE, frontier) are
    # reported in the **display currency**. Without this, the workbook mixes
    # currencies (NAV/Daily Returns in display ccy, Brinson in USD).
    if (
        ret_all_for_risk_export is not None
        and not ret_all_for_risk_export.empty
        and fx_currency != "USD"
        and df_fx is not None
    ):
        _fx_col_export = 'USDSGD' if fx_currency == 'SGD' else ('USDHKD' if fx_currency == 'HKD' else None)
        if _fx_col_export and _fx_col_export in df_fx.columns:
            _fx_ret = df_fx[_fx_col_export].copy()
            if _fx_ret.abs().max() > 1:
                _fx_ret = _fx_ret.pct_change().fillna(0)
            _common_idx = ret_all_for_risk_export.index.intersection(_fx_ret.index)
            if len(_common_idx) > 0:
                _fx_aligned = _fx_ret.loc[_common_idx]
                ret_all_for_risk_export = ret_all_for_risk_export.copy()
                for _col in ret_all_for_risk_export.columns:
                    ret_all_for_risk_export.loc[_common_idx, _col] = (
                        (1 + ret_all_for_risk_export.loc[_common_idx, _col]) * (1 + _fx_aligned) - 1
                    )

    # generate excel data
    excel_data = create_builder_excel(
        results=results,
        all_port_ret=all_port_ret,
        ret_bm=ret_bm,
        currency=fx_currency,
        top_holdings_df=top_holdings_for_export,
        sector_df=sector_for_export,
        country_df=country_for_export,
        overlap_matrix=overlap_for_export,
        config=config_for_export,
        constraints_log=constraints_for_export,
        benchmark_info=benchmark_for_export,
        universe_df=universe_for_export,
        fi_analytics_df=fi_for_export,
        asset_class_df=asset_class_for_export,
        factor_df=factor_for_export,
        df_underlying=df_underlying,
        ret_all_for_risk=ret_all_for_risk_export,
        bm_weights_series=bm_weights_series_export,
        df_sector_for_brinson=df_sector,
        df_country_for_brinson=df_country,
    )
    
    st.download_button(
        label="📥 Download Excel Report",
        data=excel_data,
        file_name=f"builder_portfolio_analysis_{fx_currency}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="builder_download_excel",
        type="primary",
        use_container_width=True
    )

