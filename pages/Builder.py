
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
                'ret_bm': ret_bm
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
def create_builder_excel(results, all_port_ret, ret_bm, currency, top_holdings_df=None, sector_df=None, country_df=None, overlap_matrix=None):
    """create excel file with all builder analysis data across multiple sheets"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # sheet 1: NAV (cumulative returns indexed to 1)
        nav_df = (1 + all_port_ret).cumprod()
        nav_df.index.name = 'Date'
        nav_df.to_excel(writer, sheet_name=f'NAV ({currency})')
        
        # sheet 2: daily returns
        daily_ret = all_port_ret.copy()
        daily_ret.index.name = 'Date'
        daily_ret.to_excel(writer, sheet_name=f'Daily Returns ({currency})')
        
        # sheet 3: daily weights (all dates)
        daily_weights_list = []
        for port_name, port_data in results.items():
            if 'weights' in port_data:
                w_df = port_data['weights'].copy()
                w_df['Portfolio'] = port_name
                daily_weights_list.append(w_df)
        
        if daily_weights_list:
            all_daily_weights = pd.concat(daily_weights_list)
            all_daily_weights.index.name = 'Date'
            all_daily_weights.columns = [c.replace('_wgt', '') for c in all_daily_weights.columns]
            all_daily_weights.to_excel(writer, sheet_name='Daily Weights')
        
        # sheet 4: rebalance weights only
        rebal_weights_list = []
        for port_name, port_data in results.items():
            if 'weights' in port_data:
                w_df = port_data['weights'].copy()
                if 'reb_flag' in w_df.columns:
                    w_rebal = w_df[w_df['reb_flag'] == True].drop(columns=['reb_flag'])
                    w_rebal['Portfolio'] = port_name
                    rebal_weights_list.append(w_rebal)
        
        if rebal_weights_list:
            all_rebal_weights = pd.concat(rebal_weights_list)
            all_rebal_weights.index.name = 'Date'
            all_rebal_weights.columns = [c.replace('_wgt', '') for c in all_rebal_weights.columns]
            all_rebal_weights.to_excel(writer, sheet_name='Rebalance Weights')
        
        # sheet 5: key summary
        summary = get_performance_summary(all_port_ret)
        summary.to_excel(writer, sheet_name='Key Summary')
        
        # sheet 6: cumulative returns
        cum_ret = (1 + all_port_ret).cumprod() - 1
        cum_ret.index.name = 'Date'
        cum_ret.to_excel(writer, sheet_name=f'Cumulative Returns ({currency})')
        
        # sheet 7: monthly returns
        monthly_ret = all_port_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_ret.index.name = 'Date'
        monthly_ret.to_excel(writer, sheet_name='Monthly Returns')
        
        # sheet 8: calendar returns (yearly)
        yearly_ret = all_port_ret.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        yearly_ret.index = yearly_ret.index.year
        yearly_ret.index.name = 'Year'
        yearly_ret.to_excel(writer, sheet_name='Calendar Returns')
        
        # sheet 9: rolling volatility (252-day)
        rolling_vol = all_port_ret.rolling(window=252).std() * np.sqrt(252)
        rolling_vol.index.name = 'Date'
        rolling_vol.to_excel(writer, sheet_name='Rolling Volatility')
        
        # sheet 10: drawdowns
        cum_ret_nav = (1 + all_port_ret).cumprod()
        running_max = cum_ret_nav.cummax()
        drawdowns = (cum_ret_nav - running_max) / running_max
        drawdowns.index.name = 'Date'
        drawdowns.to_excel(writer, sheet_name='Drawdowns')
        
        # sheet 11: correlation matrix
        corr_matrix = all_port_ret.corr()
        corr_matrix.to_excel(writer, sheet_name='Correlation Matrix')
        
        # sheet 12: risk-return summary
        risk_return = pd.DataFrame({
            'Annualized Return': all_port_ret.mean() * 252,
            'Annualized Volatility': all_port_ret.std() * np.sqrt(252),
            'Sharpe Ratio': (all_port_ret.mean() * 252) / (all_port_ret.std() * np.sqrt(252))
        })
        risk_return.to_excel(writer, sheet_name='Risk-Return')
        
        # sheet 13: benchmark NAV (if available)
        if ret_bm is not None and not ret_bm.empty:
            bm_nav = (1 + ret_bm).cumprod()
            bm_nav.index.name = 'Date'
            bm_nav.to_excel(writer, sheet_name=f'Benchmark NAV ({currency})')
        
        # sheet 14: top holdings
        if top_holdings_df is not None and not top_holdings_df.empty:
            top_holdings_df.to_excel(writer, sheet_name='Top 10 Holdings')
        
        # sheet 15: sector allocation
        if sector_df is not None and not sector_df.empty:
            sector_df.to_excel(writer, sheet_name='Sector Allocation')
        
        # sheet 16: country allocation
        if country_df is not None and not country_df.empty:
            country_df.to_excel(writer, sheet_name='Country Allocation')
        
        # sheet 17: overlap matrix
        if overlap_matrix is not None and not overlap_matrix.empty:
            overlap_matrix.to_excel(writer, sheet_name='Overlap Matrix')
    
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
    all_port_ret = all_port_ret.loc[common_idx]
    ret_bm = ret_bm.loc[common_idx]
    
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
        
        summary_formatted = summary.copy()
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
            
        w_rebal.columns = [c.replace('_wgt', '') for c in w_rebal.columns]
        w_rebal_display = w_rebal.copy()
        try:
             w_rebal_display.index = w_rebal_display.index.strftime('%Y-%m-%d')
        except: pass
        
        st.dataframe(w_rebal_display.style.background_gradient(cmap='Blues', axis=None).format("{:.2%}"), use_container_width=True)

    with tabs[1]:
        yearly_ret = all_port_ret.resample('Y').apply(lambda x: (1+x).prod()-1)
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
        roll_te = active_ret.rolling(window=252).std() * np.sqrt(252)
        
        fig_te = px.line(
            roll_te, 
            title="Rolling 12-Month Tracking Error (Annualized)", 
            labels={'value': 'Tracking Error', 'index': 'Date'}
        )
        fig_te.update_layout(yaxis_tickformat=".2%")
        st.plotly_chart(fig_te, use_container_width=True, key="builder_te_roll")
    
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
    current_holdings = {k.replace('_wgt', ''): v for k, v in last_w.items()}
    
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

    # 5. factor analysis
    st.header("5. Factor Analysis (Fama-French)")
    
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
             common_index = all_port_ret.index.intersection(df_ff_factor.index)
             
             if len(common_index) > 0:
                 port_factors = pc.factor_exposure_coefficient(
                     all_port_ret.loc[common_index],
                     df_ff_factor.loc[common_index],
                     list(results.keys())
                 )
                 
                 bm_factors = pd.DataFrame()
                 if ret_bm is not None and not ret_bm.empty:
                     common_bm = ret_bm.index.intersection(common_index)
                     if len(common_bm) > 0:
                         bm_cols = ret_bm.columns.tolist()
                         bm_factors = pc.factor_exposure_coefficient(
                             ret_bm.loc[common_bm],
                             df_ff_factor.loc[common_bm],
                             bm_cols
                         )

                 all_factors = pd.concat([port_factors, bm_factors])
                 all_factors_numeric = all_factors.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                 
                 fig_f = go.Figure()
                 for col in all_factors_numeric.index:
                     fig_f.add_trace(go.Bar(
                         name=col,
                         x=all_factors_numeric.columns, # factors on x-axis
                         y=all_factors_numeric.loc[col],
                         text=all_factors_numeric.loc[col].map('{:,.2f}'.format),
                         textposition='auto'
                     ))
                     
                 fig_f.update_layout(
                    barmode='group',
                    title='Factor Exposures Comparison',
                    xaxis_title='Factors',
                    yaxis_title='Coefficient'
                 )
                 st.plotly_chart(fig_f, use_container_width=True, key="builder_factor_chart")
                 
             else:
                 st.warning("No overlapping dates with Fama-French data.")
         else:
             st.warning("Failed to load Fama-French data.")
    except Exception as e:
        st.error(f"Error in Factor Analysis: {e}")

    # 6. individual ETF study
    st.header("6. Individual Asset Factor Analysis (In Progress)")
    st.markdown("Deep dive into Factor Exposures, Dynamic Stability, and Regression Diagnostics.")
    
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
            use_ff_mkt = st.checkbox("Use Fama-French Market Factor", value=False, 
                                     help="Tick to use standard Fama-French 'Mkt-RF'. Untick to use your Selected Benchmark (Excess Return) as the market proxy.")
            
        if study_etf and study_bench and df_ff_factor is not None:
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
                     # check scaling - if returns look like percentages
                     scale_factor = 1.0
                     if reg_df['Asset'].std() > 1.0:
                         scale_factor = 100.0
                     
                     y_asset = (reg_df['Asset'] - reg_df['RF']) / scale_factor
                     X_factors = reg_df[['SMB', 'HML', 'RMW', 'CMA']].copy()
                     
                     if use_ff_mkt:
                         X_factors.insert(0, 'Mkt-RF', reg_df['Mkt-RF'])
                     else:
                         bench_excess = (reg_df['Benchmark'] - reg_df['RF']) / scale_factor
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
                         
                         active_ret_user = (reg_df['Asset'] - reg_df['Benchmark']) / scale_factor
                         te = active_ret_user.std() * np.sqrt(252)
                         ir = (active_ret_user.mean() * 252) / te if te != 0 else 0
                         
                         st.metric("Tracking Error (vs Bench)", f"{te:.2%}", help="Annualized Standard Deviation of Active Returns (Asset - Benchmark).")
                         st.metric("Information Ratio", f"{ir:.2f}")

                     with col_res3:
                         start_d = reg_df.index[0].strftime('%Y-%m')
                         end_d = reg_df.index[-1].strftime('%Y-%m')
                         st.markdown(f"**Factor Betas (Full Period: {start_d} to {end_d})**", help="Coefficients from the OLS regression: (Asset-RF) = Alpha + β(Factors)...")
                         
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
    st.header("7. Overlap Matrix")
    
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
    st.header("8. Download Excel Report")
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
    
    # generate excel data
    excel_data = create_builder_excel(
        results=results,
        all_port_ret=all_port_ret,
        ret_bm=ret_bm,
        currency=fx_currency,
        top_holdings_df=top_holdings_for_export,
        sector_df=sector_for_export,
        country_df=country_for_export,
        overlap_matrix=overlap_for_export
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

