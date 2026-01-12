
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import sys
import os

# Import local modules
# Assuming running from root, pages/ is one level down.
# Streamlit handles pathing usually, but we might need to adjust path if utils not found.
# Actually, when running `streamlit run app.py`, pages are relative to root.
from utils.data_loader import load_static_data, get_data_path, download_data
import port_cons as pc
from utils.optimiser import create_constraints, optimize_portfolio, calc_port_perf
# Removing render_analysis import as we are implementing custom logic
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

# --- Page Config ---
st.set_page_config(
    page_title="Portfolio Builder",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Password Protection ---
pwd = st.sidebar.text_input("Access Password", type="password")
if pwd != "algo1":
    st.warning("Please enter the correct password to access the Builder.")
    st.stop()

# --- Custom CSS (Matching app.py) ---
st.markdown("""
<style>
    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Global Styles */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #172B4D;
        font-weight: 700;
    }
    
    /* Buttons */
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

# --- Initialization ---
data_path = get_data_path()
df_sec_list, df_asset_class, df_sector, df_country, df_fi_metrics, df_fx, df_daily_ret, df_risk_rating, df_underlying = load_static_data(data_path)

if df_sec_list is None:
    st.stop()

# --- Session State for Builder ---
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

# --- 1. Data Selection ---
st.header("1. Universe Selection")

# Define available_tickers early for Load logic
available_tickers = sorted(df_sec_list['Ticker'].tolist())

# --- Load Saved Settings ---
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
            # Restore State
            # 1. Tickers and Benchmark
            if 'tickers' in data:
                # Filter valid tickers
                valid_t = [t for t in data['tickers'] if t in available_tickers]
                st.session_state.builder_universe = valid_t
            
            if 'benchmark' in data and data['benchmark'] in available_tickers:
                st.session_state.builder_benchmark = data['benchmark']

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

# Filter by Asset Class (Optional Helper)
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
    # Initialize default if not in session state
    if 'builder_universe' not in st.session_state:
        st.session_state.builder_universe = ['CSPX', 'DUHP', 'MOAT', 'EFA', 'KWEB', 'MCHI', 'EIMI', 'DFAT', 'QQQ', 'XDEW']
    
    selected_tickers = st.multiselect(
        "Select Assets for Universe",
        options=available_tickers,
        key="builder_universe"
    )

with col3:
    benchmark = st.selectbox(
        "Select Benchmark",
        options=available_tickers,
        index=available_tickers.index('ACWI') if 'ACWI' in available_tickers else 0,
        key="builder_benchmark"
    )

# --- 2. Construction Settings ---
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
        # Normalize weights helper
        if 'fixed_weights' not in st.session_state.builder_portfolio:
            st.session_state.builder_portfolio['fixed_weights'] = {t: 1.0/len(selected_tickers) for t in selected_tickers}
            
        # Ensure new tickers are added
        for t in selected_tickers:
            if t not in st.session_state.builder_portfolio['fixed_weights']:
                 st.session_state.builder_portfolio['fixed_weights'][t] = 0.0
                 
        # UI
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

# --- 3. Constraints ---
st.subheader("Constraints")
# Granular constraints: Per-ETF with Operators

with st.expander("Define Constraints", expanded=True):
    st.markdown("#### per-ETF Constraints")
    
    if not selected_tickers:
        st.info("Select assets in 'Universe Selection' first.")
    else:
        # Initialize granular constraints if needed
        if 'granular_constraints' not in st.session_state.builder_portfolio or not st.session_state.builder_portfolio['granular_constraints']:
            # Set User Default: CSPX = 20%
            defaults = {
                'CSPX': {'op': 'Equal (=)', 'val': 0.20}
            }
            # Only set if empty
            st.session_state.builder_portfolio['granular_constraints'] = defaults
        
        # Merge defaults if missing? No, user might delete them. Initialized above once.
        
        # Display Row for each ticker
        grp_constraints = {}
        
        # Header Row
        h1, h2, h3 = st.columns([1, 1, 1])
        h1.markdown("**Ticker**")
        h2.markdown("**Operator**")
        h3.markdown("**Value (%)**")
        
        for t in selected_tickers:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.markdown(f"**{t}**")
            
            # Get existing val
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
            
            # Update
            if op != "None":
                grp_constraints[t] = {'op': op, 'val': val}
        
        st.session_state.builder_portfolio['granular_constraints'] = grp_constraints

    st.markdown("#### Group Constraints (Sector / Country / Tracking Error)")
    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 1])
    
    with cc1:
        const_type = st.selectbox("Type", ["Sector", "Country", "Tracking Error"])
        
    with cc2:
        # Populate options based on Type (Tracking Error doesn't need target)
        if const_type == "Sector":
            options = sorted(df_sector.index.unique().tolist())
            const_target = st.selectbox("Target", options)
        elif const_type == "Country":
            options = sorted(df_country.index.unique().tolist())
            const_target = st.selectbox("Target", options)
        else:  # Tracking Error
            st.selectbox(
                "Target", 
                ["vs Selected Benchmark"], 
                disabled=True,
                help="Tracking Error is calculated relative to the benchmark selected in Universe Selection."
            )
            const_target = "Benchmark"  # Placeholder - uses selected benchmark
        
    with cc3:
        if const_type == "Tracking Error":
            const_operator = "Max (<)"  # TE only supports max constraint
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
            
    # Display current constraints
    if st.session_state.custom_constraints:
        st.markdown("**Active Group Constraints:**")
        for i, c in enumerate(st.session_state.custom_constraints):
            col_txt, col_del = st.columns([8, 1])
            col_txt.text(f"{c['type']} - {c['target']} {c['operator']} {c['value']:.1%}")
            if col_del.button("❌", key=f"del_const_{i}"):
                st.session_state.custom_constraints.pop(i)
                st.rerun()

# --- 4. Optimisation Action ---
st.markdown("---")
# Warning for ignored constraints
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
        # ... (Data Loading same as before)
        # 1. Prepare Data
        currency = "USD"
        rebal_map = {
            "Semi-Annual (Apr/Oct)": [4, 10],
            "Annual (Apr)": [4],
            "Quarterly": [1, 4, 7, 10],
            "Monthly": list(range(1, 13))
        }
        rebal_months = rebal_map.get(rebal_freq, [4, 10])
        all_tickers_req = list(set(selected_tickers + [benchmark]))
        
        ret_all, ret_bm = download_data(
            data_source="Bloomberg",
            start_date=start_date,
            end_date=end_date,
            all_tickers=all_tickers_req,
            bm_tickers=[benchmark],
            currency=currency,
            df_sec_list=df_sec_list,
            df_fx=df_fx,
            data_path=data_path,
            df_daily_ret=df_daily_ret
        )
        
        if ret_all.empty:
            st.error("No data found.")
            st.stop()
            
        # 2. Constraints Builder
        valid_tickers = [t for t in selected_tickers if t in ret_all.columns]
        extra_constraints = []
        
        # A. ETF Level Constraints
        # Convert our granular_constraints map to list
        # Map: Ticker -> {op, val}
        # We need to construct base 'bounds' list OR specific constraints.
        # port_cons functions use `constraints_dict` which usually means equality checks or complex constraints.
        # Bounds are usually passed separately to minimize, but port_cons.get_maxSharpe_wgt handles bounds.
        # BUT `max_sharpe_rebal_wgt` only accepts `constraints_dict`.
        # AND it generates bounds internally as (0,1).
        
        # To strictly enforce per-ETF constraints, we must add them as 'ineq' (>=0) constraints.
        
        def make_etf_constraint(idx, op, val):
            # Use default argument binding to capture values correctly
            if op == "Max (<)": # w[i] <= val -> val - w[i] >= 0
                return {'type': 'ineq', 'fun': lambda w, i=idx, v=val: v - w[i]}
            elif op == "Min (>)": # w[i] >= val -> w[i] - val >= 0
                return {'type': 'ineq', 'fun': lambda w, i=idx, v=val: w[i] - v}
            elif op == "Equal (=)":
                return {'type': 'eq', 'fun': lambda w, i=idx, v=val: w[i] - v}
        
        # Apply ETF constraints
        if 'granular_constraints' in st.session_state.builder_portfolio:
            for t, c in st.session_state.builder_portfolio['granular_constraints'].items():
                if t in valid_tickers and c['op'] != "None":
                    t_idx = valid_tickers.index(t)
                    extra_constraints.append(make_etf_constraint(t_idx, c['op'], c['val']))
        
        # B. Group Constraints - Using weighted exposure (dot product)
        # The constraint should be: Σ(w[i] × exposure[i]) op limit
        # NOT: Σ w[i] for all i where exposure[i] > 0
        
        def make_group_constraint(exposure_vec, op, limit):
            """
            Create a constraint based on weighted exposure.
            exposure_vec: numpy array of exposures aligned with valid_tickers order
            """
            e = np.array(exposure_vec)  # Capture by value
            if op == "Max (<)":  # Σ(w[i] * exp[i]) <= limit → limit - dot(w, exp) >= 0
                return {'type': 'ineq', 'fun': lambda w, exp=e, lim=limit: lim - np.dot(w, exp)}
            elif op == "Min (>)":  # Σ(w[i] * exp[i]) >= limit → dot(w, exp) - limit >= 0
                return {'type': 'ineq', 'fun': lambda w, exp=e, lim=limit: np.dot(w, exp) - lim}
            elif op == "Equal (=)":  # Σ(w[i] * exp[i]) == limit
                return {'type': 'eq', 'fun': lambda w, exp=e, lim=limit: np.dot(w, exp) - lim}

        # Clear debug info for fresh run
        st.session_state['constraint_debug'] = []
        st.session_state['constraint_details'] = []  # Store structured info for display
        
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
            
            # df_country/df_sector format: Index=Category (Country/Sector), Columns=Tickers, Values=Exposure %
            
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
                # Tracking Error constraint: TE = sqrt((w - w_bm).T @ cov @ (w - w_bm)) * sqrt(252) <= limit
                try:
                    # Get returns for valid tickers
                    ret_subset = ret_all[valid_tickers].dropna()
                    
                    # Calculate covariance matrix (daily)
                    cov_matrix = ret_subset.cov().values
                    
                    # Benchmark weights: 100% in benchmark, 0% in portfolio assets
                    # For TE calculation, we treat benchmark as a separate asset
                    # But since we're constraining portfolio weights only, we use the benchmark returns directly
                    bm_ret = ret_bm.reindex(ret_subset.index).iloc[:, 0].dropna()
                    common_idx = ret_subset.index.intersection(bm_ret.index)
                    ret_subset = ret_subset.loc[common_idx]
                    bm_ret = bm_ret.loc[common_idx]
                    
                    # Calculate active returns covariance
                    # Active return = portfolio return - benchmark return
                    # We need: Var(w.T @ r - r_bm) = w.T @ Cov(r) @ w - 2*w.T @ Cov(r, r_bm) + Var(r_bm)
                    # Simplify: Create covariance of each asset with benchmark
                    asset_bm_cov = np.array([ret_subset[t].cov(bm_ret) for t in valid_tickers])
                    bm_var = bm_ret.var()
                    
                    limit = c['value']
                    
                    def te_constraint(w, cov=cov_matrix, asset_bm_c=asset_bm_cov, bm_v=bm_var, lim=limit):
                        """TE(w) = sqrt(Var(port - bm)) * sqrt(252) <= limit"""
                        port_var = w @ cov @ w
                        cross_term = 2 * np.dot(w, asset_bm_c)
                        active_var = port_var - cross_term + bm_v
                        # Ensure non-negative due to numerical issues
                        active_var = max(active_var, 0)
                        te = np.sqrt(active_var * 252)
                        return lim - te  # >= 0 for ineq constraint
                    
                    extra_constraints.append({'type': 'ineq', 'fun': te_constraint})
                    constraint_info['status'] = 'OK'
                    constraint_info['exposures'] = {'Tracking Error Target': limit}
                    
                except Exception as e:
                    constraint_info['status'] = 'ERROR'
                    constraint_info['warning'] = f"Failed to build TE constraint: {str(e)}"
            
            if exposure_row is not None:
                # Build exposure vector aligned with valid_tickers order
                exposure_vec = np.array([float(exposure_row.get(t, 0)) for t in valid_tickers])
                
                # Normalize if exposure is in percentage (0-100) to decimal (0-1)
                if exposure_vec.max() > 1:
                    exposure_vec = exposure_vec / 100
                
                # Store exposure for each ticker
                for i, t in enumerate(valid_tickers):
                    constraint_info['exposures'][t] = exposure_vec[i]
                
                # Calculate feasibility bounds
                min_possible = exposure_vec.min()  # If all weight in lowest exposure ETF
                max_possible = exposure_vec.max()  # If all weight in highest exposure ETF
                constraint_info['feasible_range'] = (min_possible, max_possible)
                
                # Check feasibility
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
        
        # Show infeasibility warnings immediately
        warnings_list = [ci for ci in st.session_state['constraint_details'] if ci.get('warning')]
        if warnings_list:
            warning_msgs = [f"{ci['type']} {ci['target']}: {ci['warning']}" for ci in warnings_list]
            st.warning("Constraint Feasibility Issues:\n" + "\n".join(warning_msgs))


        # Base Constraints (Sum=1, 0<=w<=1)
        base_constraints = create_constraints(0.0, 1.0) 
        full_constraints = base_constraints + extra_constraints
        
        # 3. Optimise
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

            # --- CONSTRAINT VALIDATION ---
            # Check first rebalance date weights against constraints
            if allocation_method in ["Max Sharpe", "Min Volatility"] and extra_constraints:
                first_rebal = weights_df[weights_df['reb_flag'] == True].iloc[0]
                w_vec = np.array([first_rebal[f"{t}_wgt"] for t in valid_tickers])
                
                violations = []
                tol = 0.02  # 2% tolerance
                
                # Check ETF constraints
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
                
                # Check Group constraints (Country/Sector) with weighted exposure
                for c in st.session_state.custom_constraints:
                    exposure_row = None
                    if c['type'] == 'Sector' and not df_sector.empty and c['target'] in df_sector.index:
                        exposure_row = df_sector.loc[c['target']]
                    elif c['type'] == 'Country' and not df_country.empty and c['target'] in df_country.index:
                        exposure_row = df_country.loc[c['target']]
                    elif c['type'] == 'Tracking Error':
                        # Validate TE constraint
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
            
            # Performance
            perf_df = calc_port_perf(ret_all[valid_tickers], weights_df, name_prefix="Builder")
            
            # Save Results - Format for render_analysis
            # render_analysis expects: {'results': {PortName: {'weights':..., 'perf':...}}, 'ret_all', 'ret_bm'}
            
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

# --- Save Configuration ---
st.markdown("---")
with st.expander("💾 Save Configuration as Template", expanded=False):
    save_name = st.text_input("Template Name", value="My Strategy", key="builder_save_name")
    if st.button("Save Template", key="builder_save_btn"):
        # Construct Save Data
        save_data = {
            'tickers': selected_tickers,
            'benchmark': benchmark,
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


# --- Helper Functions ---
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

# --- 5. Analysis & Visualization ---
if st.session_state.builder_portfolio['results']:
    res_data = st.session_state.builder_portfolio['results']
    results = res_data['results']
    # results is {PortName: {'weights':..., 'perf':..., 'tickers':...}}
    
    ret_all = res_data['ret_all']
    ret_bm = res_data['ret_bm']
    
    st.header("3. Analysis Results")
    
    # Combined Returns
    all_port_ret = pd.DataFrame()
    for p_name, p_res in results.items():
        p_ret = p_res['perf'][[c for c in p_res['perf'].columns if c.endswith('_port_ret')]].copy()
        p_ret.columns = [p_name]
        all_port_ret = pd.concat([all_port_ret, p_ret], axis=1)
        
    # Align with Benchmark
    common_idx = all_port_ret.index.intersection(ret_bm.index)
    all_port_ret = all_port_ret.loc[common_idx]
    ret_bm = ret_bm.loc[common_idx]
    
    # Tabs
    tabs = st.tabs([
        "Key Summary", "Calendar Returns", "Monthly Returns", "Cumulative Returns", 
        "Return/Risk Scatter", "Returns Distribution", "Rolling Volatility", 
        "Drawdown", "Correlation Heatmap", "Portfolio vs Benchmark", "Tracking Error"
    ])

    with tabs[0]: # Key Summary
        summary = get_performance_summary(all_port_ret)
        # Format
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
        
        # Historical Weights
        st.subheader("Historical Weights (Rebalance Dates)")
        p_select_w = st.selectbox("Select Portfolio", list(results.keys()), key="builder_w_sel")
        w_df = results[p_select_w]['weights']
        # w_df has 'reb_flag' column, but sometimes just 1 row.
        if 'reb_flag' in w_df.columns:
            w_rebal = w_df[w_df['reb_flag'] == True].drop(columns=['reb_flag'])
        else:
            w_rebal = w_df # Fallback
            
        w_rebal.columns = [c.replace('_wgt', '') for c in w_rebal.columns]
        # Format index to string for display
        w_rebal_display = w_rebal.copy()
        try:
             w_rebal_display.index = w_rebal_display.index.strftime('%Y-%m-%d')
        except: pass
        
        st.dataframe(w_rebal_display.style.background_gradient(cmap='Blues', axis=None).format("{:.2%}"), use_container_width=True)

    with tabs[1]: # Calendar Returns
        yearly_ret = all_port_ret.resample('Y').apply(lambda x: (1+x).prod()-1)
        yearly_ret.index = yearly_ret.index.year
        st.dataframe(yearly_ret.style.background_gradient(cmap='Blues', axis=None).format("{:.2%}"), use_container_width=True)

    with tabs[2]: # Monthly Returns
        fig = plot_monthly_heatmap(all_port_ret)
        st.plotly_chart(fig, use_container_width=True, key="builder_monthly")
    
    with tabs[3]: # Cumulative Returns
        cum_ret = (1 + all_port_ret).cumprod()
        fig = plot_cumulative_returns(cum_ret, benchmark_ret=ret_bm)
        st.plotly_chart(fig, use_container_width=True, key="builder_cum")
        
    with tabs[4]: # Risk/Ret
        fig = plot_scatter_risk_return(all_port_ret, benchmark_ret=ret_bm)
        st.plotly_chart(fig, use_container_width=True, key="builder_scatter")
        
    with tabs[5]: # Dist
        import plotly.figure_factory as ff
        hist_data = [all_port_ret[c].dropna() for c in all_port_ret.columns]
        group_labels = all_port_ret.columns
        try:
            fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=PALETTE[:len(group_labels)])
        except:
            fig = go.Figure() # Fallback
        fig.update_layout(title_text='Returns Distribution', plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True, key="builder_dist")
        
    with tabs[6]: # Roll Vol
        fig = plot_rolling_volatility(all_port_ret)
        st.plotly_chart(fig, use_container_width=True, key="builder_vol")

    with tabs[7]: # Drawdown
        fig = plot_drawdowns(all_port_ret)
        st.plotly_chart(fig, use_container_width=True, key="builder_dd")
        
    with tabs[8]: # Corr
        fig = plot_correlation_heatmap(all_port_ret)
        st.plotly_chart(fig, use_container_width=True, key="builder_corr")
        
    with tabs[9]: # Vs Benchmark
        rel_metrics, roll_excess = calculate_relative_metrics(all_port_ret.iloc[:, 0], ret_bm.iloc[:, 0])
        st.markdown("#### Relative Performance Metrics")
        rel_df = rel_metrics.to_frame().T
        # Format
        rel_format = {col: "{:.2f}" if 'Ratio' in col else "{:.2%}" for col in rel_df.columns}
        st.dataframe(rel_df.style.format(rel_format), use_container_width=True)
        st.markdown("#### Rolling 12M Excess Return")
        fig_roll = plot_cumulative_returns(roll_excess.to_frame(name="Rolling Excess Return"))
        st.plotly_chart(fig_roll, use_container_width=True, key="builder_excess")
        
    with tabs[10]: # Tracking Error Timeseries
        col_te_h, col_te_help = st.columns([10, 1])
        with col_te_h:
            st.markdown("#### Rolling 12M Tracking Error")
        with col_te_help:
            st.markdown("####")  # Spacing
            st.markdown(
                ":grey_question:", 
                help="This chart shows **realized** TE based on actual returns. TE constraints are applied at rebalance using *expected* TE from historical covariance. Realized TE may differ due to changing market conditions."
            )
        # TE = Std(Port - Bench) * sqrt(252)
        active_ret = all_port_ret.iloc[:, 0] - ret_bm.iloc[:, 0]
        roll_te = active_ret.rolling(window=252).std() * np.sqrt(252)
        
        fig_te = px.line(
            roll_te, 
            title="Rolling 12-Month Tracking Error (Annualized)", 
            labels={'value': 'Tracking Error', 'index': 'Date'}
        )
        fig_te.update_layout(yaxis_tickformat=".2%")
        st.plotly_chart(fig_te, use_container_width=True, key="builder_te_roll")

    
    # --- Exposure Analysis ---
    st.header("4. Exposure Analysis")
    
    # Use the first portfolio (Builder Portfolio)
    p_name = list(results.keys())[0]
    w_df = results[p_name]['weights']
    
    # Get weights at the last REBALANCE date, as requested
    # w_df has 'reb_flag' column
    if 'reb_flag' in w_df.columns:
        w_rebal_only = w_df[w_df['reb_flag'] == True]
        if not w_rebal_only.empty:
            last_w = w_rebal_only.iloc[-1].drop(['reb_flag'])
        else:
            # Fallback to last row if no rebalance flag (shouldn't happen in standard runs)
            last_w = w_df.iloc[-1].drop(['reb_flag'], errors='ignore')
    else:
        last_w = w_df.iloc[-1].drop(['reb_flag'], errors='ignore')
        
    last_w = last_w[last_w > 0]
    current_holdings = {k.replace('_wgt', ''): v for k, v in last_w.items()}
    
    import plotly.express as px
    
    # Calculate Charts (Sector, Country)
    # Ensure to use weight at rebalance (last_w comes from rebal check logic above, lines 636-644)
    # Check trace: 
    # if 'reb_flag' in w_df: w_rebal_only = ... last_w = w_rebal_only.iloc[-1]
    # So last_w IS correct. 
    # Just passing it cleanly.
    charts = get_exposure_charts(last_w, df_sector, df_country, df_asset_class=None) # df_asset_class None to skip
    
    # Row 1: Top Holdings
    st.markdown("### Top Holdings (Underlying)")
    t_topn = st.selectbox("Show Top", [10, 15, 20], key="builder_topn")

    conc_raw = pd.DataFrame() 
    if df_underlying is not None:
         # Use the helper from utils/exposure.py
         if 'reb_flag' in w_df.columns:
             last_w_full = w_df[w_df['reb_flag'] == True].iloc[-1]
         else:
             last_w_full = w_df.iloc[-1]
             
         conc_raw = get_concentration_exposure(last_w_full, df_underlying, top_n=t_topn)

    if not conc_raw.empty:
        # Sort explicitly descending just in case
        conc_raw = conc_raw.sort_values(ascending=False)
        
        # Take Top N
        conc_df = conc_raw.head(t_topn).to_frame(name='Weight')
        
        # Calculate sum strictly from the displayed top N items
        top_sum = conc_df['Weight'].sum()
        
        st.caption(f"Top {t_topn} holdings represent **{top_sum:.1%}** of the overall portfolio.")
        
        # Sort ascending for horizontal bar chart (so largest is at top/bottom depending on logic)
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

     # Row 2: Sector + Country
    st.markdown("### Allocation Analysis")
    show_bm_diff = st.checkbox("Show vs Benchmark", key="show_bm_diff")
        
    # Prepare Benchmark Data for Allocation if needed
    bm_charts = None
    
    if show_bm_diff:
        # Use the benchmark selected in Universe Selection (line 132-136)
        # The benchmark variable is available from the selection above
        try:
            bm_w = pd.Series({f"{benchmark}_wgt": 1.0})
            bm_charts = get_exposure_charts(bm_w, df_sector, df_country, df_asset_class=None)
            if bm_charts.get('sector_table') is None or bm_charts['sector_table'].empty:
                st.warning(f"Benchmark '{benchmark}' sector/country data not found. Comparison disabled.")
                bm_charts = None
        except Exception as e:
            st.warning(f"Could not load benchmark '{benchmark}' data: {str(e)}")
            bm_charts = None

    c_sec, c_ctry = st.columns(2)
    
    def plot_alloc_comparison(port_df, bm_df, title):
        # port_df index=Sector/Country, col=Weight
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

    # --- Factor Analysis ---
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
         # Load Fama French
         df_ff_factor = load_ff_data()
                     
         if df_ff_factor is not None:
             # Align using all_port_ret index (Portfolio Returns) instead of Universe Returns
             common_index = all_port_ret.index.intersection(df_ff_factor.index)
             
             if len(common_index) > 0:
                 # Portfolio Factor Exposure
                 port_factors = pc.factor_exposure_coefficient(
                     all_port_ret.loc[common_index],
                     df_ff_factor.loc[common_index],
                     list(results.keys())
                 )
                 
                 # Benchmark Factors
                 bm_factors = pd.DataFrame()
                 if ret_bm is not None and not ret_bm.empty:
                     # Align benchmark to the SAME common index
                     common_bm = ret_bm.index.intersection(common_index)
                     if len(common_bm) > 0:
                         # Use first column of benchmark if multiple (usually just one)
                         bm_cols = ret_bm.columns.tolist() 
                         bm_factors = pc.factor_exposure_coefficient(
                             ret_bm.loc[common_bm],
                             df_ff_factor.loc[common_bm],
                             bm_cols
                         )

                 # Combine
                 all_factors = pd.concat([port_factors, bm_factors])
                 all_factors_numeric = all_factors.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                 
                 # Plot (Reverted to Grouped Bar Chart)
                 fig_f = go.Figure()
                 for col in all_factors_numeric.index: # Iterate over Portfolio/Benchmark rows
                     fig_f.add_trace(go.Bar(
                         name=col,
                         x=all_factors_numeric.columns, # Factors on X-axis
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

    # --- 6. Individual ETF Study ---
    st.header("6. Individual Asset Factor Analysis (In Progress)")
    st.markdown("Deep dive into Factor Exposures, Dynamic Stability, and Regression Diagnostics.")
    
    # Selectors
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
            st.write("") # Spacer
            st.write("")
            use_ff_mkt = st.checkbox("Use Fama-French Market Factor", value=False, 
                                     help="Tick to use standard Fama-French 'Mkt-RF'. Untick to use your Selected Benchmark (Excess Return) as the market proxy.")
            
        if study_etf and study_bench and df_ff_factor is not None:
             # Align Dates
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
                     # Scaling Check (Proactive)
                     # If asset returns are like 1.5 (1.5%), std will be > 1.
                     # FF factors are 0.015 (1.5%).
                     scale_factor = 1.0
                     if reg_df['Asset'].std() > 1.0: # Likely percentage
                         scale_factor = 100.0
                     
                     # Apply Scaling
                     y_asset = (reg_df['Asset'] - reg_df['RF']) / scale_factor
                     
                     # Prepare Factors
                     X_factors = reg_df[['SMB', 'HML', 'RMW', 'CMA']].copy()
                     
                     if use_ff_mkt:
                         # Use Standard FF Market
                         X_factors.insert(0, 'Mkt-RF', reg_df['Mkt-RF'])
                     else:
                         # Use Selected Benchmark
                         bench_excess = (reg_df['Benchmark'] - reg_df['RF']) / scale_factor
                         X_factors.insert(0, 'Mkt (Bench)', bench_excess)
                         
                     X = sm.add_constant(X_factors)
                     
                     # 1. Full Period Regression
                     model = sm.OLS(y_asset, X).fit()
                     
                     # Calculate VIF
                     from statsmodels.stats.outliers_influence import variance_inflation_factor
                     vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                     vif_data = pd.Series(vifs, index=X.columns)
                     if 'const' in vif_data:
                         vif_data = vif_data.drop('const')
                     max_vif = vif_data.max() if not vif_data.empty else 0
                     
                     # Display Diagnostics
                     col_res1, col_res2, col_res3 = st.columns(3)
                     
                     with col_res1:
                         st.markdown("**Regression Diagnostics**")
                         r2 = model.rsquared_adj
                         alpha = model.params.get('const', 0) * 252 # Annualized Alpha
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

                     # 2. Dynamic Stability (Rolling Regression)
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
                             # Fix y-axis to be reasonable
                             # fig_roll_b.update_yaxes(range=[-2, 2]) 
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
            
    # --- 7. Overlap Matrix ---
    st.header("7. Overlap Matrix")
    
    if df_underlying is not None and len(current_holdings) > 1:
        st.markdown(f"Pairwise holding overlap (%) between top assets in the portfolio.")
        
        active_tickers = list(current_holdings.keys())
        
        @st.cache_data
        def calculate_overlap_matrix(tickers, _df_underlying):
            from utils.exposure import get_concentration_exposure
            
            # 1. Get breakdown for each ticker
            breakdowns = {}
            for t in tickers:
                dummy_w = pd.Series({f"{t}_wgt": 1.0})
                bd = get_concentration_exposure(dummy_w, _df_underlying, top_n=5000)
                if bd.sum() > 0:
                    bd = bd / bd.sum()
                else:
                    bd = pd.Series({t: 1.0}) 
                breakdowns[t] = bd
            
            # 2. Pairwise Overlap
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
        
        # Mask Upper Triangle (Keep Lower Triangle)
        # Using numpy to create a boolean mask for the upper triangle
        mask = np.triu(np.ones_like(overlap_mat, dtype=bool), k=0) 
        # k=0 masks the diagonal as well if we want ONLY lower off-diagonal. 
        # User said "only we need the lower triangular... anything above that is repeated".
        # Usually diagonal is 100% and redundant but sometimes useful for reference.
        # User said "lower triangular matrix", strictly that usually includes diagonal over time.
        # Let's mask k=1 (exclude diagonal from mask) -> Show Diagonal? 
        # Or k=0 (mask diagonal) -> Hide Diagonal?
        # User image shows the full square. Lower triangular typically includes diagonal.
        # Let's keep diagonal (Auto-correlation is 1).
        mask = np.triu(np.ones_like(overlap_mat, dtype=bool), k=1) 
        
        # Set masked values to NaN so Plotly doesn't plot them
        overlap_mat_masked = overlap_mat.where(~mask)
        
        # Plot with increased size
        fig_ov = px.imshow(
            overlap_mat_masked, 
            text_auto='.1%', 
            color_continuous_scale='Blues',
            zmin=0, zmax=1,
            title="ETF Overlap Matrix",
            height=800, # Increased height
            aspect="auto" # Allow non-square aspect if needed, but 'equal' is better for matrix
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

