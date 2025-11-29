
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import sys
import os

# Import local modules
# Import local modules
from utils.data_loader import load_static_data, get_data_path, download_data
from utils.optimiser import optimize_portfolio, create_constraints, calc_port_perf
from utils.plotting import (
    plot_cumulative_returns, plot_drawdowns, plot_correlation_heatmap, 
    plot_scatter_risk_return, plot_allocation_pie, plot_bar_chart, 
    plot_rolling_volatility, plot_monthly_heatmap
)
import plotly.graph_objects as go
import plotly.express as px
from utils.exposure import get_exposure_charts
from utils.performance import get_performance_summary, calculate_relative_metrics
import port_cons as pc
from styles.syfe_colors import PALETTE

# --- Page Config ---
st.set_page_config(
    page_title="Portfolio Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Syfe Branding & Remove Streamlit Branding) ---
st.markdown("""
<style>
    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hide the header decoration */
    header[data-testid="stHeader"] {
        background: transparent;
        z-index: 1;
    }
    
    /* Hide the toolbar (Deploy button, etc) */
    [data-testid="stToolbar"] {
        visibility: hidden;
    }
    
    /* Hide the sidebar toggle button to make sidebar static */
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }
    
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    button[kind="header"] {
        display: none !important;
    }
    
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
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #172B4D;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0052CC !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---
data_path = get_data_path()
df_sec_list, df_asset_class, df_sector, df_country, df_fi_metrics, df_fx = load_static_data(data_path)

if df_sec_list is None:
    st.stop()

# --- Session State ---
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = []
    # Default Portfolio
    st.session_state.portfolios.append({
        'id': 1,
        'name': 'Portfolio 1',
        'tickers': [],
        'allocation_type': 'Max sharpe',
        'rebal_freq': 'Annually',
        'min_weight': 0.0,
        'max_weight': 1.0,
        'cash_invested': 0.0,
        'fixed_weights': {}
    })

if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0

# --- Sidebar ---
with st.sidebar:
    # Syfe Logo (using text for now if image fails, or placeholder)
    st.markdown("## Syfe Portfolio Visualizer")

# --- Client Data Upload ---
st.header("1. Client Data Upload")
uploaded_file = st.file_uploader("Upload Client Portfolio CSV", type=['csv'])

if uploaded_file:
    try:
        df_client = pd.read_csv(uploaded_file)
        
        # Filter for ACTIVE portfolios
        if 'status' in df_client.columns:
            df_client = df_client[df_client['status'] == 'ACTIVE']
        
        # Mapping logic
        CLIENT_PORTFOLIO_MAPPING = {
            'GLOBAL_EQUITY_100': 'Core Equity100',
            'INCOME_ENHANCE': 'Income Enhance',
            'INCOME_PRESERVE': 'Income Preserve',
            'REIT': 'REIT 100',
            'CASH_PLUS_GUARANTEED_SGD': 'Cash SGD guaranteed',
            'SRS_INCOME_ENHANCE': 'SRS Income Enhance',
            'SRS_INCOME_PRESERVE': 'SRS Income Preserve',
            'SRS_CASH_PLUS': 'SRS Cash SGD flexi',
            'SRS_GLOBAL_EQUITY_100': 'SRS Equity100',
            'SRS_INCOME_ENHANCED': 'SRS Income Enhance', # Handle potential typo
        }
        
        # Process data to get ticker weights
        client_holdings = {}
        total_nav = 0.0
        
        for index, row in df_client.iterrows():
            p_type = row['type']
            currency = row['currency'] if 'currency' in row else 'SGD'
            nav_str = str(row['nav_in_sgd']).replace(',', '')
            try:
                nav = float(nav_str)
            except:
                nav = 0.0
            
            if nav < 50: # Ignore small balances
                continue
                
            ticker = None
            
            # Special handling for CASH_PLUS
            if p_type == 'CASH_PLUS':
                if currency == 'USD':
                    ticker = 'Cash USD flexi'
                else:
                    ticker = 'Cash SGD flexi'
            elif p_type in CLIENT_PORTFOLIO_MAPPING:
                ticker = CLIENT_PORTFOLIO_MAPPING[p_type]
            
            if ticker:
                client_holdings[ticker] = client_holdings.get(ticker, 0.0) + nav
                total_nav += nav
        
        if total_nav > 0:
            # Calculate weights
            client_weights = {k: v/total_nav for k, v in client_holdings.items()}
            
            # Calculate weights
            client_weights = {k: v/total_nav for k, v in client_holdings.items()}
            
            # Calculate Asset Allocation
            asset_allocation = {}
            for ticker, weight in client_weights.items():
                if ticker in df_asset_class.columns:
                    # Get asset breakdown for this ticker
                    ticker_assets = df_asset_class[ticker]
                    for asset, asset_weight in ticker_assets.items():
                        if asset_weight > 0:
                            asset_allocation[asset] = asset_allocation.get(asset, 0.0) + (asset_weight * weight)
                else:
                    # Fallback for unknown tickers (e.g. assume Cash or Unclassified)
                    asset_allocation['Unclassified'] = asset_allocation.get('Unclassified', 0.0) + weight

            # Display Pie Charts
            st.subheader("Client Portfolio Allocation")
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                st.markdown("**By Portfolio Type**")
                fig_client = px.pie(
                    values=list(client_weights.values()),
                    names=list(client_weights.keys()),
                    title=f"Total NAV: ${total_nav:,.2f}"
                )
                st.plotly_chart(fig_client, use_container_width=True)
            
            with col_p2:
                st.markdown("**By Asset Class**")
                # Filter out small values for cleaner chart
                final_asset_alloc = {k: v for k, v in asset_allocation.items() if v > 0.001}
                fig_asset = px.pie(
                    values=list(final_asset_alloc.values()),
                    names=list(final_asset_alloc.keys()),
                    title="Asset Allocation"
                )
                st.plotly_chart(fig_asset, use_container_width=True)
            
            # Load into Portfolio 1 Button
            if st.button("Load into Portfolio 1"):
                # Update underlying data
                st.session_state.portfolios[0]['name'] = "Client Portfolio"
                st.session_state.portfolios[0]['tickers'] = list(client_weights.keys())
                st.session_state.portfolios[0]['allocation_type'] = 'Fixed Weight'
                st.session_state.portfolios[0]['fixed_weights'] = client_weights
                st.session_state.portfolios[0]['cash_invested'] = total_nav
                
                # Update dollar amounts and decimal weights dictionaries
                st.session_state.portfolios[0]['dollar_amounts'] = client_holdings
                st.session_state.portfolios[0]['decimal_weights'] = client_weights
                
                # Increment refresh counter to force widget update
                st.session_state.refresh_counter += 1
                
                st.success("Loaded client portfolio into Portfolio 1!")
                st.rerun()
        else:
            st.warning("No valid active portfolios found with NAV >= $50.")
            
    except Exception as e:
        st.error(f"Error processing file: {e}")

# --- Base Settings ---
st.header("Base Settings")

col1, col2, col3 = st.columns(3)
with col1:
    data_source = st.radio("Data Source", ['Bloomberg', 'Yahoo Finance'])
with col2:
    start_date = st.date_input("Start Date", value=date(2015, 1, 1))
with col3:
    end_date = st.date_input("End Date", value=date.today())

col4, col5, col6 = st.columns(3)
with col4:
    bm_tickers = st.multiselect(
        "Benchmark Tickers",
        options=sorted(df_sec_list['Ticker'].tolist()),
        default=['SPY', 'ACWI', 'URTH']
    )
with col5:
    backtest_offset_days = st.number_input("Observation Window (Days)", min_value=0, value=0)
with col6:
    currency = st.selectbox("Currency", ['USD', 'SGD', 'HKD'])

st.markdown("---")



# --- Portfolio Construction ---
st.header("2. Portfolio Construction")

available_tickers = sorted(df_sec_list['Ticker'].tolist())

col_add, col_del = st.columns([1, 4])
with col_add:
    if st.button("➕ Add Portfolio"):
        new_id = len(st.session_state.portfolios) + 1
        st.session_state.portfolios.append({
            'id': new_id,
            'name': f'Portfolio {new_id}',
            'tickers': [],
            'allocation_type': 'Max sharpe',
            'rebal_freq': 'Annually',
            'min_weight': 0.0,
            'max_weight': 1.0,
            'cash_invested': 0.0,
            'fixed_weights': {}
        })
        st.rerun()



for i, port in enumerate(st.session_state.portfolios):
    with st.expander(f"{port['name']}", expanded=True):
        # Add remove button at top right
        if len(st.session_state.portfolios) > 1:
            if st.button(f"🗑️ Remove Portfolio", key=f"remove_{i}"):
                st.session_state.portfolios.pop(i)
                st.rerun()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.portfolios[i]['name'] = st.text_input(f"Name", value=port['name'], key=f"name_{i}_{st.session_state.refresh_counter}")
            st.session_state.portfolios[i]['tickers'] = st.multiselect(
                f"Select Tickers",
                options=available_tickers,
                default=port['tickers'],
                key=f"tickers_{i}_{st.session_state.refresh_counter}"
            )
        with c2:
            st.session_state.portfolios[i]['allocation_type'] = st.selectbox(
                f"Allocation", 
                ['Max sharpe', 'Min vol', 'Equal weight', 'Fixed Weight'], 
                index=['Max sharpe', 'Min vol', 'Equal weight', 'Fixed Weight'].index(port['allocation_type']) if port['allocation_type'] in ['Max sharpe', 'Min vol', 'Equal weight', 'Fixed Weight'] else 0, 
                key=f"alloc_{i}_{st.session_state.refresh_counter}"
            )
            st.session_state.portfolios[i]['rebal_freq'] = st.selectbox(
                f"Rebalance", 
                ['Monthly', 'Quarterly', 'Semi-annually', 'Annually'], 
                index=['Monthly', 'Quarterly', 'Semi-annually', 'Annually'].index(port['rebal_freq']), 
                key=f"rebal_{i}_{st.session_state.refresh_counter}"
            )
        with c3:
            st.session_state.portfolios[i]['min_weight'] = st.number_input(f"Min Weight", min_value=0.0, max_value=1.0, value=port['min_weight'], step=0.01, key=f"min_{i}_{st.session_state.refresh_counter}")
            st.session_state.portfolios[i]['max_weight'] = st.number_input(f"Max Weight", min_value=0.0, max_value=1.0, value=port['max_weight'], step=0.01, key=f"max_{i}_{st.session_state.refresh_counter}")
        
        # Cash Invested Input
        st.session_state.portfolios[i]['cash_invested'] = st.number_input(
            "Cash Invested (Optional, for Cash Flow Analysis only)",
            min_value=0.0,
            value=port.get('cash_invested', 0.0),
            step=1000.0,
            format="%.2f",
            key=f"cash_{i}_{st.session_state.refresh_counter}"
        )
        
        # Fixed Weight Inputs
        if st.session_state.portfolios[i]['allocation_type'] == 'Fixed Weight':
            input_type = st.radio("Input Type", ["Amounts ($)", "Weights (Decimal)"], key=f"input_type_{i}_{st.session_state.refresh_counter}", horizontal=True)
            
            st.markdown(f"##### Fixed Weights ({input_type})")
            fixed_cols = st.columns(3)
            fixed_weights = {}
            
            if input_type == "Amounts ($)":
                total_amount = 0.0
                dollar_amounts = {}
                
                # Initialize dollar amounts if not present
                if 'dollar_amounts' not in st.session_state.portfolios[i]:
                    st.session_state.portfolios[i]['dollar_amounts'] = {}

                for idx, ticker in enumerate(port['tickers']):
                    with fixed_cols[idx % 3]:
                        amount = st.number_input(
                            f"{ticker} ($)", 
                            min_value=0.0, 
                            value=st.session_state.portfolios[i]['dollar_amounts'].get(ticker, 1000.0),
                            step=100.0,
                            key=f"amt_{i}_{ticker}_{st.session_state.refresh_counter}"
                        )
                        dollar_amounts[ticker] = amount
                        total_amount += amount
                
                st.session_state.portfolios[i]['dollar_amounts'] = dollar_amounts
                
                if total_amount > 0:
                    st.info(f"Total Portfolio Value: ${total_amount:,.2f}")
                    # Calculate weights
                    for ticker, amount in dollar_amounts.items():
                        fixed_weights[ticker] = amount / total_amount
                else:
                    st.warning("Total amount is 0. Please enter amounts.")
            
            else: # Weights (Decimal)
                total_weight = 0.0
                
                # Initialize decimal weights if not present
                if 'decimal_weights' not in st.session_state.portfolios[i]:
                    st.session_state.portfolios[i]['decimal_weights'] = {}

                for idx, ticker in enumerate(port['tickers']):
                    with fixed_cols[idx % 3]:
                        wgt = st.number_input(
                            f"{ticker} (%)", 
                            min_value=0.0, 
                            max_value=1.0,
                            value=st.session_state.portfolios[i]['decimal_weights'].get(ticker, 1.0/len(port['tickers']) if len(port['tickers']) > 0 else 0.0),
                            step=0.05,
                            format="%.2f",
                            key=f"wgt_{i}_{ticker}_{st.session_state.refresh_counter}"
                        )
                        fixed_weights[ticker] = wgt
                        total_weight += wgt
                
                st.session_state.portfolios[i]['decimal_weights'] = fixed_weights
                
                if abs(total_weight - 1.0) > 0.001:
                    st.warning(f"Total Weight: {total_weight:.2%}. It should sum to 100%.")
                else:
                    st.success(f"Total Weight: {total_weight:.2%}")
            
            st.session_state.portfolios[i]['fixed_weights'] = fixed_weights

# Run Optimization Button
st.markdown("---")
if st.button("🚀 Run Optimization", use_container_width=True, type="primary"):
    st.session_state.run_optimization = True
else:
    st.session_state.run_optimization = False

# --- Optimization Logic ---
if st.session_state.run_optimization:
    with st.spinner("Optimizing Portfolios..."):
        # 1. Gather Tickers
        all_tickers = set(bm_tickers)
        for p in st.session_state.portfolios:
            all_tickers.update(p['tickers'])
        all_tickers = list(all_tickers)
        
        if not all_tickers:
            st.error("No tickers selected.")
            st.stop()
            
        # 2. Download Data
        ret_all, ret_bm = download_data(data_source, start_date, end_date, all_tickers, bm_tickers, currency, df_sec_list, df_fx, data_path)
        
        if ret_all.empty:
            st.error("No data downloaded.")
            st.stop()
            
        earliest_start_date = ret_all.index.min()
        start_opt_date = earliest_start_date + timedelta(days=backtest_offset_days)
        
        # 3. Optimize
        results = {}
        for p in st.session_state.portfolios:
            try:
                # Filter valid tickers
                valid_tickers = [t for t in p['tickers'] if t in ret_all.columns]
                if not valid_tickers:
                    continue
                    
                constraints = create_constraints(p['min_weight'], p['max_weight'])
                
                rebal_map = {'Annually': [1], 'Semi-annually': [1, 7], 'Quarterly': [1, 4, 7, 10], 'Monthly': list(range(1, 13))}
                rebal_months = rebal_map.get(p['rebal_freq'], [1])
                
                weights_df = optimize_portfolio(
                    ret_all[valid_tickers],
                    start_date=start_opt_date,
                    end_date=end_date,
                    constraints_dict=constraints,
                    method=p['allocation_type'].lower().replace(' ', '_'),
                    rebal_months=rebal_months,
                    fixed_weights=p['fixed_weights'] if p['allocation_type'] == 'Fixed Weight' else None,
                    min_weight=p['min_weight'],
                    max_weight=p['max_weight']
                )
                
                perf_df = calc_port_perf(ret_all[valid_tickers], weights_df, name_prefix=p['name'])
                
                results[p['name']] = {
                    'weights': weights_df,
                    'perf': perf_df,
                    'tickers': valid_tickers
                }
            except Exception as e:
                st.error(f"Error optimizing {p['name']}: {e}")
        
        st.session_state.optimization_results = {
            'results': results,
            'ret_all': ret_all,
            'ret_bm': ret_bm
        }

# --- Results Display ---
if st.session_state.optimization_results:
    st.header("3. Analysis Results")
    
    res_data = st.session_state.optimization_results
    results = res_data['results']
    ret_all = res_data['ret_all']
    ret_bm = res_data['ret_bm']
    
    # Calculate start_opt_date for analytics
    earliest_start_date = ret_all.index.min()
    start_opt_date = earliest_start_date + timedelta(days=backtest_offset_days)
    
    # Combine Returns
    all_port_ret = pd.DataFrame()
    for p_name, p_res in results.items():
        p_ret = p_res['perf'][[c for c in p_res['perf'].columns if c.endswith('_port_ret')]]
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
        "Drawdown", "Correlation Heatmap", "Portfolio vs Benchmark"
    ])
    

    with tabs[0]: # Key Summary
        summary = get_performance_summary(all_port_ret)
        
        # Custom formatting
        format_dict = {col: "{:.2%}" for col in summary.index}
        # Override for ratios
        for idx in ['Sharpe Ratio', 'Sortino Ratio', 'Information Ratio']:
            if idx in format_dict:
                format_dict[idx] = "{:.2f}"
                
        st.dataframe(summary.style.format(format_dict), use_container_width=True)
        
        st.markdown("---")
        
        # --- Historical Weights View ---
        st.subheader("Historical Weights (Rebalance Dates)")
        
        w_select = st.selectbox("Select Portfolio to View Weights", list(results.keys()), key="w_view_sel")
        
        if w_select:
            w_df = results[w_select]['weights']
            # Filter for rebalance dates
            w_rebal = w_df[w_df['reb_flag'] == True].drop(columns=['reb_flag'])
            
            # Clean column names
            w_rebal.columns = [c.replace('_wgt', '') for c in w_rebal.columns]
            
            # Format index
            w_rebal.index = w_rebal.index.strftime('%Y-%m-%d')
            
            st.dataframe(w_rebal.style.background_gradient(cmap='Blues', axis=None).format("{:.2%}"), use_container_width=True)

        st.markdown("---")

        # --- Fixed Income Analytics ---
        st.subheader("Fixed Income Analytics")
        
        # Identify FI tickers
        try:
            # Check if 'YTM / est. yield' exists in index
            if 'YTM / est. yield' in df_fi_metrics.index:
                # Get tickers with non-zero YTM
                fi_tickers = df_fi_metrics.columns[df_fi_metrics.loc['YTM / est. yield'] != 0].tolist()
                
                if fi_tickers:
                    port_fi_metrics = pd.DataFrame()
                    has_fi = False
                    
                    for p_name in results.keys():
                        # Get weights
                        w_df = results[p_name]['weights']
                        
                        # Check if portfolio has any FI tickers
                        p_tickers = [c.replace('_wgt', '') for c in w_df.columns if c.endswith('_wgt')]
                        common_fi = list(set(p_tickers).intersection(fi_tickers))
                        
                        if common_fi:
                            has_fi = True
                            # Calculate metrics
                            # exposure_analysis_mixed expects weights with _wgt suffix
                            ytm_duration_df = pc.exposure_analysis_mixed(
                                w_df, 
                                df_fi_metrics.loc[['Duration', 'YTM / est. yield']].astype(float), 
                                fi_tickers, 
                                latest_data=True, 
                                start_date=start_opt_date, 
                                end_date=end_date
                            )
                            ytm_duration_df.columns = [p_name]
                            
                            # Credit Rating
                            try:
                                cr = pc.calc_avg_credit_rating(
                                    w_df, 
                                    df_fi_metrics.loc[['Credit rating']], 
                                    latest_data=True,
                                    start_date=start_opt_date, 
                                    end_date=end_date
                                )
                                ytm_duration_df.loc['Credit rating'] = cr
                            except Exception as e:
                                ytm_duration_df.loc['Credit rating'] = 'N/A'
                                
                            port_fi_metrics = pd.concat([port_fi_metrics, ytm_duration_df], axis=1)
                    
                    # Add Benchmark FI Metrics if available
                    bm_fi_cols = [t for t in bm_tickers if t in fi_tickers]
                    if bm_fi_cols:
                         bm_metrics = df_fi_metrics.loc[['Duration', 'YTM / est. yield', 'Credit rating']][bm_fi_cols]
                         # If multiple benchmarks, maybe average or show all? 
                         # For simplicity, just show them as columns
                         port_fi_metrics = pd.concat([port_fi_metrics, bm_metrics], axis=1)

                    if has_fi and not port_fi_metrics.empty:
                        # Format
                        st.dataframe(port_fi_metrics.style.format("{:.2f}", subset=pd.IndexSlice[['Duration', 'YTM / est. yield'], :]), use_container_width=True)
                    elif not has_fi:
                        st.info("No Fixed Income assets detected in portfolios.")
                else:
                    st.info("No Fixed Income data available.")
            else:
                st.info("Fixed Income metrics data missing.")
        except Exception as e:
            st.error(f"Error calculating FI metrics: {e}")
        
    with tabs[1]: # Calendar Returns
        # Simple resampling for now
        yearly_ret = all_port_ret.resample('Y').apply(lambda x: (1+x).prod()-1)
        yearly_ret.index = yearly_ret.index.year
        st.dataframe(yearly_ret.style.background_gradient(cmap='Blues', axis=None).format("{:.2%}"), use_container_width=True)
        
    with tabs[2]: # Monthly Returns
        # Show heatmap for first portfolio or selector
        p_select = st.selectbox("Select Portfolio", all_port_ret.columns, key="monthly_sel")
        fig = plot_monthly_heatmap(all_port_ret[[p_select]])
        st.plotly_chart(fig, use_container_width=True)
        
    with tabs[3]: # Cumulative Returns
        cum_ret = (1 + all_port_ret).cumprod()
        # Pass benchmark returns for comparison
        fig = plot_cumulative_returns(cum_ret, benchmark_ret=ret_bm)
        st.plotly_chart(fig, use_container_width=True)
        
    with tabs[4]: # Return/Risk Scatter
        fig = plot_scatter_risk_return(all_port_ret, benchmark_ret=ret_bm)
        st.plotly_chart(fig, use_container_width=True)
        
    with tabs[5]: # Returns Distribution
        # Simple histogram
        import plotly.figure_factory as ff
        hist_data = [all_port_ret[c].dropna() for c in all_port_ret.columns]
        group_labels = all_port_ret.columns
        
        try:
            fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=PALETTE[:len(group_labels)])
        except np.linalg.LinAlgError:
            # Fallback for singular matrix (low variance/constant data)
            fig = go.Figure()
            for i, col in enumerate(all_port_ret.columns):
                fig.add_trace(go.Histogram(x=all_port_ret[col], name=col, marker_color=PALETTE[i % len(PALETTE)], opacity=0.7))
            fig.update_layout(barmode='overlay')
            
        fig.update_layout(title_text='Returns Distribution', plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        
    with tabs[6]: # Rolling Volatility
        fig = plot_rolling_volatility(all_port_ret)
        st.plotly_chart(fig, use_container_width=True)
        
    with tabs[7]: # Drawdown
        fig = plot_drawdowns(all_port_ret)
        st.plotly_chart(fig, use_container_width=True)
        
    with tabs[8]: # Correlation Heatmap
        fig = plot_correlation_heatmap(all_port_ret)
        st.plotly_chart(fig, use_container_width=True)
            
    with tabs[9]: # Portfolio vs Benchmark
        p_select_bm = st.selectbox("Select Portfolio", all_port_ret.columns, key="bm_sel")
        
        # Calculate Relative Metrics
        rel_metrics, roll_excess = calculate_relative_metrics(all_port_ret[p_select_bm], ret_bm.iloc[:, 0])
        
        st.markdown("#### Relative Performance Metrics")
        st.dataframe(rel_metrics.to_frame().T.style.format("{:.2%}"), use_container_width=True)
        
        st.markdown("#### Rolling 12M Excess Return")
        fig_roll = plot_cumulative_returns(roll_excess.to_frame(name="Rolling Excess Return")) # Reuse plot function
        st.plotly_chart(fig_roll, use_container_width=True)

# --- Exposure Analysis ---
if st.session_state.optimization_results:
    st.header("4. Exposure Analysis")
    
    exp_tabs = st.tabs(["Sector Exposure", "Country Exposure"])
    
    with exp_tabs[0]: # Sector Exposure
        p_select_sec = st.selectbox("Select Portfolio", list(results.keys()), key="sec_sel")
        weights_df = results[p_select_sec]['weights']
        
        # Extract last row of weights (most recent allocation)
        weights_row = weights_df.iloc[-1]
        
        charts = get_exposure_charts(weights_row, df_sector, df_country)
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts['sector_pie'], use_container_width=True)
        with c2:
            st.dataframe(charts['sector_table'].style.format("{:.2%}"), use_container_width=True)
            
    with exp_tabs[1]: # Country Exposure
        p_select_cou = st.selectbox("Select Portfolio", list(results.keys()), key="cou_sel")
        weights_df = results[p_select_cou]['weights']
        weights_row = weights_df.iloc[-1]
        charts = get_exposure_charts(weights_row, df_sector, df_country)
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts['country_pie'], use_container_width=True)
        with c2:
            st.dataframe(charts['country_table'].style.format("{:.2%}"), use_container_width=True)

# --- Factor Exposures ---
if st.session_state.optimization_results:
    st.header("5. Factor Exposures")
    
    st.markdown("""
    **Fama-French 5-Factor Model** exposure coefficients:
    - **Mkt**: Market risk premium (excess return of market over risk-free rate)
    - **SMB**: Size (Small Minus Big) - smaller firms tend to outperform larger firms
    - **HML**: Value (High Minus Low) - high book-to-market (value) vs low book-to-market (growth)
    - **RMW**: Profitability (Robust Minus Weak) - robust profitability vs weak profitability
    - **CMA**: Investment (Conservative Minus Aggressive) - conservative vs aggressive investment
    """)
    
    try:
        import requests
        import zipfile
        import io
        
        # Try to load from file first
        ff_file_path = os.path.join(os.getcwd(), 'F-F_Research_Data_5_Factors_2x3_daily.csv')
        
        try:
            df_ff_factor = pd.read_csv(ff_file_path, skiprows=3, index_col=0)
            # Filter out non-numeric rows (like copyright notice)
            df_ff_factor = df_ff_factor[pd.to_numeric(df_ff_factor.index, errors='coerce').notna()]
            df_ff_factor = df_ff_factor.astype(float) / 100
            df_ff_factor.index = pd.to_datetime(df_ff_factor.index, format='%Y%m%d')
        except FileNotFoundError:
            with st.spinner("Downloading Fama-French data..."):
                ff_link = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
                response = requests.get(ff_link, verify=False)
                
                if response.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                        with zf.open('F-F_Research_Data_5_Factors_2x3_daily.CSV') as file:
                            df_ff_factor = pd.read_csv(file, skiprows=3, index_col=0)
                            # Filter out non-numeric rows
                            df_ff_factor = df_ff_factor[pd.to_numeric(df_ff_factor.index, errors='coerce').notna()]
                            df_ff_factor = df_ff_factor.astype(float) / 100
                            df_ff_factor.index = pd.to_datetime(df_ff_factor.index, format='%Y%m%d')
                else:
                    st.error("Failed to download Fama-French data.")
                    st.stop()
        
        # Align data
        common_index = ret_all.index.intersection(df_ff_factor.index)
        
        if len(common_index) == 0:
            st.warning("No overlapping dates between portfolio data and Fama-French data.")
        else:
            # Calculate ticker factor exposures
            all_tickers_list = list(set([t for p in results.values() for t in p['tickers']]))
            
            st.subheader("Individual Ticker Factor Exposures")
            with st.spinner("Calculating ticker factor exposures..."):
                ticker_factors = pc.factor_exposure_coefficient(
                    ret_all.loc[common_index], 
                    df_ff_factor.loc[common_index], 
                    all_tickers_list
                )
                st.dataframe(
                    ticker_factors.style.background_gradient(cmap='Blues', axis=None),
                    use_container_width=True
                )
            
            # Calculate portfolio factor exposures
            common_index_2 = all_port_ret.index.intersection(df_ff_factor.index)
            
            if len(common_index_2) > 0:
                st.subheader("Portfolio Factor Exposures")
                with st.spinner("Calculating portfolio factor exposures..."):
                    # Calculate portfolio factors
                    port_factors = pc.factor_exposure_coefficient(
                        all_port_ret.loc[common_index_2], 
                        df_ff_factor.loc[common_index_2], 
                        list(results.keys())
                    )
                    
                    # Benchmark selection
                    st.markdown("**Benchmark Selection for Comparison**")
                    available_benchmarks = ret_bm.columns.tolist()
                    
                    if len(available_benchmarks) > 1:
                        selected_benchmarks = st.multiselect(
                            "Select Benchmark(s) for Factor Comparison",
                            options=available_benchmarks,
                            default=available_benchmarks,
                            key="factor_bm_select"
                        )
                    else:
                        selected_benchmarks = available_benchmarks
                    
                    # Calculate benchmark factors
                    common_index_bm = ret_bm.index.intersection(df_ff_factor.index)
                    if len(common_index_bm) > 0 and selected_benchmarks:
                        bm_factors = pc.factor_exposure_coefficient(
                            ret_bm[selected_benchmarks].loc[common_index_bm],
                            df_ff_factor.loc[common_index_bm],
                            selected_benchmarks
                        )
                        
                        # Combine for visualization
                        all_factors = pd.concat([port_factors, bm_factors])
                        
                        # Convert to numeric for plotting
                        all_factors_numeric = all_factors.apply(lambda x: pd.to_numeric(x, errors='coerce'))
                        
                        # Create grouped bar chart
                        fig = go.Figure()
                        
                        for factor in all_factors_numeric.columns:
                            fig.add_trace(go.Bar(
                                name=factor,
                                x=all_factors_numeric.index,
                                y=all_factors_numeric[factor],
                                text=all_factors[factor],
                                textposition='auto'
                            ))
                        
                        fig.update_layout(
                            barmode='group',
                            title={'text': '<b>Factor Exposures Comparison</b>', 'font': {'size': 20, 'color': '#172B4D'}},
                            xaxis_title="Portfolio / Benchmark",
                            yaxis_title="Factor Coefficient",
                            font={'family': 'Helvetica Neue, Arial', 'color': '#172B4D'},
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=500,
                            xaxis={'showgrid': True, 'gridcolor': '#F4F5F7'},
                            yaxis={'showgrid': True, 'gridcolor': '#F4F5F7'},
                            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display combined table
                        st.dataframe(
                            all_factors.style.background_gradient(cmap='Blues', axis=None),
                            use_container_width=True
                        )
                    else:
                        # Just portfolios, no benchmark comparison
                        st.dataframe(
                            port_factors.style.background_gradient(cmap='Blues', axis=None),
                            use_container_width=True
                        )
            else:
                st.warning("No overlapping dates for portfolio factor analysis.")
                
    except Exception as e:
        st.error(f"Error calculating factor exposures: {e}")
        st.exception(e)

# --- Cash Flow Analysis ---
if st.session_state.optimization_results:
    # Check if any portfolio has cash invested
    portfolios_with_cash = {
        p_name: p for p_name, p in enumerate(st.session_state.portfolios) 
        if p.get('cash_invested', 0.0) > 0
    }
    
    if portfolios_with_cash:
        st.header("6. Cash Flow Analysis")
        st.markdown("**Projected monthly cash flows based on portfolio yields and cash invested.**")
        
        # Portfolio selection
        all_portfolio_names = [p['name'] for idx, p in portfolios_with_cash.items()]
        selected_portfolios = st.multiselect(
            "Select Portfolio(s) to Display",
            options=all_portfolio_names,
            default=all_portfolio_names,
            key="cashflow_portfolio_select"
        )
        
        if not selected_portfolios:
            st.warning("Please select at least one portfolio to display cash flows.")
            st.stop()
        
        # Filter portfolios based on selection
        filtered_portfolios = {
            idx: p for idx, p in portfolios_with_cash.items() 
            if p['name'] in selected_portfolios
        }
        
        # Helper function to determine payment frequency
        def get_payment_frequency(ticker):
            """Returns 'monthly' or 'quarterly' based on ticker type."""
            # Check if ticker contains REIT indicators
            if 'REIT' in ticker.upper() or 'VNQ' in ticker or 'VNQI' in ticker:
                return 'quarterly'
            # Default to monthly for bonds and other income assets
            return 'monthly'
        
        # Calculate cash flows
        monthly_cash_flows = pd.DataFrame(0.0, 
                                         index=[p['name'] for idx, p in filtered_portfolios.items()],
                                         columns=list(range(1, 13)))
        
        ticker_cash_flows = {}  # Track by ticker for stacked chart
        
        for idx, port in filtered_portfolios.items():
            p_name = port['name']
            cash_invested = port['cash_invested']
            
            # Get portfolio weights from optimization results
            if p_name in results:
                weights_df = results[p_name]['weights']
                latest_weights = weights_df.iloc[-1]
                
                for ticker in port['tickers']:
                    weight_col = f"{ticker}_wgt"
                    if weight_col in latest_weights.index:
                        weight = latest_weights[weight_col]
                        invested_in_ticker = cash_invested * weight
                        
                        # Get yield
                        annual_yield = 0.0
                        if ticker in df_fi_metrics.columns:
                            if 'YTM / est. yield' in df_fi_metrics.index:
                                yield_val = df_fi_metrics.loc['YTM / est. yield', ticker]
                                if pd.notna(yield_val) and yield_val != 0:
                                    # Convert from percentage to decimal (e.g., 5.0 -> 0.05)
                                    annual_yield = float(yield_val) / 100
                        
                        if annual_yield > 0 and invested_in_ticker > 0:
                            payment_freq = get_payment_frequency(ticker)
                            
                            # Store ticker-level cash flows for stacked chart
                            ticker_key = f"{p_name} - {ticker}"
                            if ticker_key not in ticker_cash_flows:
                                ticker_cash_flows[ticker_key] = np.zeros(12)
                            
                            if payment_freq == 'monthly':
                                monthly_payment = (invested_in_ticker * annual_yield) / 12
                                for month in range(12):
                                    monthly_cash_flows.loc[p_name, month + 1] += monthly_payment
                                    ticker_cash_flows[ticker_key][month] += monthly_payment
                            else:  # quarterly
                                quarterly_payment = (invested_in_ticker * annual_yield) / 4
                                for quarter in range(4):
                                    month = (quarter * 3) + 2  # Months 3, 6, 9, 12 (index 2, 5, 8, 11)
                                    monthly_cash_flows.loc[p_name, month + 1] += quarterly_payment
                                    ticker_cash_flows[ticker_key][month] += quarterly_payment
        
        # Create stacked bar chart
        fig = go.Figure()
        
        # Use distinct colors for each ticker
        colors_list = PALETTE * 10  # Repeat palette if needed
        
        for i, (ticker_key, cash_flow_array) in enumerate(ticker_cash_flows.items()):
            fig.add_trace(go.Bar(
                name=ticker_key,
                x=list(range(1, 13)),
                y=cash_flow_array,
                marker_color=colors_list[i % len(colors_list)]
            ))
        
        fig.update_layout(
            barmode='stack',
            title={'text': '<b>Monthly Cash Flow Projections</b>', 'font': {'size': 20, 'color': '#172B4D'}},
            xaxis_title="Month",
            yaxis_title="Cash Flow ($)",
            xaxis={'tickmode': 'linear', 'tick0': 1, 'dtick': 1},
            yaxis={'tickformat': '$,.0f'},
            font={'family': 'Helvetica Neue, Arial', 'color': '#172B4D'},
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500,
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            legend={'orientation': 'v', 'yanchor': 'top', 'y': 1, 'xanchor': 'left', 'x': 1.02}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.subheader("Monthly Cash Flow Summary by Portfolio")
        monthly_cash_flows.columns = [f"Month {m}" for m in range(1, 13)]
        monthly_cash_flows['Total Annual'] = monthly_cash_flows.sum(axis=1)
        
        st.dataframe(
            monthly_cash_flows.style.format("${:,.2f}").background_gradient(cmap='Greens', axis=None),
            use_container_width=True
        )
        
        # Total across all portfolios
        total_monthly = monthly_cash_flows.drop(columns=['Total Annual']).sum()
        total_annual = total_monthly.sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Annual Cash Flow", f"${total_annual:,.2f}")
        with col2:
            st.metric("Average Monthly Cash Flow", f"${total_annual/12:,.2f}")
