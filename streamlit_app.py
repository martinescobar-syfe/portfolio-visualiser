import streamlit as st
import pandas as pd
import numpy as np
import math, os, io, time, pytz, re, copy
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta, datetime
import plotly.graph_objects as go
import yfinance as yf
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import warnings

# --- Page Config ---
st.set_page_config(
    page_title="Portfolio Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Aesthetics ---
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #263159;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #414e7d;
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 2rem;
    }
    .stButton>button {
        background-color: #263159;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #414e7d;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Imports from local modules ---
# Ensure local modules can be imported
sys.path.append(os.getcwd())
try:
    import port_cons as pc
except ImportError as e:
    st.error(f"Could not import 'port_cons.py'. Error: {e}")
    st.stop()

warnings.filterwarnings("ignore")

# --- Helper Functions ---
@st.cache_data
def load_static_data(data_path):
    try:
        df_sec_list = pd.read_csv(os.path.join(data_path, 'PV_sec_list.csv'))
        df_sec_list['Remarks'] = df_sec_list['Remarks'].fillna('N/A')
        
        df_asset_class = pd.read_csv(os.path.join(data_path, 'PV_asset_class.csv'), index_col=0)
        df_sector = pd.read_csv(os.path.join(data_path, 'PV_sector.csv'), index_col=0)
        df_country = pd.read_csv(os.path.join(data_path, 'PV_country.csv'), index_col=0)
        df_fi_metrics = pd.read_csv(os.path.join(data_path, 'PV_FI_metrics.csv'), index_col=0)
        
        # Load FX data
        df_fx = pd.read_csv(os.path.join(data_path, 'PV_daily_ret.csv'), header=0, index_col='Date', parse_dates=['Date'], dayfirst=True)[['USDSGD', 'USDHKD']] / 100
        
        return df_sec_list, df_asset_class, df_sector, df_country, df_fi_metrics, df_fx
    except Exception as e:
        st.error(f"Error loading static data: {e}")
        return None, None, None, None, None, None

def get_data_path():
    # Check if running in Colab (default path) or local
    if os.path.exists('/content/Projects/Data/'):
        return '/content/Projects/Data/'
    elif os.path.exists('Data/'):
        return 'Data/'
    else:
        # Fallback to current directory if Data folder not found
        return './'

# --- Initialization ---
data_path = get_data_path()
df_sec_list, df_asset_class, df_sector, df_country, df_fi_metrics, df_fx = load_static_data(data_path)

if df_sec_list is None:
    st.stop()

# --- Sidebar: Global Settings ---
with st.sidebar:
    st.image("https://assets-global.website-files.com/606f63778ec431ec1b930f1f/606f68248ec4319d2c930f6e_Syfe-Logo.svg", width=100)
    st.title("Configuration")
    
    data_source = st.radio("Data Source", ['Bloomberg', 'Yahoo Finance'])
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=date(2015, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=date.today())
        
    bm_tickers_str = st.text_input("Benchmark Tickers", value="URTH")
    bm_tickers = [x.strip() for x in bm_tickers_str.split(',')]
    
    backtest_offset_days = st.number_input("Observation Window (Days)", min_value=0, value=0)
    currency = st.selectbox("Currency", ['USD', 'SGD', 'HKD'])

# --- Main Content ---
st.markdown('<div class="main-header">Portfolio Visualizer</div>', unsafe_allow_html=True)

# --- Session State Management ---
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = []
    # Add default portfolio
    st.session_state.portfolios.append({
        'id': 1,
        'name': 'Portfolio 1',
        'tickers': [],
        'allocation_type': 'Max sharpe',
        'rebal_freq': 'Annually',
        'min_weight': 0.0,
        'max_weight': 1.0,
        'custom_weights': ''
    })

if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

if 'client_data' not in st.session_state:
    st.session_state.client_data = None

# --- Client Data Upload Section ---
st.markdown('<div class="sub-header">1. Client Data Upload</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Client Holdings (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        df_client = pd.read_csv(uploaded_file)
        st.success(f"Uploaded: {uploaded_file.name}")
        st.session_state.client_data = df_client
        
        # Display client data summary
        st.markdown("#### Portfolio Summary")
        if 'type' in df_client.columns and 'nav_in_sgd' in df_client.columns:
            # Process NAV data
            df_summary = df_client.copy()
            df_summary['nav_in_sgd'] = df_summary['nav_in_sgd'].astype(str).str.replace(',', '').astype(float)
            df_summary_grouped = df_summary.groupby('type')['nav_in_sgd'].sum().reset_index()
            df_summary_grouped['Allocation %'] = (df_summary_grouped['nav_in_sgd'] / df_summary_grouped['nav_in_sgd'].sum() * 100).round(2)
            df_summary_grouped = df_summary_grouped.rename(columns={'type': 'Portfolio Type', 'nav_in_sgd': 'NAV (SGD)'})
            df_summary_grouped = df_summary_grouped.sort_values('NAV (SGD)', ascending=False)
            
            # Display summary table
            st.dataframe(df_summary_grouped, use_container_width=True)
            
            # Pie chart of allocation
            fig_allocation = px.pie(
                df_summary_grouped, 
                values='NAV (SGD)', 
                names='Portfolio Type',
                title='Portfolio Allocation',
                hole=0.3
            )
            st.plotly_chart(fig_allocation, use_container_width=True)
        else:
            st.dataframe(df_client.head())
            
    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- Portfolio Construction Section ---
st.markdown('<div class="sub-header">2. Portfolio Construction</div>', unsafe_allow_html=True)

# Get available tickers from df_sec_list
available_tickers = sorted(df_sec_list['Ticker'].tolist())

# Add/Remove Portfolio buttons with equal sizing
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("➕ Add Portfolio", use_container_width=True):
        new_id = len(st.session_state.portfolios) + 1
        st.session_state.portfolios.append({
            'id': new_id,
            'name': f'Portfolio {new_id}',
            'tickers': [],
            'allocation_type': 'Max sharpe',
            'rebal_freq': 'Annually',
            'min_weight': 0.0,
            'max_weight': 1.0,
            'custom_weights': ''
        })
        st.rerun()

with col2:
    if st.button("➖ Remove Last", use_container_width=True):
        if len(st.session_state.portfolios) > 1:
            st.session_state.portfolios.pop()
            st.rerun()
        else:
            st.warning("Cannot remove the last portfolio.")

# Display Portfolio Widgets
for i, port in enumerate(st.session_state.portfolios):
    with st.expander(f"{port['name']}", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.portfolios[i]['name'] = st.text_input(
                f"Name", 
                value=port['name'], 
                key=f"name_{i}"
            )
            # Interactive ticker selection with multiselect
            st.session_state.portfolios[i]['tickers'] = st.multiselect(
                f"Select Tickers",
                options=available_tickers,
                default=port['tickers'] if isinstance(port['tickers'], list) else [],
                key=f"tickers_{i}",
                help="Type to search and select tickers"
            )
        with c2:
            st.session_state.portfolios[i]['allocation_type'] = st.selectbox(
                f"Allocation", 
                ['Max sharpe', 'Min vol', 'Risk parity', 'Equal weight', 'Custom weight'], 
                index=['Max sharpe', 'Min vol', 'Risk parity', 'Equal weight', 'Custom weight'].index(port['allocation_type']), 
                key=f"alloc_{i}"
            )
            st.session_state.portfolios[i]['rebal_freq'] = st.selectbox(
                f"Rebalance", 
                ['Monthly', 'Quarterly', 'Semi-annually', 'Annually'], 
                index=['Monthly', 'Quarterly', 'Semi-annually', 'Annually'].index(port['rebal_freq']), 
                key=f"rebal_{i}"
            )
        with c3:
            st.session_state.portfolios[i]['min_weight'] = st.number_input(
                f"Min Weight", 
                value=port['min_weight'], 
                key=f"min_{i}"
            )
            st.session_state.portfolios[i]['max_weight'] = st.number_input(
                f"Max Weight", 
                value=port['max_weight'], 
                key=f"max_{i}"
            )
        
        if st.session_state.portfolios[i]['allocation_type'] == 'Custom weight':
            st.session_state.portfolios[i]['custom_weights'] = st.text_input(
                f"Custom Weights (comma separated)", 
                value=port['custom_weights'], 
                key=f"custom_{i}"
            )

# --- Optimization Section ---
st.markdown('<div class="sub-header">3. Optimization & Backtest</div>', unsafe_allow_html=True)

# Helper functions for optimization (ported from test.py)
def modify_tickers_based_on_exchange(all_tickers, df_sec_list):
    modified_tickers = []
    for ticker in all_tickers:
        exchange = df_sec_list.loc[df_sec_list['Ticker'] == ticker, 'Exchange'].iloc[0] if ticker in df_sec_list['Ticker'].values else None
        if exchange == 'London':
            modified_tickers.append(ticker + '.L')
        elif exchange == 'Swiss':
            modified_tickers.append(ticker + '.SW')
        else:
            modified_tickers.append(ticker)
    return modified_tickers

def download_data(data_source, start_date, end_date, all_tickers, bm_tickers, currency, df_sec_list, df_fx, data_path):
    # Load FX data if not passed (though we pass it now)
    if df_fx is None:
         df_fx = pd.read_csv(os.path.join(data_path, 'PV_daily_ret.csv'), header=0, index_col='Date', parse_dates=['Date'], dayfirst=True)[['USDSGD', 'USDHKD']] / 100

    if data_source == 'Yahoo Finance':
        modified_tickers = modify_tickers_based_on_exchange(all_tickers, df_sec_list)
        
        # Handle date conversion for yfinance
        start_str = start_date.strftime('%Y-%m-%d') if start_date else None
        end_str = end_date.strftime('%Y-%m-%d') if end_date else None

        if start_str is None and end_str is None:
            df_all = yf.download(modified_tickers, period='max')['Adj Close']
        else:
            df_all = yf.download(modified_tickers, start=start_str, end=end_str)['Adj Close']

        df_all.index = df_all.index.tz_localize(None)
        ret_all = df_all.pct_change().dropna()
        # Map back to original tickers if needed, but yfinance returns requested tickers usually
        # If single ticker, yfinance returns Series, need DataFrame
        if isinstance(ret_all, pd.Series):
            ret_all = ret_all.to_frame()
            ret_all.columns = all_tickers
        else:
             # Try to map columns back if they were modified
             # This is a simplification; robust mapping might be needed
             if len(ret_all.columns) == len(all_tickers):
                 ret_all.columns = all_tickers
    
    elif data_source == 'Bloomberg':
        # Load from local CSV
        ret_all = pd.read_csv(os.path.join(data_path, 'PV_daily_ret.csv'), header=0, index_col='Date', parse_dates=['Date'], dayfirst=True)
        # Filter for requested tickers only
        available_tickers = [t for t in all_tickers if t in ret_all.columns]
        missing_tickers = [t for t in all_tickers if t not in ret_all.columns]
        
        if missing_tickers:
            st.warning(f"Tickers missing from Bloomberg data: {missing_tickers}")
            
        ret_all = ret_all[available_tickers] / 100
        
        first_valid_index = ret_all.apply(lambda series: series.first_valid_index()).max()
        if pd.notna(first_valid_index):
            ret_all = ret_all.loc[first_valid_index:].dropna(how='all')

        # Filter by date
        if start_date:
            ret_all = ret_all.loc[start_date.strftime('%Y-%m-%d'):]
        if end_date:
            ret_all = ret_all.loc[:end_date.strftime('%Y-%m-%d')]

    ret_all = ret_all.fillna(0)
    
    # Currency Adjustment
    for ticker in ret_all.columns:
        if ticker not in df_sec_list['Ticker'].values:
            continue
            
        base_currency = df_sec_list.loc[df_sec_list['Ticker'] == ticker, 'Base currency'].iloc[0]
        if base_currency != currency:
            # Align FX dates
            common_idx = ret_all.index.intersection(df_fx.index)
            
            if base_currency == 'USD' and currency == 'SGD':
                fx = df_fx.loc[common_idx, 'USDSGD']
                ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx) - 1
            elif base_currency == 'USD' and currency == 'HKD':
                fx = df_fx.loc[common_idx, 'USDHKD']
                ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx) - 1
            elif base_currency == 'SGD' and currency == 'USD':
                fx = df_fx.loc[common_idx, 'USDSGD']
                # Inverse approximation
                fx_inv = -fx / (1 + fx)
                ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx_inv) - 1
            elif base_currency == 'HKD' and currency == 'USD':
                fx = df_fx.loc[common_idx, 'USDHKD']
                fx_inv = -fx / (1 + fx)
                ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx_inv) - 1

    # Benchmark Returns
    ret_bm = ret_all[ [t for t in bm_tickers if t in ret_all.columns] ]
    
    return ret_all, ret_bm

def create_constraints(min_weight, max_weight):
    return ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'ineq', 'fun': lambda weights: weights - min_weight},
            {'type': 'ineq', 'fun': lambda weights: max_weight - weights})

def create_custom_weight_dict(tickers, custom_weights_str):
    try:
        weights = [float(x.strip()) for x in custom_weights_str.split(',')]
    except:
        return None
        
    if len(tickers) != len(weights):
        return None
    return dict(zip(tickers, weights))

if st.button("🚀 Run Optimization", use_container_width=True):
    with st.spinner("Downloading data and optimizing portfolios..."):
        # 1. Gather all unique tickers
        all_tickers = set(bm_tickers)
        for p in st.session_state.portfolios:
            if p['tickers']:
                # tickers is now a list from multiselect
                if isinstance(p['tickers'], list):
                    all_tickers.update(p['tickers'])
                else:
                    # Fallback for string format (shouldn't happen with multiselect)
                    p_tickers = [t.strip() for t in p['tickers'].split(',') if t.strip()]
                    all_tickers.update(p_tickers)
        
        all_tickers = list(all_tickers)
        
        if not all_tickers:
            st.error("No tickers defined in portfolios or benchmark.")
        else:
            # 2. Download Data
            ret_all, ret_bm = download_data(data_source, start_date, end_date, all_tickers, bm_tickers, currency, df_sec_list, df_fx, data_path)
            
            if ret_all.empty:
                st.error("No data downloaded. Check tickers and date range.")
            else:
                earliest_start_date = ret_all.index.min()
                
                # 3. Run Optimization for each portfolio
                optimization_results = {}
                
                progress_bar = st.progress(0)
                num_ports = len(st.session_state.portfolios)
                
                for i, p in enumerate(st.session_state.portfolios):
                    try:
                        port_name = p['name']
                        # Handle both list and string ticker formats
                        if isinstance(p['tickers'], list):
                            tickers = p['tickers']
                        else:
                            tickers = [t.strip() for t in p['tickers'].split(',') if t.strip()]
                        
                        # Filter valid tickers
                        tickers = [t for t in tickers if t in ret_all.columns]
                        
                        if not tickers:
                            st.warning(f"Skipping {port_name}: No valid tickers found.")
                            continue
                            
                        allocation_type = p['allocation_type']
                        min_w = p['min_weight']
                        max_w = p['max_weight']
                        
                        # Rebalance months
                        rebal_map = {
                            'Annually': [1],
                            'Semi-annually': [1, 7],
                            'Quarterly': [1, 4, 7, 10],
                            'Monthly': list(range(1, 13))
                        }
                        rebal_months = rebal_map.get(p['rebal_freq'], [1])
                        
                        constraints = create_constraints(min_w, max_w)
                        
                        # Calculate Weights
                        weights_df = None
                        
                        start_opt_date = earliest_start_date + timedelta(days=backtest_offset_days)
                        
                        if allocation_type == 'Max sharpe':
                            weights_df = pc.max_sharpe_rebal_wgt(
                                ret_all[tickers],
                                start_date=start_opt_date,
                                end_date=end_date,
                                constraints_dict=constraints,
                                shrink_covar=False,
                                rebal_months=rebal_months,
                                annualized=261,
                                halflife=3.5
                            )
                        elif allocation_type == 'Min vol':
                            weights_df = pc.min_vol_rebal_wgt(
                                ret_all[tickers],
                                start_date=start_opt_date,
                                end_date=end_date,
                                constraints_dict=constraints,
                                shrink_covar=False,
                                rebal_months=rebal_months,
                                annualized=261,
                                halflife=3.5
                            )
                        elif allocation_type == 'Risk parity':
                            # Assuming rebal_wgt_riskparity is available in pc or defined locally
                            # If it was defined in test.py but not port_cons.py, we need to add it here or to port_cons.py
                            # For now, let's assume it's in pc or we skip it
                            if hasattr(pc, 'rebal_wgt_riskparity'):
                                weights_df = pc.rebal_wgt_riskparity(
                                    ret_all[tickers],
                                    start_date=start_opt_date,
                                    end_date=end_date,
                                    rebal_months=rebal_months,
                                    halflife=3.5,
                                    annualized=261,
                                    shrink_covar=False
                                )
                            else:
                                st.warning(f"Risk parity not implemented in port_cons.py yet.")
                                
                        elif allocation_type == 'Equal weight':
                            weights_df = pc.equal_weight_rebal_wgt(
                                ret_all[tickers],
                                start_date=start_opt_date,
                                end_date=end_date,
                                rebal_months=rebal_months,
                            )
                        elif allocation_type == 'Custom weight':
                            custom_w_dict = create_custom_weight_dict(tickers, p['custom_weights'])
                            if custom_w_dict:
                                weights_df = pc.fixed_weight_rebal_wgt(
                                    ret_all[tickers],
                                    start_date=start_opt_date,
                                    end_date=end_date,
                                    fixed_weights=custom_w_dict,
                                    rebal_months=[1]
                                )
                            else:
                                st.error(f"Invalid custom weights for {port_name}")

                        if weights_df is not None:
                            # Calculate Performance
                            perf_df = pc.calc_port_perf(
                                ret_all.loc[start_opt_date:end_date][tickers],
                                weights_df,
                                name_prefix=port_name
                            )
                            
                            optimization_results[port_name] = {
                                'weights': weights_df,
                                'perf': perf_df,
                                'tickers': tickers
                            }
                            
                    except Exception as e:
                        st.error(f"Error optimizing {p['name']}: {e}")
                    
                    progress_bar.progress((i + 1) / num_ports)
                
                if optimization_results:
                    st.success("Optimization Complete!")
                    st.session_state.optimization_results = {
                        'results': optimization_results,
                        'ret_all': ret_all,
                        'ret_bm': ret_bm,
                        'earliest_start_date': earliest_start_date
                    }
                else:
                    st.warning("No portfolios were successfully optimized.")

# --- Results Section ---
if st.session_state.optimization_results:
    st.markdown('<div class="sub-header">4. Results Analysis</div>', unsafe_allow_html=True)
    
    results_data = st.session_state.optimization_results
    opt_results = results_data['results']
    ret_all = results_data['ret_all']
    ret_bm = results_data['ret_bm']
    earliest_start_date = results_data['earliest_start_date']
    
    # Combine Backtest Data
    port_names = list(opt_results.keys())
    
    all_port_ret = pd.DataFrame()
    all_port_wgt = pd.DataFrame()
    
    for p_name, p_data in opt_results.items():
        # Returns
        p_ret = p_data['perf'][[c for c in p_data['perf'].columns if c.endswith('_port_ret')]]
        p_ret.columns = [p_name]
        all_port_ret = pd.concat([all_port_ret, p_ret], axis=1)
        
        # Weights (Need to handle multi-index or prefixing for compatibility with port_cons functions)
        # For simplicity, we might need to adjust how port_cons expects weights
        # Assuming port_cons functions can handle the weights df as is or we concatenate them
        # Re-prefixing columns for port_cons compatibility if needed
        w_df = p_data['weights'].copy()
        # Ensure columns are tuples or formatted as expected by port_cons if it uses multi-index
        # Based on test.py, it seems to expect a specific format. 
        # Let's try to match test.py's combine_backtest_data logic roughly
        all_port_wgt = pd.concat([all_port_wgt, w_df], axis=1)

    # Align dates
    # Filter ret_bm to match portfolio dates
    common_idx = all_port_ret.index.intersection(ret_bm.index)
    all_port_ret = all_port_ret.loc[common_idx]
    ret_bm = ret_bm.loc[common_idx]
    
    # Add benchmark to all_port_ret for easy plotting/calc
    bm_col = ret_bm.columns[0]
    all_port_ret[bm_col] = ret_bm[bm_col]
    
    # --- Calculations ---
    # Performance Summary
    # Note: port_cons.performance_summary expects specific args. 
    # We might need to adapt it or call it carefully.
    # For now, let's try calling it if available, else implement simple summary
    
    try:
        perf_summary = pc.performance_summary(
            all_port_ret[port_names], 
            all_port_wgt, 
            all_port_ret[bm_col], 
            benchmark_returns=ret_bm,
            start_date=all_port_ret.index.min(), 
            end_date=all_port_ret.index.max(), 
            frequency='daily'
        )
    except Exception as e:
        st.error(f"Error calculating performance summary: {e}")
        perf_summary = pd.DataFrame()

    # Calendar Year Returns
    try:
        port_yearly_ret = pc.constituents_calendar_year_returns(all_port_ret, frequency='daily')
    except:
        port_yearly_ret = pd.DataFrame()

    # --- Tabs ---
    tabs = st.tabs(["Key Summary", "Calendar Returns", "Cumulative Returns", "Risk/Return", "Drawdown", "Correlation"])
    
    with tabs[0]:
        st.subheader("Performance Summary")
        if not perf_summary.empty:
            st.dataframe(perf_summary.style.format("{:.2%}"))
        else:
            st.write("No data available.")
        
    with tabs[1]:
        st.subheader("Calendar Year Returns")
        if not port_yearly_ret.empty:
            st.dataframe(port_yearly_ret.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2%}"))
        else:
            st.write("No data available.")
        
    with tabs[2]:
        st.subheader("Cumulative Returns")
        try:
            fig_cum = pc.plot_cumulative_returns(all_port_ret, show_data=False)
            # If pc returns a figure, display it. If it displays directly, we might need to capture it.
            # Assuming it returns a plotly figure based on test.py
            if isinstance(fig_cum, go.Figure):
                st.plotly_chart(fig_cum, use_container_width=True)
            else:
                st.write("Chart not available.")
        except Exception as e:
            st.error(f"Error plotting cumulative returns: {e}")
        
    with tabs[3]:
        st.subheader("Risk vs Return")
        try:
            fig_risk = pc.plot_return_risk_scatter_maxrange(all_port_ret)
            if isinstance(fig_risk, go.Figure):
                st.plotly_chart(fig_risk, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting risk/return: {e}")
        
    with tabs[4]:
        st.subheader("Drawdowns")
        try:
            fig_dd = pc.plot_drawdowns(all_port_ret, show_data=False)
            if isinstance(fig_dd, go.Figure):
                st.plotly_chart(fig_dd, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting drawdowns: {e}")
        
    with tabs[5]:
        st.subheader("Correlation Matrix")
        try:
            # pc.plot_correlation_heatmap might display directly. 
            # We need to check if it returns a figure. 
            # If it uses seaborn/matplotlib and plt.show(), we need to catch it.
            # Let's assume it returns a figure or we can re-implement simple heatmap here
            corr = all_port_ret.corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting correlation: {e}")


# --- Footer ---
st.markdown("---")
st.markdown("� **Note:** PDF Report generation will be implemented in a future update.")
st.markdown("Built with ❤️ using Streamlit")
