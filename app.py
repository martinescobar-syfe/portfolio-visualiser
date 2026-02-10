
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import sys
import os

# local imports
from utils.data_loader import load_static_data, get_data_path, download_data
from utils.ui_functions import (
    render_client_upload, 
    render_client_pdf_upload,
    render_recommender, 
    render_portfolio_construction, 
    render_analysis
)

# page config
st.set_page_config(
    page_title="Portfolio Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom CSS - syfe branding, hide default streamlit stuff
st.markdown("""
<style>
    /* hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* sidebar toggle on */

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
    
    /* tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: var(--secondary-background-color);
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: var(--text-color);
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# init
data_path = get_data_path()
df_sec_list, df_asset_class, df_sector, df_country, df_fi_metrics, df_fx, df_daily_ret, df_risk_rating, df_underlying = load_static_data(data_path)

if df_sec_list is None:
    st.stop()

# session state
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = []
    # default portfolio
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

# sidebar
with st.sidebar:
    # logo placeholder
    st.markdown("## Syfe Portfolio Visualizer")


# main tabs - consolidated flow

# API key (hardcoded)
GEMINI_API_KEY = "AIzaSyCZThunnMaO5Ni6fJDLnWsN24JX0yCN3p0"

# global ticker lists
available_tickers = sorted(df_sec_list['Ticker'].tolist())
bm_tickers = ['SPY', 'ACWI', 'URTH']  # defaults, user can change in base settings

with st.container():
    st.markdown("## Portfolio Visualiser")
    
    # client type toggle
    client_type = st.radio(
        "Select Client Type",
        ["New Client", "Existing Client"],
        horizontal=True,
        key="client_type_revised"
    )
    
    # base settings (shared across both flows)
    with st.expander("⚙️ Base Settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        
        # grab latest date from returns data
        if df_daily_ret is not None and not df_daily_ret.empty:
            latest_data_date = df_daily_ret.index.max().date()
        else:
            latest_data_date = date.today()
        
        data_source = c1.selectbox("Data Source", ["Bloomberg", "Yahoo Finance"], key="exist_data_source")
        currency = c2.selectbox("Currency", ["USD", "SGD"], key="exist_currency")  # USD is default
        start_date = c3.date_input("Start Date", value=date(2015, 1, 1), key="exist_start_date")
        end_date = c4.date_input("End Date", value=latest_data_date, key="exist_end_date", help=f"Latest available data: {latest_data_date}")
        
        backtest_offset_days = st.slider(
            "Backtest Offset (Days)", 
            0, 365, 90, 
            help="Skip this many days at the start for initial weight calculation before backtesting.",
            key="exist_offset"
        )
        
        # benchmark ticker selection
        bm_tickers = st.multiselect(
            "Benchmark Tickers",
            options=sorted(df_sec_list['Ticker'].tolist()),
            default=['ACWI'],  # ACWI is default benchmark
            key="rev_bm_settings"
        )
    
    # conditional flow based on client type
    if client_type == "Existing Client":
        # PDF Export section
        from utils.pdf_gen import render_export_ui
        export_modules = st.multiselect(
            "Select Modules for Export",
            [
                "1. Client Data Source",
                "2. Syfe Portfolio Recommender",
                "3. Portfolio Construction",
                "4. Analysis Results",
                "5. Exposure Analysis",
                "6. Factor Exposures",
                "7. Cash Flow Analysis and Growth Forecast Calculator"
            ],
            default=[],
            key="module_export_selector",
            help="Select sections to include in the generated PDF report."
        )
        render_export_ui(export_modules)
        
        # CSV upload
        render_client_upload(
            df_asset_class, df_sector, df_country, df_underlying,
            key_prefix="exist_upload"
        )
        
        # PDF upload (AI parsing)
        render_client_pdf_upload(
            GEMINI_API_KEY, df_asset_class, df_sector, df_country,
            key_prefix="exist_pdf"
        )
    
    # recommender (always visible)
    render_recommender(GEMINI_API_KEY, key_prefix="exist_rec")
    
    # portfolio construction (always visible)
    render_portfolio_construction(
        df_sec_list, available_tickers, bm_tickers, data_source, start_date, end_date, currency,
        backtest_offset_days, data_path, df_fi_metrics, df_fx, df_daily_ret, key_prefix="exist_cons"
    )
    
    # analysis (always visible)
    render_analysis(
        df_sec_list, df_fi_metrics, df_fx, df_sector, df_country, df_asset_class, df_underlying, data_path, bm_tickers, 
        backtest_offset_days, data_source, start_date, end_date, currency, key_prefix="exist_analysis"
    )
