"""
Advisory Tool — Streamlit landing page.

This is the *entry point* for the Syfe Advisory app and acts as a thin
orchestration layer that wires together the heavy-lifting components
defined in ``utils/ui_functions.py``. The page itself contains very
little business logic: its job is to

1. Load all reference data once (security list, asset-class / sector /
   country maps, FI metrics, FX, daily returns, risk ratings, underlying
   holdings) via :func:`load_static_data`.
2. Render shared "base settings" (date range, currency, benchmarks) that
   every downstream module consumes.
3. Branch on the *client type* radio (New vs. Existing) to decide whether
   to show the CSV / PDF upload step.
4. Delegate to the four major UI modules — recommender, portfolio
   construction, analysis, and (for existing clients) PDF export — each
   of which encapsulates its own slicing / aggregation logic.

All actual computation (optimisation, exposure aggregation, risk
analytics, etc.) lives in ``utils/``; keep this file focused on layout
and session-state plumbing.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import sys
import os

# Local imports — each render_* function below is a self-contained UI
# module that owns its own widgets, caching, and aggregation. This page
# only stitches them together in the right order.
from utils.data_loader import load_static_data, get_data_path, download_data
from utils.ui_functions import (
    render_client_upload, 
    render_client_pdf_upload,
    render_recommender, 
    render_portfolio_construction, 
    render_analysis
)

# Page config — wide layout gives the multi-column charts (exposures,
# backtest, factor) room to breathe, and the sidebar starts expanded
# because base controls / navigation live there.
st.set_page_config(
    page_title="Advisory Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — Syfe brand colours, Helvetica Neue, primary-blue
# buttons, and hidden Streamlit chrome (main menu + footer) so the
# page looks like a polished client-facing tool rather than a stock
# Streamlit app.
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

# ── Load reference data ──────────────────────────────────────────────
# A single call here hydrates every dataframe the downstream modules
# need. ``load_static_data`` is cached, so subsequent reruns are cheap.
# If the security list is missing we abort rendering — nothing else can
# work without it.
data_path = get_data_path()
df_sec_list, df_asset_class, df_sector, df_country, df_fi_metrics, df_fx, df_daily_ret, df_risk_rating, df_underlying = load_static_data(data_path)

if df_sec_list is None:
    st.stop()

# ── Session state bootstrap ──────────────────────────────────────────
# Portfolio construction, optimisation results and the refresh counter
# persist across reruns. We initialise them once with a single empty
# "Portfolio 1" so the Portfolio Construction module always has
# something to render.
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = []
    st.session_state.portfolios.append({
        'id': 1,
        'name': 'Portfolio 1',
        'tickers': [],
        'allocation_type': 'Max sharpe',
        'rebal_freq': 'Annually',
        'min_weight': 0.0,
        'max_weight': 1.0,
        'cash_invested': 0.0,
        'cash_currency': 'USD',  # Manual portfolios default to USD
        'fixed_weights': {}
    })

# Cached output of the most recent optimiser run, populated by
# ``render_portfolio_construction`` and read back by
# ``render_analysis`` so the analysis tab can render whatever the
# user last optimised without re-running the solver.
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

# Monotonic counter bumped by various render_* modules when they
# need downstream widgets to invalidate their cached state (e.g.
# changing tickers in construction forces analysis to recompute).
# Reading this counter is cheaper than diffing the full inputs.
if 'refresh_counter' not in st.session_state:
    st.session_state.refresh_counter = 0

# ── Sidebar ──────────────────────────────────────────────────────────
# Intentionally minimal today (just the app title). Reserved as the
# home for future global navigation / advisor-context controls.
with st.sidebar:
    st.markdown("## Syfe Advisory Tool")


# ── Global config shared by all downstream modules ───────────────────
# API key for Gemini-powered insights (recommender, PDF parser, etc.).
GEMINI_API_KEY = "AIzaSyCZThunnMaO5Ni6fJDLnWsN24JX0yCN3p0"

# Master ticker universe pulled from the security list — used as the
# selectable option set for every multiselect in the app.
available_tickers = sorted(df_sec_list['Ticker'].tolist())
bm_tickers = ['SPY', 'ACWI', 'URTH']  # default benchmark set, overridden in Base Settings

with st.container():
    st.markdown("## Advisory Tool")
    
    # New vs Existing changes the visible workflow:
    #   - New: jump straight to recommender + construction (no holdings to ingest)
    #   - Existing: also offer CSV / PDF upload and PDF export of the report
    client_type = st.radio(
        "Select Client Type",
        ["New Client", "Existing Client"],
        horizontal=True,
        key="client_type_revised"
    )
    
    # ── Base settings ────────────────────────────────────────────
    # Shared knobs (date range, currency, benchmarks) consumed by every
    # downstream module — kept in an expander so the page stays tidy.
    with st.expander("⚙️ Base Settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        
        # Default end-date is the most recent row in the returns frame
        # so we never request a date the underlying data can't support.
        if df_daily_ret is not None and not df_daily_ret.empty:
            latest_data_date = df_daily_ret.index.max().date()
        else:
            latest_data_date = date.today()
        
        # Data Source — selects which pricing/returns provider the
        # downstream modules pull from. Bloomberg = pre-loaded local
        # parquet; Yahoo Finance = on-demand fetch via ``download_data``.
        data_source = c1.selectbox("Data Source", ["Bloomberg", "Yahoo Finance"], key="exist_data_source")
        # Display + reporting currency. Drives FX conversion in the
        # cash-flow / performance modules; USD is the default since
        # most of the security universe is USD-denominated.
        currency = c2.selectbox("Currency", ["USD", "SGD"], key="exist_currency")
        # Start date for both optimisation history and backtest. 2015
        # is a sensible default — captures the post-QE regime, the
        # 2018 rate cycle, COVID, and the 2022 drawdown across most
        # of the security universe.
        start_date = c3.date_input("Start Date", value=date(2015, 1, 1), key="exist_start_date")
        end_date = c4.date_input("End Date", value=latest_data_date, key="exist_end_date", help=f"Latest available data: {latest_data_date}")
        
        # Backtest offset = warm-up window. We carve off the first N
        # days to compute the initial optimiser weights, then run the
        # actual backtest on the remainder. Without this, the first
        # rebalance would have no covariance history to work with.
        backtest_offset_days = st.slider(
            "Backtest Offset (Days)", 
            0, 365, 90, 
            help="Skip this many days at the start for initial weight calculation before backtesting.",
            key="exist_offset"
        )
        
        # Benchmarks selected here flow into every downstream chart
        # (cumulative returns, exposure comparison, factor exposures).
        bm_tickers = st.multiselect(
            "Benchmark Tickers",
            options=sorted(df_sec_list['Ticker'].tolist()),
            default=['ACWI'],  # ACWI is default benchmark
            key="rev_bm_settings"
        )
    
    # ── Existing-client-only modules ─────────────────────────────
    # Only existing clients have current holdings to ingest, so the
    # PDF export + CSV / PDF upload blocks are gated here. The export
    # multiselect drives which sections get baked into the generated
    # report by ``render_export_ui``.
    if client_type == "Existing Client":
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
        # Hands the selected module list off to the PDF generator,
        # which renders the matching sections from session state
        # (holdings, recommender output, construction, analysis,
        # exposures, factors, cash flow) into a single downloadable
        # client report.
        render_export_ui(export_modules)
        
        # Tabular CSV upload of current holdings.
        render_client_upload(
            df_asset_class, df_sector, df_country, df_underlying,
            key_prefix="exist_upload"
        )
        
        # Free-form PDF statement upload — Gemini parses it into the
        # same holdings shape the CSV uploader produces.
        render_client_pdf_upload(
            GEMINI_API_KEY, df_asset_class, df_sector, df_country,
            key_prefix="exist_pdf"
        )
    
    # ── Shared modules (rendered for every client type) ──────────
    # Recommender suggests a Syfe managed portfolio from the client's
    # risk / objective inputs.
    render_recommender(GEMINI_API_KEY, key_prefix="exist_rec")
    
    # Portfolio Construction is where the user assembles custom
    # portfolios, runs optimisation, and backtests them against
    # ``bm_tickers`` over [start_date, end_date].
    render_portfolio_construction(
        df_sec_list, available_tickers, bm_tickers, data_source, start_date, end_date, currency,
        backtest_offset_days, data_path, df_fi_metrics, df_fx, df_daily_ret, key_prefix="exist_cons"
    )
    
    # Analysis renders performance metrics, exposure slicing
    # (asset-class / sector / country), factor exposures, drawdowns
    # and the cash-flow / growth forecast for whatever portfolios are
    # currently stored in session state.
    render_analysis(
        df_sec_list, df_fi_metrics, df_fx, df_sector, df_country, df_asset_class, df_underlying, data_path, bm_tickers, 
        backtest_offset_days, data_source, start_date, end_date, currency, key_prefix="exist_analysis"
    )
