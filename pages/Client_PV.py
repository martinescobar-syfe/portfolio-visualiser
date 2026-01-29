
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import os

# local imports
from utils.data_loader import load_static_data, get_data_path
from utils.ui_functions import render_client_pdf_upload
from utils.pdf_helper import generate_portfolio_insights

# page config
st.set_page_config(
    page_title="Client Portfolio Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #0747A6;
    }
    
    /* cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    
    /* insight cards */
    .insight-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #0052CC;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# session state
if 'client_pv_insights' not in st.session_state:
    st.session_state.client_pv_insights = None

# API key
api_key = "AIzaSyCZThunnMaO5Ni6fJDLnWsN24JX0yCN3p0"

# load data
data_path = get_data_path()
df_sec_list, df_asset_class, df_sector, df_country, df_fi_metrics, df_fx, df_daily_ret, df_risk_rating, df_underlying = load_static_data(data_path)

# header
st.title("Portfolio Visualizer")
st.markdown("Upload your portfolio statement to analyze your holdings and receive personalized insights.")

st.divider()

# pdf upload
render_client_pdf_upload(api_key, df_asset_class, df_sector, df_country, key_prefix="client_pv")

st.divider()

# AI insights
st.header("AI-Powered Insights")

# check for parsed data
pdf_key = "client_pv_parsed_data"
if pdf_key in st.session_state:
    data = st.session_state[pdf_key]
    enriched = data.get('enriched', {})
    country_exp = data.get('country_exp', {})
    sector_exp = data.get('sector_exp', {})
    holdings_df = data.get('holdings_df', pd.DataFrame())
    
    if not holdings_df.empty:
        col_btn, col_spacer = st.columns([1, 3])
        with col_btn:
            generate_btn = st.button("Generate Insights", key="client_pv_gen_insights")
        
        if generate_btn:
            with st.spinner("Generating personalized insights..."):
                value_col = [c for c in holdings_df.columns if 'Value' in c][0]
                total_value = holdings_df[value_col].sum()
                top_holdings = holdings_df.nlargest(5, value_col)[['Ticker', 'Name', value_col]].to_dict('records')
                
                insights = generate_portfolio_insights(
                    api_key=api_key,
                    country_exposure=country_exp,
                    sector_exposure=sector_exp,
                    top_holdings=top_holdings,
                    total_value=total_value
                )
                st.session_state.client_pv_insights = insights
        
        if st.session_state.client_pv_insights:
            insights = st.session_state.client_pv_insights
            
            for i, insight in enumerate(insights):
                with st.container():
                    st.markdown(f"""
                    <div class="insight-card">
                        <strong>{insight.get('title', 'Insight')}</strong><br>
                        {insight.get('description', '')}
                        <br><br>
                        <em>Recommendation: {insight.get('recommendation', '')}</em>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Upload a portfolio PDF above to generate personalized insights.")
else:
    st.info("Upload a portfolio PDF above to generate personalized insights.")

# disclaimer
st.divider()
st.caption("""
**Disclaimer:** The insights provided are for informational purposes only and do not constitute financial advice. 
Investment decisions should be made based on your individual circumstances and in consultation with a qualified financial advisor.
""")
