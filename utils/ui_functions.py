import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.portfolio_storage import save_portfolio, load_portfolios, delete_portfolio
import plotly.graph_objects as go
from datetime import timedelta
from utils.optimiser import optimize_portfolio, create_constraints, calc_port_perf
from utils.performance import get_performance_summary, calculate_relative_metrics
from utils.exposure import get_exposure_charts, get_concentration_exposure
from utils.plotting import (
    plot_cumulative_returns, plot_drawdowns, plot_correlation_heatmap, 
    plot_scatter_risk_return, plot_allocation_pie, apply_syfe_layout, plot_bar_chart, 
    plot_rolling_volatility, plot_monthly_heatmap
)
from utils.data_loader import download_data
import port_cons as pc
from styles.syfe_colors import PALETTE
import os
from utils.llm_helper import get_gemini_recommendation, parse_recommendation_table, analyze_portfolio_appropriateness
from utils.metabase_helper import get_client_info_by_id, get_active_portfolios
import requests
import zipfile
import io


def get_portfolio_objective(portfolio_name):
    """get objective from portfolio name"""
    name_lower = portfolio_name.lower()
    
    if any(keyword in name_lower for keyword in ['china', 'disruptive', 'esg', 'clean', 'healthcare', 'equity', 'growth']):
        return 'Wealth Growth'
    if any(keyword in name_lower for keyword in ['income', 'enhance', 'reit', 'blackrock', 'monroe']):
        return 'Income Generation'
    if any(keyword in name_lower for keyword in ['preserve', 'cash', 'guaranteed']):
        return 'Capital Preservation'
    return 'Wealth Growth'


def render_client_upload(df_asset_class, df_sector, df_country, df_underlying, key_prefix="upload"):
    st.header("1. Client Data Source")
    
    # Create tabs for different data sources
    tab_csv, tab_metabase = st.tabs(["📂 CSV Upload", "🔍 Metabase Lookup"])
    
    df_client = None
    
    with tab_csv:
        uploaded_file = st.file_uploader("Upload Client Portfolio CSV", type=['csv'], key=f"{key_prefix}_file_uploader")
        if uploaded_file:
            try:
                df_client = pd.read_csv(uploaded_file)
                
                df_client.columns = df_client.columns.str.lower().str.strip()
                if 'type' in df_client.columns:
                    df_client['type'] = df_client['type'].str.replace('BlackRock', 'Blackrock', case=False)
                    df_client['type'] = df_client['type'].str.replace('Monroe', 'Monroe', case=False)
                if 'status' in df_client.columns:
                    df_client = df_client[df_client['status'].str.upper().isin(['ACTIVE', 'ACTIVE_NEW'])]
                
                client_id = 'CSV_Client'
                if 'portfolio_id' in df_client.columns and len(df_client) > 0:
                    first_id = str(df_client['portfolio_id'].iloc[0])
                    if first_id and len(first_id) > 3:
                        client_id = first_id[:10]
                st.session_state['csv_client_df'] = df_client
                st.session_state['metabase_client_df'] = df_client  # Also set this for compatibility
                st.session_state['metabase_client_info'] = {
                    'client_id': client_id,
                    'name': uploaded_file.name.replace('.csv', ''),
                    'age': None,
                    'risk_score': None
                }
                st.session_state['csv_raw_data'] = df_client.copy()
                
                st.success(f"Loaded {len(df_client)} portfolios from CSV (Client: {client_id})")
                
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    with tab_metabase:
        c_id, c_btn = st.columns([3, 1])
        with c_id:
            client_id_input = st.text_input("Client ID", key=f"{key_prefix}_meta_id")
        with c_btn:
            st.write("")
            st.write("")
            lookup_clicked = st.button("Fetch Client", key=f"{key_prefix}_meta_btn")
            
        if lookup_clicked and client_id_input:
            with st.spinner("Fetching client data from Metabase..."):
                try:
                    client_id_str = client_id_input.strip()
                    client_id = int(client_id_str)
                    client_info = get_client_info_by_id(client_id)
                    
                    if client_info.get('client_id'):
                        fetched_df = get_active_portfolios(client_info.get('client_id'))
                        st.session_state['metabase_client_info'] = client_info
                        st.session_state['metabase_client_df'] = fetched_df
                        st.session_state['client_profile'] = {
                            'name': client_info.get('name'),
                            'age': client_info.get('age'),
                            'risk_score': client_info.get('risk_score'),
                            'email': client_info.get('email')
                        }
                        
                        with st.spinner("Generating AI Analysis..."):
                            holdings_cols = []
                            for col in ['name', 'type', 'nav_in_sgd']:
                                if col in fetched_df.columns:
                                    holdings_cols.append(col)
                            holdings_list = fetched_df[holdings_cols].to_dict('records') if holdings_cols else []
                            ai_analysis = analyze_portfolio_appropriateness(
                                "AIzaSyCZThunnMaO5Ni6fJDLnWsN24JX0yCN3p0", # Hardcoded key
                                st.session_state['client_profile'],
                                holdings_list
                            )
                            st.session_state['metabase_ai_analysis'] = ai_analysis
                        st.success(f"Found client: {client_info.get('name')} (ID: {client_info.get('client_id')})")

                    else:
                        st.error("Client not found or invalid ID.")
                        
                except Exception as e:
                    st.error(f"Error fetching data: {e}")

        if st.session_state.get('metabase_client_info') and st.session_state.get('metabase_client_df') is not None:
            c_info = st.session_state['metabase_client_info']
            df_display = st.session_state['metabase_client_df']
            c1, c2, c3 = st.columns(3)
            c1.metric("Name", c_info.get('name'))
            c2.metric("Age", c_info.get('age'))
            c3.metric("Risk Score", c_info.get('risk_score'))
            
            if not df_display.empty:
                display_cols = []
                for col in ['name', 'type', 'portfolio_type', 'nav_in_sgd', 'nav', 'value', 'pnl_inception', 'pnl', 'returns']:
                    if col in df_display.columns and col not in display_cols:
                        display_cols.append(col)
                if not display_cols:
                    display_cols = list(df_display.columns)
                
                st.dataframe(df_display[display_cols], use_container_width=True)
                df_client = df_display # Set for downstream processing
            else:
                st.warning("Client has no ACTIVE or ACTIVE_NEW portfolios.")

    if df_client is not None and not df_client.empty:
        CLIENT_PORTFOLIO_MAPPING = {
            'GLOBAL_EQUITY_100': 'Core Equity100',
            'GLOBAL_EQUITY': 'Core Equity100',
            'CORE_EQUITY100': 'Core Equity100',
            'CORE_EQUITY_100': 'Core Equity100',
            'EQUITY100': 'Core Equity100',
            'CORE_GROWTH': 'Core Growth',
            'CORE_BALANCED': 'Core Balanced',
            'CORE_DEFENSIVE': 'Core Defensive',
            'DOWNSIDE_PROTECTED': 'Downside protected',
            'CHINA_GROWTH': 'China Growth',
            'DISRUPTIVE_TECHNOLOGY': 'Disruptive Technology',
            'DISRUPTIVE_TECH': 'Disruptive Technology',
            'ESG_AND_CLEAN_ENERGY': 'ESG & Clean Energy',
            'ESG_CLEAN_ENERGY': 'ESG & Clean Energy',
            'HEALTHCARE_INNOVATION': 'Healthcare Innovation',
            'INCOME_ENHANCE': 'Income Enhance',
            'INCOME_ENHANCED': 'Income Enhance',
            'INCOME_PRESERVE': 'Income Preserve',
            'REIT': 'REIT 100',
            'REIT_100': 'REIT 100',
            'REIT_RISK_MANAGED': 'REIT Risk Managed',
            'REIT_RM': 'REIT Risk Managed',
            'BLACKROCK': 'Blackrock Private Credit Fund',
            'BLACKROCK_PRIVATE_CREDIT': 'Blackrock Private Credit Fund',
            'BLACKROCK_PCF': 'Blackrock Private Credit Fund',
            'MONROE': 'Monroe Income Plus Fund',
            'MONROE_INCOME_PLUS': 'Monroe Income Plus Fund',
            'MONROE_CAPITAL': 'Monroe Income Plus Fund',
            'SILVERDALE': 'Silverdale Fund (Nov 2027)',
            'CASH_PLUS_GUARANTEED_SGD': 'Cash SGD guaranteed',
            'CASH_PLUS_GUARANTEED': 'Cash SGD guaranteed',
            'CASH_GUARANTEED': 'Cash SGD guaranteed',
            'CASH_PLUS_USD': 'Cash USD flexi',
            'CASH_USD': 'Cash USD flexi',
            'CASH_PLUS': 'Cash SGD flexi',
            'CASH_SGD': 'Cash SGD flexi',
            'CASH_PLUS_SGD': 'Cash SGD flexi',
            'SRS_INCOME_ENHANCE': 'SRS Income Enhance',
            'SRS_INCOME_ENHANCED': 'SRS Income Enhance',
            'SRS_INCOME_PRESERVE': 'SRS Income Preserve',
            'SRS_CASH_PLUS': 'SRS Cash SGD flexi',
            'SRS_CASH': 'SRS Cash SGD flexi',
            'SRS_GLOBAL_EQUITY_100': 'SRS Equity100',
            'SRS_EQUITY100': 'SRS Equity100',
            'SRS_EQUITY_100': 'SRS Equity100',
        }
        
        def normalize_portfolio_type(p_type):
            """normalize portfolio type for mapping"""
            if not isinstance(p_type, str):
                return None
            normalized = p_type.upper().strip()
            for suffix in ['_ACTIVE_NEW', '_ACTIVE', '_INACTIVE', '_CLOSED', '_PENDING']:
                if normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)]
            return normalized

        OBJECTIVE_MAPPING = {
            'Core Equity100': 'Wealth Growth',
            'SRS Equity100': 'Wealth Growth',
            'China Growth': 'Wealth Growth',
            'REIT 100': 'Income Generation',
            'Income Enhance': 'Income Generation',
            'SRS Income Enhance': 'Income Generation',
            'Blackrock PCF': 'Income Generation',
            'Income Preserve': 'Capital Preservation',
            'SRS Income Preserve': 'Capital Preservation',
            'Cash SGD guaranteed': 'Capital Preservation',
            'Cash SGD flexi': 'Capital Preservation',
            'Cash USD flexi': 'Capital Preservation'
        }
    
        client_holdings = {}
        total_nav = 0.0
        
        def safe_float(val, default=0.0):
            if pd.isna(val):
                return default
            try:
                return float(str(val).replace(',', ''))
            except (ValueError, TypeError):
                return default
    
        for index, row in df_client.iterrows():
            p_type = row.get('type', None)
            currency = str(row.get('currency', 'SGD'))
            nav = safe_float(row.get('nav_in_sgd', 0))
        
            if nav < 50:
                continue
            
            ticker = None
            normalized_type = normalize_portfolio_type(p_type)
            
            if normalized_type is None:
                continue
            if 'CASH' in normalized_type and 'PLUS' in normalized_type:
                if currency == 'USD' or 'USD' in normalized_type:
                    ticker = 'Cash USD flexi'
                elif 'GUARANTEED' in normalized_type:
                    ticker = 'Cash SGD guaranteed'
                else:
                    ticker = 'Cash SGD flexi'
            elif normalized_type in CLIENT_PORTFOLIO_MAPPING:
                ticker = CLIENT_PORTFOLIO_MAPPING[normalized_type]
            # Fallback: "contains" matching for Blackrock, Monroe, Custom, Downside (handles various CSV naming conventions)
            elif 'BLACKROCK' in normalized_type:
                ticker = 'Blackrock Private Credit Fund'
            elif 'MONROE' in normalized_type:
                ticker = 'Monroe Income Plus Fund'
            elif 'CUSTOM' in normalized_type:
                ticker = 'Custom Usd'  # Map to a recognizable name for alternatives
            elif 'DOWNSIDE' in normalized_type:
                ticker = 'Downside Protected'
            elif 'SILVER' in normalized_type or 'SILVERDALE' in normalized_type:
                # Extract date from name (e.g., "SILVER_DALE_USD_FEB2029" -> "Silverdale Fund Feb2029")
                import re
                date_match = re.search(r'(NOV|DEC|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT)\s*(\d{4})', normalized_type)
                if date_match:
                    month = date_match.group(1).title()
                    year = date_match.group(2)
                    ticker = f'Silverdale Fund {month}{year}'  # Match Excel format
                else:
                    ticker = 'Silverdale Fund Feb2029'  # Default
        
            if ticker:
                client_holdings[ticker] = client_holdings.get(ticker, 0.0) + nav
                total_nav += nav
    
        if total_nav > 0:
            # Calculate weights
            client_weights = {k: v/total_nav for k, v in client_holdings.items()}
        
            # Calculate Asset Allocation
            asset_allocation = {}
            for ticker, weight in client_weights.items():
                if ticker in df_asset_class.columns:
                    # Get asset breakdown for this ticker
                    ticker_assets = df_asset_class[ticker]
                    for asset, asset_weight in ticker_assets.items():
                        # Safely convert asset_weight to float
                        try:
                            aw = float(asset_weight) if not pd.isna(asset_weight) else 0.0
                        except (ValueError, TypeError):
                            aw = 0.0
                        if aw > 0:
                            asset_allocation[asset] = asset_allocation.get(asset, 0.0) + (aw * weight)
                else:
                    # Only CUSTOM is hardcoded as Specialized/alternatives (others should be in Excel)
                    ticker_lower = ticker.lower()
                    if 'custom' in ticker_lower:
                        asset_allocation['Specialized / alternatives'] = asset_allocation.get('Specialized / alternatives', 0.0) + weight
                    else:
                        # Fallback for other unknown tickers
                        asset_allocation['Unclassified'] = asset_allocation.get('Unclassified', 0.0) + weight

            # Calculate Objective Allocation
            objective_allocation = {}
            for ticker, weight in client_weights.items():
                obj = get_portfolio_objective(ticker)
                objective_allocation[obj] = objective_allocation.get(obj, 0.0) + weight

            st.markdown("### Portfolio Summary")
            
            # Identify top portfolio types
            sorted_ports = sorted(client_weights.items(), key=lambda x: x[1], reverse=True)
            top_port = sorted_ports[0][0] if sorted_ports else "N/A"
            
            # Identify dominant objective
            sorted_objs = sorted(objective_allocation.items(), key=lambda x: x[1], reverse=True)
            top_obj = sorted_objs[0][0] if sorted_objs else "N/A"
            
            # Identify top underlying (concentration) across all portfolios
            # We need to re-calculate concentration here for the summary text
            top_holdings_str = ""
            if df_underlying is not None and not df_underlying.empty:
                # Calculate strict top 3 for summary
                # Re-use exposure logic
                # Create a series from client_weights
                w_series = pd.Series({f"{k}_wgt": v for k, v in client_weights.items()})
                conc_exp = get_concentration_exposure(w_series, df_underlying, top_n=3)
                if not conc_exp.empty:
                    top_3 = conc_exp.head(3)
                    top_holdings_str = ", ".join([f"{n} ({w:.1%})" for n, w in top_3.items()])
            
            summary_text = f"Your current portfolio is primarily allocated to **{top_port}**, aligning with a **{top_obj}** objective."
            if top_holdings_str:
                summary_text += f" Across your holdings, your top underlying exposures are **{top_holdings_str}**."
            
            if 'Unclassified' in objective_allocation and objective_allocation['Unclassified'] > 0.1:
                summary_text += f" Note: {objective_allocation['Unclassified']:.1%} of your portfolio is in unclassified or custom assets."
            
            st.info(summary_text)

            if 'metabase_ai_analysis' in st.session_state:
                with st.expander("✨ Insights", expanded=True):
                     st.markdown(st.session_state['metabase_ai_analysis'])

            st.markdown("### Portfolio Exposures")
            st.markdown("#### Filter Exposures")
            c_filter, c_dummy = st.columns([2, 1])
            
            with c_filter:
                filter_mode = st.radio(
                    "Filter Mode", 
                    ["Show All", "By Objective", "By Portfolio Type"], 
                    horizontal=True,
                    key=f"{key_prefix}_filt_mode"
                )
            
            filtered_weights = {}
            
            if filter_mode == "Show All":
                filtered_weights = {f"{k}_wgt": v for k, v in client_weights.items()}
                
            elif filter_mode == "By Objective":
                all_objectives = sorted(list(set([get_portfolio_objective(ticker) for ticker in client_weights.keys()])))
                sel_objectives = st.multiselect("Select Objective(s)", all_objectives, default=all_objectives, key=f"{key_prefix}_filt_obj")
                
                for ticker, weight in client_weights.items():
                    obj = get_portfolio_objective(ticker)
                    if obj in sel_objectives:
                        filtered_weights[f"{ticker}_wgt"] = weight
                        
            elif filter_mode == "By Portfolio Type":
                all_port_types = sorted(list(client_weights.keys()))
                sel_port_types = st.multiselect("Select Portfolio Type(s)", all_port_types, default=all_port_types, key=f"{key_prefix}_filt_type")
                
                for ticker, weight in client_weights.items():
                    if ticker in sel_port_types:
                        filtered_weights[f"{ticker}_wgt"] = weight

            weights_series = pd.Series(filtered_weights)

            if not weights_series.empty:
                # Calculate all charts first
                charts = get_exposure_charts(weights_series, df_sector, df_country, df_asset_class, df_underlying, concentration_top_n=20)
                
                # Store exposure data in session_state for PDF export
                if 'sector_table' in charts:
                    st.session_state['sector_exposure'] = charts['sector_table']['Weight']
                if 'country_table' in charts:
                    st.session_state['country_exposure'] = charts['country_table']['Weight']
                if 'asset_table' in charts:
                    st.session_state['asset_class_exposure'] = charts['asset_table']['Weight']
                
                # Calculate filtered portfolio weights for display (remove _wgt suffix and renormalize)
                filtered_portfolio_weights = {}
                for k, v in filtered_weights.items():
                    ticker = k.replace("_wgt", "")
                    filtered_portfolio_weights[ticker] = v
                
                # Renormalize weights for the filtered selection
                total_filtered = sum(filtered_portfolio_weights.values())
                if total_filtered > 0:
                    filtered_portfolio_weights_norm = {k: v / total_filtered for k, v in filtered_portfolio_weights.items()}
                else:
                    filtered_portfolio_weights_norm = filtered_portfolio_weights
                
                # Calculate filtered NAV
                filtered_nav = sum(client_holdings.get(ticker, 0) for ticker in filtered_portfolio_weights.keys())
                
                # Calculate filtered objective allocation
                from styles.syfe_colors import get_portfolio_color, OBJECTIVE_COLORS
                filtered_obj_allocation = {}
                for ticker, weight in filtered_portfolio_weights_norm.items():
                    obj = get_portfolio_objective(ticker)
                    filtered_obj_allocation[obj] = filtered_obj_allocation.get(obj, 0) + weight
                
                # Row 1: NAV and Objective charts
                col_nav, col_objective = st.columns([1, 1])
                
                with col_nav:
                    st.markdown("**By Portfolio Type**")
                    # Use go.Figure with go.Pie for consistent sizing with Asset Class chart
                    port_colors = [get_portfolio_color(name) for name in filtered_portfolio_weights_norm.keys()]
                    fig_nav = go.Figure(data=[go.Pie(
                        labels=list(filtered_portfolio_weights_norm.keys()),
                        values=list(filtered_portfolio_weights_norm.values()),
                        hole=0.4,
                        marker=dict(colors=port_colors),
                        textinfo='none',
                        hovertemplate='%{label}: %{value:.4f}<extra></extra>'
                    )])
                    fig_nav.update_layout(
                        title={'text': f'<b>NAV: ${filtered_nav:,.2f}</b>', 'font': {'size': 20}},
                        font={'family': 'Helvetica Neue, Arial'},
                        height=500,
                        margin=dict(l=40, r=40, t=80, b=40)
                    )
                    st.plotly_chart(fig_nav, use_container_width=True, key=f"{key_prefix}_pie_nav")
                    
                    with st.expander("View Data"):
                        port_df = pd.DataFrame({
                            'Portfolio': list(filtered_portfolio_weights_norm.keys()),
                            'Weight': list(filtered_portfolio_weights_norm.values()),
                            'Est. Value ($)': [filtered_portfolio_weights_norm[k] * filtered_nav for k in filtered_portfolio_weights_norm.keys()]
                        }).set_index('Portfolio')
                        st.dataframe(
                            port_df.style.format({'Weight': '{:.2%}', 'Est. Value ($)': '${:,.2f}'}),
                            use_container_width=True
                        )
                
                with col_objective:
                    st.markdown("**By Objective**")
                    # Use go.Figure with go.Pie for consistent sizing with Asset Class chart
                    obj_colors = [OBJECTIVE_COLORS.get(name, '#666666') for name in filtered_obj_allocation.keys()]
                    fig_obj = go.Figure(data=[go.Pie(
                        labels=list(filtered_obj_allocation.keys()),
                        values=list(filtered_obj_allocation.values()),
                        hole=0.4,
                        marker=dict(colors=obj_colors),
                        textinfo='none',
                        hovertemplate='%{label}: %{value:.4f}<extra></extra>'
                    )])
                    fig_obj.update_layout(
                        title={'text': '<b>Objective Allocation</b>', 'font': {'size': 20}},
                        font={'family': 'Helvetica Neue, Arial'},
                        height=500,
                        margin=dict(l=40, r=40, t=80, b=40)
                    )
                    st.plotly_chart(fig_obj, use_container_width=True, key=f"{key_prefix}_pie_obj")
                    
                    with st.expander("View Data"):
                        obj_df = pd.DataFrame({
                            'Objective': list(filtered_obj_allocation.keys()),
                            'Weight': list(filtered_obj_allocation.values()),
                            'Est. Value ($)': [filtered_obj_allocation[k] * filtered_nav for k in filtered_obj_allocation.keys()]
                        }).set_index('Objective')
                        st.dataframe(
                            obj_df.style.format({'Weight': '{:.2%}', 'Est. Value ($)': '${:,.2f}'}),
                            use_container_width=True
                        )
                
                # Row 2: Asset Class and Top Holdings
                col_asset, col_holdings = st.columns([1, 1])
                
                with col_asset:
                    st.markdown("**Asset Class Allocation**")
                    if 'asset_pie' in charts:
                        st.plotly_chart(charts['asset_pie'], use_container_width=True, key=f"{key_prefix}_pie_asset_exp")
                        with st.expander("View Data"):
                            st.dataframe(charts['asset_table'].style.format("{:.2%}"), use_container_width=True)
                    else:
                        st.info("Asset Class data unavailable.")
                
                with col_holdings:
                    # Selector for Top N
                    t_topn = st.selectbox("Show Top", [10, 15, 20], key=f"{key_prefix}_topn_sel_client")
                    
                    if df_underlying is not None:
                        conc_raw = get_concentration_exposure(weights_series, df_underlying, top_n=t_topn)
                        if not conc_raw.empty:
                            conc_df = conc_raw.head(t_topn).to_frame(name='Weight')
                            conc_df['Est. Value ($)'] = conc_df['Weight'] * total_nav
                            total_conc = conc_df['Weight'].sum()
                            
                            # Bar Chart for Top Holdings - using consistent color scheme
                            from utils.plotting import plot_horizontal_bar_chart
                            fig_top_bar = plot_horizontal_bar_chart(
                                conc_raw.head(t_topn),
                                title=f'Top {t_topn} Holdings ({total_conc:.1%} Total)'
                            )
                            st.plotly_chart(fig_top_bar, use_container_width=True, key=f"{key_prefix}_bar_top_holdings_client")
                            
                            with st.expander("View Data"):
                                st.dataframe(
                                    conc_df.style.format({'Weight': '{:.2%}', 'Est. Value ($)': '${:,.2f}'}),
                                    use_container_width=True
                                )
                        else:
                            st.warning("No underlying holdings found.")
                    else:
                        st.info("Concentration data not available.")
                
                # Row 3: Sector and Country
                col_sector, col_country = st.columns([1, 1])
                
                with col_sector:
                    st.markdown("**Sector**")
                    if 'sector_pie' in charts:
                        st.plotly_chart(charts['sector_pie'], use_container_width=True, key=f"{key_prefix}_pie_sector")
                        with st.expander("View Data"):
                            st.dataframe(charts['sector_table'].style.format("{:.2%}"), use_container_width=True)
                    else:
                        st.info("Sector data unavailable.")
                
                with col_country:
                    st.markdown("**Country**")
                    if 'country_pie' in charts:
                        st.plotly_chart(charts['country_pie'], use_container_width=True, key=f"{key_prefix}_pie_country")
                        with st.expander("View Data"):
                            st.dataframe(charts['country_table'].style.format("{:.2%}"), use_container_width=True)
                    else:
                        st.info("Country data unavailable.")

            else:
                st.warning("No data matches the selected filters.")

            # Load into Portfolio 1 Button
            c_load1, c_load_new = st.columns(2)
            
            with c_load1:
                if st.button("Load into Portfolio 1", key=f"{key_prefix}_load_p1"):
                    # Update underlying data
                    st.session_state.portfolios[0]['name'] = "Client Portfolio"
                    st.session_state.portfolios[0]['tickers'] = list(client_weights.keys())
                    st.session_state.portfolios[0]['allocation_type'] = 'Fixed Weight'
                    st.session_state.portfolios[0]['fixed_weights'] = client_weights
                    st.session_state.portfolios[0]['cash_invested'] = total_nav
                    st.session_state.portfolios[0]['cash_currency'] = 'SGD'  # CSV nav_in_sgd is already in SGD
                
                    # Update dollar amounts and decimal weights dictionaries
                    st.session_state.portfolios[0]['dollar_amounts'] = client_holdings
                    st.session_state.portfolios[0]['decimal_weights'] = client_weights
                
                    # Increment refresh counter to force widget update
                    st.session_state.refresh_counter += 1
                
                    st.success("Loaded client portfolio into Portfolio 1!")
                    st.rerun()
            
            with c_load_new:
                if st.button("Load as New Portfolio", key=f"{key_prefix}_load_new"):
                    new_id = len(st.session_state.portfolios) + 1
                    new_port = {
                        'id': new_id,
                        'name': f"Client Upload {new_id}",
                        'tickers': list(client_weights.keys()),
                        'allocation_type': 'Fixed Weight',
                        'rebal_freq': 'Annually',
                        'min_weight': 0.0,
                        'max_weight': 1.0,
                        'cash_invested': total_nav,
                        'cash_currency': 'SGD',  # CSV nav_in_sgd is already in SGD
                        'fixed_weights': client_weights,
                        'dollar_amounts': client_holdings,
                        'decimal_weights': client_weights
                    }
                    st.session_state.portfolios.append(new_port)
                    st.session_state.refresh_counter += 1
                    st.success(f"Created new portfolio 'Client Upload {new_id}'!")
                    st.rerun()
    else:
        # If no data found and we are on the CSV tab, show warning (Metabase has its own warning)
        # We can't easily check tab activation in Streamlit without session state hacks, 
        # but we can just show generic warning if no data is present after interaction.
        pass


def render_client_pdf_upload(api_key, df_asset_class, df_sector, df_country, key_prefix="pdf_upload"):
    """
    Render a PDF upload section for parsing client portfolio PDFs using Gemini AI.
    Extracts stock holdings and provides country/sector exposure analysis via yfinance.
    Uses parallel dual-pass parsing for accuracy scoring.
    """
    from utils.pdf_helper import (
        parse_pdf_parallel,
        enrich_with_yfinance,
        calculate_aggregate_exposure,
        format_holdings_dataframe
    )
    
    st.header("1.1 Client PDF Upload")
    show_pdf_upload = st.toggle("Show PDF Upload", value=False, key=f"{key_prefix}_show_toggle")
    
    if not show_pdf_upload:
        st.info("Expand to upload a PDF document for AI-powered portfolio extraction.")
        return
    
    st.markdown("Upload a PDF document (e.g., brokerage statement) to extract stock holdings and analyze country/sector exposure.")
    
    # Presentation Currency Selector
    col_ccy, col_spacer = st.columns([1, 3])
    with col_ccy:
        presentation_ccy = st.selectbox(
            "Presentation Currency",
            ["USD", "SGD", "HKD"],
            key=f"{key_prefix}_pres_ccy",
            help="Convert all values to this currency using latest yfinance rates"
        )
    
    # Helper function to get FX rate from yfinance
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_fx_rate(from_ccy, to_ccy):
        """Get FX rate using yfinance. Retries until successful."""
        if from_ccy == to_ccy:
            return 1.0
        if not from_ccy or not to_ccy:
            return 1.0
        
        # Normalize currency codes (CNH and CNY are both Chinese Yuan)
        from_ccy = from_ccy.upper().strip()
        to_ccy = to_ccy.upper().strip()
        if from_ccy == 'CNH':
            from_ccy = 'CNY'
        if to_ccy == 'CNH':
            to_ccy = 'CNY'
        
        if from_ccy == to_ccy:
            return 1.0
        
        import yfinance as yf
        import time
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Try direct pair first (e.g., SGDUSD=X)
                ticker = f"{from_ccy}{to_ccy}=X"
                print(f"  Trying FX: {ticker} (attempt {attempt+1})")
                fx = yf.Ticker(ticker)
                hist = fx.history(period="5d")
                if not hist.empty:
                    rate = float(hist['Close'].iloc[-1])
                    if rate > 0:
                        print(f"  SUCCESS: {ticker} = {rate}")
                        return rate
                
                # Try inverse pair and invert
                ticker_inv = f"{to_ccy}{from_ccy}=X"
                print(f"  Trying FX inverse: {ticker_inv} (attempt {attempt+1})")
                fx_inv = yf.Ticker(ticker_inv)
                hist_inv = fx_inv.history(period="5d")
                if not hist_inv.empty:
                    rate_inv = float(hist_inv['Close'].iloc[-1])
                    if rate_inv > 0:
                        rate = 1.0 / rate_inv
                        print(f"  SUCCESS: 1/{ticker_inv} = {rate}")
                        return rate
                
                # Wait before retry
                if attempt < max_retries - 1:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"  ERROR on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        # After all retries failed, return 1.0 (no conversion)
        print(f"  FAILED: Could not get rate for {from_ccy} -> {to_ccy}, returning 1.0")
        return 1.0
    
    uploaded_pdf = st.file_uploader(
        "Upload Client Portfolio PDF", 
        type=['pdf'], 
        key=f"{key_prefix}_file_uploader",
        help="Upload a PDF containing stock holdings. Gemini AI will parse the document to extract tickers and values."
    )
    
    if uploaded_pdf:
        # Store in session state if not already processed
        pdf_key = f"{key_prefix}_parsed_data"
        
        # Check if we need to reprocess (new file uploaded)
        current_filename = uploaded_pdf.name
        stored_filename = st.session_state.get(f"{key_prefix}_filename", "")
        
        if current_filename != stored_filename:
            # New file - process it with parallel dual-pass parsing
            with st.spinner("Parsing PDF with Gemini AI..."):
                pdf_bytes = uploaded_pdf.read()
                holdings, accuracy_metrics, raw_responses = parse_pdf_parallel(api_key, pdf_bytes, uploaded_pdf.name, num_passes=2)
                
                if holdings:
                    with st.spinner("Downloading data from yfinance..."):
                        enriched, failed = enrich_with_yfinance(holdings)
                        country_exp, sector_exp = calculate_aggregate_exposure(enriched)
                        holdings_df = format_holdings_dataframe(enriched)
                        
                        # Store in session state
                        st.session_state[pdf_key] = {
                            'holdings': holdings,
                            'enriched': enriched,
                            'failed': failed,
                            'country_exp': country_exp,
                            'sector_exp': sector_exp,
                            'holdings_df': holdings_df,
                            'raw_response': '\n\n---\n\n'.join(raw_responses),
                            'accuracy_metrics': accuracy_metrics
                        }
                        st.session_state[f"{key_prefix}_filename"] = current_filename
                else:
                    st.error(f"Could not extract holdings from PDF.\n\n**Details:**\n{raw_responses}")
                    return
        
        # Display results if we have them
        if pdf_key in st.session_state:
            data = st.session_state[pdf_key]
            holdings_df = data['holdings_df'].copy()  # Make a copy to avoid modifying original
            country_exp = data['country_exp']
            sector_exp = data['sector_exp']
            failed = data['failed']
            enriched = data['enriched']
            
            # Apply currency conversion
            # Get unique source currencies from the data and fetch FX rates
            fx_rates = {}
            for ticker, info in enriched.items():
                source_ccy = info.get('currency', 'USD')
                if source_ccy and source_ccy not in fx_rates:
                    fx_rates[source_ccy] = get_fx_rate(source_ccy, presentation_ccy)
            
            # Print FX rates to terminal for debugging
            print(f"=== FX RATES DEBUG (presentation_ccy={presentation_ccy}) ===")
            for ccy, rate in fx_rates.items():
                print(f"  {ccy} → {presentation_ccy}: {rate}")
            
            # Convert values in holdings_df
            # The Currency column shows the original currency from yfinance
            # We convert from that currency to the presentation currency
            def convert_value(row):
                source_ccy = row.get('Currency', 'USD')
                if not source_ccy or source_ccy == presentation_ccy:
                    return row['Value ($)'], 1.0, 'No'
                rate = fx_rates.get(source_ccy, 1.0)
                converted = 'Yes' if abs(rate - 1.0) > 0.0001 else 'No'
                return row['Value ($)'] * rate, rate, converted
            
            # Apply conversion
            result = holdings_df.apply(convert_value, axis=1, result_type='expand')
            result.columns = ['Converted_Value', 'Rate_Applied', 'Converted?']
            holdings_df['Value'] = result['Converted_Value']
            holdings_df['Converted?'] = result['Converted?']
            holdings_df['FX Rate'] = result['Rate_Applied']  # Add rate column for debugging
            
            # Rename Currency to Original Currency to clarify it's the source currency
            holdings_df = holdings_df.rename(columns={'Currency': 'Original Currency'})
            
            holdings_df['Weight (%)'] = holdings_df['Value'] / holdings_df['Value'].sum()
            
            # Rename column to reflect presentation currency
            holdings_df = holdings_df.rename(columns={'Value': f'Value ({presentation_ccy})'})
            holdings_df = holdings_df.drop(columns=['Value ($)'])
            
            # Reorder columns - include FX Rate for debugging
            col_order = ['Ticker', 'Name', f'Value ({presentation_ccy})', 'Weight (%)', 'Country', 'Sector', 'Original Currency', 'FX Rate', 'Converted?']
            holdings_df = holdings_df[[c for c in col_order if c in holdings_df.columns]]
            
            total_value = holdings_df[f'Value ({presentation_ccy})'].sum()
            
            # Summary with correct currency symbol
            ccy_symbols = {'USD': '$', 'SGD': 'S$', 'HKD': 'HK$'}
            ccy_symbol = ccy_symbols.get(presentation_ccy, '$')
            st.success(f"Found {len(holdings_df)} holdings with total value of {ccy_symbol}{total_value:,.2f} {presentation_ccy}")
            
            # Display accuracy metrics from parallel parsing
            accuracy_metrics = data.get('accuracy_metrics', {})
            if accuracy_metrics:
                acc_score = accuracy_metrics.get('accuracy_score', 0)
                ticker_agree = accuracy_metrics.get('ticker_agreement', 0)
                
                col_acc1, col_acc2 = st.columns(2)
                with col_acc1:
                    st.metric("Accuracy Score", f"{acc_score:.1f}/100")
                with col_acc2:
                    st.metric("Ticker Agreement", f"{ticker_agree:.1f}%")
            
            # Failed tickers warning
            if failed:
                st.warning(f"{len(failed)} ticker(s) could not be found via yfinance: {', '.join(failed)}")
            
            # Holdings Table (removed emoji)
            st.subheader("Extracted Holdings")
            value_col = f'Value ({presentation_ccy})'
            st.dataframe(
                holdings_df.style.format({
                    value_col: f'{ccy_symbol}{{:,.2f}}',
                    'Weight (%)': '{:.2%}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Exposure Charts (removed emoji)
            st.subheader("Exposure Analysis")
            
            col_country, col_sector = st.columns(2)
            
            with col_country:
                st.markdown("**Country Exposure**")
                if country_exp:
                    # Use project's plot_allocation_pie for consistent styling
                    country_series = pd.Series(country_exp)
                    fig_country = plot_allocation_pie(country_series, title='Country Allocation')
                    st.plotly_chart(fig_country, use_container_width=True, key=f"{key_prefix}_country_pie")
                    
                    with st.expander("View Data"):
                        country_table = country_series.to_frame(name='Weight')
                        st.dataframe(
                            country_table.style.format({'Weight': '{:.2%}'}),
                            use_container_width=True
                        )
                else:
                    st.info("No country data available.")
            
            with col_sector:
                st.markdown("**Sector Exposure**")
                if sector_exp:
                    # Use project's plot_allocation_pie for consistent styling
                    sector_series = pd.Series(sector_exp)
                    fig_sector = plot_allocation_pie(sector_series, title='Sector Allocation')
                    st.plotly_chart(fig_sector, use_container_width=True, key=f"{key_prefix}_sector_pie")
                    
                    with st.expander("View Data"):
                        sector_table = sector_series.to_frame(name='Weight')
                        st.dataframe(
                            sector_table.style.format({'Weight': '{:.2%}'}),
                            use_container_width=True
                        )
                else:
                    st.info("No sector data available.")
            
            # Load into Portfolio Analysis
            st.markdown("---")
            col_load1, col_load_new = st.columns(2)
            
            with col_load1:
                if st.button("Load into Portfolio 1", key=f"{key_prefix}_load_p1"):
                    # Create weights from values
                    client_weights = {}
                    client_holdings_raw = {}
                    
                    for ticker, info in enriched.items():
                        client_weights[ticker] = info['value'] / total_value if total_value > 0 else 0
                        client_holdings_raw[ticker] = info['value']
                    
                    # Update session state
                    st.session_state.portfolios[0]['name'] = "PDF Portfolio"
                    st.session_state.portfolios[0]['tickers'] = list(client_weights.keys())
                    st.session_state.portfolios[0]['allocation_type'] = 'Fixed Weight'
                    st.session_state.portfolios[0]['fixed_weights'] = client_weights
                    st.session_state.portfolios[0]['cash_invested'] = total_value
                    st.session_state.portfolios[0]['cash_currency'] = presentation_ccy  # Use selected presentation currency
                    st.session_state.portfolios[0]['dollar_amounts'] = client_holdings_raw
                    st.session_state.portfolios[0]['decimal_weights'] = client_weights
                    
                    st.session_state.refresh_counter += 1
                    st.success("Loaded PDF portfolio into Portfolio 1!")
                    st.rerun()
            
            with col_load_new:
                if st.button("Load as New Portfolio", key=f"{key_prefix}_load_new"):
                    # Create weights from values
                    client_weights = {}
                    client_holdings_raw = {}
                    
                    for ticker, info in enriched.items():
                        client_weights[ticker] = info['value'] / total_value if total_value > 0 else 0
                        client_holdings_raw[ticker] = info['value']
                    
                    new_id = len(st.session_state.portfolios) + 1
                    new_port = {
                        'id': new_id,
                        'name': f"PDF Upload {new_id}",
                        'tickers': list(client_weights.keys()),
                        'allocation_type': 'Fixed Weight',
                        'rebal_freq': 'Annually',
                        'min_weight': 0.0,
                        'max_weight': 1.0,
                        'cash_invested': total_value,
                        'cash_currency': presentation_ccy,  # Use selected presentation currency
                        'fixed_weights': client_weights,
                        'dollar_amounts': client_holdings_raw,
                        'decimal_weights': client_weights
                    }
                    st.session_state.portfolios.append(new_port)
                    st.session_state.refresh_counter += 1
                    st.success(f"Created new portfolio 'PDF Upload {new_id}'!")
                    st.rerun()
            
            # Debug expander
            with st.expander("Debug: Raw AI Response"):
                st.text(data.get('raw_response', 'No raw response available'))


from utils.risk_engine import calculate_risk_score, get_eligible_portfolios, get_recommendation_text, get_smart_recommendation
from utils.llm_helper import get_recommendation_narrative


def render_recommender(GEMINI_API_KEY, key_prefix="rec"):
    st.header("2. Syfe Portfolio Recommender")
    show_recommender = st.toggle("Show Recommender", value=False, key=f"{key_prefix}_show_toggle")
    
    if not show_recommender:
        st.info("Expand to access the portfolio recommender.")
        return
    
    st.subheader("2.1 Personal & Financial Profile")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        age = st.number_input("Age", 18, 100, 30, key=f"{key_prefix}_age")
        
        income = st.number_input(
            "Annual Income ($)", 
            min_value=0, value=60000, step=1000, 
            help="Total annual income before tax. Used to assess financial capacity.",
            key=f"{key_prefix}_inc"
        )
        
        savings_rate = st.slider(
            "Savings Rate (% of Income)", 
            0, 100, 20, 
            format="%d%%",
            help="Percentage of monthly income saved/invested.",
            key=f"{key_prefix}_sav"
        ) / 100.0  # Convert back to decimal for calculations
        
    with col_p2:
        liquid_assets = st.number_input(
            "Net Liquid Assets ($)", 
            min_value=0, value=50000, step=1000,
            help="Total cash and investments explicitly available for this portfolio or general wealth.",
            key=f"{key_prefix}_liq"
        )
        
        concentration_high = st.checkbox(
            "Is this portfolio >50% of your Net Worth?",
            help="High concentration implies higher risk to your overall financial health if this portfolio drops.",
            key=f"{key_prefix}_conc"
        )
        
        short_term_need = st.checkbox(
            "Need to withdraw >90% within 12 months?",
            help="If yes, capital preservation is paramount regardless of other factors.",
            key=f"{key_prefix}_stn"
        )

    knowledge = st.select_slider(
        "Investment Knowledge", 
        options=["None", "Minimal", "Average", "Good", "Expert"], 
        value="Average",
        key=f"{key_prefix}_know"
    )

    st.subheader("2.2 Investment Goals")
    st.markdown("Please define your goals below.")
    
    # Default data for the table
    default_goals = pd.DataFrame(
        [
            {"Goal Name": "Retirement", "Objective": "Wealth Growth", "Target Amount ($)": 1000000, "Lump Sum ($)": 50000, "Tenure (Years)": 20, "Risk Tolerance": "Moderate (20% drawdown)"},
            {"Goal Name": "House Downpayment", "Objective": "Capital Preservation", "Target Amount ($)": 100000, "Lump Sum ($)": 20000, "Tenure (Years)": 3, "Risk Tolerance": "Low (10-15% drawdown)"},
        ]
    )
    
    goal_df = st.data_editor(
        default_goals,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Goal Name": st.column_config.TextColumn("Goal Name", required=True),
            "Objective": st.column_config.SelectboxColumn(
                "Objective",
                help="Select the primary objective for this goal.",
                width="medium",
                options=[
                    "Wealth Growth",
                    "Income Generation",
                    "Capital Preservation"
                ],
                required=True
            ),
            "Target Amount ($)": st.column_config.NumberColumn("Target Amount ($)", min_value=0, format="$%d"),
            "Lump Sum ($)": st.column_config.NumberColumn("Lump Sum ($)", min_value=0, format="$%d"),
            "Tenure (Years)": st.column_config.NumberColumn("Tenure (Years)", min_value=1, max_value=50),
            "Risk Tolerance": st.column_config.SelectboxColumn(
                "Risk Tolerance",
                help="Your willingness to accept drawdowns for this goal.",
                width="medium",
                options=[
                    "Low (10-15% drawdown)",
                    "Moderate (20% drawdown)",
                    "Higher (30% drawdown)",
                    "Maximum (30%+ drawdown)"
                ],
                required=True
            )
        },
        key=f"{key_prefix}_goals_table"
    )
    
    interest_in_ai = st.checkbox("Interested in AI / Thematic Exposure?", 
                                 help="If selected, we may allocate up to 30% to high-growth thematic portfolios (e.g. Disruptive Tech) if your risk profile allows.",
                                 key=f"{key_prefix}_ai_toggle")

    # --- Step 4: Qualitative & Check ---
    st.subheader("2.3 Additional Details")
    qualitative_details = st.text_area(
        "Additional Qualitative Inputs", 
        placeholder="Any other details relevant to your financial situation (e.g., existing liabilities, preferences, constraints)...",
        key=f"{key_prefix}_qual"
    )
    is_accredited = st.checkbox("I am an Accredited Investor (AI)", key=f"{key_prefix}_ai")
    
    st.markdown("---")
    if st.button("Generate Smart Recommendation", type="primary", key=f"{key_prefix}_calc"):
        
        # Derive risk willingness from goals table (use most conservative)
        risk_map = {
            "Low (10-15% drawdown)": 4,
            "Moderate (20% drawdown)": 3,
            "Higher (30% drawdown)": 2,
            "Maximum (30%+ drawdown)": 1
        }
        
        risk_willingness = 3  # Default moderate
        if not goal_df.empty and "Risk Tolerance" in goal_df.columns:
            # Use the most conservative (highest number = most conservative)
            risk_values = [risk_map.get(r, 3) for r in goal_df["Risk Tolerance"].dropna()]
            if risk_values:
                risk_willingness = max(risk_values)  # Most conservative
        
        # Derive horizon from goals table (use maximum tenure)
        horizon_years = 10  # Default
        if not goal_df.empty and "Tenure (Years)" in goal_df.columns:
            tenures = goal_df["Tenure (Years)"].dropna()
            if not tenures.empty:
                horizon_years = tenures.max()

        # Create profile dict
        profile = {
            'risk_willingness': risk_willingness,
            'time_horizon_years': horizon_years,
            'annual_income': income,
            'liquid_assets': liquid_assets,
            'savings_rate': savings_rate,
            'short_term_need': short_term_need,
            'concentration_high': concentration_high,
            'investment_knowledge': knowledge,
            'Interest in AI': "Yes" if interest_in_ai else "No",
            'Qualitative Details': qualitative_details,
            'Goals': goal_df.to_csv(index=False), # Pass table as CSV text to LLM (includes Risk Tolerance per goal)
            'Risk Score': 'Calculating...' 
        }
        
        # 1. Calculate Risk Score (keep this for internal validation if needed, or pass to LLM)
        score, overrides = calculate_risk_score(profile)
        profile['Risk Score'] = f"{score}/5" 
        
        # 2. Get AI Recommendation (Gemini generates the Table + Rationale)
        from utils.llm_helper import get_gemini_recommendation, parse_recommendation_table
        
        with st.spinner("Generating AI-powered Recommendation..."):
            rec_text = get_gemini_recommendation(GEMINI_API_KEY, profile)
            
        # Parse data for Logic
        portfolios, failed = parse_recommendation_table(rec_text)

        # Create maps
        goal_map = {}
        target_map = {}
        if not goal_df.empty and "Goal Name" in goal_df.columns and "Lump Sum ($)" in goal_df.columns:
            for _, row in goal_df.iterrows():
                goal_map[row["Goal Name"]] = row["Lump Sum ($)"]
                if "Target Amount ($)" in goal_df.columns:
                    target_map[row["Goal Name"]] = row["Target Amount ($)"]

        # Store in session state
        st.session_state.rec_results = {
            'score': score,
            'overrides': overrides,
            'rec_text': rec_text,
            'portfolios_data': portfolios, 
            'goal_map': goal_map,
            'target_map': target_map
        }
        st.rerun()

    # --- Display Results (Persistent) ---
    if 'rec_results' in st.session_state:
        results = st.session_state.rec_results
        score = results.get('score', 0)
        overrides = results.get('overrides', [])
        rec_text = results.get('rec_text', '')
        portfolios = results.get('portfolios_data', {})
        
        # Display Score First
        st.success(f"**Client Risk Score: {score}/5**")
        if overrides:
            st.warning(f"**Score Adjustments:** {', '.join(overrides)}")
            
        # Display Narrative (Text Only)
        st.markdown(f"### Proposed Strategy")
        
        # Parse text to highlight notes, but SKIP the markdown table lines
        lines = rec_text.split('\n')
        narrative_lines = []
        skip_table = False
        
        for line in lines:
            line_stripped = line.strip()
            # Detect table boundary
            if line_stripped.startswith("|") and "Objective" in line_stripped:
                skip_table = True
            elif skip_table and not line_stripped.startswith("|"):
                 skip_table = False
            
            if not skip_table and not line_stripped.startswith("|"):
                narrative_lines.append(line)
        
        for line in narrative_lines:
            if line.strip().startswith("Note:") or line.strip().startswith("**Note") or "should continue to invest" in line:
                st.success(line)
            elif "Detailed Analysis" in line or "Investment Strategy" in line:
                 st.markdown(f"#### {line}")
            elif line.strip(): # Skip empty lines if excessive
                st.markdown(line)

        # Render Structured Table (Deterministic Display)
        if portfolios:
            st.markdown("#### Recommended Allocation")
            
            # Convert nested dict to flat list for display
            table_rows = []
            for goal, data in portfolios.items():
                horizon = data.get('horizon', 10)
                objective = data.get('objective', 'Wealth Growth')
                
                for ticker, weight in data.get('allocations', {}).items():
                    table_rows.append({
                        "Goal": goal,
                        "Objective": objective,
                        "Horizon": f"{horizon} Yrs",
                        "Product": ticker,
                        "Allocation": weight
                    })
            
            if table_rows:
                df_disp = pd.DataFrame(table_rows)
                st.dataframe(
                    df_disp.style.format({"Allocation": "{:.1%}"}),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            # Fallback if parsing fails - show raw just in case
            st.warning("Could not structure table data. Displaying raw output:")
            st.markdown(rec_text)
        
    if 'rec_results' in st.session_state:
        st.markdown("---")
        if st.button("Add Selected Strategy to Portfolio Analysis", key=f"{key_prefix}_add_strat"):
            # Logic to add the "Primary" one or let user choose from eligible?
            # For simplicity, if Rec Text suggests "Core Equity 100", let's try to add that.
            # Or just add a placeholder/empty portfolio named after the objective/strategy.
            
            # Let's add the first eligible portfolio or specific top 1
            # Based on standard flow we need tickers.
            # Since we don't have hardcoded tickers for "Core Equity 100" inside this code block,
            # we rely on the user to construct it or we preload standard templates.
            # The previous code used LLM to get allocations.
            # Here "At no time are we changing the output of the AI recommendation" -> This implies we might still need LLM?
            # "Note at no time are we changing the output of the AI recommendation" -> This was in the prompt start.
            # BUT later: "Replace the current script... Recommender Engine Overhaul... AI generation questionnaire... simplify it... Risk Engine...".
            # The prompt is slightly contradictory. "Note at no time are we changing the output of the AI recommendation" vs "Recommender Engine Overhaul".
            # I think the user means: The *content* of the products (Core Equity 100 tickers) hasn't changed, but the *Selection Logic* has.
            # OR, maybe we should still call Gemini to get the *allocations* for "Core Equity 100"?
            # Or we assume "Core Equity 100" is a known template.
            # Given I don't have a template library loaded, I will add an empty portfolio named "Recommended Strategy" 
            # and let the user fill it, OR call Gemini just for value-add?
            # Re-reading: "Note at no time are we changing the output of the AI recommendation" -> Maybe this refers to the previous artifact's specific table format?
            # But the detailed "Risk Engine" instructions clearly mandate a DETERMINISTIC score and eligible list.
            # So the "AI Recommendation" in the prompt likely refers to the *Goal Mapper* logic which I implemented deterministically.
            
            # To be safe and useful: I will create a new portfolio named with the Top Recommendation Name.
            # If the name matches `CLIENT_PORTFOLIO_MAPPING` we used earlier, we assume user knows what it is.
            # Actually, without a template DB, I can't populate tickers.
            # I will just create a shell portfolio.
            
            results = st.session_state.rec_results
            portfolios_data = results.get('portfolios_data', {})
            goal_map = results.get('goal_map', {})
            target_map = results.get('target_map', {})
            
            if portfolios_data:
                added_count = 0
                for goal_name, p_data in portfolios_data.items():
                    # p_data = {'objective': ..., 'horizon': ..., 'allocations': {Ticker: Weight...}}
                    new_id = len(st.session_state.portfolios) + 1
                    
                    allocs = p_data.get('allocations', {})
                    tickers = list(allocs.keys())
                    
                    if not tickers:
                        continue
                        
                    # Lookup Lump Sum and Target Amount from Goal Maps
                    lump_sum = float(goal_map.get(goal_name, 100000.0))
                    target_amount = float(target_map.get(goal_name, lump_sum * 2))  # Default to 2x lump sum if not specified
                        
                    st.session_state.portfolios.append({
                        'id': new_id,
                        'name': f"Rec: {goal_name}",
                        'goal_name': goal_name,
                        'tickers': tickers,
                        'allocation_type': 'Fixed Weight',
                        'rebal_freq': 'Quarterly',
                        'min_weight': 0.0,
                        'max_weight': 1.0,
                        'cash_invested': lump_sum, # Use mapped lump sum
                        'cash_currency': 'SGD',  # Recommender lump sum is in SGD
                        'target_amount': target_amount, # Target amount from goal
                        'fixed_weights': allocs,
                        'decimal_weights': allocs, # Sync
                        'dollar_amounts': {ticker: lump_sum * weight for ticker, weight in allocs.items()}, # Proportional to lump sum
                        'objective': p_data.get('objective', 'Wealth Growth'),
                        'horizon': p_data.get('horizon', 10)
                    })
                    added_count += 1
                
                if added_count > 0:
                    st.session_state.refresh_counter += 1
                    st.success(f"Added {added_count} recommended strategies to Portfolio Construction!")
                    # Don't rerun - keep recommendation visible
                else:
                    st.warning("No valid portfolios found in recommendation to add.")
            else:
                st.warning("No structured portfolio data found. Please regenerate.")

def render_portfolio_construction(df_sec_list, available_tickers, bm_tickers, data_source, start_date, end_date, currency, backtest_offset_days, data_path, df_fi_metrics, df_fx, df_daily_ret, key_prefix="cons"):
    st.header("3. Portfolio Construction")
    
    # Calculate Data Date Range for Tooltips
    start_str = "N/A"
    end_str = "N/A"
    if df_daily_ret is not None and not df_daily_ret.empty:
        start_str = df_daily_ret.index.min().strftime('%d %b %Y')
        end_str = df_daily_ret.index.max().strftime('%d %b %Y')
    
    tooltip_data_msg = f"Data as of: {end_str}\nTime frame taken: {start_str} - {end_str}"


    col_add, col_del = st.columns([1, 4])
    with col_add:
        if st.button("➕ Add Portfolio", key=f"{key_prefix}_add_port"):
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
                'cash_currency': 'USD',  # Manual portfolios default to USD
                'fixed_weights': {}
            })
            st.rerun()

    for i, port in enumerate(st.session_state.portfolios):
        with st.expander(f"{port['name']}", expanded=True):
            # Add remove button at top right
            if len(st.session_state.portfolios) > 1:
                if st.button(f"🗑️ Remove Portfolio", key=f"{key_prefix}_remove_{i}"):
                    st.session_state.portfolios.pop(i)
                    st.rerun()
        
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.portfolios[i]['name'] = st.text_input(
                    f"Name", 
                    value=port['name'], 
                    key=f"{key_prefix}_name_{i}_{st.session_state.refresh_counter}",
                    help=tooltip_data_msg
                )
                
                # Goal Name Input
                st.session_state.portfolios[i]['goal_name'] = st.text_input(
                    "Goal Name",
                    value=port.get('goal_name', ''),
                    placeholder="e.g. Retirement, House Downpayment",
                    key=f"{key_prefix}_goal_{i}_{st.session_state.refresh_counter}"
                )
                
                # Validate defaults exist in options (Handle casing mismatches e.g. Blackrock vs Blackrock)
                # available_tickers comes from df_sec_list['Ticker']
                # port['tickers'] comes from user session or defaults
                valid_defaults = []
                for t in port['tickers']:
                     if t in available_tickers:
                         valid_defaults.append(t)
                     else:
                         # Try insensitive match
                         match = next((x for x in available_tickers if x.lower() == t.lower()), None)
                         if match:
                             valid_defaults.append(match)
                
                st.session_state.portfolios[i]['tickers'] = st.multiselect(
                    f"Select Tickers",
                    options=available_tickers,
                    default=valid_defaults,
                    key=f"{key_prefix}_tickers_{i}_{st.session_state.refresh_counter}_v2"
                )
                    
                # Objective Selection
                valid_objectives = ["Wealth Growth", "Income Generation", "Capital Preservation"]
                current_objective = port.get('objective', 'Wealth Growth')
                
                # Validate and clean the objective if it's invalid
                if current_objective not in valid_objectives:
                    # Try to extract a valid objective from combined values
                    for valid_obj in valid_objectives:
                        if valid_obj in current_objective:
                            current_objective = valid_obj
                            st.session_state.portfolios[i]['objective'] = valid_obj  # Fix it in session state
                            break
                    else:
                        # If still invalid, default to Wealth Growth
                        current_objective = 'Wealth Growth'
                        st.session_state.portfolios[i]['objective'] = 'Wealth Growth'
                
                st.session_state.portfolios[i]['objective'] = st.selectbox(
                    "Objective",
                    valid_objectives,
                    index=valid_objectives.index(current_objective),
                    key=f"{key_prefix}_obj_{i}_{st.session_state.refresh_counter}"
                )
            with c2:
                st.session_state.portfolios[i]['allocation_type'] = st.selectbox(
                    f"Allocation", 
                    ['Max sharpe', 'Min vol', 'Equal weight', 'Fixed Weight'], 
                    index=['Max sharpe', 'Min vol', 'Equal weight', 'Fixed Weight'].index(port['allocation_type']) if port['allocation_type'] in ['Max sharpe', 'Min vol', 'Equal weight', 'Fixed Weight'] else 0, 
                    key=f"{key_prefix}_alloc_{i}_{st.session_state.refresh_counter}"
                )
                st.session_state.portfolios[i]['rebal_freq'] = st.selectbox(
                    f"Rebalance", 
                    ['Monthly', 'Quarterly', 'Semi-annually', 'Annually'], 
                    index=['Monthly', 'Quarterly', 'Semi-annually', 'Annually'].index(port['rebal_freq']), 
                    key=f"{key_prefix}_rebal_{i}_{st.session_state.refresh_counter}"
                )
            with c3:
                st.session_state.portfolios[i]['min_weight'] = st.number_input(f"Min Weight", min_value=0.0, max_value=1.0, value=port['min_weight'], step=0.01, key=f"{key_prefix}_min_{i}_{st.session_state.refresh_counter}")
                st.session_state.portfolios[i]['max_weight'] = st.number_input(f"Max Weight", min_value=0.0, max_value=1.0, value=port['max_weight'], step=0.01, key=f"{key_prefix}_max_{i}_{st.session_state.refresh_counter}")
        
            # Cash Invested, Currency, and Portfolio Target Input
            c_cash, c_ccy, c_target = st.columns([2, 1, 2])
            with c_cash:
                st.session_state.portfolios[i]['cash_invested'] = st.number_input(
                    f"Cash Invested", 
                    min_value=0.0, 
                    value=float(port.get('cash_invested', 100000.0)), 
                    step=1000.0, 
                    key=f"{key_prefix}_cash_{i}_{st.session_state.refresh_counter}"
                )
            
            with c_ccy:
                ccy_options = ['SGD', 'USD', 'HKD']
                current_ccy = port.get('cash_currency', 'USD')
                ccy_index = ccy_options.index(current_ccy) if current_ccy in ccy_options else 1
                st.session_state.portfolios[i]['cash_currency'] = st.selectbox(
                    "Currency",
                    options=ccy_options,
                    index=ccy_index,
                    key=f"{key_prefix}_cash_ccy_{i}_{st.session_state.refresh_counter}"
                )
            
            with c_target:
                st.session_state.portfolios[i]['target_amount'] = st.number_input(
                    f"Portfolio Target ($)", 
                    min_value=0.0, 
                    value=float(port.get('target_amount', 0.0)), 
                    step=1000.0, 
                    key=f"{key_prefix}_target_{i}_{st.session_state.refresh_counter}",
                    help="Target goal amount for this portfolio. Used in Growth Forecast."
                )
            

            
            # Fixed Weight Inputs
            if st.session_state.portfolios[i]['allocation_type'] == 'Fixed Weight':
                input_type = st.radio("Input Type", ["Amounts ($)", "Weights (Decimal)"], key=f"{key_prefix}_input_type_{i}_{st.session_state.refresh_counter}", horizontal=True)
            
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
                            # Get value using case-insensitive lookup
                            current_dollar_amounts = st.session_state.portfolios[i].get('dollar_amounts', {})
                            existing_val = current_dollar_amounts.get(ticker)
                            if existing_val is None:
                                # Try case-insensitive fallback for tickers like Blackrock/Blackrock
                                alt_key = next((k for k in current_dollar_amounts.keys() if k.lower() == ticker.lower()), None)
                                existing_val = current_dollar_amounts.get(alt_key, 1000.0)

                            amount = st.number_input(
                                f"{ticker} ($)", 
                                min_value=0.0, 
                                value=float(existing_val),
                                step=100.0,
                                key=f"{key_prefix}_amt_{i}_{ticker}_{st.session_state.refresh_counter}"
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
                                key=f"{key_prefix}_wgt_{i}_{ticker}_{st.session_state.refresh_counter}"
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
    if st.button("🚀 Run Optimization", use_container_width=True, type="primary", key=f"{key_prefix}_run_opt"):
        st.session_state.run_optimization = True
    else:
        st.session_state.run_optimization = False

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
            ret_all, ret_bm = download_data(data_source, start_date, end_date, all_tickers, bm_tickers, currency, df_sec_list, df_fx, data_path, df_daily_ret)
        
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


def render_analysis(df_sec_list, df_fi_metrics, df_fx, df_sector, df_country, df_asset_class, df_underlying, data_path, bm_tickers, backtest_offset_days, data_source, start_date, end_date, currency, key_prefix="anal", results_data=None):
    
    # Use provided results or fallback to session state
    if results_data is None:
        results_data = st.session_state.get('optimization_results')

    if results_data:
        # Store results_data for later use in other sections
        res_data = results_data
        results = res_data['results']
        ret_all = res_data['ret_all'].copy()
        ret_bm = res_data['ret_bm'].copy()
        
        # Calculate start_opt_date for analytics
        earliest_start_date = ret_all.index.min()
        start_opt_date = earliest_start_date + timedelta(days=backtest_offset_days)

    # ============================================
    # SECTION 4: EXPOSURE ANALYSIS
    # ============================================
    if st.session_state.optimization_results:
        st.header("4. Exposure Analysis")
        
        res_data_exp = st.session_state.optimization_results
        results_exp = res_data_exp['results']
        
        st.markdown("#### Filter Exposures (Aggregated by Cash Invested)")
        
        # Filter Controls
        filter_mode_exp = st.radio(
            "Filter Mode", 
            ["Show All", "By Objective", "By Portfolio"], 
            horizontal=True,
            key=f"{key_prefix}_exp_filt_mode_4"
        )
        
        # Gather Metadata
        port_meta_exp = {}
        for p in st.session_state.portfolios:
            port_meta_exp[p['name']] = {
                'objective': p.get('objective', 'Unclassified'),
                'goal_name': p.get('goal_name', 'Unclassified'),
                'cash_invested': p.get('cash_invested', 100000.0)
            }
            if port_meta_exp[p['name']]['cash_invested'] == 0:
                port_meta_exp[p['name']]['cash_invested'] = 100000.0

        # Filter Selection Logic
        filtered_portfolios_exp = []
        scaling_factors_exp = {}

        if filter_mode_exp == "Show All":
            filtered_portfolios_exp = list(results_exp.keys())
            
        elif filter_mode_exp == "By Objective":
            all_objs_exp = sorted(list(set([m['objective'] for m in port_meta_exp.values()])))
            sel_objs_exp = st.multiselect("Select Objective(s)", all_objs_exp, default=all_objs_exp, key=f"{key_prefix}_exp_filt_obj_4")
            filtered_portfolios_exp = [p for p in results_exp.keys() if port_meta_exp[p]['objective'] in sel_objs_exp]
            
        elif filter_mode_exp == "By Portfolio":
            all_tickers_exp = set()
            for p_name in results_exp.keys():
                w_df = results_exp[p_name]['weights']
                w_row = w_df.iloc[-1]
                for ticker_wgt in w_row.index:
                    if ticker_wgt.endswith('_wgt'):
                        ticker = ticker_wgt.replace('_wgt', '')
                        all_tickers_exp.add(ticker)
            
            all_tickers_exp = sorted(list(all_tickers_exp))
            sel_tickers_exp = st.multiselect("Select Underlying Portfolio(s)", all_tickers_exp, default=all_tickers_exp, key=f"{key_prefix}_exp_filt_ticker_4")
            filtered_portfolios_exp = list(results_exp.keys())
            selected_ticker_wgts_exp = set([f"{t}_wgt" for t in sel_tickers_exp])
            
        # Level 2 Filter: By Goal
        if filtered_portfolios_exp:
            available_goals_exp = sorted(list(set([port_meta_exp[p]['goal_name'] for p in filtered_portfolios_exp])))
            if available_goals_exp:
                sel_goals_exp = st.multiselect("Filter by Goal (Optional)", available_goals_exp, default=available_goals_exp, key=f"{key_prefix}_exp_filt_goal_4")
                filtered_portfolios_exp = [p for p in filtered_portfolios_exp if port_meta_exp[p]['goal_name'] in sel_goals_exp]

        global_ticker_values_exp = {}
        total_agg_value_exp = 0.0
        
        for p_name in filtered_portfolios_exp:
            w_df = results_exp[p_name]['weights']
            w_row = w_df.iloc[-1]
            cash = port_meta_exp[p_name]['cash_invested']
            scale = scaling_factors_exp.get(p_name, 1.0)
            effective_cash = cash * scale
            
            if effective_cash > 0:
                for ticker_wgt, weight in w_row.items():
                    if filter_mode_exp == "By Portfolio" and ticker_wgt not in selected_ticker_wgts_exp:
                        continue
                    
                    val = weight * effective_cash
                    if val > 0:
                        if ticker_wgt in global_ticker_values_exp:
                            global_ticker_values_exp[ticker_wgt] += val
                        else:
                            global_ticker_values_exp[ticker_wgt] = val
                        total_agg_value_exp += val
        
        # Convert back to weights and display charts
        if total_agg_value_exp > 0:
            agg_weights_exp = pd.Series({k: v / total_agg_value_exp for k, v in global_ticker_values_exp.items()})
            
            from utils.exposure import get_exposure_charts, get_concentration_exposure
            charts_exp = get_exposure_charts(agg_weights_exp, df_sector, df_country, df_asset_class, df_underlying)
            
            col_asset_exp, col_holdings_exp = st.columns([1, 1])
            
            with col_asset_exp:
                st.markdown("**Asset Class**")
                if 'asset_pie' in charts_exp:
                    st.plotly_chart(charts_exp['asset_pie'], use_container_width=True, key=f"{key_prefix}_anal_pie_asset_4")
                    with st.expander("View Data"):
                        st.dataframe(charts_exp['asset_table'].style.format("{:.2%}"), use_container_width=True)
                else:
                    st.info("Asset Class data unavailable.")
            
            with col_holdings_exp:
                st.markdown("**Sector**")
                if 'sector_pie' in charts_exp:
                    st.plotly_chart(charts_exp['sector_pie'], use_container_width=True, key=f"{key_prefix}_anal_pie_sector_4")
                    with st.expander("View Data"):
                        st.dataframe(charts_exp['sector_table'].style.format("{:.2%}"), use_container_width=True)
                else:
                    st.info("Sector data unavailable.")
            
            col_country_exp, col_conc_exp = st.columns([1, 1])
            
            with col_country_exp:
                st.markdown("**Country**")
                if 'country_pie' in charts_exp:
                    st.plotly_chart(charts_exp['country_pie'], use_container_width=True, key=f"{key_prefix}_anal_pie_country_4")
                    with st.expander("View Data"):
                        st.dataframe(charts_exp['country_table'].style.format("{:.2%}"), use_container_width=True)
                else:
                    st.info("Country data unavailable.")
            
            with col_conc_exp:
                st.markdown("**Top Holdings**")
                if 'concentration_bar' in charts_exp:
                    st.plotly_chart(charts_exp['concentration_bar'], use_container_width=True, key=f"{key_prefix}_anal_bar_conc_4")
                    with st.expander("View Data"):
                        st.dataframe(charts_exp['concentration_table'].style.format("{:.2%}"), use_container_width=True)
                else:
                    st.info("Concentration data unavailable.")
        else:
            st.warning("No data available for selected filters.")
    
    # ============================================
    # SECTION 4.1: FIXED INCOME ANALYTICS
    # ============================================
    if results_data:
        st.subheader("4.1 Fixed Income Analytics")
        
        res_data = results_data
        results = res_data['results']
        ret_all = res_data['ret_all'].copy()
        earliest_start_date = ret_all.index.min()
        start_opt_date = earliest_start_date + timedelta(days=backtest_offset_days)
        
        try:
            # Determine which yield metric to use: YTM first, fallback to T12M
            has_ytm = 'YTM / est. yield' in df_fi_metrics.index
            has_t12m = 'T12M dividend yield' in df_fi_metrics.index
            
            if has_ytm or has_t12m:
                # Build a combined yield row: YTM first, fallback to T12M
                combined_yield = pd.Series(0.0, index=df_fi_metrics.columns, name='Yield (combined)')
                yield_source_map = {}
                
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
                    
                    if ytm_val != 0:
                        combined_yield[col] = ytm_val
                        yield_source_map[col] = 'YTM'
                    elif t12m_val != 0:
                        combined_yield[col] = t12m_val
                        yield_source_map[col] = 'T12M'
                
                fi_tickers = combined_yield[combined_yield != 0].index.tolist()
                
                # Exclude tickers that don't generate cash flow (aligned with cash flow exclusions)
                FI_EXCLUDED_PATTERNS = [
                    'CASH', 'MMF', 'FLEXI', 'GUARANTEED',  # Cash/MMF
                    'CORE', 'EQUITY', 'GROWTH',  # Growth portfolios
                    'CHINA', 'ESG', 'HEALTHCARE', 'DISRUPTIVE', 'TECHNOLOGY', 'THEMATIC',  # Thematic
                    'CUSTOM', 'STRATEGIC',  # Custom portfolios
                ]
                fi_tickers = [t for t in fi_tickers if not any(p in t.upper() for p in FI_EXCLUDED_PATTERNS)]
            
                if fi_tickers:
                    # Build a metrics df with Duration + combined yield for exposure_analysis_mixed
                    fi_metrics_for_calc = pd.DataFrame(index=['Duration', 'YTM / est. yield'], columns=df_fi_metrics.columns, dtype=float)
                    if 'Duration' in df_fi_metrics.index:
                        fi_metrics_for_calc.loc['Duration'] = df_fi_metrics.loc['Duration'].astype(float)
                    else:
                        fi_metrics_for_calc.loc['Duration'] = 0.0
                    # Use combined yield (YTM first, T12M fallback) in the YTM row
                    fi_metrics_for_calc.loc['YTM / est. yield'] = combined_yield
                    
                    port_fi_metrics = pd.DataFrame()
                    has_fi = False
                
                    for p_name in results.keys():
                        w_df = results[p_name]['weights']
                        p_tickers = [c.replace('_wgt', '') for c in w_df.columns if c.endswith('_wgt')]
                        # Also filter portfolio tickers
                        p_tickers = [t for t in p_tickers if not any(p in t.upper() for p in FI_EXCLUDED_PATTERNS)]
                        common_fi = list(set(p_tickers).intersection(fi_tickers))
                    
                        if common_fi:
                            has_fi = True
                            ytm_duration_df = pc.exposure_analysis_mixed(
                                w_df, 
                                fi_metrics_for_calc, 
                                fi_tickers, 
                                latest_data=True, 
                                start_date=start_opt_date, 
                                end_date=end_date
                            )
                            ytm_duration_df.columns = [p_name]
                            
                            # Rename the yield row to show it uses combined logic
                            ytm_duration_df = ytm_duration_df.rename(index={'YTM / est. yield': 'Yield (Estimated)'})
                            
                            # Add Credit Rating
                            try:
                                if 'Credit rating' in df_fi_metrics.index:
                                    cr = pc.calc_avg_credit_rating(
                                        w_df, 
                                        df_fi_metrics.loc[['Credit rating']], 
                                        latest_data=True,
                                        start_date=start_opt_date, 
                                        end_date=end_date
                                    )
                                    ytm_duration_df.loc['Credit rating', p_name] = cr
                            except Exception:
                                ytm_duration_df.loc['Credit rating', p_name] = 'N/A'
                            
                            port_fi_metrics = pd.concat([port_fi_metrics, ytm_duration_df], axis=1)
                
                    if has_fi:
                        # Format numeric rows, leave Credit rating as-is
                        def format_fi_value(val, row_name):
                            if row_name == 'Credit rating':
                                return val if pd.notna(val) else 'N/A'
                            try:
                                return f"{float(val):.2f}"
                            except:
                                return val
                        
                        formatted_df = port_fi_metrics.T.copy()
                        st.dataframe(formatted_df, use_container_width=True)
                    else:
                        st.info("No Fixed Income data available.")
                else:
                    st.info("No Fixed Income data available.")
            else:
                st.info("Fixed Income metrics data missing.")
        except Exception as e:
            st.error(f"Error calculating FI metrics: {e}")
    
    
    # ============================================
    # SECTION 4.3: BACKTEST RESULTS
    # ============================================
    if results_data:
        st.subheader("4.2 Backtest Results")

        # FX Methodology explanation
        fx_methodology = """
**FX Conversion Methodology:**

• **USD to SGD**: R_SGD = (1 + R_USD) × (1 + R_FX) - 1
• **SGD to USD**: R_USD = (1 + R_SGD) / (1 + R_FX) - 1
"""
        
        col_anal_ccy, col_anal_spacer = st.columns([1, 3])
        
        with col_anal_ccy:
            analysis_currency = st.selectbox(
                "Display Currency",
                options=["USD", "SGD", "HKD"],
                index=0 if currency == "USD" else 1,
                key=f"{key_prefix}_analysis_currency",
                help=fx_methodology
            )
            
            # Get latest FX rate from data if available
            latest_fx_rate = 1.27  # Default fallback for USD/SGD
            if df_fx is not None and not df_fx.empty:
                if analysis_currency == "SGD" and 'USDSGD' in df_fx.columns:
                    latest_series = df_fx['USDSGD'].dropna()
                    if not latest_series.empty:
                        latest_fx_rate = float(latest_series.iloc[-1])
                        if latest_fx_rate < 1: latest_fx_rate = 1.27
                elif analysis_currency == "HKD" and 'USDHKD' in df_fx.columns:
                     latest_series = df_fx['USDHKD'].dropna()
                     if not latest_series.empty:
                        latest_fx_rate = float(latest_series.iloc[-1])
                        if latest_fx_rate < 1: latest_fx_rate = 7.80
                     else:
                        latest_fx_rate = 7.80
                elif analysis_currency == "HKD":
                     latest_fx_rate = 7.80
            
            
            use_latest_fx_anal = st.checkbox(
                "Use Latest FX Rate",
                value=True,
                key=f"{key_prefix}_use_latest_fx_anal",
                help=f"Latest rate from data: {latest_fx_rate:.4f}"
            )
            
            if use_latest_fx_anal:
                fx_rate_anal = latest_fx_rate
                st.caption(f"USD/{analysis_currency}: {fx_rate_anal:.4f}")
            else:
                fx_rate_anal = st.number_input(
                    f"Custom USD/{analysis_currency} Rate",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.27,
                    step=0.01,
                    format="%.4f",
                    key=f"{key_prefix}_fx_rate_anal_custom"
                )
        

        st.markdown("---")
    
        # Combine Returns
        all_port_ret = pd.DataFrame()
        for p_name, p_res in results.items():
            p_ret = p_res['perf'][[c for c in p_res['perf'].columns if c.endswith('_port_ret')]].copy()
            p_ret.columns = [p_name]
            all_port_ret = pd.concat([all_port_ret, p_ret], axis=1)
    
        # Apply FX conversion if display currency differs from base currency
        # Assume base returns are in USD (most common for global portfolios)
        base_currency = currency  # The currency used during optimization
        
        if analysis_currency != base_currency:
            # Get FX returns for conversion
            fx_ticker = None
            if analysis_currency == 'SGD': fx_ticker = 'USDSGD'
            elif analysis_currency == 'HKD': fx_ticker = 'USDHKD'
            
            if df_fx is not None and not df_fx.empty and fx_ticker and fx_ticker in df_fx.columns:
                fx_returns = df_fx[fx_ticker].copy()
                
                # Check if FX data looks like returns or levels
                if fx_returns.abs().max() > 1:
                    # Looks like levels, convert to returns
                    fx_returns = fx_returns.pct_change().fillna(0)
                
                # Align dates
                common_fx_idx = all_port_ret.index.intersection(fx_returns.index)
                
                if len(common_fx_idx) > 0:
                    fx_ret_aligned = fx_returns.loc[common_fx_idx]
                    
                    if base_currency == 'USD':
                        # Convert USD returns to Target: (1+r_usd) * (1+r_fx) - 1
                        for col in all_port_ret.columns:
                            all_port_ret.loc[common_fx_idx, col] = (1 + all_port_ret.loc[common_fx_idx, col]) * (1 + fx_ret_aligned) - 1
                        for col in ret_bm.columns:
                            if col in ret_bm.columns:
                                ret_bm.loc[common_fx_idx, col] = (1 + ret_bm.loc[common_fx_idx, col]) * (1 + fx_ret_aligned) - 1
                    
                    elif base_currency == analysis_currency:
                        pass # No conversion needed
                    
                    # Note: We are simplifying by only handling Base=USD or Base=SGD cases mostly.
                    # If conversion is needed between non-USD pairs, we need cross rates.
                    # Assuming Base is usually USD or SGD.
                    
                    elif base_currency == 'SGD' and analysis_currency == 'USD':
                         # Convert SGD returns to USD: (1+r_sgd) / (1+r_fx) - 1
                         # Here fx_returns is USDSGD (USD relative to SGD? No, usually Quote Currency per Base Currency)
                         # USDSGD means 1 USD = x SGD.
                         # Return of USDSGD is return of USD against SGD.
                         # If we have SGD assets and want USD: r_USD ~ r_SGD - r_FX (approx).
                         # Exact: 1+r_USD = (1+r_SGD) / (1+r_USDSGD)
                        for col in all_port_ret.columns:
                            all_port_ret.loc[common_fx_idx, col] = (1 + all_port_ret.loc[common_fx_idx, col]) / (1 + fx_ret_aligned) - 1
                        for col in ret_bm.columns:
                            if col in ret_bm.columns:
                                ret_bm.loc[common_fx_idx, col] = (1 + ret_bm.loc[common_fx_idx, col]) / (1 + fx_ret_aligned) - 1
    
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
        
            # Display date information above the table
            try:
                exposure_start_date = pd.to_datetime(all_port_ret.index.min()).strftime('%d %b %Y')
                exposure_end_date = pd.to_datetime(all_port_ret.index.max()).strftime('%d %b %Y')
                st.markdown(f"**Data as of:** {exposure_end_date}<br>**Time frame taken:** {exposure_start_date} - {exposure_end_date}", unsafe_allow_html=True)
            except:
                st.markdown("**Data timeframe:** N/A", unsafe_allow_html=True)
        
            # Format the summary DataFrame properly
            # Metrics are in rows (index), portfolios are in columns
            # We need to format by row, so we'll create a formatted copy
            def format_summary_value(val, metric_name):
                if pd.isna(val):
                    return "N/A"
                # Ratios should be displayed as decimals
                if metric_name in ['Sharpe Ratio', 'Sortino Ratio', 'Information Ratio']:
                    return f"{val:.2f}"
                # Everything else (returns, volatility, drawdown) should be percentages
                else:
                    return f"{val:.2%}"
            
            # Create formatted DataFrame
            summary_formatted = summary.copy()
            for idx in summary.index:
                for col in summary.columns:
                    summary_formatted.loc[idx, col] = format_summary_value(summary.loc[idx, col], idx)
            
            # Rename Max Drawdown to prevent wrapping
            summary_formatted.rename(index={'Max Drawdown': 'Max DD'}, inplace=True)
            
            st.dataframe(
                summary_formatted, 
                use_container_width=True
            )
        
        # Note: Rebalance Weights moved to Section 4.4
        # Note: Fixed Income Analytics moved to Section 4.1
        
        # Old Fixed Income Analytics placeholder - now in Section 4.1
        if False:  # Disabled
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
            p_select = st.selectbox("Select Portfolio", all_port_ret.columns, key=f"{key_prefix}_monthly_sel")
            fig = plot_monthly_heatmap(all_port_ret[[p_select]])
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_monthly_heatmap")
        
        with tabs[3]: # Cumulative Returns
            # Calculate cumulative returns as percentage (subtract 1 so 1.94 becomes 0.94 = 94%)
            cum_ret = (1 + all_port_ret).cumprod() - 1
            # Pass benchmark returns for comparison
            fig = plot_cumulative_returns(cum_ret, benchmark_ret=ret_bm)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_cum_ret")
        
        with tabs[4]: # Return/Risk Scatter
            fig = plot_scatter_risk_return(all_port_ret, benchmark_ret=ret_bm)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_risk_ret")
        
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
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_dist")
        
        with tabs[6]: # Rolling Volatility
            fig = plot_rolling_volatility(all_port_ret)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_roll_vol")
        
        with tabs[7]: # Drawdown
            fig = plot_drawdowns(all_port_ret)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_dd")
        
        with tabs[8]: # Correlation Heatmap
            fig = plot_correlation_heatmap(all_port_ret)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_corr")
            
        with tabs[9]: # Portfolio vs Benchmark
            p_select_bm = st.selectbox("Select Portfolio", all_port_ret.columns, key=f"{key_prefix}_bm_sel")
        
            # Calculate Relative Metrics
            rel_metrics, roll_excess = calculate_relative_metrics(all_port_ret[p_select_bm], ret_bm.iloc[:, 0])
        
            st.markdown("#### Relative Performance Metrics")
            # Format: Tracking Error as %, Information Ratio as decimal
            rel_df = rel_metrics.to_frame().T
            rel_format = {}
            for col in rel_df.columns:
                if 'Ratio' in col:
                    rel_format[col] = "{:.2f}"
                else:
                    rel_format[col] = "{:.2%}"
            st.dataframe(rel_df.style.format(rel_format), use_container_width=True)
        
            st.markdown("#### Rolling 12M Excess Return")
            fig_roll = plot_cumulative_returns(roll_excess.to_frame(name="Rolling Excess Return")) # Reuse plot function
            st.plotly_chart(fig_roll, use_container_width=True, key=f"{key_prefix}_roll_excess")
        
    # ============================================
    # SECTION 4.4: REBALANCE WEIGHTS
    # ============================================
    if results_data:
        st.subheader("4.3 Rebalance Weights")
        show_rebal_weights = st.toggle("Show Rebalance Weights", value=False, key=f"{key_prefix}_show_rebal_toggle_44")
        
        if not show_rebal_weights:
            st.info("Expand to view historical portfolio weights at rebalance dates.")
        else:
            res_data = results_data
            results = res_data['results']
            
            w_select = st.selectbox("Select Portfolio to View Weights", list(results.keys()), key=f"{key_prefix}_w_view_sel_44")
        
            if w_select:
                w_df = results[w_select]['weights']
                # Filter for rebalance dates
                w_rebal = w_df[w_df['reb_flag'] == True].drop(columns=['reb_flag'])
            
                # Clean column names
                w_rebal.columns = [c.replace('_wgt', '') for c in w_rebal.columns]
            
                # Format index
                w_rebal.index = w_rebal.index.strftime('%Y-%m-%d')
            
                st.dataframe(w_rebal.style.background_gradient(cmap='Blues', axis=None).format("{:.2%}"), use_container_width=True)
        
    # Old Section 5 - Now handled in Section 4 above
    if False:  # Disabled - moved to Section 4
        st.header("OLD - Exposure Analysis")
        
        res_data = st.session_state.optimization_results
        results = res_data['results']
        
        st.markdown("#### Filter Exposures (Aggregated by Cash Invested)")
        
        # Filter Controls
        filter_mode = st.radio(
            "Filter Mode", 
            ["Show All", "By Objective", "By Portfolio"], 
            horizontal=True,
            key=f"{key_prefix}_exp_filt_mode"
        )
        
        # Gather Metadata
        port_meta = {}
        for p in st.session_state.portfolios:
            port_meta[p['name']] = {
                'objective': p.get('objective', 'Unclassified'),
                'goal_name': p.get('goal_name', 'Unclassified'),
                'cash_invested': p.get('cash_invested', 100000.0)
            }
            if port_meta[p['name']]['cash_invested'] == 0:
                port_meta[p['name']]['cash_invested'] = 100000.0

        # Filter Selection Logic
        filtered_portfolios = []
        scaling_factors = {}

        # Level 1 Filter
        if filter_mode == "Show All":
            filtered_portfolios = list(results.keys())
            
        elif filter_mode == "By Objective":
            all_objs = sorted(list(set([m['objective'] for m in port_meta.values()])))
            sel_objs = st.multiselect("Select Objective(s)", all_objs, default=all_objs, key=f"{key_prefix}_exp_filt_obj")
            filtered_portfolios = [p for p in results.keys() if port_meta[p]['objective'] in sel_objs]
            
        elif filter_mode == "By Portfolio":
            # Gather all unique tickers from all portfolios
            all_tickers = set()
            for p_name in results.keys():
                w_df = results[p_name]['weights']
                w_row = w_df.iloc[-1]
                for ticker_wgt in w_row.index:
                    if ticker_wgt.endswith('_wgt'):
                        ticker = ticker_wgt.replace('_wgt', '')
                        all_tickers.add(ticker)
            
            all_tickers = sorted(list(all_tickers))
            sel_tickers = st.multiselect("Select Underlying Portfolio(s)", all_tickers, default=all_tickers, key=f"{key_prefix}_exp_filt_ticker")
            
            # Filter to include all portfolios, but only selected tickers within each
            filtered_portfolios = list(results.keys())
            # Store which tickers to include for filtering weights later
            selected_ticker_wgts = set([f"{t}_wgt" for t in sel_tickers])
            
        # Level 2 Filter: By Goal (Applied on top of Level 1)
        # Only show goal filter if we have portfolios selected
        if filtered_portfolios:
            # Get goals present in the currently filtered portfolios
            available_goals = sorted(list(set([port_meta[p]['goal_name'] for p in filtered_portfolios])))
            # If there are goals to filter by
            if available_goals:
                sel_goals = st.multiselect("Filter by Goal (Optional)", available_goals, default=available_goals, key=f"{key_prefix}_exp_filt_goal")
                # Refine filtered_portfolios
                filtered_portfolios = [p for p in filtered_portfolios if port_meta[p]['goal_name'] in sel_goals]

        global_ticker_values = {}
        total_agg_value = 0.0
        
        for p_name in filtered_portfolios:
            w_df = results[p_name]['weights']
            w_row = w_df.iloc[-1]
            cash = port_meta[p_name]['cash_invested']
            scale = scaling_factors.get(p_name, 1.0)
            effective_cash = cash * scale
            
            if effective_cash > 0:
                for ticker_wgt, weight in w_row.items():
                    # If "By Portfolio" mode, only include selected tickers
                    if filter_mode == "By Portfolio" and ticker_wgt not in selected_ticker_wgts:
                        continue
                    
                    val = weight * effective_cash
                    if val > 0:
                        if ticker_wgt in global_ticker_values:
                            global_ticker_values[ticker_wgt] += val
                        else:
                            global_ticker_values[ticker_wgt] = val
                        total_agg_value += val
        
        # Convert back to weights
        if total_agg_value > 0:
            agg_weights = pd.Series({k: v / total_agg_value for k, v in global_ticker_values.items()})
            
            # Pre-calculate main charts
            from utils.exposure import get_exposure_charts, get_concentration_exposure
            charts = get_exposure_charts(agg_weights, df_sector, df_country, df_asset_class)
            
            # --- Row 1: Asset Class + Top Holdings ---
            col_asset, col_holdings = st.columns([1, 1])
            
            with col_asset:
                st.markdown("**Asset Class**")
                if 'asset_pie' in charts:
                    st.plotly_chart(charts['asset_pie'], use_container_width=True, key=f"{key_prefix}_anal_pie_asset")
                    with st.expander("View Data"):
                        st.dataframe(charts['asset_table'].style.format("{:.2%}"), use_container_width=True)
                else:
                    st.info("Asset Class data unavailable.")
            
            with col_holdings:
                # Selector for Top N
                t_topn = st.selectbox("Show Top", [10, 15, 20], key=f"{key_prefix}_anal_topn_sel")
                
                if df_underlying is not None:
                    from utils.exposure import get_concentration_exposure
                    conc_agg = get_concentration_exposure(agg_weights, df_underlying, top_n=t_topn)
                    if not conc_agg.empty:
                        conc_df = conc_agg.head(t_topn).to_frame(name='Weight')
                        conc_df['Est. Value ($)'] = conc_df['Weight'] * total_agg_value
                        total_conc = conc_df['Weight'].sum()
                        
                        # Bar Chart for Top Holdings
                        conc_df_sorted = conc_df.sort_values('Weight', ascending=True) # Sort for bar chart (bottom to top)
                        
                        fig_top_bar = px.bar(
                            conc_df_sorted, 
                            x='Weight', 
                            y=conc_df_sorted.index, 
                            orientation='h',
                            title=f'Top {t_topn} Holdings ({total_conc:.1%} Total)',
                            color='Weight',
                            color_continuous_scale=PALETTE # Use Syfe Palette
                        )
                        fig_top_bar.update_layout(showlegend=False, xaxis_tickformat='.1%')
                        st.plotly_chart(fig_top_bar, use_container_width=True, key=f"{key_prefix}_anal_bar_top_holdings")
                        
                        with st.expander("View Data"):
                            st.dataframe(
                                conc_df.style.format({'Weight': '{:.2%}', 'Est. Value ($)': '${:,.2f}'}),
                                use_container_width=True
                            )
                    else:
                        st.warning("No underlying holdings found.")
                else:
                    st.info("Concentration data not available.")
            
            # --- Row 2: Sector + Country ---
            col_sector, col_country = st.columns([1, 1])
            
            with col_sector:
                st.markdown("**Sector**")
                if 'sector_pie' in charts:
                    st.plotly_chart(charts['sector_pie'], use_container_width=True, key=f"{key_prefix}_anal_pie_sector")
                    with st.expander("View Data"):
                        st.dataframe(charts['sector_table'].style.format("{:.2%}"), use_container_width=True)
                else:
                    st.info("Sector data unavailable.")
            
            with col_country:
                st.markdown("**Country**")
                if 'country_pie' in charts:
                    st.plotly_chart(charts['country_pie'], use_container_width=True, key=f"{key_prefix}_anal_pie_country")
                    with st.expander("View Data"):
                        st.dataframe(charts['country_table'].style.format("{:.2%}"), use_container_width=True)
                else:
                    st.info("Country data unavailable.")

        else:
            st.warning("No exposure data found for the selected filters.")

    # This function is a placeholder for logic that might be moved or updated elsewhere.
    # The original instruction refers to "Add Selected Strategy" logic, which is not present
    # in the provided snippet. If this function was intended to contain that logic,
    # it needs to be provided in the context.
    def render_recommender_add_logic(key_prefix="rec_add"):
        pass


    if st.session_state.optimization_results:
        st.subheader("4.5 Factor Exposures")
        show_factor_exposures = st.toggle("Show Factor Exposures", value=False, key=f"{key_prefix}_show_factor_toggle")
    
        if not show_factor_exposures:
            st.info("Expand to view Fama-French 5-Factor Model analysis.")
        else:
            st.markdown("""
            **Fama-French 5-Factor Model** exposure coefficients:
            - **Mkt**: Market risk premium (excess return of market over risk-free rate)
            - **SMB**: Size (Small Minus Big) - smaller firms tend to outperform larger firms
            - **HML**: Value (High Minus Low) - high book-to-market (value) vs low book-to-market (growth)
            - **RMW**: Profitability (Robust Minus Weak) - robust profitability vs weak profitability
            - **CMA**: Investment (Conservative Minus Aggressive) - conservative vs aggressive investment
            """)
    
            try:
                # Try to load from file first
                ff_file_path = os.path.join(os.getcwd(), 'Data', 'F-F_Research_Data_5_Factors_2x3_daily.csv')
            
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
                                    key=f"{key_prefix}_factor_bm_select"
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
                            
                                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_factor_chart")
                            
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

    if st.session_state.optimization_results:
        # Check if any portfolio has cash invested
        portfolios_with_cash = {
            p_name: p for p_name, p in enumerate(st.session_state.portfolios) 
            if p.get('cash_invested', 0.0) > 0
        }
    
        if portfolios_with_cash:
            st.subheader("4.4 Cash Flow Analysis")
            st.markdown("**Projected monthly cash flows based on portfolio yields and cash invested.**")
        
            # Currency and Portfolio selection row
            col_ccy, col_port = st.columns([1, 3])
            
            with col_ccy:
                display_currency = st.selectbox(
                    "Display Currency",
                    options=["SGD", "USD", "HKD"],
                    index=0,
                    key=f"{key_prefix}_cashflow_currency"
                )
                
                # 1. Fetch reference USD/SGD rate (always needed for cross-rates)
                ref_usdsgd = 1.27
                if df_fx is not None and not df_fx.empty and 'USDSGD' in df_fx.columns:
                    s = df_fx['USDSGD'].dropna()
                    if not s.empty: 
                        val = float(s.iloc[-1])
                        if val > 1: ref_usdsgd = val

                # 2. Determine default rate for Selected Currency
                default_fx_rate = 1.0
                if display_currency == "SGD":
                    default_fx_rate = ref_usdsgd
                elif display_currency == "HKD":
                    default_fx_rate = 7.80
                    if df_fx is not None and not df_fx.empty and 'USDHKD' in df_fx.columns:
                        s = df_fx['USDHKD'].dropna()
                        if not s.empty:
                            val = float(s.iloc[-1])
                            if val > 1: default_fx_rate = val

                # Toggle for Latest vs Custom FX
                use_latest_fx = st.checkbox(
                    "Use Latest FX Rate",
                    value=True,
                    key=f"{key_prefix}_use_latest_fx",
                    help=f"Latest rate from data: {default_fx_rate:.4f}"
                )
                
                if use_latest_fx:
                    fx_rate = default_fx_rate
                    st.caption(f"USD/{display_currency}: {fx_rate:.4f}")
                else:
                    fx_rate = st.number_input(
                        f"Custom USD/{display_currency} Rate",
                        min_value=0.1,
                        max_value=20.0,
                        value=default_fx_rate,
                        step=0.01,
                        format="%.4f",
                        key=f"{key_prefix}_fx_rate_custom"
                    )
            
            with col_port:
                # Portfolio selection - group by goal if available
                all_portfolio_names = [p['name'] for idx, p in portfolios_with_cash.items()]
                selected_portfolios = st.multiselect(
                    "Select Portfolio(s) to Display",
                    options=all_portfolio_names,
                    default=all_portfolio_names,
                    key=f"{key_prefix}_cashflow_portfolio_select"
                )
            
            # Goal-based view toggle
            show_goal_breakdown = st.checkbox(
                "Show Goal-Based Breakdown",
                value=True,
                key=f"{key_prefix}_show_goal_breakdown",
                help="Group cash flows by investment goal"
            )
            
            # Yield source preference
            yield_preference = st.selectbox(
                "Yield Source",
                options=["YTM / Est. Yield", "T12M Dividend Yield"],
                index=0,
                key=f"{key_prefix}_yield_pref",
                help="Select which yield metric to use for cash flow projections. Primary source is used first, falls back to the other if unavailable."
            )
        
            if not selected_portfolios:
                st.warning("Please select at least one portfolio to display cash flows.")
                st.stop()
        
            # Filter portfolios based on selection
            filtered_portfolios = {
                idx: p for idx, p in portfolios_with_cash.items() 
                if p['name'] in selected_portfolios
            }
            
            # Define base currency for each portfolio/ticker type
            # SGD-denominated: REITs, Income Enhance/Preserve, Cash SGD products
            # USD-denominated: Private Credit (Blackrock, Monroe), Thematics
            def get_ticker_base_currency(ticker):
                """Determine the base currency of a ticker's cash flows."""
                ticker_upper = ticker.upper()
                
                # USD-denominated assets (check first as they're more specific)
                if 'BLACKROCK' in ticker_upper or 'MONROE' in ticker_upper:
                    return 'USD'  # Private Credit funds pay in USD
                if 'USD' in ticker_upper:  # Explicitly USD products
                    return 'USD'
                
                # SGD-denominated assets (default for Syfe Singapore products)
                if 'REIT' in ticker_upper:  # Singapore REITs pay in SGD
                    return 'SGD'
                if 'INCOME' in ticker_upper:  # Income Enhance/Preserve pay in SGD
                    return 'SGD'
                if 'SGD' in ticker_upper:  # Explicitly SGD products
                    return 'SGD'
                if 'SRS' in ticker_upper:  # SRS products are SGD
                    return 'SGD'
                    
                # Default to SGD for other Syfe products
                return 'SGD'
            
            def convert_currency(amount, from_ccy, to_ccy, target_fx, cross_usdsgd=1.27):
                """
                Convert amount. 
                target_fx: USD -> to_ccy rate (e.g. 1 USD = 7.8 HKD)
                cross_usdsgd: USD -> SGD rate (used if from_ccy is SGD)
                """
                if from_ccy == to_ccy:
                    return amount
                
                # Convert source to USD first
                usd_amount = amount
                if from_ccy == 'SGD':
                    usd_amount = amount / cross_usdsgd
                elif from_ccy == 'HKD':
                    # Not typical source, but for completeness:
                    # If we had USDHKD rate we would divide. 
                    # Assuming we don't have it separately here, but we could pass it.
                    pass 
                elif from_ccy == 'USD':
                    usd_amount = amount
                    
                # Convert USD to target
                if to_ccy == 'USD':
                    return usd_amount
                
                # For non-USD target, we multiply by USD->Target rate
                return usd_amount * target_fx
        
            # Helper function to determine payment frequency
            def get_payment_frequency(ticker):
                """Returns 'monthly' or 'quarterly' based on ticker type."""
                ticker_upper = ticker.upper()
                
                # Explicitly check for Blackrock and Monroe - monthly dividends
                if 'BLACKROCK' in ticker_upper or 'MONROE' in ticker_upper:
                    return 'monthly'
                
                # Check if ticker contains REIT indicators (including REIT Risk Managed) - quarterly dividends
                if 'REIT' in ticker_upper or 'VNQ' in ticker or 'VNQI' in ticker:
                    return 'quarterly'
                
                # Default to monthly for bonds and other income assets
                return 'monthly'
        
            # Currency symbol for display
            ccy_symbol = "$"
            if display_currency == "SGD": ccy_symbol = "S$"
            elif display_currency == "HKD": ccy_symbol = "HK$"
        
            # Calculate cash flows
            monthly_cash_flows = pd.DataFrame(0.0, 
                                             index=[p['name'] for idx, p in filtered_portfolios.items()],
                                             columns=list(range(1, 13)))
        
            ticker_cash_flows = {}  # Track by ticker for stacked chart
        
            # Track calculation breakdown for transparency
            calc_breakdown = []
        
            # Excluded patterns for cash flow analysis (aligned with portfolio_report_generator.py)
            CASHFLOW_EXCLUDED_PATTERNS = [
                'CASH', 'MMF', 'FLEXI', 'GUARANTEED',  # Cash/MMF - no dividend payout
                'CORE', 'EQUITY', 'GROWTH',  # Growth portfolios - dividends auto-reinvested
                'CHINA', 'ESG', 'HEALTHCARE', 'DISRUPTIVE', 'TECHNOLOGY', 'THEMATIC',  # Thematic - auto-reinvested
                'CUSTOM', 'STRATEGIC',  # Custom portfolios excluded
            ]
            
            for idx, port in filtered_portfolios.items():
                p_name = port['name']
                cash_invested = port['cash_invested']
                # Get the currency that cash_invested is denominated in
                # CSV loaded = SGD, Manual = USD, PDF = presentation_ccy
                cash_currency = port.get('cash_currency', 'USD')  # Default USD for backwards compatibility
                
                # Always use fixed weights (user-defined amounts/weights)
                fixed_weights = port.get('decimal_weights', port.get('fixed_weights', {}))
                
                if fixed_weights:
                    for ticker in port['tickers']:
                        # Skip excluded portfolios - no cash payout
                        ticker_upper = ticker.upper()
                        if any(pattern in ticker_upper for pattern in CASHFLOW_EXCLUDED_PATTERNS):
                            calc_breakdown.append({
                                'Portfolio': p_name,
                                'Ticker': ticker,
                                'Status': 'Excluded',
                                'Weight': '-',
                                f'Allocated ({cash_currency})': '-',
                                'Yield': '-',
                                'Frequency': '-',
                                f'Annual CF ({display_currency})': '-',
                            })
                            continue
                        
                        # Get weight from fixed weights
                        weight = float(fixed_weights.get(ticker, 0.0))
                        
                        if weight <= 0:
                            continue
                            
                        # invested_in_ticker is in cash_currency
                        invested_in_ticker = cash_invested * weight
                    
                        # Get yield based on user preference
                        # All yields in Excel are in percentage format (e.g., 0.87 = 0.87%, 10.72 = 10.72%)
                        annual_yield = 0.0
                        yield_source = 'N/A'
                        if ticker in df_fi_metrics.columns:
                            if yield_preference == "T12M Dividend Yield":
                                # Primary: T12M dividend yield, fallback: YTM
                                if 'T12M dividend yield' in df_fi_metrics.index:
                                    yield_val = df_fi_metrics.loc['T12M dividend yield', ticker]
                                    if pd.notna(yield_val) and yield_val != 0:
                                        annual_yield = float(yield_val) / 100
                                        yield_source = 'T12M Div Yield'
                                
                                if annual_yield == 0 and 'YTM / est. yield' in df_fi_metrics.index:
                                    yield_val = df_fi_metrics.loc['YTM / est. yield', ticker]
                                    if pd.notna(yield_val) and yield_val != 0:
                                        annual_yield = float(yield_val) / 100
                                        yield_source = 'YTM (fallback)'
                            else:
                                # Primary: YTM / est. yield, fallback: T12M
                                if 'YTM / est. yield' in df_fi_metrics.index:
                                    yield_val = df_fi_metrics.loc['YTM / est. yield', ticker]
                                    if pd.notna(yield_val) and yield_val != 0:
                                        annual_yield = float(yield_val) / 100
                                        yield_source = 'YTM / Est. Yield'
                                
                                if annual_yield == 0 and 'T12M dividend yield' in df_fi_metrics.index:
                                    yield_val = df_fi_metrics.loc['T12M dividend yield', ticker]
                                    if pd.notna(yield_val) and yield_val != 0:
                                        annual_yield = float(yield_val) / 100
                                        yield_source = 'T12M (fallback)'
                        
                            if annual_yield > 0 and invested_in_ticker > 0:
                                payment_freq = get_payment_frequency(ticker)
                                
                                # Get ticker's base currency (currency that dividends are paid in)
                                ticker_base_ccy = get_ticker_base_currency(ticker)
                            
                                # Convert invested amount from cash_currency to ticker's base currency
                                # This is needed because the yield is applied on the ticker's NAV currency
                                # Determine the correct target FX rate for conversion to ticker_base_ccy
                                if ticker_base_ccy == 'SGD':
                                    target_fx_for_ticker = ref_usdsgd
                                elif ticker_base_ccy == 'HKD':
                                    target_fx_for_ticker = 7.80  # Default USDHKD
                                else:  # USD
                                    target_fx_for_ticker = 1.0
                                    
                                invested_in_ticker_ccy = convert_currency(
                                    invested_in_ticker, 
                                    cash_currency, 
                                    ticker_base_ccy, 
                                    target_fx_for_ticker, 
                                    ref_usdsgd
                                )
                            
                                # Store ticker-level cash flows for stacked chart
                                ticker_key = f"{p_name} - {ticker}"
                                if ticker_key not in ticker_cash_flows:
                                    ticker_cash_flows[ticker_key] = np.zeros(12)
                                
                                annual_cf_display = 0.0
                            
                                if payment_freq == 'monthly':
                                    # Calculate payment in ticker's base currency
                                    monthly_payment = (invested_in_ticker_ccy * annual_yield) / 12
                                    # Convert to display currency
                                    monthly_payment = convert_currency(monthly_payment, ticker_base_ccy, display_currency, fx_rate, ref_usdsgd)
                                    annual_cf_display = monthly_payment * 12
                                    for month in range(12):
                                        monthly_cash_flows.loc[p_name, month + 1] += monthly_payment
                                        ticker_cash_flows[ticker_key][month] += monthly_payment
                                else:  # quarterly
                                    # Calculate payment in ticker's base currency
                                    quarterly_payment = (invested_in_ticker_ccy * annual_yield) / 4
                                    # Convert to display currency
                                    quarterly_payment = convert_currency(quarterly_payment, ticker_base_ccy, display_currency, fx_rate, ref_usdsgd)
                                    annual_cf_display = quarterly_payment * 4
                                    for quarter in range(4):
                                        month = (quarter * 3) + 2  # Months 3, 6, 9, 12 (index 2, 5, 8, 11)
                                        monthly_cash_flows.loc[p_name, month + 1] += quarterly_payment
                                        ticker_cash_flows[ticker_key][month] += quarterly_payment
                                
                                calc_breakdown.append({
                                    'Portfolio': p_name,
                                    'Ticker': ticker,
                                    'Status': 'Included',
                                    'Weight': f'{weight:.2%}',
                                    f'Allocated ({cash_currency})': f'{invested_in_ticker:,.2f}',
                                    'Ticker Ccy': ticker_base_ccy,
                                    'Yield': f'{annual_yield:.2%}',
                                    'Yield Source': yield_source,
                                    'Frequency': payment_freq,
                                    f'Annual CF ({display_currency})': f'{ccy_symbol}{annual_cf_display:,.2f}',
                                })
                            else:
                                calc_breakdown.append({
                                    'Portfolio': p_name,
                                    'Ticker': ticker,
                                    'Status': 'No yield' if annual_yield == 0 else 'Zero weight',
                                    'Weight': f'{weight:.2%}',
                                    f'Allocated ({cash_currency})': f'{invested_in_ticker:,.2f}',
                                    'Ticker Ccy': get_ticker_base_currency(ticker),
                                    'Yield': f'{annual_yield:.2%}' if annual_yield > 0 else '0.00%',
                                    'Yield Source': yield_source,
                                    'Frequency': '-',
                                    f'Annual CF ({display_currency})': f'{ccy_symbol}0.00',
                                })
        
            # Store cash flow data in session_state for PDF export
            st.session_state['cashflow_data'] = ticker_cash_flows
            st.session_state['cashflow_currency'] = display_currency
            
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
                title={'text': f'<b>Monthly Cash Flow Projections ({display_currency})</b>', 'font': {'size': 20, 'color': '#172B4D'}},
                xaxis_title="Month",
                yaxis_title=f"Cash Flow ({ccy_symbol})",
                xaxis={'tickmode': 'linear', 'tick0': 1, 'dtick': 1},
                yaxis={'tickformat': ',.0f'},
                font={'family': 'Helvetica Neue, Arial', 'color': '#172B4D'},
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=500,
                xaxis_showgrid=True,
                yaxis_showgrid=True,
                legend={'orientation': 'v', 'yanchor': 'top', 'y': 1, 'xanchor': 'left', 'x': 1.02}
            )
        
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_cashflow_chart")
        
            # Summary table
            st.subheader("Monthly Cash Flow Summary by Portfolio")
            
            if not monthly_cash_flows.empty:
                monthly_cash_flows.columns = [f"Month {m}" for m in range(1, 13)]
                monthly_cash_flows['Total Annual'] = monthly_cash_flows.sum(axis=1)
            
                # Reset index to avoid duplicate index errors if portfolio names are same
                display_df = monthly_cash_flows.reset_index().rename(columns={'index': 'Portfolio'})
                
                # Format with selected currency symbol
                format_str = f"{ccy_symbol}{{:,.2f}}"
                st.dataframe(
                    display_df.style.format(format_str, subset=display_df.columns.drop('Portfolio'))
                    .background_gradient(cmap='Greens', axis=None, subset=display_df.columns.drop(['Portfolio', 'Total Annual'])),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No portfolios selected or no cash flows generated.")
        
            # Total across all portfolios
            total_monthly = monthly_cash_flows.drop(columns=['Total Annual']).sum()
            total_annual = total_monthly.sum()
        
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"Total Annual Cash Flow ({display_currency})", f"{ccy_symbol}{total_annual:,.2f}")
            with col2:
                st.metric(f"Average Monthly Cash Flow ({display_currency})", f"{ccy_symbol}{total_annual/12:,.2f}")
            
            # Calculation breakdown for transparency
            if calc_breakdown:
                with st.expander("View Calculation Breakdown", expanded=True):
                    breakdown_df = pd.DataFrame(calc_breakdown)
                    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                    st.caption("Only tickers with 'Included' status generate cash flows. Excluded tickers (CORE, Growth, Cash, Thematic) have dividends auto-reinvested and do not produce cash payouts.")
            
            if show_goal_breakdown and ticker_cash_flows:
                with st.expander("📊 View Detailed Portfolio Contributions", expanded=False):
                    st.markdown("**Individual Portfolio Cash Flow Contributions**")
                    
                    # Build detailed breakdown data
                    contribution_data = []
                    for ticker_key, cash_flow_array in ticker_cash_flows.items():
                        annual_total = sum(cash_flow_array)
                        if annual_total > 0:
                            # Extract just the ticker name (remove portfolio prefix if present)
                            ticker_name = ticker_key.split(" - ")[-1] if " - " in ticker_key else ticker_key
                            contribution_data.append({
                                "Portfolio": ticker_key,
                                "Ticker": ticker_name,
                                **{f"Month {i+1}": f"{ccy_symbol}{cash_flow_array[i]:,.2f}" for i in range(12)},
                                f"Annual Total": f"{ccy_symbol}{annual_total:,.2f}"
                            })
                    
                    if contribution_data:
                        contribution_df = pd.DataFrame(contribution_data)
                        
                        # Add grand total row
                        total_monthly = [sum(ticker_cash_flows[k][i] for k in ticker_cash_flows) for i in range(12)]
                        grand_total = sum(total_monthly)
                        total_row = {
                            "Portfolio": "GRAND TOTAL",
                            "Ticker": "",
                            **{f"Month {i+1}": f"{ccy_symbol}{total_monthly[i]:,.2f}" for i in range(12)},
                            f"Annual Total": f"{ccy_symbol}{grand_total:,.2f}"
                        }
                        contribution_df = pd.concat([contribution_df, pd.DataFrame([total_row])], ignore_index=True)
                        
                        st.dataframe(
                            contribution_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Summary by ticker type - grouped and sorted
                        st.markdown("**Summary by Portfolio Type**")
                        
                        # Group by portfolio type (ticker name) and sum cash flows
                        grouped_cash_flows = {}
                        for ticker_key, cash_flow_array in ticker_cash_flows.items():
                            ticker_name = ticker_key.split(" - ")[-1] if " - " in ticker_key else ticker_key
                            annual_total = sum(cash_flow_array)
                            if annual_total > 0:
                                if ticker_name in grouped_cash_flows:
                                    grouped_cash_flows[ticker_name] += annual_total
                                else:
                                    grouped_cash_flows[ticker_name] = annual_total
                        
                        # Convert to list and sort by cash flow (descending)
                        summary_data = []
                        for ticker_name, annual_total in sorted(grouped_cash_flows.items(), key=lambda x: x[1], reverse=True):
                            summary_data.append({
                                "Portfolio Type": ticker_name,
                                f"Annual Cash Flow ({display_currency})": annual_total,
                                "% of Total": (annual_total / grand_total * 100) if grand_total > 0 else 0
                            })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(
                                summary_df.style.format({
                                    f"Annual Cash Flow ({display_currency})": f"{ccy_symbol}{{:,.2f}}",
                                    "% of Total": "{:.1f}%"
                                }),
                                use_container_width=True,
                                hide_index=True
                            )
        
        st.markdown("---")
        st.header("5. Growth Forecast Calculator")
        show_growth_forecast = st.toggle("Show Growth Forecast Calculator", value=False, key=f"{key_prefix}_show_growth_forecast_toggle")
        
        if not show_growth_forecast:
            st.info("Expand to project future portfolio value based on historical returns.")
        elif st.session_state.portfolios:
            # Portfolio selector
            portfolio_names = [p['name'] for p in st.session_state.portfolios]
            selected_portfolio_name = st.selectbox(
                "Select Portfolio", 
                portfolio_names,
                key=f"{key_prefix}_forecast_portfolio"
            )
            
            # Find selected portfolio
            selected_portfolio = next((p for p in st.session_state.portfolios if p['name'] == selected_portfolio_name), None)
            
            if selected_portfolio and 'tickers' in selected_portfolio and selected_portfolio['tickers']:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Auto-populate target goal from portfolio if available and non-zero
                    default_target = float(selected_portfolio.get('target_amount', 0.0))
                    if default_target == 0:
                        default_target = float(selected_portfolio.get('cash_invested', 1000000.0))
                        
                    target_goal = st.number_input(
                        "Target Goal Value ($)", 
                        min_value=0.0, 
                        value=default_target, 
                        step=10000.0,
                        key=f"{key_prefix}_target_goal_{selected_portfolio['id']}"
                    )
                
                with col2:
                    lookback_period = st.selectbox(
                        "Return Assumption (Historical)", 
                        ["1 Year", "3 Years", "5 Years"],
                        key=f"{key_prefix}_lookback_{selected_portfolio['id']}"
                    )
                
                with col3:
                    # Auto-populate from portfolio horizon if available
                    default_horizon = selected_portfolio.get('horizon', 10)
                    time_horizon = st.slider(
                        "Time Horizon (Years)", 
                        min_value=1, 
                        max_value=30, 
                        value=int(default_horizon),
                        key=f"{key_prefix}_horizon_{selected_portfolio['id']}"
                    )
                
                col4, col5 = st.columns(2)
                
                with col4:
                    solve_for_payment = st.checkbox(
                        "Solve for monthly payment required to meet goal?",
                        key=f"{key_prefix}_solve_payment"
                    )
                    
                    if not solve_for_payment:
                        monthly_contribution = st.number_input(
                            "Monthly Contribution ($)", 
                            min_value=0.0, 
                            value=0.0, 
                            step=100.0,
                            key=f"{key_prefix}_monthly_contrib"
                        )
                    else:
                        monthly_contribution = None  # Will be calculated
                
                with col5:
                    initial_value = selected_portfolio.get('cash_invested', 100000.0)
                    st.metric("Current Portfolio Value", f"${initial_value:,.2f}")
                
                if st.button("Calculate Projection", type="primary", key=f"{key_prefix}_calc_forecast"):
                    with st.spinner("Calculating future value projection..."):
                        # Calculate expected return
                        lookback_years = int(lookback_period.split()[0])
                        
                        # Get weights
                        if 'fixed_weights' in selected_portfolio:
                            weights = selected_portfolio['fixed_weights']
                        elif 'decimal_weights' in selected_portfolio:
                            weights = selected_portfolio['decimal_weights']
                        else:
                            st.error("Portfolio weights not found")
                            return
                        
                        # Calculate weighted average historical return
                        try:
                            # Get historical data if optimization was run
                            if 'optimization_results' in st.session_state and st.session_state.optimization_results:
                                ret_all = st.session_state.optimization_results.get('ret_all')
                                
                                if ret_all is not None:
                                    # Filter to lookback period
                                    end_date = ret_all.index.max()
                                    start_date = end_date - pd.DateOffset(years=lookback_years)
                                    
                                    ret_filtered = ret_all.loc[start_date:end_date]
                                    
                                    # Calculate portfolio daily returns
                                    # R_p = sum(w_i * r_i) for each day
                                    portfolio_daily_ret = pd.Series(0.0, index=ret_filtered.index)
                                    valid_tickers = []
                                    
                                    for ticker, weight in weights.items():
                                        if ticker in ret_filtered.columns:
                                            portfolio_daily_ret += ret_filtered[ticker] * weight
                                            valid_tickers.append(ticker)
                                    
                                    # Calculate annualized return (CAGR) of the portfolio
                                    if not portfolio_daily_ret.empty and len(valid_tickers) > 0:
                                        cumulative_return = (1 + portfolio_daily_ret).prod() - 1
                                        num_years = len(portfolio_daily_ret) / 252
                                        expected_return = (1 + cumulative_return) ** (1 / num_years) - 1 if num_years > 0 else 0
                                    else:
                                        expected_return = 0.07  # Default when no valid historical data
                                else:
                                    # Fallback if no historical data
                                    expected_return = 0.07  # 7% default
                            else:
                                # Fallback if optimization not run
                                expected_return = 0.07  # 7% default
                            
                            # If solving for monthly payment, calculate required contribution
                            if solve_for_payment:
                                # Solve for PMT given FV, PV, r, n
                                # FV = PV(1+r)^n + PMT * [((1+r)^n - 1) / r] * (1+r)
                                # PMT = (FV - PV(1+r)^n) / ([((1+r)^n - 1) / r] * (1+r))
                                
                                future_value_of_initial = initial_value * ((1 + expected_return) ** time_horizon)
                                gap = target_goal - future_value_of_initial
                                
                                if expected_return > 0:
                                    # Annuity factor for monthly contributions
                                    annuity_factor = (((1 + expected_return) ** time_horizon - 1) / expected_return) * (1 + expected_return)
                                    required_monthly = (gap / annuity_factor) / 12 if annuity_factor > 0 else 0
                                else:
                                    required_monthly = gap / (time_horizon * 12) if time_horizon > 0 else 0
                                
                                monthly_contribution = max(0, required_monthly)  # Don't allow negative
                                st.info(f"Required monthly contribution to meet goal: **${monthly_contribution:,.2f}**")
                            
                            # Project future values
                            years_array = np.arange(0, time_horizon + 1)
                            projected_values = []
                            
                            current_value = initial_value
                            for year in years_array:
                                if year == 0:
                                    projected_values.append(current_value)
                                else:
                                    # Add monthly contributions for the year
                                    annual_contributions = monthly_contribution * 12 if monthly_contribution else 0
                                    # Future value formula: FV = PV(1+r)^t + PMT * [((1+r)^t - 1) / r]
                                    current_value = current_value * (1 + expected_return) + annual_contributions * (1 + expected_return)
                                    projected_values.append(current_value)
                            
                            # Create visualization
                            fig = go.Figure()
                            
                            # Projected value line
                            fig.add_trace(go.Scatter(
                                x=years_array,
                                y=projected_values,
                                mode='lines+markers',
                                name='Projected Portfolio Value',
                                line=dict(color='green', width=3)
                            ))
                            
                            # Target goal line
                            fig.add_trace(go.Scatter(
                                x=years_array,
                                y=[target_goal] * len(years_array),
                                mode='lines',
                                name='Target Goal',
                                line=dict(color='red', dash='dash', width=2)
                            ))
                            
                            fig.update_layout(
                                title=f"Portfolio Growth Projection - {selected_portfolio_name}",
                                xaxis_title="Years",
                                yaxis_title="Portfolio Value ($)",
                                hovermode='x unified',
                                yaxis=dict(tickformat='$,.0f')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_forecast_chart")
                            
                            # Summary metrics
                            final_value = projected_values[-1]
                            total_contributions = monthly_contribution * 12 * time_horizon
                            investment_growth = final_value - initial_value - total_contributions
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Expected Annual Return", f"{expected_return*100:.2f}%")
                            
                            with col2:
                                st.metric("Projected Final Value", f"${final_value:,.2f}")
                            
                            with col3:
                                gap = final_value - target_goal
                                gap_pct = (gap / target_goal * 100) if target_goal > 0 else 0
                                st.metric("Gap to Goal", f"${gap:,.2f}", f"{gap_pct:+.1f}%")
                            
                            with col4:
                                st.metric("Investment Growth", f"${investment_growth:,.2f}")
                            
                            # Recommendations
                            if gap < -1:
                                st.warning(f"⚠️ Your projected value falls short of the target by ${abs(gap):,.2f}. Consider increasing contributions or extending the time horizon.")
                            elif gap > 1:
                                st.success(f"✅ You're on track to exceed your goal by ${gap:,.2f}!")
                            else:
                                st.success(f"✅ You are on track to meet your goal exactly!")
                                
                        except Exception as e:
                            st.error(f"Error calculating projection: {str(e)}")
                            # Show fallback with default return
                            st.info("Using default 7% annual return assumption")
            else:
                st.info("Please construct a portfolio first to use the Growth Forecast Calculator")
        else:
            st.info("No portfolios available. Please add portfolios in the Portfolio Construction section above.")
