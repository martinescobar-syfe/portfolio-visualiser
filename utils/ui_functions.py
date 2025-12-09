import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
from utils.llm_helper import get_gemini_recommendation, parse_recommendation_table
import requests
import zipfile
import io

def get_portfolio_objective(portfolio_name):
    """Determine objective based on portfolio name keywords."""
    name_lower = portfolio_name.lower()
    
    # Wealth Growth (Thematics + Equity)
    if any(keyword in name_lower for keyword in ['china', 'disruptive', 'esg', 'clean', 'healthcare', 'equity', 'growth']):
        return 'Wealth Growth'
    
    # Income Generation (includes AI investments like BlackRock, Monroe)
    if any(keyword in name_lower for keyword in ['income', 'enhance', 'reit', 'blackrock', 'monroe']):
        return 'Income Generation'
    
    # Capital Preservation
    if any(keyword in name_lower for keyword in ['preserve', 'cash', 'guaranteed']):
        return 'Capital Preservation'
    
    # Default to Wealth Growth for unclassified
    return 'Wealth Growth'


def render_client_upload(df_asset_class, df_sector, df_country, df_underlying, key_prefix="upload"):
    st.header("1. Client Data Upload")
    uploaded_file = st.file_uploader("Upload Client Portfolio CSV", type=['csv'], key=f"{key_prefix}_file_uploader")

    if uploaded_file:
        try:
            df_client = pd.read_csv(uploaded_file)
        
            # Filter for ACTIVE portfolios
            if 'status' in df_client.columns:
                df_client = df_client[df_client['status'] == 'ACTIVE']
        
            # Mapping logic
            CLIENT_PORTFOLIO_MAPPING = {
                'GLOBAL_EQUITY_100': 'Core Equity100',
                'CHINA_GROWTH': 'China Growth',
                'INCOME_ENHANCE': 'Income Enhance',
                'INCOME_PRESERVE': 'Income Preserve',
                'REIT': 'REIT 100',
                'BLACKROCK': 'BlackRock Private Credit Fund',
                'CASH_PLUS_GUARANTEED_SGD': 'Cash SGD guaranteed',
                'CASH_PLUS_USD': 'Cash USD flexi',
                'CASH_PLUS': 'Cash SGD flexi',  # Default SGD
                'SRS_INCOME_ENHANCE': 'SRS Income Enhance',
                'SRS_INCOME_PRESERVE': 'SRS Income Preserve',
                'SRS_CASH_PLUS': 'SRS Cash SGD flexi',
                'SRS_GLOBAL_EQUITY_100': 'SRS Equity100',
                'SRS_INCOME_ENHANCED': 'SRS Income Enhance', # Handle potential typo
            }

            # Objective Mapping
            OBJECTIVE_MAPPING = {
                'Core Equity100': 'Wealth Growth',
                'SRS Equity100': 'Wealth Growth',
                'China Growth': 'Wealth Growth',
                'REIT 100': 'Income Generation',
                'Income Enhance': 'Income Generation',
                'SRS Income Enhance': 'Income Generation',
                'BlackRock PCF': 'Income Generation',
                'Income Preserve': 'Capital Preservation',
                'SRS Income Preserve': 'Capital Preservation',
                'Cash SGD guaranteed': 'Capital Preservation',
                'Cash SGD flexi': 'Capital Preservation',
                'Cash USD flexi': 'Capital Preservation'
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
                        # Check if it's an AI investment (BlackRock, Monroe)
                        if 'blackrock' in ticker.lower() or 'monroe' in ticker.lower():
                            asset_allocation['Alternative Investments'] = asset_allocation.get('Alternative Investments', 0.0) + weight
                        else:
                            # Fallback for other unknown tickers
                            asset_allocation['Unclassified'] = asset_allocation.get('Unclassified', 0.0) + weight

                # Calculate Objective Allocation
                objective_allocation = {}
                for ticker, weight in client_weights.items():
                    obj = get_portfolio_objective(ticker)
                    objective_allocation[obj] = objective_allocation.get(obj, 0.0) + weight

                # --- Natural Summary ---
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
                
                summary_text = f"""
                Your current portfolio is primarily allocated to **{top_port}**, aligning with a **{top_obj}** objective. 
                """
                if top_holdings_str:
                    summary_text += f"\nAcross your holdings, your top underlying exposures are **{top_holdings_str}**."
                
                if 'Unclassified' in objective_allocation and objective_allocation['Unclassified'] > 0.1:
                    summary_text += f"\n\n*Note: {objective_allocation['Unclassified']:.1%} of your portfolio is in unclassified or custom assets.*"
                
                st.info(summary_text)

                # --- NEW: Tabs for Allocation vs Exposures ---
                tab_alloc, tab_exposures = st.tabs(["Client Portfolio Allocation", "Client Exposures"])

                with tab_alloc:
                    col_p1, col_p2, col_p3 = st.columns(3)
                
                    with col_p1:
                        st.markdown("**By Portfolio Type**")
                        fig_client = px.pie(
                            values=list(client_weights.values()),
                            names=list(client_weights.keys()),
                            title=f"Total NAV: ${total_nav:,.2f}"
                        )
                        st.plotly_chart(fig_client, use_container_width=True, key=f"{key_prefix}_pie_client")
                
                    with col_p2:
                        st.markdown("**By Asset Class**")
                        # Filter out small values for cleaner chart and group into Others
                        threshold = 0.01 # 1% threshold
                        final_asset_alloc = {}
                        others_val = 0.0
                        
                        for k, v in asset_allocation.items():
                            if v >= threshold:
                                final_asset_alloc[k] = v
                            else:
                                others_val += v
                        
                        if others_val > 0:
                            final_asset_alloc['Others'] = others_val
                            
                        fig_asset = px.pie(
                            values=list(final_asset_alloc.values()),
                            names=list(final_asset_alloc.keys()),
                            title="Asset Allocation"
                        )
                        st.plotly_chart(fig_asset, use_container_width=True, key=f"{key_prefix}_pie_asset")

                    with col_p3:
                        st.markdown("**By Objective**")
                        fig_obj = px.pie(
                            values=list(objective_allocation.values()),
                            names=list(objective_allocation.keys()),
                            title="Objective Allocation",
                            color_discrete_map={
                                'Wealth Growth': '#0052CC',
                                'Income Generation': '#36B37E', 
                                'Capital Preservation': '#FFAB00',
                                'Unclassified': '#C1C7D0'
                            }
                        )
                        st.plotly_chart(fig_obj, use_container_width=True, key=f"{key_prefix}_pie_obj")

                # --- NEW: Side-by-Side 2x2 Exposure Layout (User Request) ---
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
                    charts = get_exposure_charts(weights_series, df_sector, df_country, df_asset_class, df_underlying, concentration_top_n=50)
                    
                    col_exp_L, col_exp_R = st.columns([1, 1])
                    
                    options = ["Asset Class", "Sector", "Country", "Top Holdings"]
                    
                    def render_exposure_widget(column, widget_key_suffix):
                        with column:
                            choice = st.selectbox(f"View", options, key=f"{key_prefix}_exp_view_{widget_key_suffix}")
                            
                            if choice == "Asset Class":
                                if 'asset_pie' in charts:
                                    st.plotly_chart(charts['asset_pie'], use_container_width=True, key=f"{key_prefix}_pie_asset_{widget_key_suffix}")
                                    with st.expander("View Data"):
                                        st.dataframe(charts['asset_table'].style.format("{:.2%}"), use_container_width=True)
                                else:
                                    st.info("Asset Class data unavailable.")
                                    
                            elif choice == "Sector":
                                if 'sector_pie' in charts:
                                    st.plotly_chart(charts['sector_pie'], use_container_width=True, key=f"{key_prefix}_pie_sector_{widget_key_suffix}")
                                    with st.expander("View Data"):
                                        st.dataframe(charts['sector_table'].style.format("{:.2%}"), use_container_width=True)
                                else:
                                    st.info("Sector data unavailable.")
                                    
                            elif choice == "Country":
                                if 'country_pie' in charts:
                                    st.plotly_chart(charts['country_pie'], use_container_width=True, key=f"{key_prefix}_pie_country_{widget_key_suffix}")
                                    with st.expander("View Data"):
                                        st.dataframe(charts['country_table'].style.format("{:.2%}"), use_container_width=True)
                                else:
                                    st.info("Country data unavailable.")
                                    
                            elif choice == "Top Holdings":
                                # Selector for N [10, 15, 20]
                                t_topn = st.selectbox("Show Top", [10, 15, 20], key=f"{key_prefix}_topn_sel_{widget_key_suffix}")
                                
                                if df_underlying is not None:
                                    conc_raw = get_concentration_exposure(weights_series, df_underlying, top_n=t_topn)
                                    
                                    if not conc_raw.empty:
                                        conc_df = conc_raw.head(t_topn).to_frame(name='Weight')
                                        conc_df['Est. Value ($)'] = conc_df['Weight'] * total_nav
                                        
                                        # Pie chart for Top Holdings
                                        import plotly.express as px
                                        pie_df = conc_df.head(t_topn)
                                        
                                        fig_top_holdings_pie = px.pie(
                                            pie_df, 
                                            values='Weight', 
                                            names=pie_df.index, 
                                            title=f'Top {t_topn} Holdings Concentration',
                                            color_discrete_sequence=PALETTE,
                                            hole=0.3
                                        )
                                        fig_top_holdings_pie.update_traces(textposition='inside', textinfo='none')
                                        fig_top_holdings_pie.update_layout(showlegend=True)
                                        st.plotly_chart(fig_top_holdings_pie, use_container_width=True, key=f"{key_prefix}_pie_top_holdings_{widget_key_suffix}")

                                        with st.expander("View Data"):
                                            st.dataframe(
                                                conc_df.style.format({'Weight': '{:.2%}', 'Est. Value ($)': '${:,.2f}'}),
                                                use_container_width=True
                                            )
                                    else:
                                        st.warning("No underlying holdings found. Check ticker matching.")
                                        with st.expander("Debug Ticker Matching"):
                                            st.write("Input Tickers:", list(weights_series.index))
                                            st.write("Available Underlying Columns (Sample):", list(df_underlying.columns)[:10])
                                else:
                                    st.info("Concentration data not available (No underlying data found).")

                    # Render Left Widget
                    render_exposure_widget(col_exp_L, "L")
                    
                    # Render Right Widget
                    render_exposure_widget(col_exp_R, "R")

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
                            'fixed_weights': client_weights,
                            'dollar_amounts': client_holdings,
                            'decimal_weights': client_weights
                        }
                        st.session_state.portfolios.append(new_port)
                        st.session_state.refresh_counter += 1
                        st.success(f"Created new portfolio 'Client Upload {new_id}'!")
                        st.rerun()
            else:
                st.warning("No valid active portfolios found with NAV >= $50.")
            
        except Exception as e:
            st.error(f"Error processing file: {e}")

from utils.risk_engine import calculate_risk_score, get_eligible_portfolios, get_recommendation_text, get_smart_recommendation
from utils.llm_helper import get_recommendation_narrative

def render_recommender(GEMINI_API_KEY, key_prefix="rec"):
    st.header("Syfe Portfolio Recommender (Risk Engine)")

    # --- Step 1: Personal & Financial Profile ---
    st.subheader("1. Personal & Financial Profile")
    
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

    # --- Step 2: Risk Tolerance & Horizon ---
    st.subheader("2. Risk Tolerance & Horizon")
    
    # We need a primary objective to guide the risk questions, or just ask generic ones?
    # User said "Goals need to be defined and typed".
    # But Risk questions usually depend on the objective (Growth vs Income).
    # Let's add a "Primary Investment Style" here or just generic questions?
    # The previous code depended on `objective` variable from Step 1.
    # I will ask for "Primary Objective" here just to context switch the risk questions.
    
    primary_obj_context = st.radio(
        "Primary Investment Style (Context for Risk Questions)",
        ["Wealth Growth", "Income Generation", "Capital Preservation"],
        horizontal=True,
        key=f"{key_prefix}_primary_obj_ctx"
    )
    
    risk_options = []
    if primary_obj_context == "Wealth Growth":
        risk_text = "I am willing to..."
        risk_options = [
            "Take low loss risk to preserve capital (10-15% drawdown)",
            "Take moderate risk to balance risk/return (20% drawdown)",
            "Take higher risk to optimize returns (30% drawdown)",
            "Take as much risk as needed to maximize returns (30%+ drawdown)"
        ]
    elif primary_obj_context == "Income Generation":
        risk_text = "I am willing to..."
        risk_options = [
            "Take low loss risk to preserve capital (10-15% drawdown)",
            "Take moderate risk to balance risk/return (20% drawdown)",
            "Take higher risk to optimize returns (30% drawdown)",
            "Take as much risk as needed to maximize returns (30%+ drawdown)"
        ]
    else: # Capital Preservation
        risk_text = "For Capital Preservation, risk tolerance is generally low."
        risk_options = [
            "I simply want to avoid any loss (Capital Guarantee)",
            "I can accept very minor fluctuations for slightly better yield"
        ]
        
    risk_choice = st.radio(risk_text, risk_options, key=f"{key_prefix}_risk_choice")
    
    horizon_choice = st.radio(
        "General Investment Horizon (Years)",
        ["Less than 1", "3-5", "5-10", "More than 10"],
        horizontal=True,
        key=f"{key_prefix}_hor_Choice"
    )

    # --- Step 3: Investment Goals ---
    st.subheader("3. Investment Goals")
    st.markdown("Please define your goals below.")
    
    # Default data for the table
    default_goals = pd.DataFrame(
        [
            {"Goal Name": "Retirement", "Objective": "Wealth Growth", "Target Amount ($)": 1000000, "Lump Sum ($)": 50000, "Tenure (Years)": 20},
            {"Goal Name": "House Downpayment", "Objective": "Capital Preservation", "Target Amount ($)": 100000, "Lump Sum ($)": 20000, "Tenure (Years)": 3},
        ]
    )
    
    goal_df = st.data_editor(
        default_goals,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
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
            "Goal Name": st.column_config.TextColumn("Goal Name", required=True),
            "Target Amount ($)": st.column_config.NumberColumn("Target Amount ($)", min_value=0, format="$%d"),
            "Lump Sum ($)": st.column_config.NumberColumn("Lump Sum ($)", min_value=0, format="$%d"),
            "Tenure (Years)": st.column_config.NumberColumn("Tenure (Years)", min_value=1, max_value=50)
        },
        key=f"{key_prefix}_goals_table"
    )
    
    interest_in_ai = st.checkbox("Interested in AI / Thematic Exposure?", 
                                 help="If selected, we may allocate up to 30% to high-growth thematic portfolios (e.g. Disruptive Tech) if your risk profile allows.",
                                 key=f"{key_prefix}_ai_toggle")

    # --- Step 4: Qualitative & Check ---
    st.subheader("4. Additional Details")
    qualitative_details = st.text_area(
        "Additional Qualitative Inputs", 
        placeholder="Any other details relevant to your financial situation (e.g., existing liabilities, preferences, constraints)...",
        key=f"{key_prefix}_qual"
    )
    is_accredited = st.checkbox("I am an Accredited Investor (AI)", key=f"{key_prefix}_ai")
    
    # --- Generate ---
    st.markdown("---")
    if st.button("Generate Smart Recommendation", type="primary", key=f"{key_prefix}_calc"):
        
        # Map choice inputs
        # Risk Mapping
        if primary_obj_context == "Capital Preservation":
             if "avoid any loss" in risk_choice: risk_willingness = 1
             else: risk_willingness = 2
        else:
            if "30%+" in risk_choice: risk_willingness = 1
            elif "30%" in risk_choice: risk_willingness = 2
            elif "20%" in risk_choice: risk_willingness = 3
            else: risk_willingness = 4
            
        # Horizon Mapping
        if "Less than 1" in horizon_choice: horizon_years = 0.5
        elif "3-5" in horizon_choice: horizon_years = 4
        elif "5-10" in horizon_choice: horizon_years = 7.5
        else: horizon_years = 15

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
            'Goals': goal_df.to_csv(index=False), # Pass table as CSV text to LLM
            'Risk Score': 'Calculating...' 
        }
        
        # 1. Calculate Risk Score (keep this for internal validation if needed, or pass to LLM)
        score, overrides = calculate_risk_score(profile)
        profile['Risk Score'] = f"{score}/5" 
        
        # Display Score First
        st.success(f"**Client Risk Score: {score}/5**")
        if overrides:
            st.warning(f"**Score Adjustments:** {', '.join(overrides)}")
            
        # 2. Get AI Recommendation (Gemini generates the Table + Rationale)
        # Using the original flow requested by user.
        from utils.llm_helper import get_gemini_recommendation, parse_recommendation_table
        
        with st.spinner("Generating AI-powered Recommendation..."):
            rec_text = get_gemini_recommendation(GEMINI_API_KEY, profile)
            
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

        # Parse data for Logic
        portfolios, failed = parse_recommendation_table(rec_text)

        # Render Structured Table (Deterministic Display)
        if portfolios:
            st.markdown("#### Recommended Allocation")
            
            # Convert nested dict to flat list for display
            table_rows = []
            for goal, data in portfolios.items():
                horizon = data.get('horizon', 10)
                objective = data.get('objective', 'Wealth Growth')
                
                for ticker, weight in data.get('allocations', {}).items():
                    # Calculate amount if possible, or just show %
                    # We don't have the explicit goal amount here easily unless we parse it from goal_df or rec_text
                    # But the User just wants a table.
                    
                    table_rows.append({
                        "Goal": goal,
                        "Objective": objective,
                        "Horizon": f"{horizon} Yrs",
                        "Product": ticker,
                        "Allocation": weight
                    })
            
            if table_rows:
                df_disp = pd.DataFrame(table_rows)
                
                # Format
                st.dataframe(
                    df_disp.style.format({"Allocation": "{:.1%}"}),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            # Fallback if parsing fails - show raw just in case
            st.warning("Could not structure table data. Displaying raw output:")
            st.markdown(rec_text) # Only fallback
        
        if portfolios:
            # Create a map for Goal Name -> Lump Sum from the input table
            # goal_df has cols: "Goal Name", "Lump Sum ($)"
            goal_map = {}
            target_map = {}
            if not goal_df.empty and "Goal Name" in goal_df.columns and "Lump Sum ($)" in goal_df.columns:
                for _, row in goal_df.iterrows():
                    goal_map[row["Goal Name"]] = row["Lump Sum ($)"]
                    if "Target Amount ($)" in goal_df.columns:
                        target_map[row["Goal Name"]] = row["Target Amount ($)"]

            st.session_state.rec_results = {
                'score': score,
                'portfolios_data': portfolios, 
                'goal_map': goal_map,
                'target_map': target_map
            }
        else:
             pass
        
    # --- Add to Analysis Button ---
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
                    st.rerun()
                else:
                    st.warning("No valid portfolios found in recommendation to add.")
            else:
                st.warning("No structured portfolio data found. Please regenerate.")

def render_portfolio_construction(df_sec_list, available_tickers, bm_tickers, data_source, start_date, end_date, currency, backtest_offset_days, data_path, df_fi_metrics, df_fx, df_daily_ret, key_prefix="cons"):
    st.header("2. Portfolio Construction")

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
                st.session_state.portfolios[i]['name'] = st.text_input(f"Name", value=port['name'], key=f"{key_prefix}_name_{i}_{st.session_state.refresh_counter}")
                
                # Goal Name Input
                st.session_state.portfolios[i]['goal_name'] = st.text_input(
                    "Goal Name",
                    value=port.get('goal_name', ''),
                    placeholder="e.g. Retirement, House Downpayment",
                    key=f"{key_prefix}_goal_{i}_{st.session_state.refresh_counter}"
                )
                
                # Validate defaults exist in options (Handle casing mismatches e.g. BlackRock vs Blackrock)
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
        
            # Cash Invested and Portfolio Target Input
            c_cash, c_target = st.columns(2)
            with c_cash:
                st.session_state.portfolios[i]['cash_invested'] = st.number_input(
                    f"Cash Invested ($)", 
                    min_value=0.0, 
                    value=float(port.get('cash_invested', 100000.0)), 
                    step=1000.0, 
                    key=f"{key_prefix}_cash_{i}_{st.session_state.refresh_counter}"
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
                            amount = st.number_input(
                                f"{ticker} ($)", 
                                min_value=0.0, 
                                value=st.session_state.portfolios[i]['dollar_amounts'].get(ticker, 1000.0),
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

def render_analysis(df_sec_list, df_fi_metrics, df_fx, df_sector, df_country, df_asset_class, df_underlying, data_path, bm_tickers, backtest_offset_days, data_source, start_date, end_date, currency, key_prefix="anal"):
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
        
            w_select = st.selectbox("Select Portfolio to View Weights", list(results.keys()), key=f"{key_prefix}_w_view_sel")
        
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
            p_select = st.selectbox("Select Portfolio", all_port_ret.columns, key=f"{key_prefix}_monthly_sel")
            fig = plot_monthly_heatmap(all_port_ret[[p_select]])
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_monthly_heatmap")
        
        with tabs[3]: # Cumulative Returns
            cum_ret = (1 + all_port_ret).cumprod()
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
            st.dataframe(rel_metrics.to_frame().T.style.format("{:.2%}"), use_container_width=True)
        
            st.markdown("#### Rolling 12M Excess Return")
            fig_roll = plot_cumulative_returns(roll_excess.to_frame(name="Rolling Excess Return")) # Reuse plot function
            st.plotly_chart(fig_roll, use_container_width=True, key=f"{key_prefix}_roll_excess")

    # --- Exposure Analysis ---
    if st.session_state.optimization_results:
        st.header("4. Exposure Analysis")
        
        res_data = st.session_state.optimization_results
        results = res_data['results']
        
        # --- Advanced Exposure Filtering ---
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

        # --- Aggregation Logic ---
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
            
            # --- Dual Side-by-Side View (User Request) ---
            st.markdown("### Exposure Overview")
            
            col_exp_L, col_exp_R = st.columns([1, 1])
            options = ["Asset Class", "Sector", "Country", "Top Holdings"]
            
            def render_anal_exposure_widget(column, widget_key_suffix):
                with column:
                    # Header/Selector
                    choice = st.selectbox(f"View", options, key=f"{key_prefix}_anal_exp_view_{widget_key_suffix}")
                    
                    if choice == "Asset Class":
                        if 'asset_pie' in charts:
                            st.plotly_chart(charts['asset_pie'], use_container_width=True, key=f"{key_prefix}_anal_pie_asset_{widget_key_suffix}")
                            with st.expander("View Data"):
                                st.dataframe(charts['asset_table'].style.format("{:.2%}"), use_container_width=True)
                        else:
                            st.info("Asset Class data unavailable.")
                            
                    elif choice == "Sector":
                        if 'sector_pie' in charts:
                            st.plotly_chart(charts['sector_pie'], use_container_width=True, key=f"{key_prefix}_anal_pie_sector_{widget_key_suffix}")
                            with st.expander("View Data"):
                                st.dataframe(charts['sector_table'].style.format("{:.2%}"), use_container_width=True)
                        else:
                            st.info("Sector data unavailable.")
                            
                    elif choice == "Country":
                        if 'country_pie' in charts:
                            st.plotly_chart(charts['country_pie'], use_container_width=True, key=f"{key_prefix}_anal_pie_country_{widget_key_suffix}")
                            with st.expander("View Data"):
                                st.dataframe(charts['country_table'].style.format("{:.2%}"), use_container_width=True)
                        else:
                            st.info("Country data unavailable.")
                            
                    elif choice == "Top Holdings":
                        # Selector for N [10, 15, 20]
                        t_topn = st.selectbox("Show Top", [10, 15, 20], key=f"{key_prefix}_anal_topn_sel_{widget_key_suffix}")
                        
                        if df_underlying is not None:
                            conc_agg = get_concentration_exposure(agg_weights, df_underlying, top_n=t_topn)
                            
                            if not conc_agg.empty:
                                conc_df = conc_agg.to_frame(name='Weight')
                                conc_df['Est. Value ($)'] = conc_df['Weight'] * total_agg_value
                                
                                # Pie chart for Top Holdings - STRICT FILTER
                                pie_df = conc_df.head(t_topn) # Ensure only Top N are shown
                                
                                import plotly.express as px
                                fig_top_holdings_pie = px.pie(
                                    pie_df, 
                                    values='Weight', 
                                    names=pie_df.index, 
                                    title=f'Top {t_topn} Holdings Concentration',
                                    color_discrete_sequence=PALETTE,
                                    hole=0.3
                                )
                                # User Request: No text in pie chart, strict slicing (handled by top_n above)
                                fig_top_holdings_pie.update_traces(textposition='inside', textinfo='none')
                                fig_top_holdings_pie.update_layout(showlegend=True) # Legend is needed if no text
                                st.plotly_chart(fig_top_holdings_pie, use_container_width=True, key=f"{key_prefix}_anal_pie_top_holdings_{widget_key_suffix}")

                                with st.expander("View Data"):
                                    st.dataframe(
                                        conc_df.style.format({'Weight': '{:.2%}', 'Est. Value ($)': '${:,.2f}'}),
                                        use_container_width=True
                                    )
                            else:
                                st.warning("No underlying holdings found.")
                        else:
                            st.info("Concentration data not available (No underlying data found).")

            # Render Charts
            render_anal_exposure_widget(col_exp_L, "L")
            render_anal_exposure_widget(col_exp_R, "R")

        else:
            st.warning("No exposure data found for the selected filters.")

    # This function is a placeholder for logic that might be moved or updated elsewhere.
    # The original instruction refers to "Add Selected Strategy" logic, which is not present
    # in the provided snippet. If this function was intended to contain that logic,
    # it needs to be provided in the context.
    def render_recommender_add_logic(key_prefix="rec_add"):
        pass


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
                key=f"{key_prefix}_cashflow_portfolio_select"
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
                ticker_upper = ticker.upper()
                
                # Explicitly check for BlackRock and Monroe - monthly dividends
                if 'BLACKROCK' in ticker_upper or 'MONROE' in ticker_upper:
                    return 'monthly'
                
                # Check if ticker contains REIT indicators (including REIT Risk Managed) - quarterly dividends
                if 'REIT' in ticker_upper or 'VNQ' in ticker or 'VNQI' in ticker:
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
                        
                            # Get yield - check both dividend yield and YTM
                            # All yields in FI Metrics are stored as percentages (e.g., 10.72 = 10.72%)
                            annual_yield = 0.0
                            if ticker in df_fi_metrics.columns:
                                # First try T12M dividend yield (for income products like BlackRock, Monroe)
                                if 'T12M dividend yield' in df_fi_metrics.index:
                                    yield_val = df_fi_metrics.loc['T12M dividend yield', ticker]
                                    if pd.notna(yield_val) and yield_val != 0:
                                        annual_yield = float(yield_val) / 100  # Convert from % to decimal
                                
                                # Fall back to YTM if no dividend yield
                                if annual_yield == 0 and 'YTM / est. yield' in df_fi_metrics.index:
                                    yield_val = df_fi_metrics.loc['YTM / est. yield', ticker]
                                    if pd.notna(yield_val) and yield_val != 0:
                                        annual_yield = float(yield_val) / 100  # Convert from % to decimal
                        
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
        
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_cashflow_chart")
        
            # Summary table
            st.subheader("Monthly Cash Flow Summary by Portfolio")
            
            if not monthly_cash_flows.empty:
                monthly_cash_flows.columns = [f"Month {m}" for m in range(1, 13)]
                monthly_cash_flows['Total Annual'] = monthly_cash_flows.sum(axis=1)
            
                # Reset index to avoid duplicate index errors if portfolio names are same
                display_df = monthly_cash_flows.reset_index().rename(columns={'index': 'Portfolio'})
                
                st.dataframe(
                    display_df.style.format("${:,.2f}", subset=display_df.columns.drop('Portfolio'))
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
                st.metric("Total Annual Cash Flow", f"${total_annual:,.2f}")
            with col2:
                st.metric("Average Monthly Cash Flow", f"${total_annual/12:,.2f}")
        
        # --- Growth Forecast Calculator ---
        st.markdown("---")
        st.header("Growth Forecast Calculator")
        st.caption("Project future portfolio value based on historical returns")
        
        if st.session_state.portfolios:
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
                                st.info(f"💡 Required monthly contribution to meet goal: **${monthly_contribution:,.2f}**")
                            
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
