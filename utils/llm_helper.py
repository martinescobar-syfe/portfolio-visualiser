import google.generativeai as genai

SYFE_CONTEXT = """
**Syfe Product Context**

**1. Core Portfolios**
- **Core Defensive**: Low risk. Aims to beat cash/FD over long run. Mainly high-quality bond ETFs + some equity and gold. Not AI-only.
- **Core Balanced**: Medium risk. Moderate long-term growth. Mix of equity / bonds / gold. Not AI-only.
- **Core Growth**: High risk. Maximising long-term risk-adjusted returns. Equity-heavy with small bond/gold sleeve. Not AI-only.
- **Core Equity100**: Very high risk. Pure equity risk/return profile. Behaves like global equity, large drawdowns possible. Not AI-only.
*Heuristic*: More equities = more expected return, more drawdown risk.

**2. REIT+ Portfolios**
- **REIT+ 100% REITs**: High risk. Equity-like, driven by SG REIT yields + price. Concentrated in SG REITs. Not AI-only.
- **REIT+ with Risk Management**: Medium risk. REIT upside but damped by bonds. Mix of REITs + SG gov bond ETF. Lower volatility. Not AI-only.
*Intuition*: Equity income plays.

**3. Income+ Portfolios (with PIMCO)**
- **Income+ Preserve**: Low risk. Above cash / Cash+. Shorter duration, higher-quality bonds, focus on capital preservation. Not AI-only.
- **Income+ Enhance**: Moderately low risk. Higher income target than Preserve. Longer duration, slightly lower average credit quality. Not AI-only.
*Key*: Neither is capital-guaranteed, but sit below equity/REIT in risk, above Cash+.

**4. Thematic Portfolios (Syfe Select Themes)**
- **ESG & Clean Energy**: High to very high risk. Rides ESG transition. Sector-concentrated. Not AI-only.
- **Disruptive Technology**: Very high risk. AI, cloud, robotics, fintech. Growth/tech heavy. Big drawdowns possible. Not AI-only.
- **Healthcare Innovation**: High risk. Biotech, pharma, medtech. Not AI-only.
- **China Growth**: High risk. China consumer, tech. Concentrated country risk. Not AI-only.
*Note*: These are satellite plays, not core.

**5. Syfe Select (Custom Portfolios)**
- **DIY ETF Portfolio**: Build your own. Choose up to 8 ETFs. Fully customizable weights. Auto-rebalancing. Retail access.

**6. Cash Management**
- **Cash+ Flexi (SGD/USD)**: Money-market style. Daily liquidity, no lock-in. Retail access.
- **Cash+ Guaranteed (SGD/USD)**: Fixed-term capital-guaranteed (1-12 months). Retail access.

**7. Brokerage (Syfe Trade)**
- **Direct Trading**: US/SG/HK stocks & ETFs. Fractional US shares. No platform fees. Retail access.

**8. Private Wealth Products (Accredited Investor Only)**
- **Private Credit Funds**: Institutional private lending strategies (e.g., via BlackRock). AI Only.
- **Fixed Maturity Bond Portfolios**: Professional bond portfolios targeting fixed yields. AI Only.

**Accredited Investor (AI) Status**
- "AI only" products (Private Credit, Fixed Maturity Bonds) are strictly for Accredited Investors.
- AI status also implies higher risk capacity for other products.
"""

def get_gemini_recommendation(api_key, client_profile):
    """
    Generates a portfolio recommendation using Gemini.
    
    Args:
        api_key (str): Google Gemini API Key.
        client_profile (dict): Dictionary containing client details (Age, Risk, Income, etc.)
        
    Returns:
        str: The generated recommendation text.
    """
    try:
        genai.configure(api_key=api_key)
        
        # Format the complex client profile into a readable string
        import json
        profile_str = json.dumps(client_profile, indent=2)
        
        prompt = f"""
    You are a highly experienced Wealth Management Investment Counsellor at Syfe.
    Your task is to analyze the client's profile and recommend a portfolio allocation using Syfe's specific product suite.

    **Client Profile:**
    {client_profile}
    
    **Client Goals & Priorities:**
    {client_profile.get('Goals', 'No specific goals provided.')}

    **Instructions:**
    1.  **Executive Summary**: Briefly analyze the client's profile and goals.
    
    2.  **Portfolio Recommendation**:
        *   Structure this section by Goal.
        *   For EACH goal, create a recommendation table.
        *   **Format**:
            *   **Header**: "Goal: [Goal Name] ([Horizon] Years)"
            *   **Table Columns**: `Objective`, `Goal Name`, `Horizon (Yrs)`, `Product`, `Allocation (%)`, `Dollar Amount ($)`, `Rationale`.
        *   **Important Rules**:
            *   Use ONLY valid Syfe products.
            *   **Thematic Exposure**: If the client has selected "Interest in AI : Yes", you MAY allocate up to **30%** (MAXIMUM) to "Thematic: Disruptive Technology" or similar, IF their Risk Score is high enough (4 or 5). If Risk Score is low, do NOT allocate to Thematic, even if interested.
            *   **Accredited Investors**: Only recommend Private Credit/Fixed Maturity if the client is Accredited.
            *   **Total Allocation**: Must sum to 100% per goal.
    
    **Context:**
    {SYFE_CONTEXT}
    """

        # List of models to try in order of preference (Gemini 3 Pro Preview first!)
        models_to_try = [
            'gemini-3-pro-preview', 
            'gemini-2.5-flash-preview-09-2025',
            'gemini-1.5-pro', 
            'gemini-1.5-flash'
        ]
        
        last_error = None
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                last_error = e
                continue
                
        return f"Error generating recommendation: All models failed. Last error: {str(last_error)}"
    except Exception as e:
        return f"Error configuring API: {str(e)}"

def parse_recommendation_table(recommendation_text):
    """
    Parses the Markdown table from the AI recommendation to extract products and weights.
    Returns a dictionary of {Ticker: Weight(decimal)}.
    """
    import re
    
    # Mapping from AI Product Name to Ticker in PV_sec_list.csv
    product_map = {
        # Core Portfolios
        "Core Equity100": "Core Equity100",
        "Core Equity 100": "Core Equity100",
        "Core Growth": "Core Growth",
        "Core Balanced": "Core Balanced",
        "Core Defensive": "Core Defensive",
        "Downside Protected": "Downside protected",
        "Downside protected": "Downside protected",
        # REIT Portfolios
        "REIT+": "REIT 100",
        "REIT 100": "REIT 100",
        "REIT+ 100% REITs": "REIT 100",
        "REIT+ with Risk Management": "REIT Risk Managed",
        "REIT Risk Managed": "REIT Risk Managed",
        # Income Portfolios
        "Income+ Preserve": "Income Preserve",
        "Income Preserve": "Income Preserve",
        "Income+ Enhance": "Income Enhance",
        "Income Enhance": "Income Enhance",
        # Thematic Portfolios
        "Disruptive Technology": "Disruptive Technology",
        "Thematic: Disruptive Technology": "Disruptive Technology",
        "Healthcare Innovation": "Healthcare Innovation",
        "ESG & Clean Energy": "ESG & Clean Energy",
        "China Growth": "China Growth",
        # Cash Management
        "Cash+ Flexi (SGD)": "Cash SGD flexi",
        "Cash+ Flexi (USD)": "Cash USD flexi",
        "Cash+ Flexi": "Cash SGD flexi",
        "Cash+ Guaranteed": "Cash SGD guaranteed",
        "Cash SGD flexi": "Cash SGD flexi",
        "Cash USD flexi": "Cash USD flexi",
        "Cash SGD guaranteed": "Cash SGD guaranteed",
        # SRS Portfolios
        "SRS Income Preserve": "SRS Income Preserve",
        "SRS Income Enhance": "SRS Income Enhance",
        "SRS Cash SGD flexi": "SRS Cash SGD flexi",
        "SRS Equity100": "SRS Equity100",
        # Private Credit (Accredited Investor)
        "BlackRock Private Credit": "BlackRock Private Credit Fund",
        "BlackRock Private Credit Fund": "BlackRock Private Credit Fund",
        "Monroe Income Plus": "Monroe Income Plus Fund",
        "Monroe Income Plus Fund": "Monroe Income Plus Fund",
        # Silverdale
        "Silverdale Fund": "Silverdale Fund (Nov 2027)",
    }
    
    portfolios = {}
    failed_items = []
    
    try:
        lines = recommendation_text.split('\n')
        for line in lines:
            # Look for table rows with new format: | Objective | Goal Name | Horizon | Product | Allocation (%) | Dollar Amount ($) | Rationale |
            if '|' in line and '---' not in line and 'Objective' not in line and 'Goal Name' not in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 5:  # At minimum need Objective, Goal Name, Horizon, Product, Allocation
                    # New format: 0=Objective, 1=Goal Name, 2=Horizon, 3=Product, 4=Allocation %, 5=Dollar Amount (optional), 6=Rationale (optional)
                    
                    objective_raw = parts[0].strip()
                    goal_name = parts[1].strip()
                    horizon_str = parts[2].strip()
                    product_name = parts[3].strip()
                    alloc_str = parts[4].replace('%', '').strip()
                    
                    # Validate objective
                    valid_objectives = ["Wealth Growth", "Income Generation", "Capital Preservation"]
                    objective = None
                    
                    if objective_raw in valid_objectives:
                        objective = objective_raw
                    else:
                        # Try partial match
                        for valid_obj in valid_objectives:
                            if valid_obj in objective_raw:
                                objective = valid_obj
                                break
                    
                    # Fallback if no objective found
                    if objective is None:
                        objective = "Wealth Growth" # Default
                    
                    # Extract horizon as integer
                    try:
                        horizon = int(horizon_str.replace('yrs', '').replace('years', '').replace('yr', '').strip())
                    except:
                        horizon = 10  # Default
                    
                    try:
                        weight = float(alloc_str) / 100.0
                        
                        # Map product name to ticker
                        ticker = product_map.get(product_name)
                        if not ticker:
                            # Fuzzy matching or fallback
                            for key, val in product_map.items():
                                if key in product_name:
                                    ticker = val
                                    break
                        
                        if ticker and weight > 0:
                            # Initialize goal group if not exists
                            if goal_name not in portfolios:
                                portfolios[goal_name] = {
                                    'objective': objective,
                                    'horizon': horizon,
                                    'allocations': {}
                                }
                            
                            # Add to allocations
                            if ticker in portfolios[goal_name]['allocations']:
                                portfolios[goal_name]['allocations'][ticker] += weight
                            else:
                                portfolios[goal_name]['allocations'][ticker] = weight
                        else:
                            # Failed to map or weight is 0
                            if weight > 0:
                                failed_items.append((product_name, weight, goal_name))
                                
                    except ValueError:
                        continue # Skip if weight isn't a number
                        
    except Exception as e:
        print(f"Error parsing table: {e}")
        return {}, []
        
    return portfolios, failed_items

def get_recommendation_narrative(api_key, client_profile, allocation, goal_name, amount):
    """
    Generates a personalized narrative explaining the deterministic allocation.
    """
    try:
        genai.configure(api_key=api_key)
        
        # Format allocation for prompt
        alloc_str = "\n".join([f"- **{k}**: {v:.1%}" for k, v in allocation.items()])
        
        prompt = f"""
    You are a highly experienced Wealth Management Investment Counsellor at Syfe.
    
    **Context:**
    We have already determined the optimal asset allocation for this client based on their risk profile and goals using our proprietary engine.
    Your task is to **explain** this recommendation to the client in a natural, professional, and personalized way.
    
    **Client Profile:**
    - Goal: {goal_name}
    - Investment Amount: ${amount:,.2f}
    - Risk Score: {client_profile.get('Risk Score', 'N/A')} (Scale 1-5)
    - Time Horizon: {client_profile.get('Time Horizon', 'N/A')} years
    - Investment Objective: {client_profile.get('Objective', 'N/A')}
    - Interest in AI/Thematic: {client_profile.get('Interest in AI', 'No')}
    
    **Recommended Allocation:**
    {alloc_str}
    
    **Instructions:**
    1.  **Investment Strategy**: Explain *why* this specific blend of portfolios (Core + Satellite, if any) is suitable for their specific Goal "{goal_name}" and Time Horizon.
    2.  **Risk & Return**: Briefly mention the expected behavior of this portfolio (e.g., "Given your 10-year horizon, this portfolio accepts short-term volatility for higher long-term growth...").
    3.  **Thematic Exposure (if applicable)**: If the allocation includes a Thematic/AI portfolio, explain that it is capped (e.g. at 30%) to manage risk while providing exposure to high-growth sectors.
    4.  **Tone**: Professional, encouraging, and confident.
    5.  **Format**: Use Markdown. Keep it concise (2-3 paragraphs). Do NOT output a table (the table is already shown to the user). Just the narrative.
    """

        # List of models to try
        models_to_try = [
            'gemini-1.5-pro', 
            'gemini-1.5-flash'
        ]
        
        last_error = None
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                last_error = e
                continue
                
        return f"Error generating narrative: {str(last_error)}"
    except Exception as e:
        return f"Error configuring API: {str(e)}"
