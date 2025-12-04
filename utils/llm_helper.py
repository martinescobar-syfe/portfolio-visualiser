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
    1.  **Executive Summary**: Briefly analyze the client's profile, focusing on their risk tolerance, financial situation, and **specifically addressing each of their stated goals and priorities**. Explain how the recommended strategy aligns with these goals.
    
    2.  **Portfolio Recommendation**:
        *   Structure this section by Goal.
        *   For EACH goal listed in the "Client Goals & Priorities", create a specific portfolio recommendation.
        *   **Format**:
            *   **Header**: "Goal X: [Goal Name] ($[Lump Sum] Lump Sum, [Horizon] Year Horizon)"
            *   **Table**: Immediately follow the header with a markdown table for that goal.
        *   The table MUST have the following columns: `Objective`, `Goal Name`, `Horizon (Yrs)`, `Product`, `Allocation (%)`, `Dollar Amount ($)`, `Rationale`.
        *   `Objective`: One of: "Wealth Growth", "Income Generation", or "Capital Preservation".
        *   `Goal Name`: Use the EXACT name provided in the client's goals. MUST be filled for EVERY row.
        *   `Horizon (Yrs)`: The time horizon from the goal's "Horizon (Yrs)" field. MUST be filled for EVERY row.
        *   `Product`: Must be one of the valid Syfe products listed in the context below.
        *   `Allocation (%)`: The percentage of that specific goal's portfolio allocated to the product.
        *   `Dollar Amount ($)`: Calculate this by multiplying the Allocation (%) by the "Lump Sum Investment ($)" specified for that goal. Format as a dollar amount (e.g., "$50,000").
        *   `Rationale`: A brief 1-sentence explanation for this allocation.
        *   The allocations for EACH goal must sum up to 100%.
        *   **NO LATEX**: Do NOT use LaTeX formatting (e.g., do not use `$` for math blocks). Use standard text only.
        *   **CRITICAL**: You MUST fill in the 'Goal Name' and 'Horizon (Yrs)' for EVERY row. Do NOT leave them blank, do NOT use ditto marks, and do NOT merge cells. Every row must be complete.
    
    **Product Constraints:**
    *   **AI Products**: Only recommend "Private Credit" or "Fixed Maturity Bond" if "Accredited Investor" is True.
    *   **30% Limit**: "Private Credit" and "Fixed Maturity Bond" combined must NOT exceed 30% of any single goal's portfolio.
    *   **Valid Products**: Use ONLY the product names listed in the context (e.g., "Core Equity100", "REIT+ 100% REITs", "Income+ Enhance", "Cash+ Flexi", "Cash+ Guaranteed", "Private Credit", "Fixed Maturity Bond").
    
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
        "Core Equity100": "Core Equity100",
        "Core Growth": "Core Growth",
        "Core Balanced": "Core Balanced",
        "Core Defensive": "Core Defensive",
        "REIT+": "REIT 100", # Default to 100 if generic
        "REIT+ 100% REITs": "REIT 100",
        "REIT+ with Risk Management": "REIT Risk Managed",
        "Income+ Preserve": "Income Preserve",
        "Income+ Enhance": "Income Enhance",
        "Disruptive Technology": "Disruptive Technology",
        "Healthcare Innovation": "Healthcare Innovation",
        "ESG & Clean Energy": "ESG & Clean Energy",
        "China Growth": "China Growth",
        "Cash+ Flexi (SGD)": "Cash SGD flexi",
        "Cash+ Flexi (USD)": "Cash USD flexi",
        "Cash+ Flexi": "Cash SGD flexi", # Generic fallback
        "Cash+ Guaranteed": "Cash SGD guaranteed", # Proxy
        # Add more mappings as needed
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
