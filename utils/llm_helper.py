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
- **Private Credit Funds**: Institutional private lending strategies (e.g., via Blackrock). AI Only.
- **Fixed Maturity Bond Portfolios**: Professional bond portfolios targeting fixed yields. AI Only.

**Accredited Investor (AI) Status**
- "AI only" products (Private Credit, Fixed Maturity Bonds) are strictly for Accredited Investors.
- AI status also implies higher risk capacity for other products.
"""

def clean_llm_response(text):
    """
    cleans common LLM formatting issues (e.g. missing spaces between numbers and words)
    """
    import re
    # Fix "50,000and" -> "50,000 and"
    text = re.sub(r'(\d+)(and|or|dollars|years|percent)', r'\1 \2', text)
    # Fix "word50,000" -> "word 50,000" (less common but possible)
    text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)
    # Fix "\$" -> "$"
    text = text.replace(r'\$', '$')
    return text

def get_gemini_recommendation(api_key, client_profile):
    """
    generates a portfolio recommendation using gemini based on client profile
    """
    try:
        genai.configure(api_key=api_key)
        
        # format the complex client profile into a readable string
        # sanitise keys for better LLM readability
        readable_profile = {}
        key_map = {
            "risk_willingness": "Risk Willingness (1-5)",
            "risk_ability": "Risk Ability (1-5)",
            "risk_score": "Overall Risk Score (1-5)",
            "time_horizon": "Time Horizon (Years)",
        }
        
        for k, v in client_profile.items():
            new_key = key_map.get(k, k.replace('_', ' ').title())
            readable_profile[new_key] = v
            
        import json
        profile_str = json.dumps(readable_profile, indent=2)
        
        prompt = f"""
    You are a highly experienced Wealth Management Investment Counsellor at Syfe.
    Your task is to analyze the client's profile and recommend a portfolio allocation using Syfe's specific product suite.

    **Client Profile:**
    {profile_str}
    
    **Client Goals & Priorities:**
    {client_profile.get('Goals', 'No specific goals provided.')}

    **Instructions:**
    1.  **Executive Summary**: Briefly analyze the client's profile and goals in 2-3 sentences.
    
    2.  **Portfolio Recommendation**:
        *   Structure this section by Goal.
        *   For EACH goal, create a recommendation table.
        *   **Format**:
            *   **Header**: "Goal: [Goal Name] ([Horizon] Years)"
            *   **Table Columns**: `Objective`, `Goal Name`, `Horizon (Yrs)`, `Product`, `Allocation (%)`, `Dollar Amount ($)`, `Rationale`.
        *   **MANDATORY RULES**:
            *   Use ONLY valid Syfe products from the Context below.
            *   **Thematic Exposure**: If the client has selected "Interest in AI : Yes", you MAY allocate up to **30%** (MAXIMUM) to "Thematic: Disruptive Technology" or similar, IF their Risk Score is high enough (4 or 5). If Risk Score is low, do NOT allocate to Thematic, even if interested.
            *   **Accredited Investors**: Only recommend Private Credit/Fixed Maturity if the client is Accredited.
            *   **CRITICAL - 100% ALLOCATION RULE**: 
                - For EACH goal, the Allocation (%) column MUST sum to EXACTLY 100%.
                - If a client has multiple goals (e.g., Retirement, Education, Emergency Fund), EACH goal gets its own table where allocations sum to 100%.
                - Example: Goal "Retirement" might have Core Equity100 (70%) + Income Enhance (30%) = 100%. Goal "Emergency Fund" might have Cash+ Flexi (100%) = 100%.
                - DO NOT split percentages across goals. Each goal is independent with its own 100% allocation.
            *   **Objective Tagging**: Each row MUST have one of these objectives: "Wealth Growth", "Income Generation", or "Capital Preservation".
    
    ============================================================
    ABSOLUTE FORMATTING REQUIREMENTS - VIOLATION WILL CAUSE ERRORS
    ============================================================
    
    1. NEVER USE LATEX: 
       - NO dollar signs for math like this: dollar-x-dollar or double-dollar-x-double-dollar
       - Write "$50,000" as plain text (this is fine)
       - Write "20%" as plain text (this is fine)
       - DO NOT wrap numbers in dollar signs or use backslash escaping
    
    2. PLAIN TEXT ONLY:
       - Money: Write as $50,000 or 50,000 dollars
       - Percentages: Write as 20% or 20 percent
       - No mathematical notation whatsoever
       - No backslash escaping of any characters
    
    3. SIMPLE MARKDOWN TABLES:
       - Use | Column1 | Column2 | format
       - No special characters in cells
       - Keep cells simple and readable
    
    4. NO INTERNAL VARIABLE NAMES:
       - Use "Risk Willingness" not "risk_willingness"
       - Use "Time Horizon" not "time_horizon"
    
    ============================================================
    
    **Context:**
    {SYFE_CONTEXT}
    """

        # use gemini 3 pro preview (try models in order of preference)
        models_to_try = [
            'gemini-3-pro-preview',
            'gemini-2.0-flash-exp',
            'gemini-2.0-flash',
            'gemini-2.5-flash',
            'gemini-2.5-pro'
        ]
        
        last_error = None
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return clean_llm_response(response.text)
            except Exception as e:
                last_error = e
                continue
                
        return f"Error generating recommendation: All models failed. Last error: {str(last_error)}"
    except Exception as e:
        return f"Error configuring API: {str(e)}"

def parse_recommendation_table(recommendation_text):
    """
    parses the markdown table from the AI recommendation to extract products and weights, returns {goal_name: {allocations}} dict
    """
    import re
    
    # mapping from AI product name to ticker in our sec_list
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
        "Blackrock Private Credit": "Blackrock Private Credit Fund",
        "Blackrock Private Credit Fund": "Blackrock Private Credit Fund",
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
            # look for table rows with new format: | Objective | Goal Name | Horizon | Product | Allocation (%) | Dollar Amount ($) | Rationale |
            if '|' in line and '---' not in line and 'Objective' not in line and 'Goal Name' not in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 5:  # At minimum need Objective, Goal Name, Horizon, Product, Allocation
                    # new format: 0=Objective, 1=Goal Name, 2=Horizon, 3=Product, 4=Allocation %, 5=Dollar Amount (optional), 6=Rationale (optional)
                    
                    objective_raw = parts[0].strip()
                    goal_name = parts[1].strip()
                    horizon_str = parts[2].strip()
                    product_name = parts[3].strip()
                    alloc_str = parts[4].replace('%', '').strip()
                    
                    # validate objective
                    valid_objectives = ["Wealth Growth", "Income Generation", "Capital Preservation"]
                    objective = None
                    
                    if objective_raw in valid_objectives:
                        objective = objective_raw
                    else:
                        # try partial match
                        for valid_obj in valid_objectives:
                            if valid_obj in objective_raw:
                                objective = valid_obj
                                break
                    
                    # fallback if no objective found
                    if objective is None:
                        objective = "Wealth Growth" # Default
                    
                    # extract horizon as integer
                    try:
                        horizon = int(horizon_str.replace('yrs', '').replace('years', '').replace('yr', '').strip())
                    except:
                        horizon = 10  # Default
                    
                    try:
                        weight = float(alloc_str) / 100.0
                        
                        # map product name to ticker
                        ticker = product_map.get(product_name)
                        if not ticker:
                            # fuzzy matching or fallback
                            for key, val in product_map.items():
                                if key in product_name:
                                    ticker = val
                                    break
                        
                        if ticker and weight > 0:
                            # initialise goal group if not exists
                            if goal_name not in portfolios:
                                portfolios[goal_name] = {
                                    'objective': objective,
                                    'horizon': horizon,
                                    'allocations': {}
                                }
                            
                            # add to allocations
                            if ticker in portfolios[goal_name]['allocations']:
                                portfolios[goal_name]['allocations'][ticker] += weight
                            else:
                                portfolios[goal_name]['allocations'][ticker] = weight
                        else:
                            # failed to map or weight is 0
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
    generates a personalised narrative explaining the deterministic allocation
    """
    try:
        genai.configure(api_key=api_key)
        
        # format allocation for prompt
        alloc_str = "\n".join([f"- **{k}**: {v:.1%}" for k, v in allocation.items()])
        
        prompt = f"""
    You are a highly experienced Wealth Management Investment Counsellor at Syfe.
    
    **Context:**
    We have already determined the optimal asset allocation for this client based on their risk profile and goals using our proprietary engine.
    Your task is to **explain** this recommendation to the client in a natural, professional, and personalized way.
    
    **Client Profile:**
    - Goal: {goal_name}
    - Investment Amount: {amount:,.0f} dollars
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
    
    ============================================================
    ABSOLUTE FORMATTING REQUIREMENTS - VIOLATION WILL CAUSE ERRORS
    ============================================================
    
    - NEVER USE LATEX: No wrapping numbers in dollar signs or backslash escaping
    - Write money as: "50,000 dollars" or "$50,000" (plain text dollar sign is OK)
    - Write percentages as: "20%" or "20 percent" (plain text)
    - DO NOT wrap numbers in special formatting - just plain text
    - NO mathematical notation whatsoever
    - Use human-readable names like "Risk Willingness" not "risk_willingness"
    
    ============================================================
    ============================================================
    """




        # list of models to try
        models_to_try = [
            'gemini-2.5-pro',
            'gemini-2.5-flash',
            'gemini-2.0-flash',
            'gemini-2.0-flash-exp'
        ]
        
        last_error = None
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return clean_llm_response(response.text)
            except Exception as e:
                last_error = e
                continue
                
        return f"Error generating narrative: {str(last_error)}"
    except Exception as e:
        return f"Error configuring API: {str(e)}"

def analyze_portfolio_appropriateness(api_key, client_profile, current_holdings):
    """
    analyses if the current portfolio is appropriate for the client using gemini
    """
    try:
        genai.configure(api_key=api_key)
        
        # format holdings for prompt
        holdings_str = ""
        if isinstance(current_holdings, dict):
             holdings_str = "\n".join([f"- **{k}**: {v:.1%}" for k, v in current_holdings.items()])
        elif isinstance(current_holdings, list):
             holdings_str = "\n".join([f"- **{h.get('name', 'Unknown')}** ({h.get('type', 'Unknown')}): {h.get('nav_in_sgd', 0):,.2f} SGD" for h in current_holdings])
        
        prompt = f"""
    You are a Senior Risk Officer at Syfe.
    Your job is to evaluate if a client's CURRENT holdings are appropriate given their Age and Risk Score.
    
    **Client Profile:**
    - Name: {client_profile.get('name', 'Client')}
    - Age: {client_profile.get('age', 'Unknown')}
    - Risk Score: {client_profile.get('risk_score', 'Unknown')} (Scale 1-5, where 5 is Aggressive)
    
    **Current Holdings (Active Portfolios):**
    {holdings_str}
    
    **Instructions:**
    Your goal is to provide a single, actionable, and cohesive block of text (short narrative) that addresses the following:
    
    1.  **Diversification**: Is the portfolio well-diversified or concentrated? (e.g., heavily tilted to one sector/country).
    2.  **Risk & Age Fit**: Is the current risk level appropriate for their Age and Risk Score? Is it too conservative or too aggressive?
    3.  **Next Steps**: Specifically recommend where they should allocate their *next* dollar to improve their portfolio (e.g., "Consider adding fixed income to balance the high equity exposure" or "Shift excess cash into a diversified core portfolio").
    
    **Tone:** Professional, direct, and helpful.
    **Constraint:** Do NOT use bullet points. Write it as one or two cohesive paragraphs.
    
    **Context:**
    {SYFE_CONTEXT}
    """
        models_to_try = [
            'gemini-3-pro-preview',
            'gemini-2.5-pro',
            'gemini-2.5-flash',
            'gemini-2.0-flash',
            'gemini-2.0-flash-exp',
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
                
        return f"Analysis unavailable. Error: {str(last_error)}"

    except Exception as e:
        return f"Error: {str(e)}"


def generate_diversification_insights(api_key, score_result, client_weights):
    """
    generate 3 AI-powered insights from diversification scores using gemini
    """
    import json

    portfolios_list = [{"name": k, "weight": round(v, 4)}
                       for k, v in client_weights.items()]

    ac = score_result['asset_class']
    geo = score_result['geography']
    conc = score_result['concentration']
    sec = score_result['sector']

    input_payload = {
        "portfolios": portfolios_list,
        "scores": {
            "asset_class": ac['score'],
            "geography": geo['score'],
            "sector": sec['score'],
            "concentration": conc['score'],
            "overall": score_result['composite_score'],
        },
        "sub_metrics": {
            "asset_class_breadth": ac['breadth_score'],
            "asset_class_dominance": ac['dominance_score'],
            "asset_class_weights": list(ac['details'].values()) if ac['details'] else [],
            "geo_weights": [d['client_weight'] for d in geo.get('region_details', {}).values()],
            "n_holdings": conc['n_holdings'],
            "n_effective": conc['n_eff'],
            "hhi": conc['hhi'],
            "max_holding_weight": max(conc.get('top_holdings', {0: 0}).values()) if conc.get('top_holdings') else 0,
            "sector_equity_score": sec.get('equity_score'),
            "sector_fi_score": sec.get('fi_score'),
        },
    }

    prompt = f"""You are a portfolio insight engine for Syfe, a digital wealth manager regulated by the Monetary Authority of Singapore.

Given a client's portfolio diversification scores, generate exactly 3 concise, actionable insights (1-2 sentences each).

**Rules:**
1. Always include at least one strength and at least one area for improvement.
2. Map scores to specific, actionable Syfe portfolio suggestions (Core Equity100, Core Growth, Core Balanced, Core Defensive, Income+ Enhance, Income+ Preserve, REIT100, Cash+ Flexi, China Growth, Disruptive Technology, ESG & Clean Energy, Healthcare Innovation).
3. Use plain English — no jargon like HHI, N_effective, or Shannon entropy. Say things like "your portfolio behaves as if it holds only X independent positions."
4. Be specific with numbers — reference actual scores and suggest percentage ranges.
5. Tone: friendly, professional, concise. Like a wealth advisor giving a quick portfolio health check.

**Score guide:**
- Asset Class (low < 6): suggest adding missing asset classes (bonds, REITs, cash)
- Geography (low < 6): suggest broader regional exposure via Core portfolios
- Sector (low < 6): suggest diversifying via Income+ or REIT100 for missing sectors
- Concentration (low < 6): suggest broad index like Core Equity100 (3,500+ stocks)

**Input:**
{json.dumps(input_payload, indent=2)}

**Syfe Product Context:**
{SYFE_CONTEXT}

**Output format — return ONLY valid JSON, no markdown fences.**
**IMPORTANT: Do NOT use markdown formatting (no **, no *, no _) in the title or body fields. Use plain text only. Round all scores to 1 decimal place.**
[
  {{"category": "strength"|"improvement"|"observation", "title": "Short title (max 6 words)", "body": "1-2 sentence insight."}}
]"""

    try:
        genai.configure(api_key=api_key)
        models_to_try = [
            'gemini-2.5-flash',
            'gemini-2.0-flash',
            'gemini-1.5-flash',
        ]
        last_error = None
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                text = response.text.strip()
                if text.startswith('```'):
                    text = text.split('\n', 1)[1] if '\n' in text else text[3:]
                    text = text.rsplit('```', 1)[0]
                return json.loads(text)
            except Exception as e:
                last_error = e
                continue
        return [{"category": "observation", "title": "Insights unavailable",
                 "body": f"Could not generate insights: {last_error}"}]
    except Exception as e:
        return [{"category": "observation", "title": "Error",
                 "body": f"Error: {e}"}]
