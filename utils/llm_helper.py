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
        You are an expert financial advisor at Syfe. 
        Use the following **Syfe Product Context** to recommend a portfolio allocation for the client based on their detailed profile.
        
        {SYFE_CONTEXT}
        
        **Client Profile**:
        ```json
        {profile_str}
        ```
        
        **Instructions**:
        1.  **Analyze the Client**:
            -   Assess Risk Capacity, Human Capital, Goal Feasibility, and Tax/Currency needs.
            -   **Check Accredited Investor (AI) Status**: If "Yes" (Income > $300k or Assets > $2M), they qualify for **Private Wealth Products**.
            
        2.  **Recommend Syfe Portfolios**:
            -   Map each Goal to a suitable Syfe Product.
            -   **For Accredited Investors**: You **MUST** consider recommending **Private Credit Funds** or **Fixed Maturity Bond Portfolios** (from Private Wealth) for goals requiring stable yield or diversification, as these are premium products unavailable to retail.
            -   **Core Portfolio**: Select based on overall risk profile.
            -   **Satellite/Add-ons**: REIT+, Income+, Thematics.
            -   Ensure allocations are rounded to the nearest 5%.
            
        3.  **Output Format**:
            -   **Section 1: Executive Summary**
                -   Provide a concise summary (max 2 sentences per goal) explaining the recommendation rationale.
            
            -   **Section 2: Asset Allocation Strategy**
                -   **Strictly** provide a Markdown Table with the following columns:
                    -   `Goal Bucket`
                    -   `Goal Name(s)`
                    -   `Horizon`
                    -   `Syfe Product`
                    -   `Allocation Rec. (%)`
                    -   `Rationale` (1 sentence)
                -   Do not include any introductory text before or after the table.
            
        4.  **Tone & Formatting**: 
            -   Professional, direct, and deterministic.
            -   **NO LaTeX**: Do not use LaTeX formatting (e.g., no $ symbols for math). Use plain text (e.g., "$5M", not "$5M$").
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
