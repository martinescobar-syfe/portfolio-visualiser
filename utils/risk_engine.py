import pandas as pd
import numpy as np

def calculate_risk_score(profile):
    """
    Calculates Client Risk Score (1-5) based on:
    Step A: Base Assessment (Psychology + Time Horizon)
    Step B: Financial Adjusters (Enhancers/Dampeners)
    Step C: Hard Brakes (Overrides)
    
    profile: dict containing keys:
        - risk_willingness: int (1-4, mapped from questions)
        - time_horizon_years: float
        - annual_income: float
        - liquid_assets: float
        - savings_rate: float (0.0 - 1.0)
        - short_term_need: bool (withdraw >90% in 12m)
        - concentration_high: bool (>50% net worth)
        - investment_knowledge: str ('None', 'Minimal', 'Average', 'Good', 'Expert')
    """
    
    # --- Step A: Base Assessment ---
    # Psychology (Raw Score)
    # Map willingness 1-4 to raw score 1-5?
    # Prompt: "willing to accept 30%..." -> Score 5?
    # Let's map the 4 choices to Scores.
    # Choice 1 (Max risk, 30%+ dd) -> 5
    # Choice 2 (High risk, 30% dd) -> 4
    # Choice 3 (Mod risk, 20% dd) -> 3
    # Choice 4 (Low risk, 10-15% dd) -> 2
    # (Score 1 is usually reserved for pure cash/preservation)
    
    raw_score_map = {1: 5, 2: 4, 3: 3, 4: 2}
    raw_score = raw_score_map.get(profile.get('risk_willingness', 3), 3)
    
    # Time Horizon Adjuster
    # 10 Years: High Risk allowed (Score 5)
    # <3 Years: Low Risk (Score 1-2)
    horizon = profile.get('time_horizon_years', 10)
    
    # We take the MIN of Psych Score and Horizon Cap?
    # "Time Horizon: 10 Years: High Risk allowed... <3 Years: Low Risk"
    # Usually Horizon acts as a cap.
    if horizon < 3:
        # Cap at 2
        raw_score = min(raw_score, 2)
    elif horizon < 5:
        # Cap at 3?
        raw_score = min(raw_score, 3)
    elif horizon < 10:
        # Cap at 4? (Implied)
        raw_score = min(raw_score, 4)
    else:
        # Horizon >= 10, no cap from horizon
        pass
        
    score = raw_score
    
    # --- Step B: Financial Adjusters ---
    adjuster = 0
    
    # Income
    income = profile.get('annual_income', 0)
    if income < 30000:
        adjuster -= 2 # Super Dampen
    elif income < 60000:
        adjuster -= 1 # Dampen
    elif income > 300000:
        adjuster += 1 # Enhance
        
    # Liquid Assets
    assets = profile.get('liquid_assets', 0)
    if assets < 50000:
        adjuster -= 1
    elif assets > 250000:
        adjuster += 1 # Enhance (Prompt says Enhance / Super Enhance, let's stick to +1 for now)
        
    # Savings Rate
    savings = profile.get('savings_rate', 0.2)
    if savings < 0.15:
        adjuster -= 1
    elif savings > 0.35:
        adjuster += 1
        
    score += adjuster
    
    # Clamp to 1-5 range before hard brakes
    score = max(1, min(5, score))
    
    # --- Step C: Hard Brakes (Overrides) ---
    overrides = []
    
    # Short Term Need
    if profile.get('short_term_need', False):
        score = 1
        overrides.append("Short Term Need -> Score 1")
        
    # Concentration Risk (Cap at 3)
    if profile.get('concentration_high', False):
        if score > 3:
            score = 3
            overrides.append("High Concentration -> Cap at 3")
            
    # Investment Knowledge (Cap at 2 if Minimal/None)
    knowledge = profile.get('investment_knowledge', 'Average')
    if knowledge in ['None', 'Minimal']:
        if score > 2:
            score = 2
            overrides.append("Low Experience -> Cap at 2")
            
    return score, overrides

def get_eligible_portfolios(risk_score, objective_category, is_accredited=False):
    """
    Returns list of eligible portfolios based on Risk Score and Objective.
    
    Risk Score 1: Defensive
    Risk Score 2: Moderately Low
    Risk Score 3: Moderate
    Risk Score 4: Moderately High
    Risk Score 5: Aggressive
    """
    
    # Portfolio Metadata
    # Structure: Name, RiskTier (1-5), Category, AccreditedOnly
    # Note: Prompt lists specific suggestions.
    
    # Catalog
    # Format: (Name, MinRiskScore, Category)
    catalog = [
        # Score 1
        ("Cash+ Guaranteed", 1, "Capital Preservation"),
        ("Cash+ Flexi", 1, "Capital Preservation"),
        ("Income+ Preserve", 1, "Income Generation"),
        ("Core Defensive", 1, "Wealth Growth"), # Assuming Core Defensive fits here
        
        # Score 2
        ("Core Balanced", 2, "Wealth Growth"),
        ("Income+ Enhance", 2, "Income Generation"),
        
        # Score 3
        ("Core Growth", 3, "Wealth Growth"),
        ("Downside Protected", 3, "Wealth Growth"), # Prompt says "Growth & Income Balance"
        ("REIT+ (Risk Mgmt)", 3, "Income Generation"), # "REITs with Risk Management"
        
        # Score 4
        ("Core Equity 100", 4, "Wealth Growth"),
        ("REIT 100", 4, "Income Generation"), # "100% REITs"
        
        # Score 5
        ("Core Equity 100", 5, "Wealth Growth"), # Duplicate for 5? Or just eligible.
        ("Thematic: ESG", 5, "Wealth Growth"),
        ("Thematic: Disruptive Tech", 5, "Wealth Growth"),
        ("Thematic: Healthcare", 5, "Wealth Growth"),
        ("China Growth", 5, "Wealth Growth"),
    ]
    
    # Accredited
    if is_accredited:
        # Prompt: "Qualitative write up... can include Alternative Investments"
        # We can add them to list or handle separately as "Waitlist/Manual"
        pass

    results = []
    
    # Logic: User can buy products with Product Score <= Client Score
    # But typically we recommend products MATCHING the score or slightly lower for safety.
    # The prompt says: "A user cannot buy a product with a Product Risk Score higher than their Client Risk Score."
    
    for name, p_score, cat in catalog:
        if p_score <= risk_score and cat == objective_category:
            results.append({"name": name, "min_score": p_score, "category": cat})
            
    return results

def get_recommendation_text(risk_score, objective, time_horizon):
    """
    Returns specific recommendation logic based on "The Goal Mapper".
    """
    rec_text = ""
    
    if objective == "Wealth Growth":
        if time_horizon >= 10 and risk_score >= 4:
            rec_text = "Primary: **Core Equity 100** (Long term growth)."
        elif time_horizon >= 5:
            if risk_score >= 3:
                rec_text = "Primary: **Core Growth** (Balanced growth)."
            else:
                rec_text = "Primary: **Core Balanced** (Conservative growth)."
        else:
            rec_text = "Primary: **Downside Protected** (Short-medium term growth with protection)."
            
    elif objective == "Income Generation":
        if risk_score >= 4:
            rec_text = "Primary: **REIT 100** (Max yield)."
        elif risk_score >= 2:
            rec_text = "Primary: **Income+ Enhance** (Balanced yield/risk)."
        else:
            rec_text = "Primary: **Income+ Preserve** (Stable income)."
            
    elif objective == "Capital Preservation":
        rec_text = "Primary: **Cash+ Guaranteed** or **Cash+ Flexi** (No loss expected)."
        
    return rec_text

def get_smart_recommendation(risk_score, objective, time_horizon, interest_in_ai=False):
    """
    Deterministically calculates a recommended portfolio allocation.
    
    Logic:
    1. Select a 'Core' portfolio based on Risk Score and Objective.
    2. If interest_in_ai is True AND Risk Score is sufficient (>=4?), allocate to Satellite.
       - Satellite Cap: 30% max.
       - Typically 70% Core / 30% Satellite.
    """
    
    # 1. Determine Core Portfolio
    core_port = "Core Balanced" # Default fallback
    
    if objective == "Wealth Growth":
        if risk_score == 5:
            core_port = "Core Equity 100"
        elif risk_score == 4:
            core_port = "Core Equity 100"
        elif risk_score == 3:
            core_port = "Core Growth"
        elif risk_score == 2:
            core_port = "Core Balanced"
        else:
            core_port = "Core Defensive"
            
    elif objective == "Income Generation":
        if risk_score >= 4:
            core_port = "REIT 100"
        elif risk_score >= 2:
            core_port = "Income+ Enhance"
        else:
            core_port = "Income+ Preserve"
            
    elif objective == "Capital Preservation":
        core_port = "Cash+ Guaranteed"

    # 2. Allocation Logic
    allocation = {}
    
    # Check eligibility for AI/Thematic (needs higher risk tolerance)
    # Let's say needs Risk Score >= 4 for purely thematic "Disruptive Tech" 
    # But user might insist. The prompt says "cap of 30%".
    # Let's enforce Risk Score >= 3 for any satellite to be safe, or just follow user?
    # Let's use Risk Score >= 4 as a realistic constraint for "Disruptive Tech".
    
    if interest_in_ai and objective == "Wealth Growth" and risk_score >= 4:
        # Satellite Allocation
        satellite_port = "Thematic: Disruptive Technology"
        satellite_weight = 0.30
        core_weight = 0.70
        
        allocation[core_port] = core_weight
        allocation[satellite_port] = satellite_weight
        
    else:
        # 100% Core
        allocation[core_port] = 1.0
        
    return allocation
