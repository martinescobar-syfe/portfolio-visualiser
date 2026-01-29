import pandas as pd
import numpy as np

def calculate_risk_score(profile):
    """
    calc client risk score 1-5 based on psychology, time horizon, 
    financials and hard brakes
    """
    # step A: base assessment
    raw_score_map = {1: 5, 2: 4, 3: 3, 4: 2}
    raw_score = raw_score_map.get(profile.get('risk_willingness', 3), 3)
    
    # time horizon caps
    horizon = profile.get('time_horizon_years', 10)
    if horizon < 3:
        raw_score = min(raw_score, 2)
    elif horizon < 5:
        raw_score = min(raw_score, 3)
    elif horizon < 10:
        raw_score = min(raw_score, 4)
        
    score = raw_score
    
    # step B: financial adjusters
    adjuster = 0
    income = profile.get('annual_income', 0)
    if income < 30000:
        adjuster -= 2 # Super Dampen
    elif income < 60000:
        adjuster -= 1 # Dampen
    elif income > 300000:
        adjuster += 1
    assets = profile.get('liquid_assets', 0)
    if assets < 50000:
        adjuster -= 1
    elif assets > 250000:
        adjuster += 1
    savings = profile.get('savings_rate', 0.2)
    if savings < 0.15:
        adjuster -= 1
    elif savings > 0.35:
        adjuster += 1
        
    score += adjuster
    score = max(1, min(5, score))
    
    # step C: hard brakes
    overrides = []
    if profile.get('short_term_need', False):
        score = 1
        overrides.append("Short Term Need -> Score 1")
    
    if profile.get('concentration_high', False):
        if score > 3:
            score = 3
            overrides.append("High Concentration -> Cap at 3")
    knowledge = profile.get('investment_knowledge', 'Average')
    if knowledge in ['None', 'Minimal']:
        if score > 2:
            score = 2
            overrides.append("Low Experience -> Cap at 2")
            
    return score, overrides

def get_eligible_portfolios(risk_score, objective_category, is_accredited=False):
    """get eligible portfolios for risk score and objective"""
    catalog = [
        ("Cash+ Guaranteed", 1, "Capital Preservation"),
        ("Cash+ Flexi", 1, "Capital Preservation"),
        ("Income+ Preserve", 1, "Income Generation"),
        ("Core Defensive", 1, "Wealth Growth"),
        ("Core Balanced", 2, "Wealth Growth"),
        ("Income+ Enhance", 2, "Income Generation"),
        ("Core Growth", 3, "Wealth Growth"),
        ("Downside Protected", 3, "Wealth Growth"),
        ("REIT+ (Risk Mgmt)", 3, "Income Generation"),
        ("Core Equity 100", 4, "Wealth Growth"),
        ("REIT 100", 4, "Income Generation"),
        ("Core Equity 100", 5, "Wealth Growth"),
        ("Thematic: ESG", 5, "Wealth Growth"),
        ("Thematic: Disruptive Tech", 5, "Wealth Growth"),
        ("Thematic: Healthcare", 5, "Wealth Growth"),
        ("China Growth", 5, "Wealth Growth"),
    ]
    
    if is_accredited:
        pass

    results = []
    for name, p_score, cat in catalog:
        if p_score <= risk_score and cat == objective_category:
            results.append({"name": name, "min_score": p_score, "category": cat})
            
    return results

def get_recommendation_text(risk_score, objective, time_horizon):
    """recommendation logic based on goal mapper"""
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
    """deterministic portfolio allocation rec"""
    core_port = "Core Balanced"
    
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

    allocation = {}
    
    if interest_in_ai and objective == "Wealth Growth" and risk_score >= 4:
        satellite_port = "Thematic: Disruptive Technology"
        satellite_weight = 0.30
        core_weight = 0.70
        
        allocation[core_port] = core_weight
        allocation[satellite_port] = satellite_weight
    else:
        allocation[core_port] = 1.0
        
    return allocation
