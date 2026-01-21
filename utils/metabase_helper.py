"""
Helper functions for interacting with Metabase API to fetch client data.
"""
import requests
import streamlit as st
import pandas as pd

METABASE_URL = "https://metabase.internal.syfe.com"
API_KEY = "mb_5fsl6dNzjFaylX86lm9rsJnEHc+0dVDIrP+e+tYjetA="

HEADERS = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

def run_card(card_id, params=None):
    """Run a Metabase card/query with optional parameters."""
    try:
        url = f"{METABASE_URL}/api/card/{card_id}/query"
        data = {"parameters": params} if params else {}
        response = requests.post(url, headers=HEADERS, json=data, timeout=15)
        
        if response.status_code in [200, 202]:
            result = response.json()
            data = result.get('data', {})
            rows = data.get('rows', [])
            cols = [c.get('name', c.get('display_name')) for c in data.get('cols', [])]
            return [dict(zip(cols, row)) for row in rows]
        else:
            st.error(f"Metabase API Error {response.status_code}: {response.text}")
            return []
    except requests.exceptions.ConnectTimeout:
        st.error("Connection to Metabase timed out. Please ensure you are connected to the Syfe VPN.")
        return []
    except requests.exceptions.ConnectionError:
        st.error("Connection to Metabase failed. Please check your network and VPN connection.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred connecting to Metabase: {str(e)}")
        return []

def get_client_info_by_email(email):
    """
    Get client info including age and risk score (Card 1642).
    """
    params = [{"type": "category", "value": email, "target": ["variable", ["template-tag", "email"]]}]
    rows = run_card(1642, params)
    
    # Card 1642 returns key-value pairs
    info = {
        'age': None,
        'risk_score': None,
        'client_id': None,
        'name': None
    }
    
    for row in rows:
        key = row.get('key')
        value = row.get('value')
        if key:
            info[key] = value
            
    return info

def get_email_by_id(client_id):
    """
    Get client email by ID using Card 4905.
    """
    params = [{"type": "text", "value": str(client_id), "target": ["variable", ["template-tag", "investor_id"]]}]
    rows = run_card(4905, params)
    if rows and 'email' in rows[0]:
        return rows[0]['email']
    return None

def get_client_info_by_id(client_id):
    """
    Get client info by ID. First fetches email, then fetches details using email.
    Returns dict with age, risk_score, name, email, client_id.
    """
    email = get_email_by_id(client_id)
    if not email:
        return {'client_id': client_id, 'email': None, 'age': None, 'risk_score': None, 'name': None}
        
    info = get_client_info_by_email(email)
    info['client_id'] = client_id
    info['email'] = email
    return info

def get_active_portfolios(client_id):
    """
    Get active portfolios for a client (Card 1470).
    Returns a dataframe normalized to match the expected format.
    """
    if not client_id:
        return pd.DataFrame()
        
    params = [{"type": "number", "value": client_id, "target": ["variable", ["template-tag", "client_id"]]}]
    rows = run_card(1470, params)
    
    # Filter for ACTIVE portfolios only
    active_rows = [r for r in rows if r.get('status') == 'ACTIVE']
    
    if not active_rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(active_rows)
    
    # Rename columns to match what the app expects (from CSV)
    # App expects: 'name', 'type', 'nav_in_sgd', 'invested_amount', 'pnl_inception', 'status'
    # The Metabase result already has these names mostly correct
    
    # Ensure numeric columns are actually numeric
    numeric_cols = ['nav_in_sgd', 'invested_amount', 'pnl_inception', 'balance']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df
