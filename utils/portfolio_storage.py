import json
import os
import streamlit as st
from supabase import create_client, Client

from utils.app_secrets import get_secret

def get_supabase_client() -> Client:
    """connect to supabase; returns None (local JSON fallback) if SUPABASE_URL/KEY are not configured"""
    try:
        url = get_secret("SUPABASE_URL")
        key = get_secret("SUPABASE_KEY")
        
        if not url or not key:
            return None
        
        return create_client(url, key)
    except Exception:
        return None

# local fallback if supabase unavailable
DATA_FILE = "Data/saved_portfolios.json"

def load_portfolios():
    """load saved portfolios from supabase (or local JSON fallback)"""
    client = get_supabase_client()
    
    if client:
        try:
            response = client.table('portfolios').select('name, data').execute()
            return {row['name']: row['data'] for row in response.data}
        except Exception as e:
            st.warning(f"Supabase read failed, using local: {e}")
    
    # fallback to local JSON
    if not os.path.exists(DATA_FILE):
        return {}
    
    try:
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading saved portfolios: {e}")
        return {}

def save_portfolio(name, portfolio_data):
    """save portfolio to supabase (or local JSON fallback)"""
    client = get_supabase_client()
    
    if client:
        try:
            # upsert (insert or update if name already exists)
            client.table('portfolios').upsert({
                'name': name,
                'data': portfolio_data
            }, on_conflict='name').execute()
            return True
        except Exception as e:
            st.warning(f"Supabase write failed, saving locally: {e}")
    
    # fallback to local JSON
    data = load_portfolios()
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    data[name] = portfolio_data
    
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving portfolio: {e}")
        return False

def delete_portfolio(name):
    """delete portfolio from supabase (or local JSON fallback)"""
    client = get_supabase_client()
    
    if client:
        try:
            client.table('portfolios').delete().eq('name', name).execute()
            return True
        except Exception as e:
            st.warning(f"Supabase delete failed: {e}")
    
    # fallback to local JSON
    data = load_portfolios()
    
    if name in data:
        del data[name]
        try:
            with open(DATA_FILE, 'w') as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            st.error(f"Error deleting portfolio: {e}")
            return False
    return False
