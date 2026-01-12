import json
import os
import streamlit as st

DATA_FILE = "data/saved_portfolios.json"

def load_portfolios():
    """Load saved portfolios from JSON file."""
    if not os.path.exists(DATA_FILE):
        return {}
    
    try:
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading saved portfolios: {e}")
        return {}

def save_portfolio(name, portfolio_data):
    """Save a portfolio configuration to JSON file."""
    data = load_portfolios()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    
    # Add/Update
    data[name] = portfolio_data
    
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving portfolio: {e}")
        return False

def delete_portfolio(name):
    """Delete a saved portfolio."""
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
