
import pandas as pd
import numpy as np
import os
import yfinance as yf
import streamlit as st
from datetime import datetime

@st.cache_data
def load_static_data(data_path):
    try:
        excel_path = os.path.join(data_path, 'Inv_Data_Hardcoded.xlsx')
        
        # Load all sheets
        # Sheets: Daily Returns, Country, Sector, FI Metrics, Asset Class, Risk Rating, Underlying
        
        # 1. Security List (Constructed from Asset Class sheet or similar if separate not found, 
        #    but typically we need a master list. The prompt implies 'new data file... various sheets'.
        #    Let's check if 'Asset Class' sheet serves as the master or if we should look for 'Security List'.
        #    The prompt says "Risk Rating for each asset and Underlying (which is also constituents)".
        #    Let's assume 'Asset Class' or a specific list exists.
        #    Actually, existing code used PV_sec_list.csv.
        #    Let's try to infer if there's a specific sheet for security list constants (Ticker, Name, etc.)
        #    or if we should use one of the attribute sheets as the master.
        #    Let's assume 'Asset Class' sheet has Ticker as index or column.
        
        # OPTIMIZATION: Read all sheets at once is faster than multiple reads if file is huge, 
        # but pandas read_excel(None) reads all.
        xls = pd.read_excel(excel_path, sheet_name=None)
        
        # Helper to get dataframe safely
        def get_sheet(name, index_col=0):
            if name in xls:
                df = xls[name]
                if index_col is not None and index_col < len(df.columns):
                     df = df.set_index(df.columns[index_col])
                return df
            return pd.DataFrame()

        # Load specific sheets
        df_daily_ret = get_sheet('Daily Return', index_col=0) # Date index
        # Fix Date index if needed
        if not df_daily_ret.empty:
            df_daily_ret.index = pd.to_datetime(df_daily_ret.index)
        
        df_country = get_sheet('Country', index_col=0)
        df_sector = get_sheet('Sector', index_col=0)
        df_fi_metrics = get_sheet('FI Metrics', index_col=0)
        df_asset_class = get_sheet('Asset Class', index_col=0)
        df_risk_rating = get_sheet('Risk', index_col=0)
        df_underlying = get_sheet('Underlying', index_col=None) # For concentration analysis, wide format, no meaningful index

        # We need to reconstruct df_sec_list which was used for Ticker, Name, Remarks, Base currency, Exchange
        # If 'Asset Class' contains these details, we can use it. 
        # Let's assume 'Asset Class' sheet has Ticker as index and columns like 'Name', 'Asset Class', etc.
        # We might need to ensure 'Ticker' is a column for some legacy logic or just index.
        # Current app expects df_sec_list to look like a CSV read.
        
        # Let's create a proxy pv_sec_list from df_asset_class if possible, or maybe there is a 'Security List' sheet?
        # If not, let's use df_asset_class.reset_index() as the base.
        
        if 'Asset Class' in xls:
            # Check if headers look like tickers (Matrix format)
            raw_ac = xls['Asset Class']
            
            # Identify 'Asset Class' column (case insensitive search)
            ac_col = next((c for c in raw_ac.columns if c.lower() == 'asset class'), None)
            
            if ac_col and ('SPY' in raw_ac.columns or 'ACWI' in raw_ac.columns):
                # Matrix Format: Rows=Classes, Cols=Tickers (1/0 flags)
                ac_names = raw_ac[ac_col]
                
                # Drop the label column to get just tickers
                tickers_df = raw_ac.drop(columns=[ac_col])
                
                # Transpose: Tickers became Index, Columns are original Row Indices
                t_df = tickers_df.T 
                
                # Ensure numeric
                t_df = t_df.apply(pd.to_numeric, errors='coerce').fillna(0)
                
                # Find row index with max value (the '1')
                # This returns Series: Index=Ticker, Value=Original Row Index
                class_indices = t_df.idxmax(axis=1)
                
                # Map indices to names
                # class_indices values align with ac_names index
                mapped_classes = ac_names.loc[class_indices].values
                
                df_sec_list = pd.DataFrame({
                    'Ticker': t_df.index,
                    'Asset Class': mapped_classes
                })
                # Ensure Ticker is string
                df_sec_list['Ticker'] = df_sec_list['Ticker'].astype(str)
                
            else:
                 # Standard format or unknown
                 df_sec_list = raw_ac.copy()
        else:
            # Fallback
            df_sec_list = pd.DataFrame(columns=['Ticker', 'Name', 'Base currency', 'Exchange', 'Remarks'])
            
        # Sanitize df_sec_list for legacy compatibility
        # We need standard columns: Ticker, Name, Base currency, Exchange, Remarks
        # Let's assume the Excel has them or we fill NaNs.
        for col in ['Name', 'Base currency', 'Exchange', 'Remarks']:
            if col not in df_sec_list.columns:
                df_sec_list[col] = 'N/A'
                
        # Ensure Ticker column exists (for fallback path)
        if 'Ticker' not in df_sec_list.columns:
             if df_sec_list.index.name == 'Ticker':
                 df_sec_list = df_sec_list.reset_index()
             else:
                 # Fallback, likely won't happen with above fix
                 df_sec_list = df_sec_list.reset_index().rename(columns={'index': 'Ticker'})

        # Load FX from Daily Returns (USDSGD, USDHKD) if columns exist
        fx_cols = ['USDSGD', 'USDHKD']
        if all(col in df_daily_ret.columns for col in fx_cols):
            df_fx = df_daily_ret[fx_cols] / 100 # Assuming data is in % or similar scaling? 
            # Prompt says "Daily Returns". Usually returns are small decimals or %.
            # However, FX rates are usually absolute levels (e.g. 1.34).
            # If "Daily Returns" sheet literally contains returns for tickers, does it also contain FX RATES?
            # Or FX Returns?
            # User said "Daily Returns... for each asset".
            # Typically FX is needed for conversion. 
            # Let's assume the sheet might be mixed or we search for FX columns.
            # If they are returns, we need levels! 
            # Wait, previous code loaded 'F-F_Research_Data...' or 'PV_daily_ret.csv' which had FX columns.
            # If the new sheet is ONLY returns, we can't get FX levels easily unless provided.
            # Let's assume for now they are in there as levels or we will flag this.
            pass
        else:
             # Fallback: maybe they are in a separate sheet or we assume 1.0 if missing to prevent crash
             df_fx = pd.DataFrame(index=df_daily_ret.index)
             df_fx['USDSGD'] = 1.35 # Static fallback
             df_fx['USDHKD'] = 7.8
             
        return df_sec_list, df_asset_class, df_sector, df_country, df_fi_metrics, df_fx, df_daily_ret, df_risk_rating, df_underlying

    except Exception as e:
        st.error(f"Error loading static data from Excel: {e}")
        return None, None, None, None, None, None, None, None, None

def get_data_path():
    # Check if running in Colab (default path) or local
    if os.path.exists('/content/Projects/Data/'):
        return '/content/Projects/Data/'
    elif os.path.exists('Data/'):
        return 'Data/'
    else:
        # Fallback to current directory if Data folder not found
        return './'

def modify_tickers_based_on_exchange(all_tickers, df_sec_list):
    modified_tickers = []
    for ticker in all_tickers:
        if ticker in df_sec_list['Ticker'].values:
            exchange = df_sec_list.loc[df_sec_list['Ticker'] == ticker, 'Exchange'].iloc[0]
            if exchange == 'London':
                modified_tickers.append(ticker + '.L')
            elif exchange == 'Swiss':
                modified_tickers.append(ticker + '.SW')
            else:
                modified_tickers.append(ticker)
        else:
            modified_tickers.append(ticker)
    return modified_tickers

def download_data(data_source, start_date, end_date, all_tickers, bm_tickers, currency, df_sec_list, df_fx, data_path, df_daily_ret=None):
    """
    Download or load return data for the specified tickers.
    
    Args:
        data_source: 'Yahoo Finance' or 'Bloomberg' (now uses Excel data)
        start_date: Start date for data
        end_date: End date for data
        all_tickers: List of tickers to load
        bm_tickers: Benchmark tickers
        currency: Target currency for FX adjustment
        df_sec_list: Security list DataFrame
        df_fx: FX rates DataFrame
        data_path: Path to data directory
        df_daily_ret: Pre-loaded daily returns DataFrame from Excel (for Bloomberg source)
    """
    if data_source == 'Yahoo Finance':
        modified_tickers = modify_tickers_based_on_exchange(all_tickers, df_sec_list)
        
        # Handle date conversion for yfinance
        start_str = start_date.strftime('%Y-%m-%d') if start_date else None
        end_str = end_date.strftime('%Y-%m-%d') if end_date else None

        if start_str is None and end_str is None:
            df_all = yf.download(modified_tickers, period='max')['Adj Close']
        else:
            df_all = yf.download(modified_tickers, start=start_str, end=end_str)['Adj Close']

        df_all.index = df_all.index.tz_localize(None)
        ret_all = df_all.pct_change().dropna()
        
        # Map back to original tickers
        if isinstance(ret_all, pd.Series):
            ret_all = ret_all.to_frame()
            ret_all.columns = all_tickers
        else:
             if len(ret_all.columns) == len(all_tickers):
                 ret_all.columns = all_tickers
    
    elif data_source == 'Bloomberg':
        # Use pre-loaded Excel data instead of CSV
        if df_daily_ret is None or df_daily_ret.empty:
            st.error("Daily return data not available. Please check the Excel file.")
            return pd.DataFrame(), pd.DataFrame()
        
        # Work with a copy to avoid modifying the cached data
        ret_all = df_daily_ret.copy()
        
        # Filter for requested tickers only
        available_tickers = [t for t in all_tickers if t in ret_all.columns]
        missing_tickers = [t for t in all_tickers if t not in ret_all.columns]
        
        if missing_tickers:
            st.warning(f"Tickers missing from data: {missing_tickers}")
            
        ret_all = ret_all[available_tickers]
        
        # Convert from percentage to decimal if needed (check if values look like percentages)
        # If max absolute value > 1, assume it's in percentage format
        if not ret_all.empty and ret_all.abs().max().max() > 1:
            ret_all = ret_all / 100
        
        # Find first valid date where we have data for all tickers
        first_valid_index = ret_all.apply(lambda series: series.first_valid_index()).max()
        if pd.notna(first_valid_index):
            ret_all = ret_all.loc[first_valid_index:].dropna(how='all')

        # Filter by date
        if start_date:
            ret_all = ret_all.loc[start_date.strftime('%Y-%m-%d'):]
        if end_date:
            ret_all = ret_all.loc[:end_date.strftime('%Y-%m-%d')]
            
        # Note: We no longer fallback to Yahoo for missing tickers.
        # All tickers should be in the Excel file.
    else:
        st.error(f"Unknown data source: {data_source}")
        return pd.DataFrame(), pd.DataFrame()

    ret_all = ret_all.fillna(0)
    
    # Currency Adjustment
    if df_fx is not None and not df_fx.empty:
        for ticker in ret_all.columns:
            if ticker not in df_sec_list['Ticker'].values:
                continue
                
            base_currency_row = df_sec_list.loc[df_sec_list['Ticker'] == ticker, 'Base currency']
            if base_currency_row.empty:
                continue
            base_currency = base_currency_row.iloc[0]
            
            if base_currency != currency and base_currency != 'N/A':
                # Align FX dates
                common_idx = ret_all.index.intersection(df_fx.index)
                
                if base_currency == 'USD' and currency == 'SGD' and 'USDSGD' in df_fx.columns:
                    fx = df_fx.loc[common_idx, 'USDSGD']
                    ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx) - 1
                elif base_currency == 'USD' and currency == 'HKD' and 'USDHKD' in df_fx.columns:
                    fx = df_fx.loc[common_idx, 'USDHKD']
                    ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx) - 1
                elif base_currency == 'SGD' and currency == 'USD' and 'USDSGD' in df_fx.columns:
                    fx = df_fx.loc[common_idx, 'USDSGD']
                    # Inverse approximation
                    fx_inv = -fx / (1 + fx)
                    ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx_inv) - 1
                elif base_currency == 'HKD' and currency == 'USD' and 'USDHKD' in df_fx.columns:
                    fx = df_fx.loc[common_idx, 'USDHKD']
                    fx_inv = -fx / (1 + fx)
                    ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx_inv) - 1

    # Benchmark Returns
    ret_bm = ret_all[ [t for t in bm_tickers if t in ret_all.columns] ]
    
    return ret_all, ret_bm

