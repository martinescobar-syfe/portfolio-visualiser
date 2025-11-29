
import pandas as pd
import numpy as np
import os
import yfinance as yf
import streamlit as st
from datetime import datetime

@st.cache_data
def load_static_data(data_path):
    try:
        df_sec_list = pd.read_csv(os.path.join(data_path, 'PV_sec_list.csv'))
        df_sec_list['Remarks'] = df_sec_list['Remarks'].fillna('N/A')
        
        df_asset_class = pd.read_csv(os.path.join(data_path, 'PV_asset_class.csv'), index_col=0)
        df_sector = pd.read_csv(os.path.join(data_path, 'PV_sector.csv'), index_col=0)
        df_country = pd.read_csv(os.path.join(data_path, 'PV_country.csv'), index_col=0)
        df_fi_metrics = pd.read_csv(os.path.join(data_path, 'PV_FI_metrics.csv'), index_col=0)
        
        # Load FX data
        df_fx = pd.read_csv(os.path.join(data_path, 'PV_daily_ret.csv'), header=0, index_col='Date', parse_dates=['Date'], dayfirst=True)[['USDSGD', 'USDHKD']] / 100
        
        return df_sec_list, df_asset_class, df_sector, df_country, df_fi_metrics, df_fx
    except Exception as e:
        st.error(f"Error loading static data: {e}")
        return None, None, None, None, None, None

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

def download_data(data_source, start_date, end_date, all_tickers, bm_tickers, currency, df_sec_list, df_fx, data_path):
    # Load FX data if not passed
    if df_fx is None:
         df_fx = pd.read_csv(os.path.join(data_path, 'PV_daily_ret.csv'), header=0, index_col='Date', parse_dates=['Date'], dayfirst=True)[['USDSGD', 'USDHKD']] / 100

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
        # Load from local CSV
        ret_all = pd.read_csv(os.path.join(data_path, 'PV_daily_ret.csv'), header=0, index_col='Date')
        ret_all.index = pd.to_datetime(ret_all.index, dayfirst=True) # Apply dayfirst=True here
        
        # Filter for requested tickers only
        available_tickers = [t for t in all_tickers if t in ret_all.columns]
        missing_tickers = [t for t in all_tickers if t not in ret_all.columns]
        
        if missing_tickers:
            st.warning(f"Tickers missing from Bloomberg data: {missing_tickers}")
            
        ret_all = ret_all[available_tickers] / 100
        
        first_valid_index = ret_all.apply(lambda series: series.first_valid_index()).max()
        if pd.notna(first_valid_index):
            ret_all = ret_all.loc[first_valid_index:].dropna(how='all')

        # Filter by date
        if start_date:
            ret_all = ret_all.loc[start_date.strftime('%Y-%m-%d'):]
        if end_date:
            ret_all = ret_all.loc[:end_date.strftime('%Y-%m-%d')]
            
        # Fallback to Yahoo Finance for missing tickers
        if missing_tickers:
            st.warning(f"Fetching missing tickers from Yahoo Finance: {missing_tickers}")
            try:
                modified_missing = modify_tickers_based_on_exchange(missing_tickers, df_sec_list)
                
                # Fetch from Yahoo
                start_str = start_date.strftime('%Y-%m-%d') if start_date else None
                end_str = end_date.strftime('%Y-%m-%d') if end_date else None
                
                if start_str is None and end_str is None:
                    df_missing = yf.download(modified_missing, period='max')['Adj Close']
                else:
                    df_missing = yf.download(modified_missing, start=start_str, end=end_str)['Adj Close']
                
                if not df_missing.empty:
                    df_missing.index = df_missing.index.tz_localize(None)
                    ret_missing = df_missing.pct_change().dropna()
                    
                    # Handle Series vs DataFrame
                    if isinstance(ret_missing, pd.Series):
                        ret_missing = ret_missing.to_frame()
                        if len(missing_tickers) == 1:
                            ret_missing.columns = missing_tickers
                    else:
                        # Map columns back if needed (yf might return modified tickers)
                        # This is a simplification; ideally we map back exactly
                        if len(ret_missing.columns) == len(missing_tickers):
                             ret_missing.columns = missing_tickers
                    
                    # Merge with ret_all
                    # Align indices
                    ret_all = ret_all.join(ret_missing, how='outer')
                    
            except Exception as e:
                st.error(f"Error fetching missing tickers from Yahoo Finance: {e}")

    ret_all = ret_all.fillna(0)
    
    # Currency Adjustment
    for ticker in ret_all.columns:
        if ticker not in df_sec_list['Ticker'].values:
            continue
            
        base_currency = df_sec_list.loc[df_sec_list['Ticker'] == ticker, 'Base currency'].iloc[0]
        if base_currency != currency:
            # Align FX dates
            common_idx = ret_all.index.intersection(df_fx.index)
            
            if base_currency == 'USD' and currency == 'SGD':
                fx = df_fx.loc[common_idx, 'USDSGD']
                ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx) - 1
            elif base_currency == 'USD' and currency == 'HKD':
                fx = df_fx.loc[common_idx, 'USDHKD']
                ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx) - 1
            elif base_currency == 'SGD' and currency == 'USD':
                fx = df_fx.loc[common_idx, 'USDSGD']
                # Inverse approximation
                fx_inv = -fx / (1 + fx)
                ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx_inv) - 1
            elif base_currency == 'HKD' and currency == 'USD':
                fx = df_fx.loc[common_idx, 'USDHKD']
                fx_inv = -fx / (1 + fx)
                ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx_inv) - 1

    # Benchmark Returns
    ret_bm = ret_all[ [t for t in bm_tickers if t in ret_all.columns] ]
    
    return ret_all, ret_bm
