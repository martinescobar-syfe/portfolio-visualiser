
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
        
        # load all sheets at once
        xls = pd.read_excel(excel_path, sheet_name=None)
        
        def get_sheet(name, index_col=0):
            if name in xls:
                df = xls[name]
                if index_col is not None and index_col < len(df.columns):
                     df = df.set_index(df.columns[index_col])
                return df
            return pd.DataFrame()

        df_daily_ret = get_sheet('Daily Return', index_col=0)
        if not df_daily_ret.empty:
            df_daily_ret.index = pd.to_datetime(df_daily_ret.index)
        
        df_country = get_sheet('Country', index_col=0)
        df_sector = get_sheet('Sector', index_col=0)
        df_fi_metrics = get_sheet('FI Metrics', index_col=0)
        df_asset_class = get_sheet('Asset Class', index_col=0)
        df_risk_rating = get_sheet('Risk', index_col=0)
        df_underlying = get_sheet('Underlying', index_col=None)

        # build sec_list from asset class sheet
        if 'Asset Class' in xls:
            raw_ac = xls['Asset Class']
            ac_col = next((c for c in raw_ac.columns if c.lower() == 'asset class'), None)
            
            if ac_col and ('SPY' in raw_ac.columns or 'ACWI' in raw_ac.columns):
                # matrix format: rows=classes, cols=tickers
                ac_names = raw_ac[ac_col]
                tickers_df = raw_ac.drop(columns=[ac_col])
                t_df = tickers_df.T 
                t_df = t_df.apply(pd.to_numeric, errors='coerce').fillna(0)
                class_indices = t_df.idxmax(axis=1)
                mapped_classes = ac_names.loc[class_indices].values
                
                df_sec_list = pd.DataFrame({
                    'Ticker': t_df.index,
                    'Asset Class': mapped_classes
                })
                # Ensure Ticker is string
                df_sec_list['Ticker'] = df_sec_list['Ticker'].astype(str)
            else:
                 df_sec_list = raw_ac.copy()
        else:
            df_sec_list = pd.DataFrame(columns=['Ticker', 'Name', 'Base currency', 'Exchange', 'Remarks'])
            
        # ensure required columns exist
        for col in ['Name', 'Base currency', 'Exchange', 'Remarks']:
            if col not in df_sec_list.columns:
                df_sec_list[col] = 'N/A'
                
        if 'Ticker' not in df_sec_list.columns:
             if df_sec_list.index.name == 'Ticker':
                 df_sec_list = df_sec_list.reset_index()
             else:
                 df_sec_list = df_sec_list.reset_index().rename(columns={'index': 'Ticker'})

        # FX data
        fx_cols = ['USDSGD', 'USDHKD']
        if all(col in df_daily_ret.columns for col in fx_cols):
            df_fx = df_daily_ret[fx_cols] / 100
        else:
             # fallback to static rates
             df_fx = pd.DataFrame(index=df_daily_ret.index)
             df_fx['USDSGD'] = 1.35
             df_fx['USDHKD'] = 7.8
             
        return df_sec_list, df_asset_class, df_sector, df_country, df_fi_metrics, df_fx, df_daily_ret, df_risk_rating, df_underlying

    except Exception as e:
        st.error(f"Error loading static data from Excel: {e}")
        return None, None, None, None, None, None, None, None, None

def get_data_path():
    # colab vs local
    if os.path.exists('/content/Projects/Data/'):
        return '/content/Projects/Data/'
    elif os.path.exists('Data/'):
        return 'Data/'
    else:
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
    """load return data for tickers from yahoo or our excel data"""
    if data_source == 'Yahoo Finance':
        modified_tickers = modify_tickers_based_on_exchange(all_tickers, df_sec_list)
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
        # use pre-loaded excel data
        if df_daily_ret is None or df_daily_ret.empty:
            st.error("Daily return data not available. Please check the Excel file.")
            return pd.DataFrame(), pd.DataFrame()
        
        ret_all = df_daily_ret.copy()
        available_tickers = [t for t in all_tickers if t in ret_all.columns]
        missing_tickers = [t for t in all_tickers if t not in ret_all.columns]
        
        if missing_tickers:
            st.warning(f"Tickers missing from data: {missing_tickers}")
            
        ret_all = ret_all[available_tickers]
        
        # convert % to decimal if needed
        if not ret_all.empty and ret_all.abs().max().max() > 1:
            ret_all = ret_all / 100
        
        first_valid_index = ret_all.apply(lambda series: series.first_valid_index()).max()
        if pd.notna(first_valid_index):
            ret_all = ret_all.loc[first_valid_index:].dropna(how='all')
        if start_date:
            ret_all = ret_all.loc[start_date.strftime('%Y-%m-%d'):]
        if end_date:
            ret_all = ret_all.loc[:end_date.strftime('%Y-%m-%d')]
    else:
        st.error(f"Unknown data source: {data_source}")
        return pd.DataFrame(), pd.DataFrame()

    ret_all = ret_all.fillna(0)
    
    # fx adjustment
    if df_fx is not None and not df_fx.empty:
        for ticker in ret_all.columns:
            if ticker not in df_sec_list['Ticker'].values:
                continue
                
            base_currency_row = df_sec_list.loc[df_sec_list['Ticker'] == ticker, 'Base currency']
            if base_currency_row.empty:
                continue
            base_currency = base_currency_row.iloc[0]
            
            if base_currency != currency and base_currency != 'N/A':
                common_idx = ret_all.index.intersection(df_fx.index)
                
                if base_currency == 'USD' and currency == 'SGD' and 'USDSGD' in df_fx.columns:
                    fx = df_fx.loc[common_idx, 'USDSGD']
                    ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx) - 1
                elif base_currency == 'USD' and currency == 'HKD' and 'USDHKD' in df_fx.columns:
                    fx = df_fx.loc[common_idx, 'USDHKD']
                    ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx) - 1
                elif base_currency == 'SGD' and currency == 'USD' and 'USDSGD' in df_fx.columns:
                    fx = df_fx.loc[common_idx, 'USDSGD']
                    fx_inv = -fx / (1 + fx)
                    ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx_inv) - 1
                elif base_currency == 'HKD' and currency == 'USD' and 'USDHKD' in df_fx.columns:
                    fx = df_fx.loc[common_idx, 'USDHKD']
                    fx_inv = -fx / (1 + fx)
                    ret_all.loc[common_idx, ticker] = (1 + ret_all.loc[common_idx, ticker]) * (1 + fx_inv) - 1

    ret_bm = ret_all[ [t for t in bm_tickers if t in ret_all.columns] ]
    
    return ret_all, ret_bm

