
import pandas as pd
import plotly.graph_objects as go
from utils.plotting import plot_bar_chart, plot_allocation_pie, plot_horizontal_bar_chart

def calculate_exposure(weights, df_exposure):
    """calc aggregate exposure by sector/country via matrix mult"""
    # ensure weights is a Series
    if isinstance(weights, pd.DataFrame):
        weights = weights.iloc[-1] # Take latest weights if DataFrame
    
    # strip _wgt suffix, filter out non-ticker cols
    ticker_weights = {}
    for col in weights.index:
        if col.endswith('_wgt'):
            ticker = col.replace('_wgt', '')
            # ensure weight is numeric
            try:
                w = float(weights[col])
                ticker_weights[ticker] = w
            except (ValueError, TypeError):
                continue
    
    weights = pd.Series(ticker_weights)
    
    if len(weights) == 0:
        return pd.Series()
    
    # find missing tickers
    all_tickers = set(weights.index)
    known_tickers = set(df_exposure.columns)
    missing_tickers = all_tickers - known_tickers
    common_tickers = weights.index.intersection(df_exposure.columns)
    agg_exposure = pd.Series(dtype=float)
    
    if len(common_tickers) > 0:
        weights_aligned = weights[common_tickers]
        exposure_matrix = df_exposure[common_tickers]
        exposure_matrix = exposure_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)
        # weighted exposure
        agg_exposure = exposure_matrix.dot(weights_aligned)
    
    # handle missing tickers
    if missing_tickers:
        is_asset_class = 'Specialized / alternatives' in df_exposure.index or 'Equity' in df_exposure.index
        fallback_weights = {}
        
        for ticker in missing_tickers:
            weight = weights[ticker]
            ticker_lower = ticker.lower()
            
            if 'custom' in ticker_lower:
                if is_asset_class:
                    fallback_cat = 'Specialized / alternatives'
                else:
                    fallback_cat = 'Others'
            else:
                fallback_cat = 'Others'
            
            fallback_weights[fallback_cat] = fallback_weights.get(fallback_cat, 0) + weight
        
        agg_dict = agg_exposure.to_dict()
        for cat, weight in fallback_weights.items():
            agg_dict[cat] = agg_dict.get(cat, 0) + weight
        
        agg_exposure = pd.Series(agg_dict)
    
    agg_exposure = agg_exposure.sort_values(ascending=False)
    
    return agg_exposure

def group_tail(series, top_n=9):
    """group tail of series into 'Others' bucket"""
    if len(series) <= top_n:
        return series
    
    top = series.head(top_n)
    others_val = series.iloc[top_n:].sum()
    
    if 'Others' in top.index:
        top_dict = top.to_dict()
        top_dict['Others'] += others_val
        return pd.Series(top_dict)
    else:
        others = pd.Series({'Others': others_val})
        return pd.concat([top, others])

def clean_holding_name(name):
    """clean holding name to unify share classes (GOOG vs GOOGL etc)"""
    if not isinstance(name, str):
        return str(name)
        
    name_upper = name.upper()
    suffixes = [' CLASS A', ' CLASS B', ' CLASS C', ' CL A', ' CL B', ' CL C', 
                ' INC', ' CORP', ' LTD', ' PLC', ' NV', ' AG', ' SA', ' SE', 
                ' CO', ' HLDG', ' HOLDINGS', ' GROUP', ' GRP']
    
    for suffix in suffixes:
        if name_upper.endswith(suffix):
            name_upper = name_upper[:-len(suffix)].strip()
            
    return name_upper

def get_concentration_exposure(weights, df_underlying, top_n=10):
    """calc weighted underlying exposure across all holdings, unifying share classes (i.e. GOOG vs GOOGL)"""
    aggregated_holdings = {}
    
    for col_name in weights.index:
        if not col_name.endswith('_wgt'):
            continue
            
        ticker = col_name.replace('_wgt', '')
        
        try:
            weight_in_port = float(weights[col_name])
        except (ValueError, TypeError):
            continue
        
        if weight_in_port == 0:
            continue
        
        # handle different ticker formats (e.g. "SPY US", "SPY Equity")
        candidates = [ticker]
        clean_t = ticker.replace(" US", "").replace(" Equity", "").strip()
        if clean_t != ticker:
            candidates.append(clean_t)
            
        found_cols = False
        name_col = ""
        w_col = ""
        
        # look for matching columns in the underlying data
        for cand in candidates:
            name_patterns = [f"{cand} Holdings", f"{cand} Holding"]
            weight_patterns = [f"{cand} Holdings Weights", f"{cand} Holding Weights", f"{cand} Weights"]
            
            for n_pattern in name_patterns:
                if n_pattern in df_underlying.columns:
                    name_col = n_pattern
                    break
            
            for w_pattern in weight_patterns:
                if w_pattern in df_underlying.columns:
                    w_col = w_pattern
                    break
            
            if name_col and w_col:
                found_cols = True
                break
        
        if not found_cols:
            # no breakdown available - use the ticker itself
            aggregated_holdings[ticker] = aggregated_holdings.get(ticker, 0.0) + weight_in_port
            continue
        
        sub_df = df_underlying[[name_col, w_col]].dropna(subset=[name_col])
        
        # check if weights are in percentage (0-100) or decimal (0-1) format
        valid_weights = []
        for _, row in sub_df.iterrows():
            h_name = row[name_col]
            h_val = row[w_col]
            if pd.isna(h_name) or str(h_name).strip() in ['', '0']:
                continue
            try:
                h_val_float = float(h_val) if not pd.isna(h_val) else 0.0
                if h_val_float > 0:
                    valid_weights.append(h_val_float)
            except (ValueError, TypeError):
                continue
        
        # if weights sum > 1.5, they're probably in percentage format
        total_weights = sum(valid_weights)
        is_percentage_format = total_weights > 1.5
        
        for _, row in sub_df.iterrows():
            h_name = row[name_col]
            h_val = row[w_col]
            
            if pd.isna(h_name) or str(h_name).strip() in ['', '0']:
                continue
            
            try:
                h_val_float = float(h_val) if not pd.isna(h_val) else 0.0
            except (ValueError, TypeError):
                continue
            
            if h_val_float <= 0:
                continue
            
            clean_name = clean_holding_name(h_name)
            
            # convert to decimal if needed
            if is_percentage_format:
                contribution = (h_val_float / 100.0) * weight_in_port
            else:
                contribution = h_val_float * weight_in_port
            
            aggregated_holdings[clean_name] = aggregated_holdings.get(clean_name, 0.0) + contribution

    if not aggregated_holdings:
        return pd.Series()
    
    grouped_exposure = pd.Series(aggregated_holdings)
    grouped_exposure = grouped_exposure.sort_values(ascending=False)
    
    return grouped_exposure

def get_exposure_charts(weights, df_sector, df_country, df_asset_class=None, df_underlying=None, concentration_top_n=10):
    """returns dict of charts/tables for sector, country, asset class, concentration"""
    charts = {}
    
    # sector
    sector_exp = calculate_exposure(weights, df_sector)
    sector_exp = group_tail(sector_exp, top_n=9)
    
    charts['sector_table'] = sector_exp.to_frame(name='Weight')
    charts['sector_bar'] = plot_bar_chart(sector_exp.to_frame(name='Weight').T, title="Sector Exposure", barmode='relative')
    charts['sector_pie'] = plot_allocation_pie(sector_exp, title="Sector Allocation")
    
    # country
    country_exp = calculate_exposure(weights, df_country)
    country_exp = group_tail(country_exp, top_n=9)
    
    charts['country_table'] = country_exp.to_frame(name='Weight')
    charts['country_bar'] = plot_bar_chart(country_exp.to_frame(name='Weight').T, title="Country Exposure", barmode='relative')
    charts['country_pie'] = plot_allocation_pie(country_exp, title="Country Allocation")
    
    # asset class
    if df_asset_class is not None:
        asset_exp = calculate_exposure(weights, df_asset_class)
        asset_exp = group_tail(asset_exp, top_n=9)
        
        charts['asset_table'] = asset_exp.to_frame(name='Weight')
        charts['asset_bar'] = plot_bar_chart(asset_exp.to_frame(name='Weight').T, title="Asset Class Exposure", barmode='relative')
        charts['asset_pie'] = plot_allocation_pie(asset_exp, title="Asset Class Allocation")

    # concentration (Top Holdings) - use horizontal bar chart with color gradient
    if df_underlying is not None:
        conc_exp = get_concentration_exposure(weights, df_underlying, top_n=concentration_top_n)
        conc_top = group_tail(conc_exp, top_n=concentration_top_n)
        total_concentration = conc_top.sum()
        charts['concentration_table'] = conc_top.to_frame(name='Weight')
        charts['concentration_bar'] = plot_horizontal_bar_chart(
            conc_top, 
            title=f"Top {concentration_top_n} Holdings ({total_concentration:.1%} Total)"
        )
        
    return charts
