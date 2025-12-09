
import pandas as pd
import plotly.graph_objects as go
from utils.plotting import plot_bar_chart, plot_allocation_pie

def calculate_exposure(weights, df_exposure):
    """
    Calculates aggregate exposure by category (Sector, Country) using matrix multiplication.
    weights: Series with ticker index and weight values
    df_exposure: DataFrame with Category index and Ticker columns (Exposure Matrix)
    """
    # Ensure weights is a Series
    if isinstance(weights, pd.DataFrame):
        weights = weights.iloc[-1] # Take latest weights if DataFrame
    
    # Strip _wgt suffix and filter out non-ticker columns
    # The weights index has format like 'SPY_wgt', 'ACWI_wgt', 'reb_flag'
    ticker_weights = {}
    for col in weights.index:
        if col.endswith('_wgt'):
            ticker = col.replace('_wgt', '')
            ticker_weights[ticker] = weights[col]
    
    # Convert to Series with clean ticker names
    weights = pd.Series(ticker_weights)
    
    if len(weights) == 0:
        return pd.Series()
    
    # Align tickers
    common_tickers = weights.index.intersection(df_exposure.columns)
    
    if len(common_tickers) == 0:
        return pd.Series()

    # Filter and align
    weights_aligned = weights[common_tickers]
    exposure_matrix = df_exposure[common_tickers]
    
    # Calculate weighted exposure: (Categories x Tickers) @ (Tickers x 1) -> (Categories x 1)
    # We use dot product
    agg_exposure = exposure_matrix.dot(weights_aligned)
    
    # Sort by weight
    agg_exposure = agg_exposure.sort_values(ascending=False)
    
    return agg_exposure

def group_tail(series, top_n=9):
    """
    Groups the tail of the series into an 'Others' category.
    """
    if len(series) <= top_n:
        return series
    
    top = series.head(top_n)
    others_val = series.iloc[top_n:].sum()
    
    # Create a new series for Others
    others = pd.Series({'Others': others_val})
    
    return pd.concat([top, others])

def clean_holding_name(name):
    """
    Cleans holding name to unify share classes (e.g. GOOG vs GOOGL).
    This is a heuristic approach.
    """
    if not isinstance(name, str):
        return str(name)
        
    name_upper = name.upper()
    
    # Common suffixes to remove/ignore for aggregation
    starts_with_to_keep = name_upper
    
    # Dict of common mappings manually if needed, or algorithmic
    # For now, let's strip common suffixes
    suffixes = [' CLASS A', ' CLASS B', ' CLASS C', ' CL A', ' CL B', ' CL C', 
                ' INC', ' CORP', ' LTD', ' PLC', ' NV', ' AG', ' SA', ' SE', 
                ' CO', ' HLDG', ' HOLDINGS', ' GROUP', ' GRP']
    
    for suffix in suffixes:
        if name_upper.endswith(suffix):
            name_upper = name_upper[:-len(suffix)].strip()
            
    return name_upper

def get_concentration_exposure(weights, df_underlying, top_n=10):
    """
    Calculates weighted underlying exposure.
    Unifies share classes.
    Handles 'Wide' format df_underlying: {Ticker} Holdings, {Ticker} Holding Weights
    """
    aggregated_holdings = {}
    
    # 1. Iterate through portfolio tickers
    # Weights index is like 'CSPX_wgt'
    for col_name in weights.index:
        if not col_name.endswith('_wgt'):
            continue
            
        ticker = col_name.replace('_wgt', '')
        weight_in_port = weights[col_name]
        
        if weight_in_port == 0:
            continue
            
        # Ticker Matching Logic
        # User upload might have "SPY US", "SPY Equity" etc.
        # underlying sheet has "SPY Holdings"
        
        candidates = [ticker]
        # remove common suffixes
        clean_t = ticker.replace(" US", "").replace(" Equity", "").strip()
        if clean_t != ticker:
            candidates.append(clean_t)
            
        found_cols = False
        name_col = ""
        w_col = ""
        use_fallback_weights = False
        
        for cand in candidates:
            n_c = f"{cand} Holdings"
            w_c = f"{cand} Holding Weights"
            if n_c in df_underlying.columns and w_c in df_underlying.columns:
                name_col = n_c
                w_col = w_c
                found_cols = True
                break
            # Fallback: If Holdings exists but Weights doesn't, use 100% weight
            elif n_c in df_underlying.columns:
                name_col = n_c
                w_col = None
                found_cols = True
                use_fallback_weights = True
                break
        
        if not found_cols:
            # Try fuzzy or logging
            # print(f"[DEBUG] Missing underlying data for {ticker}. Tried: {[f'{c} Holdings' for c in candidates]}")
            continue
            
        # Extract data
        # Drop entry rows (NaN names)
        # Note: df_underlying might be jagged
        if use_fallback_weights:
            # Only holdings column, no weights - assume 100% to single holding
            sub_df = df_underlying[[name_col]].dropna(subset=[name_col])
            for _, row in sub_df.iterrows():
                h_name = row[name_col]
                clean_name = clean_holding_name(h_name)
                # Assume 100% of the portfolio weight goes to this holding
                contribution = weight_in_port
                aggregated_holdings[clean_name] = aggregated_holdings.get(clean_name, 0.0) + contribution
        else:
            sub_df = df_underlying[[name_col, w_col]].dropna(subset=[name_col])
            # Process
            for _, row in sub_df.iterrows():
                h_name = row[name_col]
                h_val = row[w_col]
                
                # Clean name
                clean_name = clean_holding_name(h_name)
                
                # Calculate net weight
                # Underlying weights might be in % (0-100) or decimal (0-1).
                # If 7.5 -> 7.5%, then (7.5/100) * port_weight
                # BUT we need to be careful if data is already decimal.
                # Usually vendors provide %. Let's assume % for now as per previous observation (Nvidia 7.63).
                contribution = (h_val / 100.0) * weight_in_port
                
                aggregated_holdings[clean_name] = aggregated_holdings.get(clean_name, 0.0) + contribution

    if not aggregated_holdings:
        return pd.Series()
    
    # Convert to Series
    grouped_exposure = pd.Series(aggregated_holdings)
    
    # 3. Sort descending
    grouped_exposure = grouped_exposure.sort_values(ascending=False)
    
    return grouped_exposure

def get_exposure_charts(weights, df_sector, df_country, df_asset_class=None, df_underlying=None, concentration_top_n=10):
    """
    Returns a dictionary of charts and tables for Sector, Country, Asset Class, AND Concentration.
    """
    charts = {}
    
    # Sector Exposure
    sector_exp = calculate_exposure(weights, df_sector)
    sector_exp = group_tail(sector_exp, top_n=9)
    
    charts['sector_table'] = sector_exp.to_frame(name='Weight')
    charts['sector_bar'] = plot_bar_chart(sector_exp.to_frame(name='Weight').T, title="Sector Exposure", barmode='relative')
    charts['sector_pie'] = plot_allocation_pie(sector_exp, title="Sector Allocation")
    
    # Country Exposure
    country_exp = calculate_exposure(weights, df_country)
    country_exp = group_tail(country_exp, top_n=9)
    
    charts['country_table'] = country_exp.to_frame(name='Weight')
    charts['country_bar'] = plot_bar_chart(country_exp.to_frame(name='Weight').T, title="Country Exposure", barmode='relative')
    charts['country_pie'] = plot_allocation_pie(country_exp, title="Country Allocation")
    
    # Asset Class Exposure (if df_asset_class provided)
    if df_asset_class is not None:
        asset_exp = calculate_exposure(weights, df_asset_class)
        asset_exp = group_tail(asset_exp, top_n=9)
        
        charts['asset_table'] = asset_exp.to_frame(name='Weight')
        charts['asset_bar'] = plot_bar_chart(asset_exp.to_frame(name='Weight').T, title="Asset Class Exposure", barmode='relative')
        charts['asset_pie'] = plot_allocation_pie(asset_exp, title="Asset Class Allocation")

    # Concentration Exposure (if df_underlying provided)
    if df_underlying is not None:
        conc_exp = get_concentration_exposure(weights, df_underlying, top_n=concentration_top_n)
        # For concentration, we usually just show the Top N, and maybe 'Others' if requested, 
        # but typically Top N table is the key requirement.
        
        # Taking top N for display
        conc_top = group_tail(conc_exp, top_n=concentration_top_n)
        
        charts['concentration_table'] = conc_top.to_frame(name='Weight')
        # Maybe a bar chart for concentration too?
        charts['concentration_bar'] = plot_bar_chart(conc_top.to_frame(name='Weight').T, title=f"Top {concentration_top_n} Holdings", barmode='relative')
        
    return charts
