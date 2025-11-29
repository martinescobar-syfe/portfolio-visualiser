
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

def get_exposure_charts(weights, df_sector, df_country):
    """
    Returns a dictionary of charts and tables for Sector and Country exposure.
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
    
    return charts
