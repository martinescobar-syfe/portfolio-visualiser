
import pandas as pd
import numpy as np
from utils.plotting import plot_cumulative_returns, plot_rolling_volatility, plot_drawdowns

def calculate_metrics(df_ret, annualized=261):
    """
    Calculates performance metrics for a return series.
    Returns metrics as decimals (e.g., 0.0785 for 7.85%).
    
    Metrics returned:
    - Cumulative Return, Annualized Return, Annualized Volatility, Max Drawdown: decimals (multiply by 100 for %)
    - Sharpe Ratio, Sortino Ratio: ratios (no conversion needed)
    """
    metrics = {}
    
    # Cumulative Return
    metrics['Cumulative Return'] = (1 + df_ret).prod() - 1
    
    # Annualized Return
    n_years = len(df_ret) / annualized
    metrics['Annualized Return'] = (1 + metrics['Cumulative Return']) ** (1 / n_years) - 1
    
    # Annualized Volatility
    metrics['Annualized Volatility'] = df_ret.std() * np.sqrt(annualized)
    
    # Sharpe Ratio (rf=0)
    metrics['Sharpe Ratio'] = metrics['Annualized Return'] / metrics['Annualized Volatility']
    
    # Sortino Ratio (rf=0)
    downside_ret = df_ret[df_ret < 0]
    downside_vol = downside_ret.std() * np.sqrt(annualized)
    metrics['Sortino Ratio'] = metrics['Annualized Return'] / downside_vol
    
    # Max Drawdown
    cum_ret = (1 + df_ret).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    metrics['Max Drawdown'] = drawdown.min()
    
    return pd.Series(metrics)

# Define which metrics are percentages vs ratios for formatting
PERCENTAGE_METRICS = ['Cumulative Return', 'Annualized Return', 'Annualized Volatility', 'Max Drawdown', 'Tracking Error']
RATIO_METRICS = ['Sharpe Ratio', 'Sortino Ratio', 'Information Ratio']

def format_metric_value(value, metric_name):
    """
    Format a metric value for display.
    
    Args:
        value: The numeric value
        metric_name: Name of the metric to determine formatting
        
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return "N/A"
    
    if metric_name in PERCENTAGE_METRICS:
        return f"{value:.2%}"  # e.g., 0.0785 -> "7.85%"
    elif metric_name in RATIO_METRICS:
        return f"{value:.2f}"  # e.g., 1.234 -> "1.23"
    else:
        # Default to 2 decimal places
        return f"{value:.2f}"

def calculate_relative_metrics(port_ret, bm_ret, annualized=261):
    """
    Calculates relative metrics (Tracking Error, IR, etc.)
    """
    metrics = {}
    
    # Align dates
    common_idx = port_ret.index.intersection(bm_ret.index)
    port_ret = port_ret.loc[common_idx]
    bm_ret = bm_ret.loc[common_idx]
    
    excess_ret = port_ret - bm_ret
    
    # Tracking Error
    metrics['Tracking Error'] = excess_ret.std() * np.sqrt(annualized)
    
    # Information Ratio
    metrics['Information Ratio'] = (excess_ret.mean() * annualized) / metrics['Tracking Error']
    
    # Rolling 12M Excess Return
    rolling_excess = excess_ret.rolling(window=annualized).sum()
    
    return pd.Series(metrics), rolling_excess

def get_performance_summary(all_port_ret, bm_ret=None):
    """
    Generates a summary DataFrame of metrics for all portfolios.
    """
    summary_list = []
    
    if all_port_ret.empty or len(all_port_ret.columns) == 0:
        return pd.DataFrame()
    
    for col in all_port_ret.columns:
        metrics = calculate_metrics(all_port_ret[col])
        summary_list.append(metrics)
        
    summary_df = pd.concat(summary_list, axis=1)
    summary_df.columns = all_port_ret.columns
    
    if bm_ret is not None:
        # Calculate relative metrics for first portfolio vs benchmark as example
        # Or for all against benchmark
        pass
        
    return summary_df
