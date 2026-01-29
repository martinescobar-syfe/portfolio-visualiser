
import pandas as pd
import numpy as np
from utils.plotting import plot_cumulative_returns, plot_rolling_volatility, plot_drawdowns

def calculate_metrics(df_ret, annualized=261):
    """calc performance metrics for return series"""
    metrics = {}
    
    metrics['Cumulative Return'] = (1 + df_ret).prod() - 1
    n_years = len(df_ret) / annualized
    metrics['Annualized Return'] = (1 + metrics['Cumulative Return']) ** (1 / n_years) - 1
    metrics['Annualized Volatility'] = df_ret.std() * np.sqrt(annualized)
    metrics['Sharpe Ratio'] = metrics['Annualized Return'] / metrics['Annualized Volatility']
    
    downside_ret = df_ret[df_ret < 0]
    downside_vol = downside_ret.std() * np.sqrt(annualized)
    metrics['Sortino Ratio'] = metrics['Annualized Return'] / downside_vol
    cum_ret = (1 + df_ret).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    metrics['Max Drawdown'] = drawdown.min()
    
    return pd.Series(metrics)

PERCENTAGE_METRICS = ['Cumulative Return', 'Annualized Return', 'Annualized Volatility', 'Max Drawdown', 'Tracking Error']
RATIO_METRICS = ['Sharpe Ratio', 'Sortino Ratio', 'Information Ratio']

def format_metric_value(value, metric_name):
    """format metric value for display"""
    if pd.isna(value):
        return "N/A"
    
    if metric_name in PERCENTAGE_METRICS:
        return f"{value:.2%}"  # e.g., 0.0785 -> "7.85%"
    elif metric_name in RATIO_METRICS:
        return f"{value:.2f}"  # e.g., 1.234 -> "1.23"
    else:
        return f"{value:.2f}"

def calculate_relative_metrics(port_ret, bm_ret, annualized=261):
    """calc TE, IR and other relative metrics"""
    metrics = {}
    common_idx = port_ret.index.intersection(bm_ret.index)
    port_ret = port_ret.loc[common_idx]
    bm_ret = bm_ret.loc[common_idx]
    
    excess_ret = port_ret - bm_ret
    metrics['Tracking Error'] = excess_ret.std() * np.sqrt(annualized)
    metrics['Information Ratio'] = (excess_ret.mean() * annualized) / metrics['Tracking Error']
    rolling_excess = excess_ret.rolling(window=annualized).sum()
    
    return pd.Series(metrics), rolling_excess

def get_performance_summary(all_port_ret, bm_ret=None):
    """summary df of metrics for all portfolios"""
    summary_list = []
    
    if all_port_ret.empty or len(all_port_ret.columns) == 0:
        return pd.DataFrame()
    
    for col in all_port_ret.columns:
        metrics = calculate_metrics(all_port_ret[col])
        summary_list.append(metrics)
        
    summary_df = pd.concat(summary_list, axis=1)
    summary_df.columns = all_port_ret.columns
    
    if bm_ret is not None:
        pass
        
    return summary_df
