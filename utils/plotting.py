
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from styles.syfe_colors import PALETTE, HEATMAP_COLORS

def apply_syfe_layout(fig, title="", height=500):
    fig.update_layout(
        title={'text': f'<b>{title}</b>', 'font': {'size': 20, 'color': '#172B4D'}},
        font={'family': 'Helvetica Neue, Arial', 'color': '#172B4D'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        xaxis={'showgrid': True, 'gridcolor': '#F4F5F7'},
        yaxis={'showgrid': True, 'gridcolor': '#F4F5F7'},
        colorway=PALETTE
    )
    return fig

def plot_cumulative_returns(df_cum_ret, benchmark_ret=None):
    fig = go.Figure()
    for i, col in enumerate(df_cum_ret.columns):
        fig.add_trace(go.Scatter(
            x=df_cum_ret.index, 
            y=df_cum_ret[col], 
            mode='lines', 
            name=col,
            line=dict(width=2, color=PALETTE[i % len(PALETTE)])
        ))
    
    if benchmark_ret is not None:
        # Benchmark Colors (Dark/Grey scales to contrast with colorful portfolios)
        BM_COLORS = ['#000000', '#525252', '#969696', '#252525']
        
        if isinstance(benchmark_ret, (pd.Series, pd.DataFrame)):
            bm_cum = (1 + benchmark_ret).cumprod()
            if isinstance(bm_cum, pd.DataFrame):
                for i, col in enumerate(bm_cum.columns):
                    fig.add_trace(go.Scatter(
                        x=bm_cum.index,
                        y=bm_cum[col],
                        mode='lines',
                        name=f"{col} (BM)",
                        line=dict(width=2, color=BM_COLORS[i % len(BM_COLORS)], dash='dash')
                    ))
            else:
                 fig.add_trace(go.Scatter(
                    x=bm_cum.index,
                    y=bm_cum,
                    mode='lines',
                    name="Benchmark",
                    line=dict(width=2, color=BM_COLORS[0], dash='dash')
                ))

    fig = apply_syfe_layout(fig, title="Cumulative Returns")
    fig.update_yaxes(tickformat=".2%")
    return fig

def plot_drawdowns(df_ret):
    cum_ret = (1 + df_ret).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    
    fig = go.Figure()
    for i, col in enumerate(drawdown.columns):
        fig.add_trace(go.Scatter(
            x=drawdown.index, 
            y=drawdown[col], 
            mode='lines', 
            name=col,
            fill='tozeroy',
            line=dict(width=1, color=PALETTE[i % len(PALETTE)])
        ))
        
    fig = apply_syfe_layout(fig, title="Drawdowns")
    fig.update_yaxes(tickformat=".2%")
    return fig

def plot_correlation_heatmap(df_ret):
    corr = df_ret.corr()
    
    fig = px.imshow(
        corr, 
        text_auto=".2f", 
        aspect="auto", 
        color_continuous_scale=HEATMAP_COLORS,
        zmin=-1, 
        zmax=1
    )
    fig = apply_syfe_layout(fig, title="Correlation Matrix")
    return fig

def plot_scatter_risk_return(df_ret, annualized=261, benchmark_ret=None):
    summary = pd.DataFrame()
    summary['Return'] = df_ret.mean() * annualized
    summary['Risk'] = df_ret.std() * np.sqrt(annualized)
    summary['Sharpe'] = summary['Return'] / summary['Risk']
    summary['Type'] = 'Portfolio'
    
    if benchmark_ret is not None:
        if isinstance(benchmark_ret, pd.Series):
            benchmark_ret = benchmark_ret.to_frame()
            
        bm_summary = pd.DataFrame()
        bm_summary['Return'] = benchmark_ret.mean() * annualized
        bm_summary['Risk'] = benchmark_ret.std() * np.sqrt(annualized)
        bm_summary['Sharpe'] = bm_summary['Return'] / bm_summary['Risk']
        bm_summary['Type'] = 'Benchmark'
        
        summary = pd.concat([summary, bm_summary])
    
    fig = px.scatter(
        summary, 
        x='Risk', 
        y='Return', 
        text=summary.index, 
        size=[15]*len(summary),
        color=summary.index,
        color_discrete_sequence=PALETTE if benchmark_ret is None else PALETTE + ['#000000'],
        symbol='Type' if benchmark_ret is not None else None
    )
    
    fig.update_traces(textposition='top center')
    fig = apply_syfe_layout(fig, title="Risk vs Return")
    fig.update_xaxes(tickformat=".2%", title="Annualized Risk (Vol)")
    fig.update_yaxes(tickformat=".2%", title="Annualized Return")
    return fig

def plot_allocation_pie(weights, title="Portfolio Allocation"):
    # Filter non-zero weights
    weights = weights[weights > 0.001]
    
    fig = go.Figure(data=[go.Pie(
        labels=weights.index, 
        values=weights.values, 
        hole=.4,
        marker=dict(colors=PALETTE),
        textinfo='none', # No text on chart
        hoverinfo='label+percent+value'
    )])
    
    fig = apply_syfe_layout(fig, title=title)
    return fig

def plot_bar_chart(df, title="", barmode='group'):
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Bar(
            x=df.index,
            y=df[col],
            name=col,
            marker_color=PALETTE[i % len(PALETTE)]
        ))
    
    fig.update_layout(barmode=barmode)
    fig = apply_syfe_layout(fig, title=title)
    fig.update_yaxes(tickformat=".2%")
    return fig

def plot_rolling_volatility(df_ret, window=126, annualized=261):
    rolling_vol = df_ret.rolling(window=window).std() * np.sqrt(annualized)
    
    fig = go.Figure()
    for i, col in enumerate(rolling_vol.columns):
        fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol[col],
            mode='lines',
            name=col,
            line=dict(width=2, color=PALETTE[i % len(PALETTE)])
        ))
        
    fig = apply_syfe_layout(fig, title=f"Rolling {window}-Day Volatility (Annualized)")
    fig.update_yaxes(tickformat=".2%")
    return fig

def plot_monthly_heatmap(df_ret):
    # Resample to monthly returns
    monthly_ret = df_ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_ret['Year'] = monthly_ret.index.year
    monthly_ret['Month'] = monthly_ret.index.month_name()
    
    # Pivot for heatmap
    # Taking the first column (assuming single portfolio passed or handle multiple)
    # If multiple, we need a dropdown or separate plots. 
    # For now, assume df_ret is single series or we take first col
    
    col_name = df_ret.columns[0]
    pivot_table = monthly_ret.pivot(index='Year', columns='Month', values=col_name)
    
    # Reorder months
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    pivot_table = pivot_table.reindex(columns=months_order)
    
    fig = px.imshow(
        pivot_table,
        text_auto=".2%",
        aspect="auto",
        color_continuous_scale=HEATMAP_COLORS,
        zmin=max(-0.1, pivot_table.min().min()),
        zmax=min(0.1, pivot_table.max().max())
    )
    fig = apply_syfe_layout(fig, title=f"Monthly Returns ({col_name})")
    return fig
