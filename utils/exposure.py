
import re
import pandas as pd
import plotly.graph_objects as go
from utils.plotting import plot_bar_chart, plot_allocation_pie, plot_horizontal_bar_chart


# sentinel label used when a client row is a custom DIY portfolio
CUSTOM_PORTFOLIO_LABEL = 'CUSTOM'


def parse_custom_holdings(raw):
    """Parse an `actual_holding_weightage` string into {ticker: decimal_weight}.

    The Metabase / CSV export stores custom holdings as a single comma-separated
    string like ``"SMH 33.67%, VTV 14.67%, XLE 17.24%"``. Some exports may use
    semicolons instead of commas, decimal commas (European style), or omit the
    trailing ``%``. We normalize those here.
    """
    if not isinstance(raw, str):
        return {}
    s = raw.strip()
    if not s or s.lower() in ('nan', 'none', 'null'):
        return {}

    # treat both ',' and ';' as separators; keep decimals that use '.' only
    # (the pct numbers themselves don't have commas in this dataset).
    parts = re.split(r'[;,]\s*', s)
    out = {}
    for p in parts:
        p = p.strip().strip(',').strip(';')
        if not p:
            continue
        # split from the right so tickers with spaces (rare, e.g. "SPY US") still parse
        m = re.match(r'^(?P<ticker>.+?)\s+(?P<pct>-?\d+(?:[.,]\d+)?)\s*%?\s*$', p)
        if not m:
            continue
        ticker = m.group('ticker').strip().upper()
        pct_str = m.group('pct').replace(',', '.')
        try:
            pct = float(pct_str)
        except ValueError:
            continue
        if pct <= 0:
            continue
        # string is expressed as percentages (0-100); normalize to decimal
        out[ticker] = out.get(ticker, 0.0) + pct / 100.0
    return out


def expand_custom_in_weights_dict(weights_dict, custom_etf_breakdown, custom_label=CUSTOM_PORTFOLIO_LABEL):
    """Replace the aggregate ``CUSTOM`` bucket with its constituent ETF weights.

    Both ``weights_dict`` and ``custom_etf_breakdown`` are ticker→weight dicts
    (no ``_wgt`` suffix). The breakdown can be any non-negative numbers (NAV
    dollars or fractions); it will be renormalized internally.
    """
    if not weights_dict or custom_label not in weights_dict:
        return dict(weights_dict) if weights_dict else {}
    cust_w = float(weights_dict.get(custom_label, 0) or 0)
    result = {k: v for k, v in weights_dict.items() if k != custom_label}
    if cust_w <= 0 or not custom_etf_breakdown:
        return result
    total = sum(v for v in custom_etf_breakdown.values() if v and v > 0)
    if total <= 0:
        # no usable breakdown — keep the CUSTOM bucket so downstream fallbacks
        # (specialized / alternatives) still kick in
        result[custom_label] = cust_w
        return result
    for etf, nav in custom_etf_breakdown.items():
        if not nav or nav <= 0:
            continue
        frac = nav / total
        result[etf] = result.get(etf, 0.0) + cust_w * frac
    return result


def benchmark_exposure_series(benchmark_ticker, df_exposure):
    """Return a normalized Series of a single-ticker benchmark's exposure.

    Uses the same df_sector / df_country / df_asset_class matrices as the
    client exposure calc so the comparison is like-for-like. Returns an
    empty Series if the benchmark ticker is unknown.
    """
    if not benchmark_ticker or df_exposure is None:
        return pd.Series(dtype=float)
    if benchmark_ticker not in df_exposure.columns:
        return pd.Series(dtype=float)
    col = pd.to_numeric(df_exposure[benchmark_ticker], errors='coerce').fillna(0.0)
    col = col[col > 0]
    total = col.sum()
    if total <= 0:
        return pd.Series(dtype=float)
    # always renormalise so the comparison is on the same scale regardless
    # of whether the source column was stored as percentages or decimals
    return (col / total).sort_values(ascending=False)


def expand_custom_in_weights_series(weights_series, custom_etf_breakdown, custom_label=CUSTOM_PORTFOLIO_LABEL):
    """Same as ``expand_custom_in_weights_dict`` but for a ``_wgt``-suffixed pd.Series."""
    if weights_series is None or len(weights_series) == 0:
        return weights_series
    suffix_key = f"{custom_label}_wgt"
    if suffix_key not in weights_series.index:
        return weights_series
    # unwrap suffix (suffix-safe so unusual ticker names aren't corrupted), expand, re-wrap
    raw = {k[:-4]: float(v) for k, v in weights_series.items() if isinstance(k, str) and k.endswith('_wgt')}
    expanded = expand_custom_in_weights_dict(raw, custom_etf_breakdown, custom_label=custom_label)
    return pd.Series({f"{k}_wgt": v for k, v in expanded.items()})


def calculate_exposure(weights, df_exposure):
    """calc aggregate exposure by sector/country via matrix mult"""
    # ensure weights is a Series
    if isinstance(weights, pd.DataFrame):
        weights = weights.iloc[-1] # Take latest weights if DataFrame
    
    # strip trailing _wgt suffix (suffix-safe), filter out non-ticker cols
    ticker_weights = {}
    for col in weights.index:
        if isinstance(col, str) and col.endswith('_wgt'):
            ticker = col[:-4]
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
        # some tickers (e.g. "Equity Alpha") have their country/sector columns
        # stored as percentages (sum≈100) while the rest are stored as
        # decimals (sum≈1). Normalise any percent-style column down to
        # decimal form so the weighted sum doesn't blow up 100×.
        col_sums = exposure_matrix.sum(axis=0)
        pct_cols = col_sums[col_sums > 1.5].index
        if len(pct_cols) > 0:
            exposure_matrix.loc[:, pct_cols] = exposure_matrix.loc[:, pct_cols] / 100.0
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

CATCH_ALL_HOLDINGS = {
    'OTHERS', 'OTHER', 'CASH AND/OR DERIVATIVES', 'CASH',
    'NOT CLASSIFIED', 'N/A', 'UNKNOWN', 'REMAINDER',
}

def clean_holding_name(name):
    """clean holding name to unify share classes (GOOG vs GOOGL etc)"""
    if not isinstance(name, str):
        return str(name).strip()
        
    name_upper = name.strip().upper()
    # Strip ONLY share-class designators. Stripping corporate-form suffixes
    # (INC/LTD/SA/CO/...) merges unrelated firms that share a base name — e.g.
    # "Titan Co Ltd" (India) + "Titan SA" (Greece) would both collapse to "TITAN".
    suffixes = [' CLASS A', ' CLASS B', ' CLASS C', ' CL A', ' CL B', ' CL C']
    
    for suffix in suffixes:
        if name_upper.endswith(suffix):
            name_upper = name_upper[:-len(suffix)].strip()
            
    return name_upper

def get_concentration_exposure(weights, df_underlying, top_n=10):
    """calc weighted underlying exposure across all holdings, unifying share classes (i.e. GOOG vs GOOGL)"""
    aggregated_holdings = {}
    
    for col_name in weights.index:
        if not (isinstance(col_name, str) and col_name.endswith('_wgt')):
            continue

        ticker = col_name[:-4]
        
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
            
            if clean_name in CATCH_ALL_HOLDINGS:
                continue
            
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

    # concentration (Top Holdings) — show only actual named holdings, no "Others" bucket
    if df_underlying is not None:
        conc_exp = get_concentration_exposure(weights, df_underlying, top_n=concentration_top_n)
        conc_top = conc_exp.head(concentration_top_n)
        total_concentration = conc_top.sum()
        charts['concentration_table'] = conc_top.to_frame(name='Weight')
        charts['concentration_bar'] = plot_horizontal_bar_chart(
            conc_top, 
            title=f"Top {concentration_top_n} Holdings ({total_concentration:.1%} Total)"
        )
        
    return charts
