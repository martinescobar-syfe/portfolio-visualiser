"""Risk and attribution analytics helpers for the Portfolio Builder.

All return values use **decimal** units (e.g. 0.05 == 5%) and assume daily
return inputs unless otherwise noted. Annualization factor is 252 (trading
days) by default.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _align_weights_and_cov(weights: pd.Series, cov: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Restrict to common tickers and reorder so vector / matrix line up."""
    common = [t for t in weights.index if t in cov.index and t in cov.columns]
    if len(common) == 0:
        raise ValueError("No overlap between weight index and covariance matrix.")
    w = weights.loc[common].astype(float)
    c = cov.loc[common, common].astype(float)
    return w, c


def annualised_cov_from_returns(returns_df: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """Annualised covariance matrix from a daily return DataFrame."""
    returns_df = returns_df.dropna(how='all').apply(pd.to_numeric, errors='coerce')
    return returns_df.cov() * periods_per_year


# ---------------------------------------------------------------------------
# 1) Risk contribution per asset (MCTR / Component Vol)
# ---------------------------------------------------------------------------
def compute_risk_contributions(weights: pd.Series, cov_matrix: pd.DataFrame) -> pd.DataFrame:
    """Marginal and component contributions to portfolio volatility.

    Standard Euler decomposition:
        sigma_p = sqrt(w' Sigma w)
        MCTR_i  = (Sigma w)_i / sigma_p          (change in sigma_p per unit weight in i)
        CCTR_i  = w_i * MCTR_i                   (component contribution; sums to sigma_p)
        % share = CCTR_i / sigma_p               (sums to 1.0 = 100%)

    Both MCTR and CCTR are annualised when ``cov_matrix`` is annualised.
    """
    w, cov = _align_weights_and_cov(weights, cov_matrix)
    w_vec = w.values
    sigma_w = cov.values @ w_vec                   # shape (n,)
    port_var = float(w_vec @ sigma_w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))

    if port_vol == 0:
        mctr = np.zeros_like(w_vec)
        cctr = np.zeros_like(w_vec)
        share = np.zeros_like(w_vec)
    else:
        mctr = sigma_w / port_vol                  # MCTR
        cctr = w_vec * mctr                        # component contribution; sums to port_vol
        share = cctr / port_vol                    # sums to 1.0

    out = pd.DataFrame({
        'Weight': w_vec,
        'MCTR (annualised)': mctr,
        'Component Risk (annualised)': cctr,
        '% of Total Risk': share,
    }, index=w.index)
    out.attrs['portfolio_vol'] = port_vol
    return out


# ---------------------------------------------------------------------------
# 2) Marginal contribution to tracking error (active risk)
# ---------------------------------------------------------------------------
def compute_active_risk_contributions(
    port_weights: pd.Series,
    bm_weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Decomposition of tracking error into per-asset contributions.

        active_w = w_p - w_b   (over the union of tickers, missing -> 0)
        TE       = sqrt(active_w' Sigma active_w)
        MCTE_i   = (Sigma active_w)_i / TE
        CCTE_i   = active_w_i * MCTE_i           (sums to TE)
        % share  = CCTE_i / TE                   (sums to 1.0; can be negative)
    """
    universe = sorted(set(port_weights.index) | set(bm_weights.index))
    universe = [t for t in universe if t in cov_matrix.index and t in cov_matrix.columns]
    if len(universe) == 0:
        raise ValueError("No overlap between weights and covariance matrix.")

    p = port_weights.reindex(universe).fillna(0.0).astype(float)
    b = bm_weights.reindex(universe).fillna(0.0).astype(float)
    a = p - b
    cov = cov_matrix.loc[universe, universe].astype(float)

    sigma_a = cov.values @ a.values
    te_var = float(a.values @ sigma_a)
    te = float(np.sqrt(max(te_var, 0.0)))

    if te == 0:
        mcte = np.zeros_like(a.values)
        ccte = np.zeros_like(a.values)
        share = np.zeros_like(a.values)
    else:
        mcte = sigma_a / te
        ccte = a.values * mcte
        share = ccte / te

    out = pd.DataFrame({
        'Portfolio Weight': p.values,
        'Benchmark Weight': b.values,
        'Active Weight': a.values,
        'MCTE (annualised)': mcte,
        'Component TE (annualised)': ccte,
        '% of Total TE': share,
    }, index=universe)
    out.attrs['tracking_error'] = te
    return out


# ---------------------------------------------------------------------------
# 3) VaR & CVaR (historical & parametric)
# ---------------------------------------------------------------------------
def compute_var_cvar(
    returns: pd.Series,
    confidence_levels: Iterable[float] = (0.95, 0.99),
    horizon_days: int = 1,
    annualise: bool = False,
) -> pd.DataFrame:
    """Compute historical and parametric Gaussian VaR/CVaR.

    Sign convention: VaR / CVaR are reported as **negative** numbers (losses).
    For example a 95% VaR of -0.025 means "5% of the time you lose more than 2.5%".

    The ``annualise=True`` flag is treated as a shorthand for ``horizon_days=252``
    and is **mutually exclusive** with a manually supplied ``horizon_days`` --
    if both are passed, ``annualise=True`` overrides ``horizon_days``.

    Scaling rules applied uniformly:
      * Parametric Gaussian (h-day):
            VaR  = mu*h + sigma*sqrt(h)*z
            CVaR = mu*h - sigma*sqrt(h)*phi(z)/alpha
      * Historical (h-day):
            scale 1-day quantile/CVaR by sqrt(h) (sqrt-time approximation,
            assumes i.i.d. returns; will mis-state under serial correlation).
    """
    s = returns.dropna().astype(float)
    if s.empty:
        return pd.DataFrame()

    mu = float(s.mean())
    sigma = float(s.std(ddof=1))
    h = TRADING_DAYS if annualise else int(horizon_days)
    sqrt_h = np.sqrt(h)

    rows = []
    for cl in confidence_levels:
        alpha = 1.0 - cl                           # tail prob (e.g. 0.05)

        # historical 1-day -> scale by sqrt(h)
        hist_var_1d = float(s.quantile(alpha))
        tail = s[s <= hist_var_1d]
        hist_cvar_1d = float(tail.mean()) if not tail.empty else float('nan')
        hist_var = hist_var_1d * sqrt_h
        hist_cvar = hist_cvar_1d * sqrt_h

        # parametric Gaussian h-day
        z = norm.ppf(alpha)                        # negative for alpha < 0.5
        param_var = mu * h + sigma * sqrt_h * z
        param_cvar = mu * h - sigma * sqrt_h * norm.pdf(z) / alpha

        rows.append({
            'Confidence': cl,
            'Historical VaR': hist_var,
            'Historical CVaR': hist_cvar,
            'Parametric VaR': param_var,
            'Parametric CVaR': param_cvar,
        })

    df = pd.DataFrame(rows).set_index('Confidence')
    df.attrs['horizon_days'] = h
    return df


# ---------------------------------------------------------------------------
# 4) Drawdown episodes
# ---------------------------------------------------------------------------
def compute_drawdown_episodes(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Identify the ``top_n`` worst drawdown episodes by depth.

    Returns a DataFrame with peak / trough / recovery dates, depth (negative
    decimal), peak-to-trough days, peak-to-recovery days, and an "Ongoing"
    flag for any episode that has not yet recovered by the last data point.
    """
    s = returns.dropna().astype(float)
    if s.empty:
        return pd.DataFrame()

    nav = (1 + s).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0                           # <= 0

    in_dd = dd < 0
    episodes = []
    start_idx = None                                # first day inside current episode
    peak_dt = None                                  # day when peak was last set (= day before start)

    for i, dt in enumerate(dd.index):
        if in_dd.iloc[i] and start_idx is None:
            start_idx = i
            peak_dt = dd.index[i - 1] if i > 0 else dd.index[i]
        elif not in_dd.iloc[i] and start_idx is not None:
            # recovery on this date (NAV >= prior peak again)
            seg = dd.iloc[start_idx:i + 1]
            trough_dt = seg.idxmin()
            depth = float(seg.min())
            episodes.append({
                'Peak Date': peak_dt,
                'Trough Date': trough_dt,
                'Recovery Date': dt,
                'Depth': depth,
                'Peak-to-Trough Days': (trough_dt - peak_dt).days,
                'Peak-to-Recovery Days': (dt - peak_dt).days,
                'Ongoing': False,
            })
            start_idx = None
            peak_dt = None

    # Tail episode that never recovered
    if start_idx is not None:
        seg = dd.iloc[start_idx:]
        trough_dt = seg.idxmin()
        depth = float(seg.min())
        episodes.append({
            'Peak Date': peak_dt,
            'Trough Date': trough_dt,
            'Recovery Date': pd.NaT,
            'Depth': depth,
            'Peak-to-Trough Days': (trough_dt - peak_dt).days,
            'Peak-to-Recovery Days': np.nan,
            'Ongoing': True,
        })

    if not episodes:
        return pd.DataFrame()

    df = pd.DataFrame(episodes)
    df = df.sort_values('Depth', ascending=True).head(top_n).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = 'Rank'
    return df


# ---------------------------------------------------------------------------
# 5) Up/Down capture ratios + hit / batting ratios
# ---------------------------------------------------------------------------
def compute_capture_ratios(
    port_returns: pd.Series,
    bm_returns: pd.Series,
    frequency: str = 'M',
) -> pd.Series:
    """Up / down capture, hit ratio and batting average vs benchmark.

    ``frequency`` aggregates daily returns to monthly ('M'), quarterly ('Q'),
    or annual ('Y') before computing capture. Capture uses **geometric**
    average period returns to be robust to outliers.
    """
    p = pd.Series(port_returns).dropna().astype(float)
    b = pd.Series(bm_returns).dropna().astype(float)
    common = p.index.intersection(b.index)
    p = p.loc[common]
    b = b.loc[common]
    if p.empty:
        return pd.Series(dtype=float)

    # aggregate to chosen frequency using geometric compounding;
    # drop buckets with zero observations (otherwise empty groups silently
    # produce 0% period returns and pollute capture / hit ratios)
    rule = {'M': 'ME', 'Q': 'QE', 'Y': 'YE'}.get(frequency.upper(), 'ME')

    def _safe_compound(x):
        if len(x) == 0:
            return float('nan')
        return float((1 + x).prod() - 1)

    counts = p.resample(rule).count()
    p_per = p.resample(rule).apply(_safe_compound)
    b_per = b.resample(rule).apply(_safe_compound)
    valid_periods = (counts > 0)
    p_per = p_per[valid_periods]
    b_per = b_per[valid_periods]
    # also align indices in case resample produced different bucket sets
    common_idx = p_per.index.intersection(b_per.index)
    p_per = p_per.loc[common_idx]
    b_per = b_per.loc[common_idx]
    if p_per.empty:
        return pd.Series(dtype=float)

    up_mask = b_per > 0
    dn_mask = b_per < 0

    def _geo_mean(x):
        x = x.dropna()
        if x.empty:
            return float('nan')
        compound = (1 + x).prod()
        if compound <= 0:
            # any period with return <= -100% blows up the geometric mean
            return float('nan')
        return float(compound ** (1 / len(x)) - 1)

    bm_up = _geo_mean(b_per[up_mask])
    bm_dn = _geo_mean(b_per[dn_mask])
    p_up = _geo_mean(p_per[up_mask])
    p_dn = _geo_mean(p_per[dn_mask])

    def _safe_div(num, den):
        if not (np.isfinite(num) and np.isfinite(den)) or den == 0:
            return float('nan')
        return float(num / den)

    up_capture = _safe_div(p_up, bm_up)
    dn_capture = _safe_div(p_dn, bm_dn)
    ud_ratio = _safe_div(up_capture, dn_capture)

    hit_ratio = float((p_per > b_per).mean())                       # % periods strictly beating
    batting_avg = float((p_per >= b_per).mean())                    # % periods >= benchmark

    out = pd.Series({
        'Up Capture': up_capture,
        'Down Capture': dn_capture,
        'Up/Down Capture Ratio': ud_ratio,
        'Hit Ratio': hit_ratio,
        'Batting Average': batting_avg,
        '# Up Periods': int(up_mask.sum()),
        '# Down Periods': int(dn_mask.sum()),
        '# Total Periods': int(len(p_per)),
        'Avg Excess Return (period)': float((p_per - b_per).mean()),
        'Frequency': frequency.upper(),
    })
    return out


# ---------------------------------------------------------------------------
# 6) Brinson-Fachler attribution (single-period)
# ---------------------------------------------------------------------------
def _bucket_weights_from_etf(weights: pd.Series, exposure_table: pd.DataFrame) -> pd.Series:
    """Aggregate ETF-level weights into bucket-level weights via the exposure matrix.

    ``exposure_table`` is shaped (buckets x tickers); values are expected as
    decimals summing to ~1 down each ticker column (handles 0-100 % style too).
    """
    common = [t for t in weights.index if t in exposure_table.columns]
    if len(common) == 0:
        return pd.Series(dtype=float)
    sub = exposure_table[common].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    # normalise any column that looks like percentages (sum >> 1)
    col_sums = sub.sum(axis=0)
    pct_cols = col_sums[col_sums > 1.5].index
    if len(pct_cols):
        sub.loc[:, pct_cols] = sub.loc[:, pct_cols] / 100.0
    bucket_w = sub.dot(weights[common].astype(float))
    return bucket_w


def _bucket_returns_from_etf(
    weights: pd.Series,
    asset_returns: pd.Series,                       # period return per ETF (decimal)
    exposure_table: pd.DataFrame,
) -> pd.Series:
    """Approximate per-bucket return as the weighted average of ETF returns
    where weights are the *contribution* of each ETF to that bucket.

    For bucket b:
        R_b = sum_i ( w_i * exp_{i,b} * r_i )  /  sum_i ( w_i * exp_{i,b} )

    This is the look-through-style approximation of Brinson when the only
    return data we have is at the ETF level.
    """
    common = [t for t in weights.index if t in exposure_table.columns and t in asset_returns.index]
    if len(common) == 0:
        return pd.Series(dtype=float)
    exp_sub = exposure_table[common].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    col_sums = exp_sub.sum(axis=0)
    pct_cols = col_sums[col_sums > 1.5].index
    if len(pct_cols):
        exp_sub.loc[:, pct_cols] = exp_sub.loc[:, pct_cols] / 100.0
    w_vec = weights.loc[common].astype(float)
    r_vec = asset_returns.loc[common].astype(float)

    contributions = exp_sub.multiply(w_vec.values, axis=1)              # buckets x tickers
    weighted_returns = contributions.multiply(r_vec.values, axis=1)
    numer = weighted_returns.sum(axis=1)                                 # buckets
    denom = contributions.sum(axis=1)                                    # buckets
    out = pd.Series(np.where(denom > 0, numer / denom.replace(0, np.nan), 0.0), index=numer.index)
    return out.fillna(0.0)


def compute_brinson_attribution(
    port_weights: pd.Series,
    bm_weights: pd.Series,
    asset_returns: pd.Series,                       # period return per ETF (decimal)
    exposure_table: pd.DataFrame,                   # buckets x tickers
) -> pd.DataFrame:
    """Brinson-Fachler attribution per bucket.

    Returns a DataFrame indexed by bucket with these columns:
        Port Weight, Bench Weight, Active Weight,
        Port Return, Bench Return,
        Allocation, Selection, Interaction, Total

    Where, per bucket i:
        Allocation_i  = (w_p_i - w_b_i) * (R_b_i - R_b_total)
        Selection_i   =  w_b_i           * (R_p_i - R_b_i)
        Interaction_i = (w_p_i - w_b_i) * (R_p_i - R_b_i)
        Total_i       = Allocation_i + Selection_i + Interaction_i

    The Brinson-Fachler allocation effect uses (R_b_i - R_b_total) so that
    over- or under-weighting an above-average benchmark sector is rewarded /
    penalised correctly. Sum(Allocation + Selection + Interaction) reconciles
    to the portfolio's *active* return for the period.
    """
    w_p = _bucket_weights_from_etf(port_weights, exposure_table)
    w_b = _bucket_weights_from_etf(bm_weights, exposure_table)
    r_p = _bucket_returns_from_etf(port_weights, asset_returns, exposure_table)
    r_b = _bucket_returns_from_etf(bm_weights, asset_returns, exposure_table)

    buckets = sorted(set(w_p.index) | set(w_b.index) | set(r_p.index) | set(r_b.index))
    w_p = w_p.reindex(buckets).fillna(0.0)
    w_b = w_b.reindex(buckets).fillna(0.0)
    r_p = r_p.reindex(buckets).fillna(0.0)
    r_b = r_b.reindex(buckets).fillna(0.0)

    bm_total_return = float((w_b * r_b).sum())
    port_total_return = float((w_p * r_p).sum())

    allocation = (w_p - w_b) * (r_b - bm_total_return)
    selection = w_b * (r_p - r_b)
    interaction = (w_p - w_b) * (r_p - r_b)
    total = allocation + selection + interaction

    out = pd.DataFrame({
        'Port Weight': w_p,
        'Bench Weight': w_b,
        'Active Weight': w_p - w_b,
        'Port Return': r_p,
        'Bench Return': r_b,
        'Allocation': allocation,
        'Selection': selection,
        'Interaction': interaction,
        'Total': total,
    })
    out.attrs['port_total_return'] = port_total_return
    out.attrs['bench_total_return'] = bm_total_return
    out.attrs['active_return'] = port_total_return - bm_total_return
    out = out.sort_values('Total', ascending=False)
    return out


# ---------------------------------------------------------------------------
# 7) Efficient frontier (long-only, no leverage)
# ---------------------------------------------------------------------------
def compute_efficient_frontier(
    returns_df: pd.DataFrame,
    n_points: int = 30,
    periods_per_year: int = TRADING_DAYS,
    weight_bounds: Tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """Long-only efficient frontier from the empirical mean / covariance.

    Sweeps target returns between min and max of individual asset annualised
    means and solves min-variance for each target. Returns a DataFrame with
    columns ``Return``, ``Vol``, ``Sharpe``, plus one column per asset for
    the optimised weights.
    """
    rets = returns_df.dropna(how='all').apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    if rets.empty or rets.shape[1] < 2:
        return pd.DataFrame()

    mu = rets.mean() * periods_per_year                                # annualised
    cov = rets.cov() * periods_per_year                                # annualised
    n = len(mu)
    bnds = tuple(weight_bounds for _ in range(n))
    eq_cons = [{'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0)}]

    def _port_var(w):
        return float(w @ cov.values @ w)

    # global min variance
    res_min = minimize(
        _port_var,
        x0=np.full(n, 1.0 / n),
        method='SLSQP',
        bounds=bnds,
        constraints=eq_cons,
        options={'maxiter': 200, 'ftol': 1e-9},
    )
    if not res_min.success:
        return pd.DataFrame()
    min_ret = float(mu.values @ res_min.x)
    max_ret = float(mu.max())

    if max_ret <= min_ret + 1e-9:
        targets = np.array([min_ret])
    else:
        targets = np.linspace(min_ret, max_ret, n_points)

    rows = []
    weights_rows = []
    for tr in targets:
        cons = eq_cons + [{'type': 'eq', 'fun': lambda w, tr=tr: float(mu.values @ w - tr)}]
        r = minimize(
            _port_var,
            x0=np.full(n, 1.0 / n),
            method='SLSQP',
            bounds=bnds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-9},
        )
        if not r.success:
            continue
        w_opt = r.x
        vol = float(np.sqrt(max(w_opt @ cov.values @ w_opt, 0.0)))
        ret_real = float(mu.values @ w_opt)
        rows.append({'Return': ret_real, 'Vol': vol, 'Sharpe': ret_real / vol if vol > 0 else float('nan')})
        weights_rows.append(pd.Series(w_opt, index=mu.index))

    if not rows:
        return pd.DataFrame()
    frontier = pd.DataFrame(rows)
    weights_df = pd.DataFrame(weights_rows).reset_index(drop=True)
    frontier = pd.concat([frontier, weights_df], axis=1)
    return frontier


def evaluate_portfolio_on_frontier(
    weights: pd.Series,
    returns_df: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS,
) -> Tuple[float, float]:
    """Annualised (return, vol) of a fixed portfolio using the same mean/cov
    used for the frontier. Useful to overlay the actual portfolio onto the
    frontier chart.
    """
    rets = returns_df.dropna(how='all').apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    common = [t for t in weights.index if t in rets.columns]
    if len(common) == 0:
        return float('nan'), float('nan')
    sub = rets[common]
    w = weights.loc[common].astype(float).values
    mu = sub.mean().values * periods_per_year
    cov = sub.cov().values * periods_per_year
    ret = float(mu @ w)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return ret, vol
