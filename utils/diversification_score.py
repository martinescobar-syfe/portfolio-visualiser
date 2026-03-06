from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Union


# How we map the Excel's asset class rows to the 4 scoring buckets
ASSET_CLASS_MAP = {
    'Equity': 'Equity',
    'Fixed income': 'Bonds',
    'REITs': 'Real Assets',
    'Commodities': 'Real Assets',
    'Cash / MMF': 'Cash',
    'Specialized / alternatives': 'Alternatives',
    'Cryptocurrency': 'Alternatives',
}

SCORING_ASSET_CLASSES = ['Equity', 'Bonds', 'Real Assets', 'Cash']

# portfolios to exclude from diversification score calculation
# these are typically custom portfolios whose underlying ETFs aren't yet
# decomposed in the static data files
IGNORED_PORTFOLIOS = {'Custom Usd'}

# region mapping — simplified for now, will be replaced by a backend lookup
# only a short list of countries counts as Developed ex-US; everything else
# defaults to Emerging Markets. if US exposure is >90%, we treat the whole
# portfolio as US
DEVELOPED_EX_US = {
    'Japan', 'Britain', 'France', 'Germany', 'Canada', 'Australia',
    'Switzerland', 'Netherlands', 'Sweden', 'Denmark', 'Finland',
    'Norway', 'Ireland', 'Italy', 'Spain', 'Belgium', 'Austria',
    'Singapore', 'Hong Kong', 'Israel', 'New Zealand', 'Portugal',
    'Luxembourg', 'European Union', 'Eurozone',
}

# if US weight is above this threshold, skip per-country mapping
US_DOMINANCE_THRESHOLD = 0.90

GEO_BENCHMARK = {
    'United States': 0.65,
    'Developed ex-US': 0.25,
    'Emerging Markets': 0.10,
}

# sector benchmarks for scoring
EQUITY_SECTOR_BENCHMARK = {
    'Information Technology': 26.27,
    'Financials': 17.41,
    'Industrials': 11.33,
    'Consumer Discretionary': 9.80,
    'Health Care': 8.90,
    'Communication Services': 8.71,
    'Consumer Staples': 5.37,
    'Materials': 4.03,
    'Energy': 3.77,
    'Utilities': 2.58,
    'Real Estate': 1.78,
    'Not Classified': 0.04,
    'Commodities': 0.00,
}

# fixed income benchmark (AGGU). BICS naming differs slightly from GICS
# e.g. "Technology" instead of "Information Technology"
# source: Bloomberg AGGU sector weights
FI_SECTOR_BENCHMARK = {
    'Government': 69.94,
    'Financials': 13.05,
    'Not Classified': 2.95,
    'Utilities': 2.38,
    'Industrials': 1.85,
    'Health Care': 1.58,
    'Communications': 1.53,
    'Consumer Discretionary': 1.41,
    'Energy': 1.41,
    'Consumer Staples': 1.28,
    'Technology': 0.99,
    'Real Estate': 0.91,
    'Materials': 0.73,
    'High Yield bonds': 0.00,
}

# map the sector names we see in the Excel to their BICS equivalents
# NOTE: the Excel sheet will eventually be updated so FI sectors also follow
# GICS-style naming. until then, this map bridges the gap
FI_SECTOR_MAP = {

    'US Gov. Related bonds': 'Government',
    'Treasuries': 'Government',
    'Sovereign': 'Government',
    'Government-related': 'Government',
    'Singapore Government Bonds': 'Government',

    'Securitized': 'Not Classified',
    'ABS': 'Not Classified',
    'CMBS': 'Not Classified',
    'MBS Pass-Through': 'Not Classified',
    'CLO': 'Not Classified',
    'CMO': 'Not Classified',
    'CDO': 'Not Classified',
    'Housing': 'Not Classified',
    'Agency': 'Not Classified',
    'Covered': 'Not Classified',

    'Investment Grade bonds': 'Financials',
    'Corporates': 'Financials',


    'Non-USD Developed bonds': 'Not Classified',
    'Emerging Markets bonds': 'Not Classified',

    'Supranational': 'Government',
    'Local Authority': 'Government',

    'Other': 'Not Classified',
    'Cash and/or Derivatives': 'Not Classified',
    'Cash / Money Market Funds': 'Not Classified',

    'High Yield Bonds': 'High Yield bonds',
}

# GICS equity sectors. used to route the equity portion of mixed portfolios
# (like Core Defensive) to the equity benchmark instead of the FI one
EQUITY_GICS_SECTORS = {
    'Information Technology', 'Financials', 'Industrials',
    'Consumer Discretionary', 'Health Care', 'Communication Services',
    'Consumer Staples', 'Materials', 'Energy', 'Utilities', 'Real Estate',
    'Commodities',
}

IGNORED_SECTORS = {'Cash and/or Derivatives'}

# map GICS sector names to their BICS equivalents for FI portfolios whose
# sector data has been updated to GICS naming (e.g. Income Enhance)
# "Information Technology" in GICS → "Technology" in BICS, etc
GICS_TO_BICS_MAP = {
    'Information Technology': 'Technology',
    'Communication Services': 'Communications',
    'Government': 'Government',
}
# sectors that share the same name in both GICS and BICS need no mapping —
# Financials, Industrials, Consumer Discretionary, Health Care,
# Consumer Staples, Materials, Energy, Utilities, Real Estate


# ---------------------------------------------------------------------------
# 1. Asset Class Score (40% weight)
# ---------------------------------------------------------------------------

def calculate_asset_class_score(asset_weights: dict) -> dict:
    """calc asset class diversification score (1-10 scale). takes in asset_weights dict where weights sum to ~1"""
    if not asset_weights:
        return {'score': None, 'breadth_score': None, 'dominance_score': None, 'details': {},
                'error': 'No asset class data available — check source data.'}

    weights = {k: v for k, v in asset_weights.items() if v > 0}
    if not weights:
        return {'score': None, 'breadth_score': None, 'dominance_score': None, 'details': {},
                'error': 'All asset class weights are zero — check source data.'}

    # part A — how many meaningful asset classes are present?
    meaningful = {k: v for k, v in weights.items() if v >= 0.05}
    m = len(meaningful)

    if m <= 1:
        b_base = 1
    elif m == 2:
        b_base = 4
    elif m == 3:
        b_base = 6
    else:
        b_base = 8

    w_min = min(meaningful.values()) if meaningful else 0
    if w_min < 0.10:
        uplift = 0
    elif w_min < 0.15:
        uplift = 1 + (w_min - 0.10) / 0.05
    else:
        uplift = 2

    breadth_score = min(10, b_base + uplift)

    # part B — is any single asset class too dominant?
    w_max = max(weights.values())
    if w_max <= 0.70:
        dominance_score = 10
    elif w_max <= 0.90:
        dominance_score = 10 - 6 * (w_max - 0.70) / 0.20
    else:
        dominance_score = 4 - 3 * (w_max - 0.90) / 0.10

    dominance_score = max(1, dominance_score)

    # combine breadth and dominance 50/50
    score = 0.50 * breadth_score + 0.50 * dominance_score

    return {
        'score': round(score, 2),
        'breadth_score': round(breadth_score, 2),
        'dominance_score': round(dominance_score, 2),
        'details': weights,
    }


# ---------------------------------------------------------------------------
# 2. Geography Score (30% weight)
# ---------------------------------------------------------------------------

def _map_country_to_region(country) -> str:
    """map a single country name to one of our three regions (US, Developed ex-US, Emerging Markets)"""
    if not isinstance(country, str) or pd.isna(country):
        return 'Not Classified'
    stripped = country.strip().lower()
    if stripped in ('', 'nan', 'none', 'others'):
        return 'Not Classified'
    if 'not classified' in stripped:
        return 'Not Classified'
    if country == 'United States':
        return 'United States'
    if country in DEVELOPED_EX_US:
        return 'Developed ex-US'
    # default: treat all remaining countries as EM
    return 'Emerging Markets'


def calculate_geography_score(country_weights: dict, benchmark: Optional[dict] = None) -> dict:
    """calc geography diversification score (0-10 scale) based on country weights vs our benchmark"""
    benchmark = benchmark or GEO_BENCHMARK
    T = 10.0

    # if US exposure alone exceeds the dominance threshold, classify the
    # entire portfolio as US (handles single-country ETFs like SPY)
    us_weight = country_weights.get('United States', 0.0)
    total_weight = sum(v for v in country_weights.values()
                       if isinstance(v, (int, float)) and not pd.isna(v))
    if total_weight > 0 and (us_weight / total_weight) >= US_DOMINANCE_THRESHOLD:
        region_weights = {'United States': 1.0}
    else:
        region_weights = {}
        for country, w in country_weights.items():
            region = _map_country_to_region(country)
            region_weights[region] = region_weights.get(region, 0.0) + w

    # exclude unclassified but keep raw weights (don't renormalize)
    # missing country data should reduce coverage, not inflate other regions
    region_norm = {k: v for k, v in region_weights.items() if k != 'Not Classified'}

    region_details = {}
    total_score = 0.0
    for region, b_r in benchmark.items():
        p_max = b_r * T
        c_r = region_norm.get(region, 0.0)
        coverage = min(1.0, c_r / b_r) if b_r > 0 else 0.0
        contribution = p_max * coverage
        total_score += contribution
        region_details[region] = {
            'client_weight': round(c_r, 4),
            'benchmark_weight': round(b_r, 4),
            'coverage': round(coverage, 4),
            'points': round(contribution, 2),
            'max_points': round(p_max, 2),
        }

    return {
        'score': round(min(total_score, 10.0), 2),
        'region_weights': region_norm,
        'region_details': region_details,
    }


# ---------------------------------------------------------------------------
# 3. Sector Score (20% weight)
# ---------------------------------------------------------------------------

def _classify_portfolio_type(ticker: str, df_asset_class) -> str:
    """determine if a portfolio is equity, fixed_income, reit, or cash"""
    if df_asset_class is None or ticker not in df_asset_class.columns:
        name_lower = ticker.lower()
        if any(kw in name_lower for kw in ['equity', 'growth', 'balanced', 'china', 'disruptive', 'esg', 'healthcare', 'srs equity']):
            return 'equity'
        if any(kw in name_lower for kw in ['income', 'defensive', 'downside', 'blackrock', 'monroe', 'silverdale']):
            return 'fixed_income'
        if 'reit' in name_lower:
            return 'reit'
        if 'cash' in name_lower:
            return 'cash'
        return 'equity'

    # mixed/structured portfolios that should stay in the equity bucket
    # because they are growth-oriented products
    name_lower = ticker.lower()
    if any(kw in name_lower for kw in ['balanced', 'growth', 'defensive', 'downside']):
        return 'equity'

    col = df_asset_class[ticker]
    try:
        col_numeric = pd.to_numeric(col, errors='coerce').fillna(0)
    except Exception:
        return 'equity'

    equity_wt = col_numeric.get('Equity', 0)
    fi_wt = col_numeric.get('Fixed income', 0)
    reit_wt = col_numeric.get('REITs', 0)
    cash_wt = col_numeric.get('Cash / MMF', 0)

    max_wt = max(equity_wt, fi_wt, reit_wt, cash_wt)
    if max_wt == 0:
        # none of the 4 standard classes matched (e.g. Specialized / alternatives)
        # fall back to name-based classification
        name_lower = ticker.lower()
        if any(kw in name_lower for kw in ['blackrock', 'monroe', 'silverdale', 'income', 'defensive', 'downside']):
            return 'fixed_income'
        if 'reit' in name_lower:
            return 'reit'
        if 'cash' in name_lower:
            return 'cash'
        return 'equity'
    if max_wt == reit_wt:
        return 'reit'
    if max_wt == fi_wt:
        return 'fixed_income'
    if max_wt == cash_wt:
        return 'cash'
    return 'equity'


def _map_fi_sector(sector_name: str) -> str:
    """map excel FI sector names to benchmark categories"""
    return FI_SECTOR_MAP.get(sector_name, sector_name)


import re as _re

def _normalize_ticker_name(name: str) -> str:
    """strip parentheses and collapse month+year for fuzzy matching (e.g. 'Silverdale Fund (Feb 2029)' → 'silverdale fund feb2029')"""
    n = name.lower()
    n = _re.sub(r'[()]', '', n)
    n = _re.sub(r'\s+', ' ', n).strip()
    n = _re.sub(
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})',
        r'\1\2', n,
    )
    return n


def calculate_sector_score(
    client_weights: dict,
    df_sector,
    df_asset_class=None,
) -> dict:
    """calc sector diversification score (0-10 scale). classifies each portfolio, computes sector exposure per type, then compares to the appropriate benchmark. REIT portfolios use the ACWI equity benchmark"""
    if df_sector is None or not client_weights:
        return {'score': None, 'equity_score': None, 'fi_score': None,
                'details': {}, 'unmapped_sectors': {},
                'error': 'No sector data available — check source data.'}

    equity_sector_weights = {}
    fi_sector_weights = {}
    equity_total = 0.0
    fi_total = 0.0
    unmapped_sectors = {}
    skipped_tickers = []
    zero_sector_tickers = []

    for ticker, port_weight in client_weights.items():
        ptype = _classify_portfolio_type(ticker, df_asset_class)

        if ptype == 'cash':
            # cash portfolios may still have sector exposure (e.g. Cash SGD
            # flexi holds money-market securities). include them in the FI
            # bucket when sector data is available; skip only when there is
            # no data to contribute
            ptype = 'fixed_income'

        # flexible column lookup: case-insensitive, then normalised
        # (handles e.g. "Silverdale Fund Feb2029" vs "Silverdale Fund (Feb 2029)")
        matched_col = None
        if ticker in df_sector.columns:
            matched_col = ticker
        else:
            ticker_lower = ticker.lower()
            for col in df_sector.columns:
                if col.lower() == ticker_lower:
                    matched_col = col
                    break
            if matched_col is None:
                ticker_norm = _normalize_ticker_name(ticker)
                for col in df_sector.columns:
                    if _normalize_ticker_name(col) == ticker_norm:
                        matched_col = col
                        break

        if matched_col is None:
            skipped_tickers.append(ticker)
            continue

        sector_col = pd.to_numeric(df_sector[matched_col], errors='coerce').fillna(0)

        # flag portfolios whose non-ignored sector data sums to zero
        valid_sector_sum = sum(
            sw for sector, sw in sector_col.items()
            if sw > 0 and sector not in IGNORED_SECTORS
        )
        if valid_sector_sum == 0:
            zero_sector_tickers.append(ticker)
            continue

        if ptype in ('equity', 'reit'):
            # equity/REIT portfolios: most sectors stay in the equity
            # bucket. "Government" is the exception — it represents bond
            # exposure inside balanced/growth funds and should be scored
            # against the FI benchmark proportionately
            EQUITY_TO_FI_SECTORS = {'Government'}
            for sector, sw in sector_col.items():
                if sw <= 0 or sector in IGNORED_SECTORS:
                    continue
                if sector in EQUITY_TO_FI_SECTORS:
                    mapped = _map_fi_sector(sector)
                    fi_sector_weights[mapped] = fi_sector_weights.get(mapped, 0.0) + sw * port_weight
                    fi_total += sw * port_weight
                else:
                    equity_sector_weights[sector] = equity_sector_weights.get(sector, 0.0) + sw * port_weight
                    if sector not in EQUITY_SECTOR_BENCHMARK and sector not in EQUITY_GICS_SECTORS:
                        unmapped_sectors.setdefault(sector, []).append((ticker, sw * port_weight))
            equity_total += port_weight
        elif ptype == 'fixed_income':
            eq_sub = 0.0
            fi_sub = 0.0
            for sector, sw in sector_col.items():
                if sw <= 0 or sector in IGNORED_SECTORS:
                    continue
                # FI portfolios may now have GICS sector names (e.g. Income
                # Enhance updated to GICS). map them to BICS so they land in
                # the FI benchmark comparison instead of the equity bucket
                if sector in GICS_TO_BICS_MAP:
                    mapped = GICS_TO_BICS_MAP[sector]
                    fi_sector_weights[mapped] = fi_sector_weights.get(mapped, 0.0) + sw * port_weight
                    fi_sub += sw
                elif sector in FI_SECTOR_MAP or sector in FI_SECTOR_BENCHMARK:
                    mapped = _map_fi_sector(sector)
                    fi_sector_weights[mapped] = fi_sector_weights.get(mapped, 0.0) + sw * port_weight
                    if mapped not in FI_SECTOR_BENCHMARK:
                        unmapped_sectors.setdefault(sector, []).append((ticker, sw * port_weight))
                    fi_sub += sw
                elif sector in EQUITY_GICS_SECTORS:
                    # GICS sector that shares the same name in BICS (e.g.
                    # Financials, Industrials) — route to FI bucket directly
                    bics_name = sector
                    fi_sector_weights[bics_name] = fi_sector_weights.get(bics_name, 0.0) + sw * port_weight
                    if bics_name not in FI_SECTOR_BENCHMARK:
                        unmapped_sectors.setdefault(sector, []).append((ticker, sw * port_weight))
                    fi_sub += sw
                else:
                    mapped = _map_fi_sector(sector)
                    fi_sector_weights[mapped] = fi_sector_weights.get(mapped, 0.0) + sw * port_weight
                    if mapped not in FI_SECTOR_BENCHMARK:
                        unmapped_sectors.setdefault(sector, []).append((ticker, sw * port_weight))
                    fi_sub += sw
            equity_total += port_weight * eq_sub
            fi_total += port_weight * fi_sub

    # normalise to percentages within each type
    def normalise_to_pct(d, total):
        if total <= 0:
            return {}
        return {k: (v / total) * 100 for k, v in d.items()}

    # use actual sector weight sums so each bucket's client % sums to 100
    eq_sector_sum = sum(equity_sector_weights.values())
    fi_sector_sum = sum(fi_sector_weights.values())

    eq_pct = normalise_to_pct(equity_sector_weights, eq_sector_sum) if eq_sector_sum > 0 else {}
    fi_pct = normalise_to_pct(fi_sector_weights, fi_sector_sum) if fi_sector_sum > 0 else {}

    # score each type against its benchmark
    def benchmark_gap_score(client_pct: dict, benchmark: dict):
        raw = 0.0
        details = {}
        for sector, bw in benchmark.items():
            cw = client_pct.get(sector, 0.0)
            contribution = min(cw, bw)
            raw += contribution
            details[sector] = {
                'client': round(cw, 2),
                'benchmark': round(bw, 2),
                'contribution': round(contribution, 2),
            }
        # flag client sectors that have no benchmark match
        for sector, cw in client_pct.items():
            if sector not in benchmark:
                details[sector] = {
                    'client': round(cw, 2),
                    'benchmark': 0.0,
                    'contribution': 0.0,
                    'warning': 'No benchmark match',
                }
        return raw / 10.0, details  # scale 0-100 → 0-10

    equity_score, eq_details = (None, {})
    fi_score, fi_details = (None, {})

    if equity_total > 0:
        equity_score, eq_details = benchmark_gap_score(eq_pct, EQUITY_SECTOR_BENCHMARK)
    if fi_total > 0:
        fi_score, fi_details = benchmark_gap_score(fi_pct, FI_SECTOR_BENCHMARK)

    # weighted combination — use actual sector sums to avoid double-counting
    # from equity portfolios whose FI sectors are routed to the FI bucket
    combined_sum = eq_sector_sum + fi_sector_sum
    if combined_sum > 0:
        eq_contrib = (equity_score or 0) * (eq_sector_sum / combined_sum)
        fi_contrib = (fi_score or 0) * (fi_sector_sum / combined_sum)
        score = eq_contrib + fi_contrib
    else:
        score = 0.0

    return {
        'score': round(min(score, 10.0), 2),
        'equity_score': round(equity_score, 2) if equity_score is not None else None,
        'fi_score': round(fi_score, 2) if fi_score is not None else None,
        'equity_weight': round(equity_total, 4),
        'fi_weight': round(fi_total, 4),
        'equity_details': eq_details,
        'fi_details': fi_details,
        'unmapped_sectors': unmapped_sectors,
        'skipped_tickers': skipped_tickers,
        'zero_sector_tickers': zero_sector_tickers,
    }


# ---------------------------------------------------------------------------
# 4. Concentration Score (10% weight)
# ---------------------------------------------------------------------------

def calculate_concentration_score(
    holding_weights: Union[pd.Series, dict],
    n_target: int = 1000,
) -> dict:
    """calc concentration score using HHI / effective-N approach. takes holding weights summing to ~1 and target n_eff"""
    if isinstance(holding_weights, dict):
        holding_weights = pd.Series(holding_weights)

    holding_weights = holding_weights[holding_weights > 0].astype(float)
    if holding_weights.empty:
        return {'score': None, 'hhi': None, 'n_eff': None, 'n_holdings': 0,
                'error': 'No underlying holdings found — check source data.'}

    total = holding_weights.sum()
    if total <= 0:
        return {'score': None, 'hhi': None, 'n_eff': None, 'n_holdings': 0,
                'error': 'Holdings weights sum to zero — check source data.'}
    w = holding_weights / total

    hhi = (w ** 2).sum()
    n_eff = 1.0 / hhi if hhi > 0 else 1.0

    if n_eff <= 1:
        s_raw = 0.0
    else:
        s_raw = min(1.0, np.log(n_eff) / np.log(n_target))

    score = max(1.0, round(s_raw * 10, 2))

    return {
        'score': score,
        'hhi': round(hhi, 6),
        'n_eff': round(n_eff, 2),
        'n_holdings': len(w),
        'top_holdings': w.head(10).to_dict(),
    }


# ---------------------------------------------------------------------------
# Composite score — weighted average of all four sub-scores
# ---------------------------------------------------------------------------

WEIGHTS = {
    'asset': 0.40,
    'geo': 0.30,
    'sector': 0.20,
    'conc': 0.10,
}


def calculate_diversification_score(
    client_weights: dict,
    df_asset_class,
    df_sector,
    df_country,
    df_underlying,
    concentration_holdings: Optional[pd.Series] = None,
) -> dict:
    """orchestrator: compute the four sub-scores (asset, geo, sector, conc) and the weighted composite"""
    from utils.exposure import calculate_exposure, get_concentration_exposure

    # filter out ignored portfolios but keep raw weights (don't renormalize)
    # the gap from excluded portfolios should reduce coverage scores, not
    # inflate the remaining ones
    client_weights = {k: v for k, v in client_weights.items() if k not in IGNORED_PORTFOLIOS}

    # 1. Asset class
    weights_series = pd.Series({f"{k}_wgt": v for k, v in client_weights.items()})

    if df_asset_class is not None:
        raw_ac = calculate_exposure(weights_series, df_asset_class)
        ac_buckets = {}
        for ac_name, wt in raw_ac.items():
            bucket = ASSET_CLASS_MAP.get(ac_name, 'Alternatives')
            ac_buckets[bucket] = ac_buckets.get(bucket, 0.0) + wt
        asset_result = calculate_asset_class_score(ac_buckets)
    else:
        asset_result = calculate_asset_class_score({})

    # 2. Geography
    if df_country is not None:
        raw_country = calculate_exposure(weights_series, df_country)
        country_dict = raw_country.to_dict()
        geo_result = calculate_geography_score(country_dict)
    else:
        geo_result = calculate_geography_score({})

    # 3. Sector
    sector_result = calculate_sector_score(client_weights, df_sector, df_asset_class)

    # 4. Concentration — equity/REIT portfolios only
    if concentration_holdings is not None and not concentration_holdings.empty:
        conc_result = calculate_concentration_score(concentration_holdings)
    elif df_underlying is not None:
        equity_weights = {}
        for ticker, w in client_weights.items():
            ptype = _classify_portfolio_type(ticker, df_asset_class)
            if ptype in ('equity', 'reit'):
                equity_weights[f"{ticker}_wgt"] = w
        if equity_weights:
            eq_series = pd.Series(equity_weights)
            conc_holdings = get_concentration_exposure(eq_series, df_underlying, top_n=9999)
            conc_result = calculate_concentration_score(conc_holdings)
        else:
            conc_result = {'score': None, 'hhi': None, 'n_eff': None, 'n_holdings': 0,
                           'error': 'No equity/REIT portfolios — concentration not applicable.'}
    else:
        conc_result = calculate_concentration_score({})

    # data completeness: flag portfolios whose country or sector data
    # doesn't sum to ~100%. tolerance of 2pp handles minor rounding
    COMPLETENESS_TOL = 0.02
    data_warnings = []
    for ticker in client_weights:
        # country completeness
        if df_country is not None and ticker in df_country.columns:
            col = pd.to_numeric(df_country[ticker], errors='coerce').fillna(0)
            country_total = col.sum()
            if abs(country_total - 1.0) > COMPLETENESS_TOL:
                data_warnings.append(
                    f"**{ticker}** country data sums to {country_total:.1%} (expected ~100%)"
                )
        elif df_country is not None:
            data_warnings.append(f"**{ticker}** has no country data in the backend")

        # sector completeness — use the same normalised lookup as the scorer
        if df_sector is not None:
            matched_col = None
            if ticker in df_sector.columns:
                matched_col = ticker
            else:
                for col_name in df_sector.columns:
                    if col_name.lower() == ticker.lower():
                        matched_col = col_name
                        break
                if matched_col is None:
                    ticker_norm = _normalize_ticker_name(ticker)
                    for col_name in df_sector.columns:
                        if _normalize_ticker_name(col_name) == ticker_norm:
                            matched_col = col_name
                            break
            if matched_col is not None:
                col = pd.to_numeric(df_sector[matched_col], errors='coerce').fillna(0)
                sector_total = col.sum()
                # data is in decimal format (0-1). flag if it doesn't sum to ~1.0
                if abs(sector_total - 1.0) > COMPLETENESS_TOL:
                    data_warnings.append(
                        f"**{ticker}** sector data sums to {sector_total * 100:.1f}% (expected ~100%)"
                    )
            else:
                data_warnings.append(f"**{ticker}** has no sector data in the backend")

    # Composite: rescale weights to exclude any None sub-scores so
    # missing dimensions don't drag the overall score down unfairly.
    _sub_results = [
        ('asset', asset_result),
        ('geo', geo_result),
        ('sector', sector_result),
        ('conc', conc_result),
    ]
    available = [(k, r) for k, r in _sub_results if r.get('score') is not None]
    if available:
        raw_weight_sum = sum(WEIGHTS[k] for k, _ in available)
        composite = sum(
            (WEIGHTS[k] / raw_weight_sum) * r['score'] for k, r in available
        )
    else:
        composite = 0.0

    return {
        'composite_score': round(composite, 2),
        'asset_class': asset_result,
        'geography': geo_result,
        'sector': sector_result,
        'concentration': conc_result,
        'weights_used': WEIGHTS,
        'data_warnings': data_warnings,
    }
