"""Holdings ingestion — raw rows → canonical {portfolio: nav} mapping.

The Streamlit page ``pages/Client_PV.py`` does this inside ``process_client_data``.
This module exposes the same logic as plain functions that take primitive
inputs (a CSV path, a list of row dicts, or a Metabase client_id) and
return JSON-serialisable dicts.

Type 1 + 2 functions (data access plus a tiny bit of normalisation).
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.exposure import CUSTOM_PORTFOLIO_LABEL, parse_custom_holdings  # noqa: E402


# ── Portfolio code → display-name mapping (mirrors Client_PV.CLIENT_PORTFOLIO_MAPPING) ──
# Kept in this module on purpose so MCP tools don't take a runtime dependency
# on the Streamlit page. If you add a new managed portfolio, update both.
CLIENT_PORTFOLIO_MAPPING: dict[str, str] = {
    'GLOBAL_EQUITY_100': 'Core Equity100', 'GLOBAL_EQUITY': 'Core Equity100',
    'CORE_EQUITY100': 'Core Equity100', 'CORE_EQUITY_100': 'Core Equity100',
    'EQUITY100': 'Core Equity100',
    'CORE_GROWTH': 'Core Growth', 'CORE_BALANCED': 'Core Balanced',
    'CORE_DEFENSIVE': 'Core Defensive', 'DOWNSIDE_PROTECTED': 'Downside protected',
    'CHINA_GROWTH': 'China Growth',
    'DISRUPTIVE_TECHNOLOGY': 'Disruptive Technology', 'DISRUPTIVE_TECH': 'Disruptive Technology',
    'ESG_AND_CLEAN_ENERGY': 'ESG & Clean Energy', 'ESG_CLEAN_ENERGY': 'ESG & Clean Energy',
    'HEALTHCARE_INNOVATION': 'Healthcare Innovation',
    'INCOME_ENHANCE': 'Income Enhance', 'INCOME_ENHANCED': 'Income Enhance',
    'INCOME_PRESERVE': 'Income Preserve',
    'REIT': 'REIT 100', 'REIT_100': 'REIT 100',
    'REIT_RISK_MANAGED': 'REIT Risk Managed', 'REIT_RM': 'REIT Risk Managed',
    'BLACKROCK': 'Blackrock Private Credit Fund',
    'BLACKROCK_PRIVATE_CREDIT': 'Blackrock Private Credit Fund',
    'BLACKROCK_PCF': 'Blackrock Private Credit Fund',
    'MONROE': 'Monroe Income Plus Fund',
    'MONROE_INCOME_PLUS': 'Monroe Income Plus Fund',
    'MONROE_CAPITAL': 'Monroe Income Plus Fund',
    'SILVERDALE': 'Silverdale Fund (Nov 2027)',
    'CASH_PLUS_GUARANTEED_SGD': 'Cash SGD guaranteed',
    'CASH_PLUS_GUARANTEED': 'Cash SGD guaranteed', 'CASH_GUARANTEED': 'Cash SGD guaranteed',
    'CASH_PLUS_USD': 'Cash USD flexi', 'CASH_USD': 'Cash USD flexi',
    'CASH_PLUS': 'Cash SGD flexi', 'CASH_SGD': 'Cash SGD flexi',
    'CASH_PLUS_SGD': 'Cash SGD flexi',
    'SRS_INCOME_ENHANCE': 'SRS Income Enhance', 'SRS_INCOME_ENHANCED': 'SRS Income Enhance',
    'SRS_INCOME_PRESERVE': 'SRS Income Preserve',
    'SRS_CASH_PLUS': 'SRS Cash SGD flexi', 'SRS_CASH': 'SRS Cash SGD flexi',
    'SRS_GLOBAL_EQUITY_100': 'SRS Equity100', 'SRS_EQUITY100': 'SRS Equity100',
    'SRS_EQUITY_100': 'SRS Equity100',
}


def _norm_type(p_type: Any) -> str | None:
    """Strip lifecycle suffixes off a raw portfolio-type code (private helper)."""
    if not isinstance(p_type, str):
        return None
    n = p_type.upper().strip()
    for sfx in ('_ACTIVE_NEW', '_ACTIVE', '_INACTIVE', '_CLOSED', '_PENDING'):
        if n.endswith(sfx):
            n = n[:-len(sfx)]
    return n


def _safe_float(val: Any, default: float = 0.0) -> float:
    if pd.isna(val):
        return default
    try:
        return float(str(val).replace(',', ''))
    except (ValueError, TypeError):
        return default


def aggregate_client_holdings(rows: list[dict]) -> dict:
    """Aggregate raw client portfolio rows into canonical holdings.

    Use when you have a list of raw client portfolio rows (from a CSV
    or a Metabase query) and need them rolled up by portfolio name in
    the canonical shape every other tool expects.

    Each row should contain at minimum:
        - ``type``     — raw portfolio code (e.g. ``CORE_GROWTH_ACTIVE``)
        - ``nav_in_sgd`` — current NAV in SGD
        - ``currency`` — optional, used to disambiguate Cash variants
        - ``actual_holding_weightage`` — optional, used for CUSTOM portfolios

    Sub-$50 dust positions are dropped.

    Args:
        rows: list of dicts (one per raw portfolio row).

    Returns:
        ``{
            "weights":         {portfolio_name: weight 0–1},
            "holdings":        {portfolio_name: nav_in_sgd},
            "total_nav":       float,
            "custom_etf_nav":  {etf_ticker: nav_in_sgd},
        }``

        Returns empty dicts and ``total_nav == 0`` when nothing usable.
    """
    holdings: dict[str, float] = {}
    custom_etf_nav: dict[str, float] = {}
    total = 0.0

    for row in rows:
        p_type = row.get('type')
        currency = str(row.get('currency', 'SGD'))
        nav = _safe_float(row.get('nav_in_sgd', 0))
        if nav < 50:
            continue
        nt = _norm_type(p_type)
        if nt is None:
            continue

        # Map raw code → canonical display name (mirrors Client_PV.process_client_data).
        ticker = None
        if 'CASH' in nt and 'PLUS' in nt:
            if currency == 'USD' or 'USD' in nt:
                ticker = 'Cash USD flexi'
            elif 'GUARANTEED' in nt:
                ticker = 'Cash SGD guaranteed'
            else:
                ticker = 'Cash SGD flexi'
        elif nt in CLIENT_PORTFOLIO_MAPPING:
            ticker = CLIENT_PORTFOLIO_MAPPING[nt]
        elif 'BLACKROCK' in nt:
            ticker = 'Blackrock Private Credit Fund'
        elif 'MONROE' in nt:
            ticker = 'Monroe Income Plus Fund'
        elif 'CUSTOM' in nt:
            ticker = CUSTOM_PORTFOLIO_LABEL
        elif 'DOWNSIDE' in nt:
            ticker = 'Downside protected'
        elif 'SILVER' in nt or 'SILVERDALE' in nt:
            m = re.search(r'(NOV|DEC|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT)\s*(\d{4})', nt)
            ticker = (
                f'Silverdale Fund {m.group(1).title()}{m.group(2)}'
                if m else 'Silverdale Fund Feb2029'
            )

        if ticker:
            holdings[ticker] = holdings.get(ticker, 0.0) + nav
            total += nav
            if ticker == CUSTOM_PORTFOLIO_LABEL:
                parsed = parse_custom_holdings(row.get('actual_holding_weightage'))
                if parsed:
                    for etf, pct in parsed.items():
                        custom_etf_nav[etf] = custom_etf_nav.get(etf, 0.0) + nav * pct

    if total <= 0:
        return {"weights": {}, "holdings": {}, "total_nav": 0.0, "custom_etf_nav": {}}

    weights = {k: v / total for k, v in holdings.items()}
    return {
        "weights": weights,
        "holdings": holdings,
        "total_nav": total,
        "custom_etf_nav": custom_etf_nav,
    }


def parse_holdings_csv(csv_path: str) -> dict:
    """Load a client-holdings CSV and aggregate into canonical holdings.

    Use when the client's portfolio is supplied as a CSV (e.g. the dummy
    file under ``mcp/dummy_data/`` or an advisor-exported file).

    The CSV must have at minimum a ``type`` column and a ``nav_in_sgd``
    column. Rows with status ``ACTIVE`` / ``ACTIVE_NEW`` are kept; all
    other statuses are dropped.

    Args:
        csv_path: absolute or relative path to a CSV file.

    Returns:
        Same shape as :func:`aggregate_client_holdings`.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()
    if 'type' in df.columns:
        df['type'] = df['type'].str.replace('BlackRock', 'Blackrock', case=False)
    if 'status' in df.columns:
        df = df[df['status'].str.upper().isin(['ACTIVE', 'ACTIVE_NEW'])]
    return aggregate_client_holdings(df.to_dict('records'))


def get_client_holdings_metabase(client_id: int) -> dict:
    """Fetch a client's active portfolios from Metabase and aggregate them.

    Use when the chatbot has a Syfe client ID and needs the current
    book of holdings.

    Requires Syfe VPN (Metabase endpoint is on the internal network)
    and a working API key inside ``utils/metabase_helper.py``.

    Args:
        client_id: integer Syfe client ID.

    Returns:
        Same shape as :func:`aggregate_client_holdings`, plus a
        ``"client_info"`` key with age / risk_score / name pulled from
        the matching Metabase profile card.

        If the client is not found, returns ``{}`` for every key and
        ``"client_info": None``.
    """
    from utils.metabase_helper import get_active_portfolios, get_client_info_by_id

    info = get_client_info_by_id(int(client_id))
    if not info or not info.get('client_id'):
        return {
            "weights": {}, "holdings": {}, "total_nav": 0.0,
            "custom_etf_nav": {}, "client_info": None,
        }

    df = get_active_portfolios(info['client_id'])
    rows = [] if df is None or df.empty else df.to_dict('records')
    result = aggregate_client_holdings(rows)
    result["client_info"] = {
        "client_id": info.get('client_id'),
        "name": info.get('name'),
        "age": info.get('age'),
        "risk_score": info.get('risk_score'),
    }
    return result
