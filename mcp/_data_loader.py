"""Lazy, process-global reference-data loader for MCP tools.

The Streamlit pages call ``utils.data_loader.load_static_data`` directly and
let ``@st.cache_data`` keep it warm. MCP tools run outside Streamlit, so we
need our own minimal cache: load the workbook on first request, hand back
the same dataframes thereafter.

Public API
----------
``get_reference_data()`` → ``ReferenceData`` namedtuple with these fields::

    sec_list       df_sec_list
    asset_class    df_asset_class
    sector         df_sector
    country        df_country
    fi_metrics     df_fi_metrics
    fx             df_fx
    daily_ret      df_daily_ret
    risk_rating    df_risk_rating
    underlying     df_underlying

``set_reference_data(...)`` lets tests / notebooks inject a custom dataset
without touching the Excel file. Calling it with ``None`` clears the cache.

This loader is intentionally **not** an MCP tool itself — it's plumbing
that the tool wrappers use internally.
"""

from __future__ import annotations

import os
import sys
from typing import NamedTuple, Optional

import pandas as pd

# Make the project root importable so we can reuse ``utils/``.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.data_loader import get_data_path, load_static_data  # noqa: E402


class ReferenceData(NamedTuple):
    """Bundle of all reference dataframes loaded from Inv_Data_Hardcoded.xlsx."""

    sec_list: pd.DataFrame
    asset_class: pd.DataFrame
    sector: pd.DataFrame
    country: pd.DataFrame
    fi_metrics: pd.DataFrame
    fx: pd.DataFrame
    daily_ret: pd.DataFrame
    risk_rating: pd.DataFrame
    underlying: pd.DataFrame


_CACHE: Optional[ReferenceData] = None


def get_reference_data(data_path: Optional[str] = None) -> ReferenceData:
    """Return the bundle of reference dataframes, loading once per process.

    ``data_path`` defaults to whatever ``utils.data_loader.get_data_path()``
    resolves to (Colab → local ``Data/`` → cwd). Pass an explicit path to
    point at a different copy of the workbook (useful in tests).
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    path = data_path or get_data_path()
    sl, ac, sec, cty, fi, fx, dret, rr, und = load_static_data(path)
    if sl is None:
        raise RuntimeError(
            f"Could not load reference data from {path!r}. "
            "Confirm Inv_Data_Hardcoded.xlsx exists under that directory."
        )
    _CACHE = ReferenceData(
        sec_list=sl,
        asset_class=ac,
        sector=sec,
        country=cty,
        fi_metrics=fi,
        fx=fx,
        daily_ret=dret,
        risk_rating=rr,
        underlying=und,
    )
    return _CACHE


def set_reference_data(data: Optional[ReferenceData]) -> None:
    """Inject a custom ReferenceData bundle (or clear the cache with ``None``)."""
    global _CACHE
    _CACHE = data
