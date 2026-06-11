#!/usr/bin/env python3
"""
Syfe client-exposure MCP server
===============================

Exposes the Portfolio Visualiser's **deterministic look-through exposure**
calculators as MCP tools the advisor bot can call. Given a client's
``{product: weight}`` mix (ANY distribution of weights), the bot can ask for the
blended country / sector / asset-class breakdown and the top underlying
holdings. The math is identical to what the Streamlit app shows
(``mcp/exposure.py`` -> ``utils/exposure.py``), backed by
``Data/Inv_Data_Hardcoded.xlsx``. No LLM is involved in the numbers — the tools
are pure, deterministic matrix maths.

Example the bot can now answer deterministically
------------------------------------------------
Client holds 50% Core Equity100 + 50% Equity Alpha and asks "what's my
exposure?":

    country_exposure({"Core Equity100": 0.5, "Equity Alpha": 0.5})
    sector_exposure({"Core Equity100": 0.5, "Equity Alpha": 0.5})
    top_holdings({"Core Equity100": 0.5, "Equity Alpha": 0.5}, top_n=10)

The ``mcp`` package-name collision (and how we resolve it)
----------------------------------------------------------
This repo ships a LOCAL package named ``mcp/`` (the calculator library:
``mcp.exposure``, ``mcp._data_loader``, ...). The official **MCP SDK** also
installs a top-level package named ``mcp`` (``mcp.types``, ``mcp.server``, ...).
They collide on the name ``mcp``.

Switching to the standalone ``fastmcp`` distribution does NOT dodge this:
``fastmcp`` 3.x DEPENDS ON the SDK and does ``import mcp.types`` internally. With
the repo root on ``sys.path`` (so ``mcp.exposure`` resolves), a naive import makes
fastmcp's ``import mcp.types`` resolve to the LOCAL ``mcp/`` package, which has no
``types`` submodule -> ``ModuleNotFoundError: No module named 'mcp.types'``.

Fix = path merge. The two packages have DISJOINT submodules (SDK: types/server/
client; local: exposure/_data_loader/holdings/...), so we can safely union them
under one namespace: import the SDK ``mcp`` FIRST (repo root temporarily off
``sys.path``), then append the local ``mcp/`` directory to ``mcp.__path__`` so
``mcp.types`` (SDK) and ``mcp.exposure`` (local) both import. See the bootstrap
below.

Install
-------
    pip install -r requirements-mcp.txt        # fastmcp (pulls in the mcp SDK)

Run it
------
    cd "<this repo root>"
    python portfolio_exposure_mcp.py            # stdio transport (default)

Register with the advisor agent (Claude Code example, stdio):

    claude mcp add syfe-exposure -- python "<abs path>/portfolio_exposure_mcp.py"

or commit the project-scoped ``.mcp.json`` at the repo root (already provided).

NAV-FRIENDLY, NORMALISED OUTPUTS
--------------------------------
The country / sector / asset-class tools accept the input ``weights`` as
normalised weights, percentages, OR raw dollar NAVs — the breakdown is rescaled
by the sum of the inputs, so ``{"Core Equity100": 60000, "Equity Alpha": 40000}``
and ``{"Core Equity100": 0.6, "Equity Alpha": 0.4}`` give identical results. This
is what lets the advisor bot pull a client's managed-portfolio NAVs straight from
the playground and feed them in unchanged. Outputs are clipped non-negative and
always sum to 1.0; any weight that does not map to a known bucket is collected
into an ``"Others"`` bucket.

DATA-COVERAGE CAVEATS — read before trusting a breakdown
--------------------------------------------------------
A product only resolves if it has a column in the relevant sheet of
Inv_Data_Hardcoded.xlsx. As of this writing:

  * COUNTRY        — has BOTH 'Core Equity100' and 'Equity Alpha'.
  * SECTOR         — has BOTH 'Core Equity100' and 'Equity Alpha'.
  * ASSET CLASS    — has BOTH 'Core Equity100' and 'Equity Alpha'
                     (Equity Alpha = 100% Equity).
                     -> full look-through for the 50/50 example above across
                        country, sector and asset class.
  * TOP HOLDINGS   — the Underlying sheet carries per-ETF holdings
                     (CSPX, QQQ, XDEW, ...) but NO holdings column for the
                     managed portfolios themselves.

What that means in practice:
  - Any product WITHOUT a column in a sheet is folded into the "Others"
    bucket — "Others" is a data gap, NOT a real sector / asset class. With
    Equity Alpha now in every slice sheet, a 50/50 Equity100 + Equity Alpha
    book resolves cleanly (negligible "Others") in country, sector and
    asset-class views.
  - In top_holdings, a managed product with no holdings column passes
    through as its OWN name (e.g. "Equity Alpha": 0.5) instead of resolving
    to single stocks.

Remaining gap to close (data work, not code):
  - add a '<Product> Holdings' / '<Product> Holding Weights' column pair to
    the Underlying sheet for each managed portfolio, OR pass the product's
    ETF composition via ``custom_etf_nav`` so each ETF's existing holdings
    column is used for true single-name look-through.

These tools faithfully mirror the app; they do NOT paper over data gaps.
Inspect the returned weights — a large "Others" or a product-name passthrough
is the signal that a column is missing.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Dict, Optional

# Repo root = this file's directory. The local `mcp/` and `utils/` packages must
# import no matter what CWD the agent launches from.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_LOCAL_MCP = os.path.join(_REPO_ROOT, "mcp")

# --- resolve the `mcp` package-name collision (see module docstring) ----------
# fastmcp depends on the SDK and does `import mcp.types`. If the local `mcp/`
# shadows the SDK, that import dies. So: import the SDK `mcp` FIRST with the repo
# root OFF sys.path, then append the local `mcp/` dir to the SDK package's
# __path__. The submodule sets are disjoint, so afterwards `mcp.types` (SDK) and
# `mcp.exposure` (local) both resolve under one hybrid namespace.
_saved_path = list(sys.path)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _REPO_ROOT)]
try:
    import mcp as _sdk_mcp  # official MCP SDK — fastmcp needs mcp.types/.server
finally:
    sys.path[:] = _saved_path
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _LOCAL_MCP not in _sdk_mcp.__path__:
    _sdk_mcp.__path__.append(_LOCAL_MCP)

# The local `mcp.exposure` chain pulls in streamlit / plotly / yfinance, and the
# one-time workbook load can emit banners or "no runtime" warnings. On a stdio
# MCP transport, ANY stray write to real stdout corrupts the JSON-RPC channel,
# so we redirect stdout -> stderr for the whole startup (imports + data prime),
# then hand a pristine stdout to FastMCP when server.run() takes over. Priming
# the cache here with an explicit absolute path makes the tools CWD-independent
# and fails fast at startup (clear RuntimeError) if the workbook is missing.
_DATA_DIR = os.path.join(_REPO_ROOT, "Data")
with contextlib.redirect_stdout(sys.stderr):
    from fastmcp import FastMCP

    from mcp._data_loader import get_reference_data
    from mcp.exposure import (
        get_asset_class_exposure,
        get_country_exposure,
        get_sector_exposure,
        get_top_holdings,
    )

    get_reference_data(data_path=_DATA_DIR)

server = FastMCP("syfe-exposure")


# ---------------------------------------------------------------------------
# Product-name alias map: playground product_id / catalog display_name -> the
# exact column header used in Inv_Data_Hardcoded.xlsx (Country/Sector/Asset
# Class sheets). This is what lets the advisor bot take get_portfolios() output
# verbatim — where ``product`` is a product_id like "equity_alpha" and each row
# carries a ``nav`` — and feed ``{product_id: nav}`` straight into these tools.
# Keys are matched case-insensitively (and stripped). Anything not in the map
# (already-correct sheet names, raw ETF tickers, "CUSTOM") passes through
# unchanged, so this is purely additive — it never breaks a correct name.
#
# Composition-only sleeves ("select_custom", "brokerage") have NO managed column
# of their own; resolve them by passing their underlying ETF/stock composition
# via ``custom_etf_nav`` (keyed "CUSTOM") instead of relying on this map.
_PRODUCT_ALIASES = {
    # Core managed (product_id -> sheet column; display_name already identical)
    "core_defensive": "Core Defensive",
    "core_balanced": "Core Balanced",
    "core_growth": "Core Growth",
    "core_equity100": "Core Equity100",
    "equity_alpha": "Equity Alpha",
    # REIT+
    "reit_plus_100": "REIT 100",
    "reit+ (100% reits)": "REIT 100",
    "reit_plus_managed": "REIT Risk Managed",
    "reit+ with risk management": "REIT Risk Managed",
    # Income+
    "income_plus_preserve": "Income Preserve",
    "income+ preserve": "Income Preserve",
    "income_plus_enhance": "Income Enhance",
    "income+ enhance": "Income Enhance",
    # Select Themes
    "select_themes_tech": "Disruptive Technology",
    "select themes — disruptive tech": "Disruptive Technology",
    "select_themes_china": "China Growth",
    "select themes — china growth": "China Growth",
    # Cash+ (money-market / guaranteed)
    "cash_plus_flexi_sgd": "Cash SGD flexi",
    "cash+ flexi sgd": "Cash SGD flexi",
    "cash_plus_flexi_usd": "Cash USD flexi",
    "cash+ flexi usd": "Cash USD flexi",
    "cash_plus_guaranteed_sgd": "Cash SGD guaranteed",
    "cash+ guaranteed sgd": "Cash SGD guaranteed",
}


def _resolve_product_keys(
    weights: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    """Remap playground product_ids / display_names to sheet column headers.

    Case-insensitive, whitespace-tolerant. Unknown keys pass through unchanged
    so already-correct sheet names, ETF tickers and "CUSTOM" are untouched. If
    two input keys collapse to the same canonical name their weights are summed.
    """
    if not weights:
        return weights
    out: Dict[str, float] = {}
    for k, v in weights.items():
        canonical = _PRODUCT_ALIASES.get(str(k).strip().lower(), k)
        out[canonical] = out.get(canonical, 0.0) + float(v)
    return out


@server.tool()
def country_exposure(
    weights: Dict[str, float],
    custom_etf_nav: Optional[Dict[str, float]] = None,
) -> dict:
    """Blended country / region breakdown of a client's portfolio (look-through).

    Use when the client asks about geographic exposure ("how much US?",
    "what's my China / emerging-markets allocation?"). Works for any
    distribution of product weights.

    Args:
        weights: ``{product_name: weight_or_nav}`` — normalised weights,
            percentages, or raw NAVs (output is rescaled by their sum), e.g.
            ``{"Core Equity100": 0.5, "Equity Alpha": 0.5}`` or
            ``{"Core Equity100": 60000, "Equity Alpha": 40000}``. Use the exact
            product names from the data sheet (e.g. "Core Equity100", not
            "Equity100").
        custom_etf_nav: optional ``{etf_ticker: nav}`` to expand a "CUSTOM"
            (DIY) bucket into its constituent ETFs before computing exposure.

    Returns:
        ``{country: weight_0_to_1}`` sorted high to low, clipped non-negative
        and summing to 1.0. Any weight mapping to products/tickers not present
        in the Country sheet is collected under "Others".
    """
    return get_country_exposure(_resolve_product_keys(weights), custom_etf_nav)


@server.tool()
def sector_exposure(
    weights: Dict[str, float],
    custom_etf_nav: Optional[Dict[str, float]] = None,
) -> dict:
    """Blended sector breakdown of a client's portfolio (e.g. Technology, Financials).

    Use when the client asks which sectors they're exposed to or whether
    they're over/underweight a sector.

    COVERAGE: a product with no column in the Sector sheet is folded into an
    "Others" bucket (a data gap, not a real sector). Both 'Core Equity100' and
    'Equity Alpha' now have Sector columns, so a book of those resolves cleanly.
    If you do see a large "Others", flag it to the client as a data gap rather
    than presenting it as a genuine sector.

    Args:
        weights: ``{product_name: weight_or_nav}`` — normalised weights,
            percentages, or raw NAVs (output is rescaled by their sum).
        custom_etf_nav: optional CUSTOM-bucket expansion map.

    Returns:
        ``{sector: weight_0_to_1}`` sorted high to low, clipped non-negative and
        summing to 1.0 (unmapped weight collected under "Others").
    """
    return get_sector_exposure(_resolve_product_keys(weights), custom_etf_nav)


@server.tool()
def asset_class_exposure(
    weights: Dict[str, float],
    custom_etf_nav: Optional[Dict[str, float]] = None,
) -> dict:
    """Blended asset-class breakdown (Equity, Fixed income, Cash, REITs, ...).

    Use when the client asks how their money is split across asset classes.

    COVERAGE: a product with no column in the Asset Class sheet is folded into
    "Others". Both 'Core Equity100' and 'Equity Alpha' now have Asset Class
    columns ('Equity Alpha' = 100% Equity), so a book of those resolves cleanly.

    Args:
        weights: ``{product_name: weight_or_nav}`` — normalised weights,
            percentages, or raw NAVs (output is rescaled by their sum).
        custom_etf_nav: optional CUSTOM-bucket expansion map.

    Returns:
        ``{asset_class: weight_0_to_1}`` sorted high to low, clipped non-negative
        and summing to 1.0 (unmapped weight collected under "Others").
    """
    return get_asset_class_exposure(_resolve_product_keys(weights), custom_etf_nav)


@server.tool()
def top_holdings(
    weights: Dict[str, float],
    top_n: int = 10,
    custom_etf_nav: Optional[Dict[str, float]] = None,
) -> dict:
    """Top-N underlying single-name holdings across the whole portfolio (look-through).

    Use when the client asks "what stocks am I actually holding?" or "what's
    my biggest single-name exposure?". The same security held inside two
    different ETFs is summed; share-class variants (GOOG / GOOGL) are unified.

    COVERAGE: look-through only happens for tickers that have a holdings column
    in the Underlying sheet (per-ETF: CSPX, QQQ, XDEW, ...). A managed product
    with NO holdings column (currently 'Core Equity100' and 'Equity Alpha')
    passes through as its OWN name instead of resolving to single stocks. To
    get true stock-level look-through, add a '<Product> Holdings' /
    '<Product> Holding Weights' column pair, or pass the product's ETF
    composition via ``custom_etf_nav``.

    Args:
        weights: ``{product_name: weight}``.
        top_n: how many holdings to return (default 10).
        custom_etf_nav: optional CUSTOM-bucket expansion map.

    Returns:
        ``{"holdings": {name: weight_0_to_1}, "top_n": int,
           "total_weight_in_top_n": float}`` — holdings sorted high to low.
    """
    return get_top_holdings(
        _resolve_product_keys(weights), top_n=top_n, custom_etf_nav=custom_etf_nav
    )


if __name__ == "__main__":
    server.run()
