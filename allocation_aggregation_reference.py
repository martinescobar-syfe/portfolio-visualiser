"""
Reference implementation — Digital Advisory composition allocation (asset class, geography, sector).

Aligned with Client PV matrix math (``utils/exposure.calculate_exposure``) when holdings are
expressed as fractions of total: ``w_i = holding_value_i / base``.

Spec summary (portfolio & plan)
-------------------------------
- **Portfolio level:** For each dimension, for each category *c*:

  ``value_c = Σ_i holding_value_i × pct_{c,i}``

  ``base`` = **holding value of the portfolio** (sum of constituent / line-item holdings —
  **not** fund NAV unless holdings are recorded at NAV). ``pct_{c,i}`` is the share of
  constituent *i* allocated to category *c* (decimals summing to 1 per constituent per
  dimension after Investment Team normalization).

  ``%_c = value_c / base``.

- **Investment plan level:** Same formulas with base = Σ (portfolio holding values in the
  plan) and values summed across all constituents of all portfolios in the plan.

- **MVP:** Aggregation at **portfolio-constituent** (or product-line) granularity; columns in
  the matrix match spreadsheet keys. Future **security master**: swap column IDs; formulas
  unchanged.

Normalization & rounding (product)
----------------------------------
- Store weights as decimals with **6 d.p.**; **Σ = 1.000000** per dimension after normalization.
- UI shows **2 d.p.** percentages (percent points).
- Rounding residual: add to **Other** if present; else **largest** bucket; **log** adjustment.

This module implements the formulas, base handling options, and rounding rules.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping, MutableMapping

import pandas as pd

logger = logging.getLogger(__name__)

# Labels treated as "Other" for residual absorption (case-sensitive; extend if needed)
OTHER_LABELS = frozenset({"Others", "Other", "others", "other"})


# ---------------------------------------------------------------------------
# Copied from ``portfolio-visualiser/utils/exposure.py`` (Client PV)
# ---------------------------------------------------------------------------


def calculate_exposure(weights, df_exposure: pd.DataFrame) -> pd.Series:
    """Aggregate exposure by sector / country / asset class via matrix multiply.

    ``weights``: Series with optional ``*_wgt`` suffixes (PV). Values can be **normalized**
    weights (sum 1) or **dollar** holdings; output is in the same units.

    ``df_exposure``: rows = categories, columns = keys (portfolio products or constituents),
    cell = fraction in ``[0, 1]`` for that category (columns should sum to ~1 per key).

    Returns ``Σ_i w_i × pct_{category,i}`` per row.
    """
    if isinstance(weights, pd.DataFrame):
        weights = weights.iloc[-1]

    ticker_weights: dict[str, float] = {}
    for col in weights.index:
        if col.endswith("_wgt"):
            ticker = col.replace("_wgt", "")
            try:
                ticker_weights[ticker] = float(weights[col])
            except (ValueError, TypeError):
                continue

    weights = pd.Series(ticker_weights)

    if len(weights) == 0:
        return pd.Series(dtype=float)

    all_tickers = set(weights.index)
    missing_tickers = all_tickers - set(df_exposure.columns)
    common_tickers = weights.index.intersection(df_exposure.columns)
    agg_exposure = pd.Series(dtype=float)

    if len(common_tickers) > 0:
        weights_aligned = weights[common_tickers]
        exposure_matrix = df_exposure[common_tickers]
        exposure_matrix = exposure_matrix.apply(pd.to_numeric, errors="coerce").fillna(0)
        agg_exposure = exposure_matrix.dot(weights_aligned)

    if missing_tickers:
        is_asset_class = "Specialized / alternatives" in df_exposure.index or "Equity" in df_exposure.index
        fallback_weights: dict[str, float] = {}

        for ticker in missing_tickers:
            weight = weights[ticker]
            ticker_lower = ticker.lower()

            if "custom" in ticker_lower:
                fallback_cat = "Specialized / alternatives" if is_asset_class else "Others"
            else:
                fallback_cat = "Others"

            fallback_weights[fallback_cat] = fallback_weights.get(fallback_cat, 0) + weight

        agg_dict = agg_exposure.to_dict()
        for cat, weight in fallback_weights.items():
            agg_dict[cat] = agg_dict.get(cat, 0) + weight

        agg_exposure = pd.Series(agg_dict)

    return agg_exposure.sort_values(ascending=False)


def group_tail(series: pd.Series, top_n: int = 9) -> pd.Series:
    """Group tail of sorted exposure into ``Others`` (PV charts — presentation only)."""
    if len(series) <= top_n:
        return series

    top = series.head(top_n)
    others_val = series.iloc[top_n:].sum()

    if "Others" in top.index:
        top_dict = top.to_dict()
        top_dict["Others"] += others_val
        return pd.Series(top_dict)
    others = pd.Series({"Others": others_val})
    return pd.concat([top, others])


# ---------------------------------------------------------------------------
# Spec: portfolio & plan (holding value base)
# ---------------------------------------------------------------------------


def series_from_holdings_and_matrix(
    holding_value_by_key: Mapping[str, float],
    df_dimension: pd.DataFrame,
    *,
    portfolio_base: float | None = None,
) -> tuple[pd.Series, float]:
    """
    Portfolio-level: ``value_c = Σ_i holding_value_i × pct_{c,i}`` for keys *i* present as
    columns in ``df_dimension``.

    **Base (denominator for %)** — per spec, base is **holding value of the portfolio**
    (sum of all line items), not NAV.

    - If ``portfolio_base`` is **None**, ``base`` defaults to ``Σ holding_value`` over keys
      that appear in the matrix (same as "only mapped holdings"). Use this when the matrix
      covers all constituents.
    - If some holdings are **not** in the spreadsheet, pass ``portfolio_base =`` the full
      sum of all holding values; then ``fractions_from_values`` still uses this base so
      category % sum to (mapped_value_total / portfolio_base) ≤ 1; the gap is
      **unmapped** mass (handle via ``Unclassified`` row in data or separate line).

    Returns ``(values_per_category, base)``.
    """
    hv = pd.Series({k: float(v) for k, v in holding_value_by_key.items() if float(v) != 0.0})
    common = hv.index.intersection(df_dimension.columns)
    if len(common) == 0:
        b = float(portfolio_base) if portfolio_base is not None else float(hv.sum())
        return pd.Series(dtype=float), b

    m = df_dimension[common].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values = m.dot(hv[common])

    if portfolio_base is not None:
        base = float(portfolio_base)
        if base <= 0:
            return pd.Series(dtype=float), 0.0
    else:
        base = float(hv[common].sum())

    return values.sort_values(ascending=False), base


def fractions_from_values(values: pd.Series, base: float) -> pd.Series:
    """``%_c = value_c / base`` (asset class, geography, sector)."""
    if base <= 0:
        return pd.Series(dtype=float)
    return values / base


def aggregate_investment_plan(
    portfolio_holding_maps: Mapping[str, Mapping[str, float]],
    df_dimension: pd.DataFrame,
    *,
    portfolio_bases: Mapping[str, float] | None = None,
) -> tuple[pd.Series, float]:
    """
    Plan level: base = Σ (holding value of each portfolio in the plan).

    ``portfolio_holding_maps``: ``plan_portfolio_id → { constituent_or_product_key: hv }``.

    If ``portfolio_bases`` is provided, ``portfolio_bases[pid]`` overrides the base for that
    portfolio (full portfolio HV); otherwise each portfolio's base is the sum of its mapped
    keys (same rules as ``series_from_holdings_and_matrix``).
    """
    total_values: MutableMapping[str, float] = {}
    base_total = 0.0

    for pid, hv_map in portfolio_holding_maps.items():
        pb = portfolio_bases.get(pid) if portfolio_bases is not None else None
        part, b = series_from_holdings_and_matrix(hv_map, df_dimension, portfolio_base=pb)
        base_total += b
        for cat, v in part.items():
            total_values[cat] = total_values.get(cat, 0.0) + float(v)

    ser = pd.Series(total_values).sort_values(ascending=False)
    return ser, base_total


def assert_column_sums_unit(
    df_dimension: pd.DataFrame,
    tol: float = 1e-5,
    label: str = "dimension",
) -> list[str]:
    """
    Optional QA: each column should sum to ~1 (constituent fully allocated in that dimension).
    Returns list of column names where ``|sum - 1| > tol``.
    """
    bad: list[str] = []
    for col in df_dimension.columns:
        s = pd.to_numeric(df_dimension[col], errors="coerce").fillna(0).sum()
        if abs(float(s) - 1.0) > tol:
            bad.append(str(col))
    if bad:
        logger.warning(
            "%s matrix: column sum differs from 1.0 for: %s",
            label,
            bad[:20] + (["..."] if len(bad) > 20 else []),
        )
    return bad


# ---------------------------------------------------------------------------
# Normalization & rounding (product rules)
# ---------------------------------------------------------------------------


def normalize_weights_storage(
    weights: Mapping[str, float] | pd.Series,
    decimals: int = 6,
    *,
    other_labels: Iterable[str] = OTHER_LABELS,
) -> pd.Series:
    """
    Store weights as decimals with ``decimals`` d.p. and **Σ = 1** exactly.

    After rounding each bucket, residual ``1 - Σ rounded`` is applied to:
    1) a bucket whose key is in ``other_labels`` if present, else
    2) the **largest** bucket.

    Logs when a non-zero adjustment is applied.
    """
    s = pd.Series(weights, dtype=float).dropna()
    s = s[s != 0]
    total = float(s.sum())
    if total <= 0:
        return pd.Series(dtype=float)

    raw = s / total
    rounded = raw.round(decimals)
    drift = 1.0 - float(rounded.sum())

    if abs(drift) <= 10 ** (-(decimals + 2)):
        return rounded

    rounded = rounded.copy()
    other_key = next((k for k in rounded.index if k in other_labels), None)
    if other_key is not None:
        rounded.loc[other_key] = rounded.loc[other_key] + drift
        logger.info(
            "Storage normalization (%d d.p.): shifted residual %s to Other bucket %r",
            decimals,
            drift,
            other_key,
        )
    else:
        idx = rounded.idxmax()
        rounded.loc[idx] = rounded.loc[idx] + drift
        logger.info(
            "Storage normalization (%d d.p.): no Other bucket; shifted residual %s to largest %r",
            decimals,
            drift,
            idx,
        )

    return rounded


def display_percentages_two_dp(
    fractions: Mapping[str, float] | pd.Series,
    *,
    other_labels: Iterable[str] = OTHER_LABELS,
) -> tuple[dict[str, float], float]:
    """
    UI: percentage points at **2 d.p.**; if Σ ≠ 100.00, residual goes to **Other** if present,
    else **largest** bucket; log.
    """
    s = pd.Series(fractions, dtype=float)
    pct = (s * 100.0).round(2)
    residual = 100.0 - float(pct.sum())
    out = pct.to_dict()

    if abs(residual) < 0.005:
        return out, residual

    other_key = next((k for k in out if k in other_labels), None)
    if other_key is not None:
        out[other_key] = round(out[other_key] + residual, 2)
        logger.info(
            "UI rounding: shifted %.4f percentage points to Other %r",
            residual,
            other_key,
        )
    else:
        largest = max(out, key=lambda k: out[k])
        out[largest] = round(out[largest] + residual, 2)
        logger.info(
            "UI rounding: absorbed %.4f pp in largest bucket %r",
            residual,
            largest,
        )
    return out, residual


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df_ac = pd.DataFrame(
        {
            "Core Equity100": [0.98, 0.02],
            "Core Balanced": [0.55, 0.45],
        },
        index=["Equity", "Fixed Income"],
    )

    holdings = {"Core Equity100": 60_000.0, "Core Balanced": 40_000.0}
    vals, base = series_from_holdings_and_matrix(holdings, df_ac)
    fr = fractions_from_values(vals, base)
    print("Values:", vals.to_dict(), "base:", base)
    print("Fractions:", fr.to_dict())

    w_series = pd.Series({"Core Equity100_wgt": 0.6, "Core Balanced_wgt": 0.4})
    print("PV calculate_exposure:", calculate_exposure(w_series, df_ac).to_dict())

    disp, _ = display_percentages_two_dp(fr)
    print("UI %:", disp)
