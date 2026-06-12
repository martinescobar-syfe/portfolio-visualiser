"""Pin ``portfolio_exposure_mcp._PRODUCT_ALIASES`` to the data sheets.

The advisor bot feeds ``{product_id: nav}`` straight from the playground into the
exposure tools; ``_resolve_product_keys`` remaps those product_ids / display
names onto the exact managed-portfolio column headers in
``Data/Inv_Data_Hardcoded.xlsx`` (Asset Class / Sector / Country sheets). If the
map and the sheets ever drift apart — a typo'd target, a renamed column, a newly
added managed sleeve nobody aliased — a client's NAV silently lands in the
``"Others"`` bucket instead of resolving. These tests make that drift a loud test
failure instead of a silent wrong number.

The whole module imports ``portfolio_exposure_mcp``, which pulls in ``fastmcp``
(and the MCP SDK). CI installs only ``requirements.txt`` (no fastmcp), so we
``importorskip`` to skip there; the local ``scripts/pre-push`` hook has fastmcp
installed and enforces these invariants before every push.
"""

from __future__ import annotations

import pytest

# fastmcp is an MCP-server-only dependency (requirements-mcp.txt). Skip the whole
# module wherever it isn't installed (e.g. the GitHub Actions test job).
pytest.importorskip("fastmcp")

import portfolio_exposure_mcp as pem  # noqa: E402  (import after importorskip)

# The first managed-portfolio column in every slice sheet. Everything from this
# column to the end of the sheet is a managed product (the columns before it are
# raw ETF / single-stock tickers). This mirrors the user's framing: "the
# portfolio names starting from core equity 100 up till the end".
_FIRST_MANAGED_COLUMN = "Core Equity100"


def _managed_columns(df) -> list[str]:
    """Columns from ``Core Equity100`` to the end of a slice sheet (the managed
    block), in sheet order."""
    cols = list(df.columns)
    assert _FIRST_MANAGED_COLUMN in cols, (
        f"{_FIRST_MANAGED_COLUMN!r} not found in sheet columns; the managed block "
        "anchor has moved — update _FIRST_MANAGED_COLUMN."
    )
    return cols[cols.index(_FIRST_MANAGED_COLUMN):]


@pytest.fixture(scope="module")
def slice_sheets():
    """The three look-through slice sheets, keyed by name. Reuses the cache that
    ``portfolio_exposure_mcp`` primes at import time."""
    ref = pem.get_reference_data()
    return {
        "Asset Class": ref.asset_class,
        "Sector": ref.sector,
        "Country": ref.country,
    }


def test_alias_targets_exist_in_every_slice_sheet(slice_sheets) -> None:
    """Every canonical value in the alias map must be a real column in ALL three
    slice sheets — otherwise that product resolves in one view but leaks into
    "Others" in another (exactly the BlackRock/Blackrock casing bug)."""
    targets = set(pem._PRODUCT_ALIASES.values())
    for sheet_name, df in slice_sheets.items():
        cols = set(df.columns)
        missing = sorted(t for t in targets if t not in cols)
        assert not missing, (
            f"alias targets absent from the {sheet_name} sheet: {missing}. "
            "Either the sheet column was renamed or the alias value has a typo."
        )


def test_all_managed_columns_are_aliased(slice_sheets) -> None:
    """Every managed-portfolio column (Core Equity100 → end) must be reachable as
    an alias target. Adding a managed sleeve to the sheets without aliasing it
    should fail here — that is the "match my data sheets" guarantee."""
    targets = set(pem._PRODUCT_ALIASES.values())
    for sheet_name, df in slice_sheets.items():
        managed = _managed_columns(df)
        uncovered = sorted(c for c in managed if c not in targets)
        assert not uncovered, (
            f"{sheet_name} sheet has managed columns with no alias target: "
            f"{uncovered}. Add them to _PRODUCT_ALIASES so client NAVs resolve."
        )


def test_managed_block_is_consistent_across_sheets(slice_sheets) -> None:
    """All three sheets must expose the SAME set of managed columns (order may
    differ). A mismatch means a product exists in one view but not another."""
    blocks = {name: set(_managed_columns(df)) for name, df in slice_sheets.items()}
    ref_name, ref_set = next(iter(blocks.items()))
    for name, s in blocks.items():
        assert s == ref_set, (
            f"managed-column set differs between {ref_name} and {name}: "
            f"only in {ref_name}={sorted(ref_set - s)}, only in {name}={sorted(s - ref_set)}"
        )


def test_alias_keys_are_normalised() -> None:
    """Lookup lowercases+strips the input key, so every alias KEY must already be
    lowercase and stripped or it can never be hit."""
    bad = [k for k in pem._PRODUCT_ALIASES if k != k.strip().lower()]
    assert not bad, f"alias keys must be lowercase & stripped to ever match: {bad}"


def test_resolve_known_product_id() -> None:
    """A playground product_id resolves to its exact sheet column."""
    assert pem._resolve_product_keys({"equity_alpha": 1.0}) == {"Equity Alpha": 1.0}
    assert pem._resolve_product_keys({"core_equity100": 0.5}) == {"Core Equity100": 0.5}


def test_resolve_is_case_and_whitespace_insensitive() -> None:
    """Mixed case and surrounding whitespace still resolve."""
    assert pem._resolve_product_keys({"  Equity_Alpha  ": 1.0}) == {"Equity Alpha": 1.0}
    assert pem._resolve_product_keys({"CHINA GROWTH": 0.25}) == {"China Growth": 0.25}


def test_resolve_passthrough_for_unknown_keys() -> None:
    """Unknown keys (raw ETF tickers, already-correct sheet names, CUSTOM) pass
    through unchanged so the map is purely additive."""
    assert pem._resolve_product_keys({"CSPX": 0.4}) == {"CSPX": 0.4}
    assert pem._resolve_product_keys({"Equity Alpha": 0.4}) == {"Equity Alpha": 0.4}
    assert pem._resolve_product_keys({"CUSTOM": 0.4}) == {"CUSTOM": 0.4}


def test_resolve_sums_duplicate_targets() -> None:
    """Two input keys that collapse to the same canonical column have their
    weights summed (e.g. a product_id and its display name)."""
    out = pem._resolve_product_keys({"equity_alpha": 0.3, "Equity Alpha": 0.2})
    assert out == {"Equity Alpha": pytest.approx(0.5)}


def test_resolve_handles_empty_and_none() -> None:
    """Empty / None inputs pass straight through (no crash)."""
    assert pem._resolve_product_keys(None) is None
    assert pem._resolve_product_keys({}) == {}


# --- normalisation-fallback (slug) resolution -------------------------------
# Multi-client testing showed alt-fund product_id slugs (not in the playground
# catalog yet) leaked 100% into "Others". The resolver now has a normalised-slug
# fallback; these tests pin that behaviour and its safety invariants.

def test_normalisation_is_injective_over_canonical_names() -> None:
    """The fallback must not collapse two DISTINCT managed columns onto the same
    normalised key — otherwise a slug could silently resolve to the wrong
    product. (This is what makes deriving matches from canonical names safe.)"""
    seen: dict[str, str] = {}
    for c in sorted(set(pem._PRODUCT_ALIASES.values())):
        n = pem._norm_key(c)
        assert n not in seen, (
            f"normalisation collision: {c!r} and {seen[n]!r} both normalise to {n!r}")
        seen[n] = c


def test_alt_fund_underscore_slugs_resolve() -> None:
    """Private Wealth alt-fund slugs must resolve to their sheet columns via the
    normalisation fallback — the leak found in multi-client testing."""
    cases = {
        "blackrock_private_credit": "Blackrock Private Credit Fund",
        "blackrock_private_credit_fund": "Blackrock Private Credit Fund",
        "monroe_income_plus": "Monroe Income Plus Fund",
        "monroe_income_plus_fund": "Monroe Income Plus Fund",
        "silverdale_nov_2027": "Silverdale Fund (Nov 2027)",
        "silverdale_oct_2028": "Silverdale Fund (Oct 2028)",
        "silverdale_feb_2029": "Silverdale Fund (Feb 2029)",
    }
    for slug, canonical in cases.items():
        assert pem._resolve_product_keys({slug: 1.0}) == {canonical: 1.0}, slug


def test_slug_resolved_targets_are_real_columns(slice_sheets) -> None:
    """The columns the alt-fund slugs resolve to must exist in every slice sheet
    (else they'd still land in Others despite resolving)."""
    targets = {"Blackrock Private Credit Fund", "Monroe Income Plus Fund",
               "Silverdale Fund (Nov 2027)", "Silverdale Fund (Oct 2028)",
               "Silverdale Fund (Feb 2029)"}
    for sheet_name, df in slice_sheets.items():
        missing = sorted(t for t in targets if t not in set(df.columns))
        assert not missing, f"{sheet_name} sheet missing: {missing}"


def test_normalisation_only_runs_on_miss() -> None:
    """Exact matches and unknown tickers are unaffected by the fallback."""
    assert pem._resolve_product_keys({"equity_alpha": 1.0}) == {"Equity Alpha": 1.0}
    # bare tickers normalise to themselves and aren't in the index -> passthrough
    assert pem._resolve_product_keys({"QQQ": 1.0}) == {"QQQ": 1.0}
    assert pem._resolve_product_keys({"AGGG": 1.0}) == {"AGGG": 1.0}
    assert pem._resolve_product_keys({"CUSTOM": 1.0}) == {"CUSTOM": 1.0}


def test_resolve_drops_nonfinite_and_clamps_magnitude() -> None:
    """NaN / inf NAVs are dropped and absurd magnitudes are clamped, so a garbage
    value can't overflow a downstream sum to +inf and leak NaN (stress-test Bug A)."""
    assert pem._resolve_product_keys({"core_equity100": float("nan"), "equity_alpha": 1.0}) == {"Equity Alpha": 1.0}
    assert pem._resolve_product_keys({"core_equity100": float("inf"), "equity_alpha": 1.0}) == {"Equity Alpha": 1.0}
    assert pem._resolve_product_keys({"core_equity100": 1e308}) == {"Core Equity100": 1e15}
    assert pem._resolve_product_keys({"core_equity100": -1e308}) == {"Core Equity100": -1e15}
