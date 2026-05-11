# Tests

Two layers of testing protect the analytics math:

1. **Property-based tests** (`pytest` + `hypothesis`) — assert invariants that
   should hold for *any* well-formed input (e.g. "MCTR component contributions
   always sum to portfolio vol", "VaR(99) ≤ VaR(95)", "Sortino is undefined
   when there are no negative returns").
2. **Pinned regression tests** — assert *exact* numerical outputs for a
   deterministic synthetic dataset (`tests/fixtures/reference_dataset.py`).
   If any formula changes — intentionally or accidentally — these will fail
   with a clear numeric diff.

## Run the suite

```bash
# from the project root
pytest tests/                       # full suite
pytest tests/ -v                    # verbose
pytest tests/test_performance.py    # one file
pytest tests/ -k brinson            # filter by name
pytest tests/ -m pinned             # only pinned regression tests
```

The full suite takes ~10 s.

## Files

| File | What it covers |
|---|---|
| `conftest.py` | Shared fixtures (synthetic returns, weights, benchmark, exposure table, lagged portfolio returns). |
| `fixtures/reference_dataset.py` | Deterministic dataset generator (seed=42). **Do not** change the seed without regenerating pinned values. |
| `test_performance.py` | `calculate_metrics`, `calculate_relative_metrics`. |
| `test_risk_analytics.py` | MCTR, MCTE, VaR/CVaR, drawdowns, capture ratios, Brinson, efficient frontier. |
| `test_optimiser.py` | `calc_port_perf` lookahead check, `min_vol` bounds, `create_constraints`, drift safety. |
| `test_factor.py` | FF5 fixed-spec OLS, Mkt-RF rename, missing-factor failure mode. |
| `test_exposure.py` | `calculate_exposure` matrix math, percent vs decimal auto-detect, overlap min kernel. |
| `test_fi_analytics.py` | `exposure_analysis_mixed` FI sleeve renormalisation, `calc_avg_credit_rating`. |
| `test_pinned_reference.py` | Hard-coded regression numbers — fail on any deviation. |
| `test_fx_overlay.py` | `(1+R_USD)(1+R_FX)−1` conversion: identity, multiplicative, self-inverse, double-apply guard. |
| `test_full_pipeline.py` | **End-to-end pinned values for every UI section** — Section 3 (Key Summary, Calendar, Cumulative, Rolling Vol, Drawdowns, Correlation, Relative Metrics, Rolling Excess geometric, Rolling TE), Section 3.1 (FI), Section 4 (exposure dot product, look-through), Section 5 (MCTR/MCTE sums, full VaR table, Drawdown ordering, Capture, Brinson reconciliation, Frontier extents + portfolio overlay), Section 6 (FF5 alpha/Mkt/R²), Section 8 (overlap kernel). Plus **cross-section reconciliation tests** (Max DD ↔ Drawdown Top1, Cumulative Return ↔ NAV-1, Distribution Stats ↔ Key Summary). |
| `test_excel_exporter.py` | End-to-end smoke: workbook builds, expected sheets present, Key Summary matches in-memory, Distribution Stats uses CAGR, Rolling Excess is geometric, Brinson uses time-averaged weights. |

## Updating pinned values after an intentional change

When you've **intentionally** changed a formula and the pinned tests now fail,
regenerate the expected values for both pinned files:

```bash
PINNED_REGEN=1 pytest tests/test_pinned_reference.py::test_print_for_regen -s
PINNED_REGEN=1 pytest tests/test_full_pipeline.py::test_print_for_regen -s
```

Each prints a JSON block. Open the corresponding test file
(`test_pinned_reference.py` and `test_full_pipeline.py`), paste the new
numbers into the `PINNED` / `EXPECTED` dict at the top of the file, and
re-run the suite. **Read the diff** before committing — pinned tests exist
precisely to make formula changes visible.

## Adding a new metric

1. Implement the formula in the appropriate module under `utils/` or
   `port_cons.py`.
2. Add **property tests** in `tests/test_<area>.py`. At minimum cover:
   - Sign / range invariants (e.g. "≥ 0", "≤ 0", "in [0, 1]").
   - Self-cases that should yield trivial results (e.g. portfolio == benchmark
     ⇒ TE = 0).
   - Known limit cases (e.g. all positive returns ⇒ Sortino diverges).
3. Add a **pinned expectation** for at least one realistic case to
   `test_pinned_reference.py` so future formula drift is caught.
4. Run `pytest tests/`. The CI gate is the same command.

## Hypothesis settings

The Hypothesis tests use `max_examples=50` and `deadline=None` for speed.
If you want more aggressive coverage during development, run:

```bash
pytest tests/ --hypothesis-show-statistics
```

or bump `max_examples` in the `@settings(...)` decorator on the relevant test.
