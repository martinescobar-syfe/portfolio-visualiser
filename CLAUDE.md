# CLAUDE.md — Portfolio Visualiser

> Onboarding + agent-context document. Read this first before making any
> change. If you change a major formula, page layout, data source, or
> security/secret, **update the relevant section here** so the next
> reader stays oriented.

---

## 1. What this is

**Portfolio Visualiser** is a Streamlit app used by Syfe advisors to:

1. **Recommend** Syfe-managed portfolios for a new or existing client.
2. **Construct and backtest** custom multi-ETF portfolios with mean-variance optimisation, constraints, and risk attribution.
3. **Visualise an existing client's book** — current holdings, slice-and-dice exposures, diversification scoring, goal planning (growth / income), and projected cash flow.

The app is **three Streamlit pages** sharing a flat `utils/` layer, one Excel-backed reference dataset, and a handful of external integrations (Metabase, Gemini, Supabase, yfinance, Bloomberg-exported returns).

| Page | File | Audience |
|---|---|---|
| **Advisory Tool** (default landing) | `Advisory_Tool.py` | Advisors building recommendations + PDF reports |
| **Portfolio Builder** (password-gated) | `pages/Builder.py` | Power users constructing & backtesting custom portfolios |
| **Client Portfolio Visualizer** | `pages/Client_PV.py` | Advisors analysing an existing client's book + planning goals |

Run locally: `streamlit run Advisory_Tool.py` from inside `portfolio-visualiser/`. To host it for the team, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 2. Repository layout

```
portfolio-visualiser/
├── Advisory_Tool.py             # Entry-point Streamlit page (advisory flow)
├── pages/
│   ├── Builder.py               # Portfolio builder + backtester (~3,094 lines)
│   └── Client_PV.py             # Client-level dashboard + goal planner
├── utils/                       # Flat module layer (no __init__.py)
│   ├── app_secrets.py           # get_secret / require_secret (st.secrets → env)
│   ├── data_loader.py           # Excel ingest + yfinance/Bloomberg returns
│   ├── exposure.py              # Sector/country/asset-class aggregation, CUSTOM expansion
│   ├── diversification_score.py # 0–10 sub-scores + composite (with LLM insight inputs)
│   ├── cash_flow.py             # FI yield-based monthly income projection
│   ├── performance.py           # CAGR, vol, Sharpe, Sortino, TE, IR, rolling 12M excess
│   ├── risk_analytics.py        # MCTR, MCTE, VaR/CVaR, drawdowns, capture, Brinson, frontier
│   ├── risk_engine.py           # Deterministic client risk scoring + product eligibility
│   ├── optimiser.py             # Max-Sharpe / Min-Vol / Equal / Fixed rebalancing
│   ├── plotting.py              # Plotly chart builders (Syfe-branded)
│   ├── ui_functions.py          # Big UI layer for Advisory Tool (~4k lines)
│   ├── llm_helper.py            # Gemini integration (recommendations, insights)
│   ├── metabase_helper.py       # Live client lookups via Metabase cards
│   ├── portfolio_storage.py     # Supabase + local JSON fallback for saved Builder portfolios
│   ├── pdf_gen.py               # Streamlit PDF export UI (Advisory Tool)
│   ├── pdf_helper.py            # Brokerage-PDF parsing via Gemini + yfinance enrichment
│   └── portfolio_report_generator.py  # ReportLab Portfolio Summary PDF
├── mcp/                         # Headless calculator package behind the advisor MCP layer
│   ├── exposure.py              # NAV-friendly, normalised country/sector/asset-class slices
│   ├── _data_loader.py          # Loads Inv_Data_Hardcoded.xlsx into cached reference frames
│   ├── holdings.py              # CUSTOM expansion + client-holding aggregation
│   └── ...                      # cash_flow / diversification / forecast / performance / recommender
├── portfolio_exposure_mcp.py    # FastMCP server (stdio) — exposes exposure tools to the advisor bot (§13)
├── port_cons.py                 # Legacy optimisation + FI + FF5 helpers (used by Builder + ui_functions)
├── styles/
│   └── syfe_colors.py           # Brand palette + helper functions
├── Data/                        # Reference data (TRACKED in git, not ignored)
│   ├── Inv_Data_Hardcoded.xlsx              # Canonical static dataset (committed; powers the MCP exposure tools)
│   ├── F-F_Research_Data_5_Factors_2x3_daily.csv  # Kenneth French FF5 factors
│   ├── saved_portfolios.json                # Local fallback for Builder templates
│   ├── Portfolio_Visualiser_*.pdf           # PDF templates (cover, landscape, fonts)
│   └── ProximaNova-*.ttf                    # Syfe brand fonts
├── PDF Format/
│   └── Portfolio_Visualiser_Appendix.pdf
├── tests/                       # pytest + hypothesis suite (see tests/README.md)
├── scripts/pre-push             # Local git hook that runs pytest
├── .github/workflows/tests.yml  # CI on push + PR to main
├── .cursor/rules/
│   └── builder-module.mdc       # Deep-dive rule for Builder.py (auto-applies)
├── forecast_reference.py        # Offline batch script (Windows paths). NOT imported.
├── allocation_aggregation_reference.py  # Spec for exposure math. NOT imported.
├── Syfe_Context.md              # Long-form product catalogue (human ref; not imported)
├── insight_generation_instructions.md   # Prompt spec for diversification insights (human ref)
├── .mcp.json                    # Project-scoped MCP registration (the syfe-exposure server)
├── pytest.ini
├── requirements.txt
├── requirements-mcp.txt         # Extra deps to run the MCP server (fastmcp + base stack)
├── Dockerfile / .dockerignore   # Container image for internal hosting (not yet build-tested)
└── DEPLOYMENT.md                # Hosting guide: secrets, network, access control, smoke test
```

---

## 3. The three pages

### 3.1 Advisory Tool (`Advisory_Tool.py`)

**Purpose.** Default landing page. Thin orchestration: loads reference data once, renders base controls, branches on New vs Existing client, then delegates to the five big `render_*` functions in `utils/ui_functions.py`.

**Flow:**

```
load_static_data → base settings (dates, currency, benchmarks)
  → client-type radio
    ├─ Existing → render_export_ui (PDF) → render_client_upload (CSV)
    │              → render_client_pdf_upload (Gemini-parsed PDF)
    └─ shared  → render_recommender (LLM + risk engine)
               → render_portfolio_construction (build + backtest)
               → render_analysis (perf + exposures + factors + cashflow)
```

**Where to edit common things:**

| What you want | File / lines |
|---|---|
| Add/remove a base setting | `Advisory_Tool.py` around `with st.expander("⚙️ Base Settings"`) |
| Change the recommender behaviour | `utils/ui_functions.py::render_recommender` + `utils/risk_engine.py` |
| Add an analysis tab | `utils/ui_functions.py::render_analysis` |
| Add a PDF export section | `utils/pdf_gen.py::generate_report_content` |
| Change the LLM prompt | `utils/llm_helper.py` (also update `insight_generation_instructions.md`) |

**Caveats.** Most actual logic lives in `utils/ui_functions.py` (~4k lines). The page file itself is intentionally tiny. The `bm_tickers` default in `Advisory_Tool.py` is overridden by the Base Settings widget.

---

### 3.2 Portfolio Builder (`pages/Builder.py`)

**Purpose.** Power-user workflow for assembling, backtesting, and analysing ETF portfolios. Sidebar-gated by the `BUILDER_PASSWORD` secret (page stays locked if it is unset).

**The page is split into numbered sections that map directly to the UI headers:**

```
1. Universe Selection         → tickers, dates, benchmark blend, template load/save
2. Portfolio Construction     → allocation method, rebal frequency, fixed weights
   Constraints                → per-ETF, sector/country, tracking error
   🚀 Build & Optimise        → runs download → optimise → backtest → store results
3. Analysis Results           → Key Summary tabs + historical/drift weights
   3.1 Fixed Income Analytics → duration / YTM / T12M / credit
4. Exposure Analysis          → top holdings, sector/country vs benchmark
5. Risk & Attribution         → MCTR, MCTE, VaR/CVaR, drawdowns, capture, Brinson, frontier
6. Factor Analysis (FF5)      → portfolio + benchmark Fama-French regressions
7. Individual Asset Factor    → single-asset OLS + rolling betas
8. Overlap Matrix             → pairwise ETF holding overlap heatmap
9. Download Excel Report      → multi-sheet workbook export
```

**Deep details live in [`.cursor/rules/builder-module.mdc`](.cursor/rules/builder-module.mdc)** — that file is auto-applied to chats and is the source of truth for Builder formulas, conventions, FX overlay logic, pinned tests, and pitfalls. Always read it before changing any Builder calculation.

**Quick navigation:**

| Change | Line range |
|---|---|
| Optimiser logic / constraint validation | 465–759 (build button handler) |
| Add a new Analysis Results tab | 1319–1497 |
| Add an Excel sheet | 828–1255 (`create_builder_excel`) + 2824–3082 (export prep) |
| Default universe / constraints | 97–99, 205, 363–365 |
| Password gate | top of file, reads `BUILDER_PASSWORD` via `utils.app_secrets.get_secret` |

**Session-state primary key:** `st.session_state.builder_portfolio['results']` holds optimised weights, backtest perf, and ticker list.

---

### 3.3 Client Portfolio Visualizer (`pages/Client_PV.py`)

**Purpose.** End-to-end client dashboard. Three workflows in one page:

1. **Ingest holdings** — CSV upload *or* Metabase lookup, normalised by `process_client_data` into `{portfolio: nav}` plus expanded CUSTOM-ETF breakdown.
2. **Slice and dice exposures** — asset-class / geography / sector / concentration tabs, driven by `utils/exposure.get_exposure_charts` and `utils/diversification_score.calculate_diversification_score`. The "Filter Exposures" radio (Show All / By Objective / By Portfolio Type) drives the slice.
3. **Plan goals** — Growth or Income goals, each mixing existing portfolios (re-using client NAV) with new portfolios funded by lump sums + monthly top-ups. Goal detail view renders:
   - Monte-Carlo growth forecast (5th / 50th / 95th percentile bands).
   - Yield-based income forecast (binary-searched suggested top-up).
   - Backtested performance vs optional benchmark (cumulative / calendar heatmap / drawdown).
   - Goal-scoped diversification + exposure tabs.
   - For income goals: FI metrics table + monthly cash-flow chart.

**Forecast convention.** `_goal_weighted_params` blends per-portfolio capital-market assumptions (CMAs) by NAV weight in arithmetic-return space, then converts to log space for ONE log-normal projection. This is **Approach 1** (blend inputs, then forecast once) — *not* Approach 2 (forecast each sleeve then blend forecasts). Vol assumes zero cross-sleeve correlation (documented in `_goal_weighted_params` docstring). Performance backtest section separately builds `r_C(t) = Σ wᵢ·rᵢ(t)` from daily returns.

**Where to edit common things:**

| What you want | File / lines |
|---|---|
| Add a new portfolio mapping (raw type → display name) | `CLIENT_PORTFOLIO_MAPPING` at top of `Client_PV.py` |
| Update a portfolio's CMA | `PORTFOLIO_ESTIMATES` dict |
| Change the dust threshold | `process_client_data` (current: `nav < 50`) |
| Add a diversification sub-score | `utils/diversification_score.py` (also update `WEIGHTS` and the renderer in `Client_PV.py`) |
| Change the Monte-Carlo formula | `Growth Forecast` block in `Client_PV.py` |
| Edit the goal form | "GOAL FORM (unified)" section, ~line 2010+ |

**Session-state keys** are all prefixed `cpv_*` (e.g. `cpv_goals`, `cpv_active_goal_id`, `cpv_client_weights`, `cpv_df_client`). No cross-page sharing.

---

## 4. Data sources

### 4.1 Canonical reference data — `Data/Inv_Data_Hardcoded.xlsx`

Loaded **once** via `utils.data_loader.load_static_data(data_path)` and cached with `@st.cache_data`. Returns a 9-tuple:

| # | Variable | Excel sheet | Purpose |
|---|---|---|---|
| 1 | `df_sec_list` | Derived from Asset Class | Ticker / name / base currency / exchange / remarks |
| 2 | `df_asset_class` | **Asset Class** | Per-ticker asset-class exposure matrix |
| 3 | `df_sector` | **Sector** | Sector exposure matrix |
| 4 | `df_country` | **Country** | Country/region exposure matrix |
| 5 | `df_fi_metrics` | **FI Metrics** | Duration, YTM, T12M dividend yield, credit rating |
| 6 | `df_fx` | USDSGD/USDHKD cols from Daily Return; fallback `1.35` / `7.8` | FX for display conversion |
| 7 | `df_daily_ret` | **Daily Return** | All ticker + index daily returns (percent → decimal **once** at ingest) |
| 8 | `df_risk_rating` | **Risk** | Per-product risk ratings. **Sheet currently missing → empty df → PDF falls back to risk rating 2.0** |
| 9 | `df_underlying` | **Underlying** | ETF look-through holdings (paired `Holdings` / `Holding Weights` columns) |

**Important:** `Inv_Data_Hardcoded.xlsx` is **not in the repo** (supplied separately). The app fails loudly with `st.error` if missing. Path resolution order in `get_data_path()`: Colab `/content/Projects/Data/` → local `Data/` → `./`.

**Returns are decimals everywhere downstream** (e.g. `0.05` = 5%). The `/100` happens exactly once in `load_static_data`. There is a defensive secondary `/100` in `Client_PV.py` / `download_data` triggered only when `abs_max > 1` — leave it in place but don't introduce a third rescaling.

### 4.2 Fama-French factors — `Data/F-F_Research_Data_5_Factors_2x3_daily.csv`

Loaded by `pages/Builder.py::load_ff_data()` (cached) and by `utils/ui_functions.py`. Source is the Kenneth French data library. Format: `skiprows=3`, percent → decimal `/100`, index = `%Y%m%d`. Columns: `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `RF`. Used by Builder Sections 6 / 7 and the Excel `Factor Exposures (FF5)` sheet.

### 4.3 Live client data — Metabase

`utils/metabase_helper.py` calls saved cards on `https://metabase.internal.syfe.com` (Syfe VPN required). API key is read at call time from the `METABASE_API_KEY` secret (`utils/app_secrets.py`).

| Card | Template var | Purpose |
|---|---|---|
| **1642** | `email` | Client profile (age, risk_score, client_id, name) |
| **4905** | `investor_id` | Client ID → email lookup |
| **1470** | `client_id` | Active portfolios (filtered to `ACTIVE` / `ACTIVE_NEW`) |

Consumers: `pages/Client_PV.py` (Metabase Lookup tab) and `utils/ui_functions.py::render_client_upload`.

### 4.4 LLM — Gemini (`utils/llm_helper.py`)

`google.generativeai` with the `GEMINI_API_KEY` secret (read via `utils.app_secrets.get_secret`, passed in to the helper functions). Models are tried in fallback order (typical chain: `gemini-2.5-pro` → `2.5-flash` → `2.0-flash` → `1.5-flash`). Used for:

- Portfolio recommendations (`get_gemini_recommendation`, returns markdown tables that `parse_recommendation_table` converts via `product_map`).
- Plain-English narrative on a deterministic `risk_engine` allocation.
- Diversification insights (called from `Client_PV.py` in a **background thread** so the main script never blocks).
- Portfolio appropriateness check vs client risk profile.
- Brokerage-PDF parsing (`utils/pdf_helper.py`, multi-pass via `ThreadPoolExecutor`).

**All credentials come from `utils/app_secrets.py`** (`get_secret` / `require_secret`: `st.secrets` first, then environment variables). Nothing sensitive is hardcoded. See `.streamlit/secrets.toml.example` for the full list: `GEMINI_API_KEY`, `METABASE_API_KEY`, `BUILDER_PASSWORD`, optional `SUPABASE_URL` / `SUPABASE_KEY`. Keys that were hardcoded before this change remain in git history and must be treated as compromised — rotate them.

### 4.5 Saved Builder portfolios — Supabase + local JSON

`utils/portfolio_storage.py`. Primary: Supabase `portfolios` table (`name`, `data` JSON). Fallback: `Data/saved_portfolios.json`. Reads `SUPABASE_URL` / `SUPABASE_KEY` via `get_secret`; with neither set it uses the local JSON fallback.

### 4.6 yfinance

Used in two places:

1. `utils/data_loader.py::download_data` when `data_source == 'Yahoo Finance'` — pulls `Adj Close`, computes daily returns, suffixes `.L` / `.SW` for London / Swiss listings via `modify_tickers_based_on_exchange`.
2. `utils/pdf_helper.py::enrich_with_yfinance` to add country/sector/industry to brokerage-PDF-parsed holdings (rate-limited with `time.sleep`).

### 4.7 PDF templates and fonts

- Landscape template: `Data/Portfolio_Visualiser_Template_Landscape.pdf`
- Cover: `Data/Portfolio_Visualiser_Cover.pdf`
- Appendix: **`PDF Format/Portfolio_Visualiser_Appendix.pdf`** (relative to CWD, not `data_path`)
- Fonts: `Data/ProximaNova-{Bold,Regular}.ttf`

Generated under CWD: `temp_content.pdf`, `temp_templated.pdf`, `temp_merged.pdf`, `temp_cover_with_client.pdf` (cleaned up after generation).

---

## 5. Utility modules — one-line reference

| Module | What it is for |
|---|---|
| `data_loader` | Loads `Inv_Data_Hardcoded.xlsx` (cached) + builds yfinance/Bloomberg return panels with optional FX overlay. |
| `exposure` | Sector / country / asset-class aggregation, CUSTOM-portfolio expansion, look-through concentration, and exposure chart builders. |
| `diversification_score` | Composite 0–10 diversification score with four sub-scores (asset class, geography, sector, concentration). Emits `data_warnings` when exposure columns don't sum to 100%. |
| `cash_flow` | FI-eligible portfolio detection, blended yield retrieval, 12-month income projection, FI metrics table, monthly cash-flow plot. |
| `performance` | Standard daily-returns metrics: CAGR, vol, Sharpe, Sortino, max drawdown, TE, IR, rolling 12M excess (geometric). |
| `risk_analytics` | Builder-only advanced risk: MCTR, MCTE, VaR/CVaR, drawdown episodes, capture ratios, Brinson-Fachler, efficient frontier. |
| `risk_engine` | Deterministic client questionnaire → risk score (1–5) → eligible-portfolio filter and smart-recommendation dict. **No LLM.** |
| `optimiser` | SLSQP mean-variance optimisation with periodic rebalancing. `calc_port_perf` lags weights to avoid lookahead. |
| `plotting` | Plotly chart builders. `apply_syfe_layout` enforces the brand colour scheme via `styles/syfe_colors.PALETTE`. |
| `ui_functions` | Huge Streamlit UI layer for Advisory Tool (recommender, construction, analysis, upload, PDF upload). ~4k lines. |
| `llm_helper` | All Gemini calls + parsing + `SYFE_CONTEXT` product catalogue string. |
| `metabase_helper` | Live Metabase API calls — VPN required. |
| `portfolio_storage` | Supabase (primary) + local JSON (fallback) save/load/delete for Builder templates. |
| `pdf_gen` | Streamlit "Generate PDF Report" UI in Advisory Tool — assembles content blocks into a downloadable PDF. |
| `pdf_helper` | Brokerage-statement PDF parsing via Gemini, multi-pass + threaded, yfinance enrichment. |
| `portfolio_report_generator` | ReportLab Syfe-branded Portfolio Summary PDF (donut + cash-flow charts + grouped holdings table). |

**Dependency note.** `port_cons.py` (at project root, not under `utils/`) is the legacy analytics engine for rebalancing optimisers, FI exposure, credit ratings, and FF regression. It is imported by Builder, `ui_functions`, and many tests — **not** by `pages/Client_PV.py`. Don't move or delete it without checking every importer.

---

## 6. Running the app

```bash
# from the repo root (portfolio-visualiser/)
pip install -r requirements.txt           # Python 3.11 (matches CI)
streamlit run Advisory_Tool.py            # default landing
streamlit run pages/Builder.py            # Builder directly (still password-gated)
streamlit run pages/Client_PV.py          # Client_PV directly
```

The sidebar will show all three pages once any of them is the entry point because they all live under `pages/` or the project root (Streamlit auto-discovers `pages/`).

**Dependencies you'll likely hit issues with:**

- `Inv_Data_Hardcoded.xlsx` must exist under `Data/` (or wherever `get_data_path()` resolves).
- `GEMINI_API_KEY` is needed for any LLM-powered feature (recommender, insights, PDF parsing). Provide it via `.streamlit/secrets.toml` (copy from `secrets.toml.example`), the hosting platform's secrets, or the environment.
- Metabase lookup requires Syfe VPN.
- Supabase URL / key for cloud-stored Builder portfolios (otherwise fallback to local JSON).

---

## 7. Tests + CI

Full details in [`tests/README.md`](tests/README.md). Quick reference:

```bash
pytest tests/                       # full suite (~10 s)
pytest tests/ -v                    # verbose
pytest tests/test_performance.py    # one file
pytest tests/ -k brinson            # filter by name
pytest tests/ -m pinned             # only pinned regression tests
```

**Pinned regression tests** (`test_pinned_reference.py`, `test_full_pipeline.py`) lock down the exact numeric outputs of every UI section against a deterministic synthetic dataset (`tests/fixtures/reference_dataset.py`, seed=42). They fail loudly when any formula changes.

**After an intentional formula change**:

```bash
PINNED_REGEN=1 pytest tests/test_pinned_reference.py::test_print_for_regen -s
PINNED_REGEN=1 pytest tests/test_full_pipeline.py::test_print_for_regen -s
```

Paste the printed JSON into the `PINNED` / `EXPECTED` dicts in the respective test files and **commit the formula change and the new pinned values together**.

**Automation:**

- Local pre-push hook: `scripts/pre-push` runs the full suite before every `git push`. Install: `ln -sf ../../scripts/pre-push .git/hooks/pre-push && chmod +x scripts/pre-push`.
- CI: `.github/workflows/tests.yml` runs the same on every push/PR to `main`. Python 3.11, 10-minute timeout.

---

## 8. Conventions you must respect

These are enforced by tests and load-bearing for cross-section consistency. Breaking any of them silently corrupts displayed numbers.

1. **Returns are decimals**, not percent. `0.05` = 5%. The `/100` happens **once** in `data_loader.load_static_data` at ingest. Defensive guards (`if abs_max > 1: /100`) exist downstream — leave them in place but don't introduce additional rescalings.
2. **Annualisation = 252 trading days** everywhere in the Builder pipeline. `utils/performance.TRADING_DAYS_PER_YEAR = 252`. Legacy `261` in `port_cons.py` is stranded — don't reuse it in new Builder code.
3. **`_wgt` suffix stripping**: always use `c[:-4] if isinstance(c, str) and c.endswith('_wgt') else c`. Never `c.replace('_wgt', '')` — unsafe for unusual ticker names.
4. **Lagged weights for backtests**: `r_p,t = Σ_i w_{i,t-1} × r_{i,t}` with day-1 return set to 0. Both `utils/optimiser.calc_port_perf` and the legacy `port_cons.calc_port_perf` do `weights.shift(1)`. **Never use same-day weights × same-day returns for backtest performance.**
5. **Sign conventions**: Max Drawdown ≤ 0; VaR / CVaR are **negative** numbers (losses); skew/kurtosis are pandas defaults (Fisher excess).
6. **`ddof=1`** (sample std) for all volatility calculations.
7. **CAGR everywhere as "Annualized Return"** — not arithmetic mean × 252. Distribution Stats (Excel) and Return/Risk scatter both use CAGR. Switching to arithmetic-mean × 252 will desync cross-section reconciliation tests.
8. **FX overlay scope (Builder)**: applied to `all_port_ret`, `ret_bm`, `rets_period` (Brinson), `ret_all_for_risk_export` (Excel) — all displayed in user-selected currency. **FF regression always uses USD** (`*_base_ccy` returns) because Ken French factors are USD-denominated.
9. **LLM prompts**: no LaTeX, no escaped `$`. `utils/llm_helper.clean_llm_response()` fixes common artefacts; if you change a prompt, also update `insight_generation_instructions.md` for human reviewers.
10. **Forecast (Client_PV)**: blend CMAs at the *input* level then forecast once. Vol blend assumes zero cross-sleeve correlation by design.

---

## 9. Debugging cheat sheet

| Symptom | First place to look |
|---|---|
| `st.error("Inv_Data_Hardcoded.xlsx not found")` | Confirm CWD is `portfolio-visualiser/` and `Data/` contains the file. Check `get_data_path()` resolution. |
| Returns look 100× too large/small | Search for an accidental second `/100`. The legitimate one is in `data_loader.load_static_data`. |
| Empty analytics for a known ticker | Check it exists in the **Daily Return** sheet (~286 columns) and is spelled identically. |
| Risk ratings always show 2.0 in PDF | The **Risk** sheet is missing from the Excel file — add it or accept the fallback. |
| Metabase lookup fails | Syfe VPN connected? API key still valid? Card IDs unchanged (1642 / 4905 / 1470)? |
| LLM call hangs / fails | Model name change — check fallback chain in `llm_helper.py`. Confirm `GEMINI_API_KEY` is set. |
| Pinned tests fail unexpectedly | A formula changed somewhere. Read the diff — if intentional, regen pins (Section 7). If not, find the regression. |
| Builder shows wrong backtest after a constraint change | Check `constraint_debug` / `constraint_details` in session state for feasibility logs. |
| `_wgt` shows up in a chart label | Someone used `.replace('_wgt', '')` instead of the suffix-safe slice. |
| Brinson active ≠ realised active | Expected — Brinson is a single-period bucket model on time-averaged weights. Always show the Residual cell. |
| FX-overlaid chart looks wrong | Did you add a new asset-level returns chart without applying the `(1+R_USD)(1+R_FX)−1` overlay? See `_fx_col_risk` / `_fx_col_ef` / `fx_col_brin` in `Builder.py`. |
| Goal forecast gives 0 / NaN | Check `goal_navs` — all-zero goals fall back to equal weights to keep the page renderable. CMA fallback is `_DEFAULT_EST` (60/40 profile). |
| CUSTOM portfolios show as "unknown" in exposure | `actual_holding_weightage` column wasn't parseable by `utils.exposure.parse_custom_holdings`. Inspect the raw string format (`"SMH 33.67%, VTV 14.67%"` expected). |

---

## 10. Security and ops debt (known)

These are real but tolerated for now. Heads up before you commit changes that touch them.

1. **Leaked historical secrets**: code now reads secrets from config (§4.4), but the old Gemini key, Metabase key, Supabase key and Builder password `algo1` are still in git history from before the move. Rotate them; do not rely on them. History was not rewritten.
2. **Network requirements**:
   - Metabase: Syfe VPN.
   - Gemini, yfinance, Supabase, Ken French factor download: open internet.
3. **Missing canonical data**: `Inv_Data_Hardcoded.xlsx` is gitignored; supplied out-of-band.
4. **Stale comment**: `Advisory_Tool.py` Base Settings says "Bloomberg = pre-loaded local parquet"; actually it reads from the Excel.
5. **Duplicated cash-flow logic**: `utils/ui_functions.py` has an inline cash-flow projection in the Advisory Tool that *should* call `utils/cash_flow.compute_portfolio_cash_flows` but partially duplicates it. Consolidate when convenient.
6. **`pdf_gen.render_portfolio_summary_export_ui`** is defined but unreferenced — dead entry point.

---

## 11. When you want to change X, edit Y

| Goal | Files |
|---|---|
| Add a new Syfe managed portfolio | `CLIENT_PORTFOLIO_MAPPING` + `ALL_MANAGED_PORTFOLIOS` + `PORTFOLIO_ESTIMATES` in `pages/Client_PV.py`; `PORTFOLIO_TO_*` in `styles/syfe_colors.py`; product mapping in `utils/llm_helper.py` (`SYFE_CONTEXT` + `product_map` for recommender parsing). |
| Add a new ticker to the universe | Add a column to the **Daily Return** sheet, a row to **Asset Class** / **Sector** / **Country** / **FI Metrics** / **Underlying** as relevant in `Inv_Data_Hardcoded.xlsx`. No code change needed for most pages. |
| Change a CMA (expected return / vol / yield) | `PORTFOLIO_ESTIMATES` in `pages/Client_PV.py`. |
| Change a performance metric formula | `utils/performance.py` — then run `pytest tests/test_performance.py` and (if intentional) regen pinned values. |
| Change a Builder optimiser | `port_cons.py` for the engine; `pages/Builder.py` line ~465 for the wiring. |
| Add a new exposure dimension | Add a sheet to the Excel + a loader line in `utils/data_loader.py` + a renderer to `utils/exposure.get_exposure_charts` + a tab in the page. |
| Change what the advisor MCP exposure tools return | `mcp/exposure.py` (normalise / NAV-friendly logic) + `portfolio_exposure_mcp.py` (tool wrappers + product-name alias map). To map a new playground `product_id` to its sheet column, add an entry to `_PRODUCT_ALIASES`. See §13. |
| Reword a diversification insight | `utils/llm_helper.generate_diversification_insights` (prompt) **and** `insight_generation_instructions.md` (human-readable spec). |
| Add a new section to the PDF report | `utils/pdf_gen.generate_report_content` (Advisory Tool flow) or `utils/portfolio_report_generator.PortfolioReportGenerator.generate_report` (Portfolio Summary). |
| Tweak Syfe brand colours | `styles/syfe_colors.py`. Pages that hardcode hex (e.g. `Advisory_Tool.py` CSS) will need manual updates too. |
| Add or change a secret | Read it with `utils.app_secrets.get_secret` / `require_secret` (never hardcode), and add the name to `.streamlit/secrets.toml.example`. |

---

## 12. Reference / scratch files at the project root

These are intentionally **not imported** by the app but live in the repo for reference:

| File | Role |
|---|---|
| `port_cons.py` | **Production code** — legacy analytics engine. Imported by Builder + ui_functions. Do not delete. |
| `forecast_reference.py` | Offline batch quant script with Windows paths. Reference for production data pipeline. |
| `allocation_aggregation_reference.py` | Spec + reference implementation for the Digital Advisory allocation math (rounding to Other, 6 d.p., etc.). Use when validating exposure aggregation. |
| `Syfe_Context.md` | Long-form product catalogue. Shorter inline duplicate lives in `llm_helper.SYFE_CONTEXT`. |
| `insight_generation_instructions.md` | Prompt source-of-truth for diversification insights. Duplicated in `llm_helper.generate_diversification_insights`. **Update both** when changing insight behaviour. |
| `.cursor/rules/builder-module.mdc` | Deep-dive rule for Builder.py — auto-applies to chats. Source of truth for Builder formulas. |
| `tests/README.md` | Test-suite workflow. |

---

## 13. Advisor MCP — deterministic exposure + cashflow tools (`portfolio_exposure_mcp.py`)

**What it is.** A FastMCP (stdio) server exposing the look-through exposure math
(`mcp/exposure.py` → `utils/exposure.py`, backed by `Data/Inv_Data_Hardcoded.xlsx`)
as tools the advisor bot can call. The numbers are pure deterministic matrix
maths — no LLM in the loop.

**Tools (server name `syfe-exposure`):**

| Tool | Returns |
|---|---|
| `country_exposure(weights)` | `{country: weight}` — sums to 1, non-negative |
| `sector_exposure(weights)` | `{sector: weight}` — sums to 1, non-negative |
| `asset_class_exposure(weights)` | `{asset_class: weight}` — sums to 1, non-negative |
| `top_holdings(weights, top_n=10)` | top-N underlying single names (look-through); sums to what the slice captures, NOT forced to 1 |
| `weighted_dividend_yield(weights, base_currencies=None)` | NAV-weighted trailing-12M dividend yield over the WHOLE book → `{weighted_yield, weighted_yield_pct, annual_income, total_nav, coverage_pct, no_yield_data, breakdown}` (`no_yield_data` = sleeves with no yield figure) |
| `cash_flow(weights)` | projected next-12-months income → `{annual_total, monthly_average, effective_yield, monthly:{Jan..Dec}, breakdown}` (income sleeves only) |

The four exposure tools accept an optional `custom_etf_nav` map for CUSTOM-portfolio expansion.

**Two properties that make it advisor-friendly:**

- **NAV-friendly.** `weights` may be normalised weights, percentages, OR raw
  dollar NAVs — the output is rescaled by the input sum, so
  `{"Core Equity100": 60000, "Equity Alpha": 40000}` ≡
  `{"Core Equity100": 0.6, "Equity Alpha": 0.4}`.
- **Normalised + "Others".** Negatives are clipped to 0; the three slice tools
  always sum to 1.0; any input with no column in the relevant sheet is collected
  into `"Others"`. **A large `"Others"` is a data-coverage gap, not a real bucket.**

**Cashflow / yield tools.** `weighted_dividend_yield` is the blended portfolio
dividend yield — every sleeve's trailing-12M yield (`yield_preference="T12M"`
default; `"YTM"` available) weighted by NAV, **including** equity sleeves' small
dividends, so a growth-heavy book correctly shows a low yield. `cash_flow`
projects the 12-month income schedule but filters to income sleeves, so its
`effective_yield` (income ÷ total NAV) is **not** the whole-book dividend yield —
use `weighted_dividend_yield` for that. Both FX-convert internally: pass
`base_currencies={product_id: ccy}` (straight from `get_portfolios`) plus
`display_currency` and `usdsgd`/`usdhkd`, and a multi-currency book is exact — no
manual FX pre-normalisation needed (unlike the exposure tools). `coverage_pct`
reports the share of NAV carrying a non-zero yield (low = mostly growth/cash, not
a data gap).

**Auto-call recipe — how an exposure question gets answered.** When a client asks
"what's my country / sector / asset-class exposure?" or "what am I actually holding?":

1. `mcp__syfe-advisor-playground__get_portfolios` for the client → read each
   managed holding's product id and its NAV.
2. Build `{product_id: nav}`. Product ids like `equity_alpha`,
   `income_plus_enhance`, `cash_plus_guaranteed_sgd` are mapped to their sheet
   columns by `_PRODUCT_ALIASES` inside the server — **pass them through
   unchanged.** **FX first (multi-currency books):** the tools do NOT convert
   currencies, and `get_portfolios` returns each NAV in its own `base_currency`.
   Single-currency book → raw NAVs are fine. Mixed-currency book (e.g. Vincent:
   USD `equity_alpha` + SGD income/cash) → convert every NAV to one currency
   BEFORE calling, or the weights skew toward the larger-nominal-currency sleeves
   (Vincent reads 44% equity on raw NAVs vs ~51% once SGD→USD normalised).
3. Call `country_exposure` / `sector_exposure` / `asset_class_exposure` /
   `top_holdings` with that dict.
4. Read back the breakdown; flag any meaningful `"Others"` as missing data, not a
   real allocation.

Sleeves that are NOT managed-column products — `select_custom` and `brokerage`
rows — won't resolve via the alias map. Expand them through the holdings path
instead: `get_brokerage_holdings` / a CUSTOM row's `composition`, fed via
`custom_etf_nav` (see `mcp/holdings.aggregate_client_holdings`).

Worked example — Vincent's managed book
`{equity_alpha: 1.85M, income_plus_enhance: 1.30M, cash_plus_guaranteed_sgd: 1.05M}`
resolves to ≈ 44% Equity / 31% Fixed income / 25% Cash with `Others ≈ 0`.

**Product-name bridge.** The playground returns product *ids* (`equity_alpha`);
the Excel sheets use column *names* (`Equity Alpha`). `_PRODUCT_ALIASES` +
`_resolve_product_keys` in `portfolio_exposure_mcp.py` bridge the two and sum
duplicates. To support a new product, add one `product_id → "Sheet Column"`
entry to `_PRODUCT_ALIASES`.

**The `mcp` name collision (do not "simplify" the bootstrap).** This repo's local
`mcp/` package and the MCP SDK's top-level `mcp` collide. fastmcp depends on the
SDK (`import mcp.types`), so the server imports the SDK `mcp` first (repo root off
`sys.path`), then appends the local `mcp/` dir to `mcp.__path__` (the submodule
sets are disjoint → safe union). Flatten that and the server stops importing
(`ModuleNotFoundError: No module named 'mcp.types'`). Full rationale in the module
docstring.

**Run / register.**

```
pip install -r requirements-mcp.txt          # fastmcp + base stack
python portfolio_exposure_mcp.py             # stdio (manual smoke test)
```

Project-scoped registration lives in `.mcp.json` (server `syfe-exposure`, launched
as `python3 ${CLAUDE_PROJECT_DIR}/portfolio_exposure_mcp.py`); each user approves
the project MCP server once. The server locates its data via `__file__` (CWD
doesn't matter) and redirects stdout → stderr during import so the JSON-RPC
channel stays clean.

---

## 14. How to use this file as an agent

- **Skim section 1–3** to know which page handles what.
- **Section 4** is your data map. Always check this before assuming where a value comes from.
- **Section 5** is the utils cheatsheet — use it to know which module to read before changing behaviour.
- **Section 8** is non-negotiable. Breaking conventions = breaking pinned tests = breaking displayed numbers.
- **Section 9** is the first stop when a bug is reported.
- **Section 11** is the answer-key for "where do I make this change?".

When you edit a major surface area (page layout, formula, data source, secret, dependency), **update the relevant section of this file in the same commit**. CLAUDE.md is only useful if it's accurate.
