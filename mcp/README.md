# mcp/ — calculator functions ready to wrap as MCP tools

This folder contains the **pure computational / data-access functions** from
the Portfolio Visualiser that are good first candidates to expose as MCP
tools to a chatbot. The rest of the repo is left untouched.

## The philosophy (the boring but important part)

Following the rule of thumb in the meeting:

- **Type 1 — pure computation.** Inputs in → outputs out. No DB, no UI,
  no Streamlit. → **MCP candidate.**
- **Type 2 — data access.** Reads a CSV / DB / API and returns rows.
  → **MCP candidate.**
- **Type 3 — glue / orchestration / UI controllers.** → **NOT** an MCP tool.
  The LLM becomes the new orchestrator.

Every file in this folder is Type 1 or Type 2 and obeys the four rules:

1. **Importable in isolation** — `from mcp.exposure import get_asset_class_exposure` works in a fresh Python file.
2. **Plain-Python args** — numbers, strings, dicts, lists. No DataFrames in
   the public signature.
3. **JSON-serialisable returns** — always dicts of primitives / lists /
   nested dicts. Never a `pd.DataFrame` out the door.
4. **LLM-friendly docstring** — first line is a verb-phrase summary; then
   a `Use when ...` line; then `Args` and `Returns`. The LLM reads this
   to decide which tool to call.

These functions **delegate** to the existing `utils/` modules instead of
re-implementing anything. The math should always match what the Streamlit
pages display — if it doesn't, that's a bug.

## File layout

| File | What it exposes | Underlying logic from |
|---|---|---|
| `_data_loader.py` | `get_reference_data()` — lazy-loads `Inv_Data_Hardcoded.xlsx` once per process | `utils/data_loader.py` |
| `holdings.py` | `parse_holdings_csv()`, `aggregate_client_holdings()`, `get_client_holdings_metabase()` | `pages/Client_PV.process_client_data` + `utils/metabase_helper` |
| `exposure.py` | `get_asset_class_exposure()`, `get_sector_exposure()`, `get_country_exposure()`, `get_top_holdings()` | `utils/exposure.py` |
| `diversification.py` | `get_diversification_score()` | `utils/diversification_score.py` |
| `performance.py` | `get_portfolio_metrics()`, `get_period_return()`, `get_relative_metrics()` | `utils/performance.py` |
| `cash_flow.py` | `project_monthly_cash_flow()`, `get_fixed_income_metrics()` | `utils/cash_flow.py` |
| `forecast.py` | `project_portfolio_growth()`, `project_income_payout()`, `suggested_topup_for_target()` | extracted from `pages/Client_PV.py` (Growth + Income forecast blocks) |
| `recommender.py` | `score_client_risk()`, `recommend_portfolio()`, `get_eligible_portfolios_for_risk()` | `utils/risk_engine.py` |

Plus:

- `dummy_data/sample_client_holdings.csv` — realistic shape, fake values,
  used by the test harness. Swap for a real Metabase pull when you're
  ready.
- `examples/run_all_tools.py` — runs every tool end-to-end on the dummy
  data. Use this to confirm nothing is broken before wiring an MCP server.

## How to use this from an MCP server

```python
# mcp_server.py (lives outside this folder; not included here)
from mcp.server.fastmcp import FastMCP
from mcp.exposure import get_asset_class_exposure
from mcp.forecast import project_portfolio_growth
from mcp.diversification import get_diversification_score
# ... etc.

server = FastMCP("syfe")

@server.tool()
def asset_class_breakdown(weights: dict) -> dict:
    """Return the asset-class breakdown of a client's portfolio."""
    return get_asset_class_exposure(weights)

@server.tool()
def project_growth(initial: float, monthly: float, years: int, mix: dict) -> dict:
    """Project a client portfolio's future value..."""
    return project_portfolio_growth(initial, monthly, years, mix)

if __name__ == "__main__":
    server.run()
```

The MCP server file is intentionally **not** in this folder — keep this
folder as a clean library that's usable from a server, a notebook, a
test, or anywhere else. The MCP server is just one more consumer.

## How to add a new tool

1. Find the existing function in `utils/` or in a Streamlit page that
   already does the calculation you need.
2. Decide which file in `mcp/` it belongs in (or add a new one).
3. Wrap it. The wrapper should:
   - Take primitive args.
   - Call the existing function.
   - Convert the result to a JSON-serialisable dict.
   - Have an LLM-friendly docstring.
4. Add a smoke call to `examples/run_all_tools.py`.
5. Run `python -m examples.run_all_tools` from the repo root. If it
   prints "OK" for your tool, you're done.

## What this folder is NOT

- Not a refactor of the existing repo. Pages and utils stay as-is.
- Not a re-implementation of the math. Wrappers delegate.
- Not the MCP server. That's a separate ~30-line file the user adds.
- Not production data access. Default is the dummy CSV. Swap to
  Metabase when ready.
