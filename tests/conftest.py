"""Shared pytest fixtures.

The reference dataset is pinned: a fixed seed produces an identical return
panel and weight history every run. This lets us write **pinned regression
tests** that catch any numeric change to a metric, even ones a code reviewer
might miss.

For property tests we generate fresh random panels through Hypothesis
strategies (see individual test files).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the project root importable so tests can do
# `from utils.performance import ...` etc. without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fixtures.reference_dataset import (  # noqa: E402  (import after sys.path tweak)
    BENCHMARK_TICKER,
    TICKERS,
    make_benchmark_weights,
    make_exposure_table,
    make_returns,
    make_weights,
)


@pytest.fixture(scope="session")
def returns_panel() -> pd.DataFrame:
    """Daily return DataFrame for assets + benchmark."""
    return make_returns()


@pytest.fixture(scope="session")
def asset_returns(returns_panel: pd.DataFrame) -> pd.DataFrame:
    return returns_panel[list(TICKERS)]


@pytest.fixture(scope="session")
def bench_returns(returns_panel: pd.DataFrame) -> pd.Series:
    return returns_panel[BENCHMARK_TICKER]


@pytest.fixture(scope="session")
def weights_history(returns_panel: pd.DataFrame) -> pd.DataFrame:
    """Daily weights with `reb_flag` and `_wgt` suffixes — matches what
    the optimiser pipeline produces in the live app."""
    return make_weights(returns_panel)


@pytest.fixture(scope="session")
def latest_weights(weights_history: pd.DataFrame) -> pd.Series:
    """Last rebalance row, with `reb_flag` dropped and `_wgt` suffix stripped."""
    rebal_rows = weights_history[weights_history["reb_flag"] == True]
    last = rebal_rows.iloc[-1].drop("reb_flag")
    last.index = [c[:-4] if isinstance(c, str) and c.endswith("_wgt") else c for c in last.index]
    return last[last > 0]


@pytest.fixture(scope="session")
def benchmark_weights() -> pd.Series:
    return make_benchmark_weights()


@pytest.fixture(scope="session")
def exposure_table() -> pd.DataFrame:
    return make_exposure_table()


@pytest.fixture(scope="session")
def portfolio_returns(asset_returns: pd.DataFrame, weights_history: pd.DataFrame) -> pd.Series:
    """Realised daily portfolio return using lagged weights (no lookahead).

    Mirrors what `utils.optimiser.calc_port_perf` should produce.
    """
    weights_only = weights_history[[c for c in weights_history.columns if c.endswith("_wgt")]].copy()
    weights_only.columns = [c[:-4] for c in weights_only.columns]
    weights_lagged = weights_only.shift(1)
    port_ret = (asset_returns * weights_lagged).sum(axis=1)
    port_ret.iloc[0] = 0.0
    port_ret.name = "Portfolio"
    return port_ret
