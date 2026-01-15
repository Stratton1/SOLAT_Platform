# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SOLAT** (Self-Optimizing Local Algorithmic Trading Platform) is an autonomous algorithmic trading system that operates on local hardware with zero infrastructure costs. It implements the Ichimoku Kinko Hyo trend-following strategy augmented by a novel "Evolutionary Layer" that dynamically reallocates computational resources to assets demonstrating superior fitness metrics.

Unlike static trading bots, SOLAT is an adaptive system that mimics natural selection—continuously evaluating asset performance and pruning low-fitness assets from the active trading pool.

## Architectural Vision

SOLAT rejects the monolithic script approach in favor of a **Split-Process Concurrency Model**:

1. **The Sentinel (Backend)**: A headless Python process responsible for heavy computation—data ingestion, signal generation, evolutionary scoring, and database writes. It is the primary "Writer" to the database.

2. **The Observer (Frontend)**: A Streamlit-based dashboard for visualization and monitoring. It reads from the database and can inject manual commands (e.g., "Force Sell") via a commands table. The Observer polls the database every 10-30 seconds for reactive updates.

Both processes communicate exclusively through a **shared SQLite database in WAL (Write-Ahead Logging) mode**, eliminating the need for message brokers like Redis while ensuring concurrent read/write access without blocking.

## Core Constraints & Technology Standards

1. **Zero-Budget Data**: Relies exclusively on yfinance (equities) and ccxt (crypto). No paid data subscriptions.
2. **Local Execution**: Single-machine deployment. Uses pandas_ta for vectorized technical analysis to work around Python's GIL.
3. **Database Concurrency**: SQLite with WAL mode (`PRAGMA journal_mode=WAL;`) is mandatory to prevent "database is locked" errors during concurrent access.
4. **Type Safety**: Strict type hinting required throughout all Python modules.
5. **Indicator Library**: Always use `pandas_ta` for technical indicators, never raw pandas calculations.

## File Structure Organization

```
SOLAT_Platform/
├── config/
│   ├── __init__.py
│   ├── settings.py             # Global constants (Timeouts, Thresholds)
│   ├── assets_seed.json        # Initial "Gene Pool" of tickers
│   └── secrets.toml            # API Keys (excluded from version control)
├── data/
│   ├── db/
│   │   ├── trading_engine.db   # Primary SQLite DB
│   │   ├── trading_engine.db-shm  # Shared memory file (WAL artifact)
│   │   └── trading_engine.db-wal  # Write-Ahead Log (WAL artifact)
│   └── logs/
│       ├── sentinel.log        # Backend logs
│       └── observer.log        # UI logs
├── src/
│   ├── __init__.py
│   ├── adapters/               # DATA INGESTION LAYER
│   │   ├── __init__.py
│   │   ├── interface.py        # Abstract Base Class for Adapters
│   │   ├── yfinance_lib.py     # YFinance implementation with MultiIndex fix
│   │   └── ccxt_lib.py         # CCXT implementation with normalization
│   ├── core/                   # TRADING LOGIC LAYER
│   │   ├── __init__.py
│   │   ├── engine.py           # Main Sentinel Event Loop
│   │   ├── ichimoku.py         # Strategy Logic & Pandas_TA wrappers
│   │   ├── evolution.py        # Fitness & Ranking Algorithms
│   │   └── risk.py             # Position Sizing Logic
│   └── database/               # PERSISTENCE LAYER
│       ├── __init__.py
│       ├── schema.py           # SQL DDL & Table Definitions
│       └── repository.py       # Connection handling & WAL setup
├── dashboard/                  # UI LAYER
│   ├── __init__.py
│   ├── app.py                  # Streamlit Entry Point
│   ├── views.py                # Page Layouts
│   └── charts.py               # Plotly Visualizations
├── run_sentinel.py             # Entry point for Backend
├── run_dashboard.py            # Entry point for Frontend
└── requirements.txt            # Dependency Pinning
```

## Data Ingestion Layer

The system ingests data from two sources through a **Unified Adapter Pattern**:

- **YFinance Adapter** (`src/adapters/yfinance_lib.py`): Handles the MultiIndex problem in recent yfinance versions. Always flattens MultiIndex DataFrames and renames columns to lowercase (Close → close, Volume → volume). Ensures UTC timezone-awareness.

- **CCXT Adapter** (`src/adapters/ccxt_lib.py`): Converts raw list-of-lists OHLCV data into normalized DataFrames with proper DatetimeIndex and column naming.

Both adapters must return a standardized format:
- **Index**: DatetimeIndex (UTC, timezone-aware)
- **Columns**: `open`, `high`, `low`, `close`, `volume` (lowercase, float64)

Implement **resilient request decorators** with exponential backoff for rate limit handling (1s → 2s → 4s → 8s).

## Database Schema

Four primary tables:

1. **assets**: Symbol, Source, Status (Active/Normal/Dormant), Fitness_Score, Last_Scan
2. **market_snapshots**: Symbol, Close_Price, Cloud_Status, TK_Cross, Chikou_Conf, Updated_At
3. **trades**: ID, Symbol, Side (BUY/SELL), Entry_Price, Exit_Price, PnL, Exit_Reason
4. **evolution_metrics**: Symbol, Win_Rate, Profit_Factor, Max_Drawdown

Every database connection must execute:
```python
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")
```

This is implemented in `src/database/repository.py:init_db()`.

## The Ichimoku Strategy

Ichimoku Cloud components calculated using `pandas_ta`:

- **Tenkan-sen** (9-period): (Max(9) + Min(9)) / 2
- **Kijun-sen** (26-period): (Max(26) + Min(26)) / 2
- **Senkou Span A**: (Tenkan + Kijun) / 2, shifted forward 26 periods
- **Senkou Span B** (52-period): (Max(52) + Min(52)) / 2, shifted forward 26 periods
- **Chikou Span**: Close price shifted backward 26 periods

**Signal Logic**:
- **Strong Buy**: Price > Cloud AND Tenkan > Kijun AND Chikou > Price (26 periods ago)
- **Cloud Breakout**: Price closes above Cloud (Bullish) or below Cloud (Bearish)
- **TK Cross**: Tenkan crosses above Kijun (Bullish) or below (Bearish)

## The Evolutionary Layer

The system periodically recalculates asset "Fitness Scores" to dynamically prune low-performing assets and allocate resources to high-performing ones.

**Fitness Formula**:
```
F(x) = (0.4 * WinRate) + (0.4 * ProfitFactor) - (0.2 * MaxDrawdown)
```

**Selection Mechanism** (runs every 4 hours):
- Top 20% of assets → **Active** (increased scan frequency)
- Middle 60% → **Normal** (standard scan frequency)
- Bottom 20% → **Dormant** (reduced scan frequency, saves API credits)

This self-optimization ensures SOLAT only trades markets where the Ichimoku strategy is currently effective.

## The Sentinel Event Loop

The Sentinel runs the core trading engine:

1. **Initialize**: Load assets from database
2. **Filter Candidates**: Select assets where `current_time - last_scan > scan_interval`
3. **Batch Processing**: For each candidate:
   - Fetch OHLCV data via adapter
   - Calculate Ichimoku indicators
   - Check for trading signals
   - If signal triggered, log trade to database
   - Update market snapshot
   - Update last_scan timestamp
4. **Evolution Check**: Every 4 hours, run the Evolutionary Optimizer to recalculate fitness scores and rerank assets
5. **Smart Sleep**: Avoid CPU thrashing between scan cycles

## Project Status

**Current Implementation Status**:
- ✓ Database layer (repository.py) - Complete, WAL-enabled
- ✓ Backend entry point (run_sentinel.py) - Scaffolded, ready for logic injection
- ✓ Data adapters interface (interface.py) - Complete
- ✓ YFinance adapter (yfinance_lib.py) - Complete with MultiIndex fix
- ✓ CCXT adapter (ccxt_lib.py) - Complete with normalization
- ✗ Trading engine (engine.py, ichimoku.py, evolution.py, risk.py) - Not started
- ✗ Frontend dashboard (app.py, views.py, charts.py) - Not started
- ✗ Configuration modules (settings.py, assets_seed.json) - Not started
- ✗ Test suite - Not started

**Lines of Code**: ~600+ lines implemented across 5 files

## Development Commands

**Installation**:
```bash
pip install -r requirements.txt
```

**Running the Application** (once trading engine is implemented):
```bash
# Terminal 1: Backend trading engine
python run_sentinel.py

# Terminal 2: Frontend dashboard
streamlit run run_dashboard.py
```

**Testing the Data Adapters**:
```bash
# Verify adapters work correctly with your data sources
python test_data.py
```

**Quality Tools** (to be installed):
```bash
# Code formatting
pip install black
black src/ dashboard/

# Type checking
pip install mypy
mypy src/ dashboard/

# Testing (when test suite is created)
pip install pytest
pytest tests/
```

**Current Limitations**:
- The Sentinel event loop is not yet implemented
- The Observer dashboard is not yet implemented
- Linting/formatting tools are not yet configured in the repo

## Getting Started with Development

Before implementing new modules, ensure familiarity with these concepts:

1. **Adapter Pattern**: Review `src/adapters/interface.py` and understand how YFinance and CCXT normalize data through a unified interface.
2. **Database Concurrency**: Understand WAL mode limitations and SQLite's concurrent read/write semantics. Always enable WAL on new connections.
3. **Ichimoku Calculations**: Review pandas_ta's `ichimoku()` output structure (8 columns with many NaN at the start due to shifting).
4. **Event Loop Design**: The Sentinel processes one asset at a time, not concurrently (GIL constraint). Use smart sleep intervals to prevent CPU thrashing.
5. **Evolutionary Ranking**: Fitness score recalculation happens every 4 hours, not in real-time. Pre-compute fitness metrics from trade history.

## Key Patterns and Conventions

1. **Abstract Adapter Pattern**: All data sources inherit from `src/adapters/interface.py` and implement the `get_ohlcv()` method. Return a DataFrame with:
   - **Index**: UTC-aware DatetimeIndex
   - **Columns**: `open`, `high`, `low`, `close`, `volume` (lowercase, float64)

2. **Resilient Requests**: The `resilient_request` decorator in `interface.py` handles rate limiting with exponential backoff (1s → 2s → 4s → 8s). All API calls should use this decorator.

3. **Database Transactions**: Always use connection context managers to ensure proper transaction handling and WAL synchronization.

4. **Logging**: Use Python's `logging` module with separate loggers for `sentinel` and `observer`. Log file paths: `data/logs/sentinel.log` and `data/logs/observer.log`.

5. **Configuration**: All global constants live in `config/settings.py`. Secrets go in `config/secrets.toml` (git-ignored).

6. **Signal Definitions**: Trade signals are immutable enums or constants defined in the strategy modules, not hardcoded in the event loop.

7. **Datetime Handling**: Always use UTC-aware DatetimeIndex. Never use naive datetimes. Ensure consistent timezone handling across adapters.

## Type Hints Examples

All function signatures must include type hints:

```python
from typing import Optional
import pandas as pd

def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Returns OHLCV data with UTC DatetimeIndex and lowercase columns."""
    pass

def calculate_fitness(win_rate: float, profit_factor: float, max_drawdown: float) -> float:
    """Returns weighted fitness score."""
    return (0.4 * win_rate) + (0.4 * profit_factor) - (0.2 * max_drawdown)

def get_connection() -> sqlite3.Connection:
    """Returns a WAL-configured SQLite connection."""
    pass
```

## Common Pitfalls to Avoid

1. **MultiIndex DataFrames**: Recent yfinance versions return MultiIndex DataFrames. Always flatten and rename to lowercase columns in the adapter.
2. **Timezone Naive Datetimes**: Never use `pd.Timestamp` without timezone info. Always use UTC and set as the DataFrame index.
3. **Raw pandas_ta Output**: The `ichimoku()` function returns 8 columns with many NaN at the start due to shifting. Drop NaN rows before analysis.
4. **SQLite Locking**: Always use WAL mode. Without it, the Observer and Sentinel will deadlock during concurrent read/write.
5. **Rate Limiting**: yfinance and CCXT have rate limits. Use the `resilient_request` decorator—don't retry immediately.
6. **Config Files vs Database**: Settings go in `config/settings.py`. Runtime state (current trades, fitness scores) goes in the database.
7. **CCXT Timestamp Handling**: CCXT returns timestamps in milliseconds. Always convert to `pd.to_datetime(..., unit='ms', utc=True)`.

## Important Rules

- **Type Hints**: Strict type annotations required on all function signatures. Use `from typing import ...` for complex types.
- **SQLite WAL**: Every database initialization must explicitly enable WAL mode and set `synchronous=NORMAL`.
- **pandas_ta Only**: All technical indicators must use `pandas_ta`, never raw pandas rolling calculations.
- **No Blocking Operations**: The Sentinel must use non-blocking I/O and smart sleep intervals to avoid blocking the Observer.
- **Immutable Config**: `config/settings.py` should contain only immutable, read-only configuration. Runtime state belongs in the database.
- **Adapter Output Format**: All adapters must return DataFrames with identical structure: UTC DatetimeIndex + lowercase OHLCV columns.

## Glossary

- **The Sentinel**: Backend trading engine process, primary database writer
- **The Observer**: Streamlit frontend, database reader with command injection capability
- **Gene Pool**: The dynamic watchlist of assets, evolving based on fitness scores
- **Fitness Score**: Metric combining win rate, profit factor, and drawdown; determines asset priority
- **Selection Epoch**: Periodic (4-hour) recalculation of asset fitness and reranking
- **WAL Mode**: Write-Ahead Logging; SQLite configuration enabling concurrent reads and writes
- **WAL Artifacts**: The `.db-shm` and `.db-wal` files created alongside the main database file; do not delete
- **Scan Interval**: Time between data fetches for a specific asset (varies by status: Active/Normal/Dormant)
- **Market Adapter**: Unified interface for fetching OHLCV data from different sources (yfinance, CCXT)
