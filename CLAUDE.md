# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SOLAT** (Self-Optimizing Local Algorithmic Trading Platform) is an autonomous algorithmic trading system that operates on local hardware with zero infrastructure costs. It implements the Ichimoku Kinko Hyo strategy augmented by:
- **Evolutionary Layer**: Dynamically reallocates resources to assets with superior fitness metrics
- **HMM Regime Detection**: Uses Hidden Markov Models to identify Bull/Bear/Chop market conditions

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the trading engine (Sentinel backend)
python run_sentinel.py

# Run the dashboard (Observer frontend) - separate terminal
streamlit run run_dashboard.py

# Test data adapters (YFinance + CCXT)
python test_data.py

# Test core trading pipeline (single scan)
python test_core.py

# Test evolutionary optimizer
python test_evolution.py

# Validate settings configuration
python src/config/settings.py
```

## Architecture: Split-Process Concurrency Model

```
┌─────────────────────────────────────────────────────────────┐
│                    THE SENTINEL (Backend)                    │
│  Python process: data ingestion → strategy → evolution      │
│  Primary Writer to SQLite                                    │
└────────────────────────┬────────────────────────────────────┘
                         │ SQLite WAL Mode
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    trading_engine.db                         │
│  Tables: assets, market_snapshots, trades, evolution_metrics│
└────────────────────────┬────────────────────────────────────┘
                         │ Read-only (mode=ro)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    THE OBSERVER (Frontend)                   │
│  Streamlit dashboard: visualization, monitoring              │
│  Polls database every 30 seconds                            │
└─────────────────────────────────────────────────────────────┘
```

Both processes communicate **exclusively through SQLite in WAL mode**. No message brokers required.

## Key Source Files

| File | Purpose |
|------|---------|
| `src/core/engine.py` | Sentinel event loop, orchestrates scanning and evolution |
| `src/core/ichimoku.py` | Ichimoku Cloud strategy using `ta.trend.IchimokuIndicator` |
| `src/core/evolution.py` | Fitness scoring (F = 0.4·WinRate + 0.4·ProfitFactor - 0.2·Drawdown) |
| `src/core/regime.py` | HMM regime detector (Bull/Bear/Chop) using `hmmlearn` |
| `src/adapters/yfinance_lib.py` | Equities data with MultiIndex fix |
| `src/adapters/ccxt_lib.py` | Crypto data via Binance public API |
| `src/database/repository.py` | WAL-enabled SQLite connection handling |
| `src/config/settings.py` | All constants: intervals, thresholds, weights |
| `src/knowledge/brain.py` | Local RAG system using FAISS + sentence-transformers |
| `dashboard/app.py` | Streamlit multi-page dashboard with traffic light regime display |

## Signal Flow

1. **Data Fetch**: Adapters return standardized DataFrame (UTC DatetimeIndex, lowercase OHLCV columns)
2. **Ichimoku Analysis**: Generate BUY/SELL/NEUTRAL based on price vs cloud + TK cross
3. **Regime Filter**: HMM blocks signals misaligned with market regime (e.g., BUY in Bear)
4. **Persistence**: Log to `market_snapshots` and `trades` tables
5. **Evolution** (every 4 hours): Recalculate fitness, promote/demote assets

## Critical Constraints

### SQLite WAL Mode (Required)
Every database connection must execute:
```python
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")
```

### Adapter Output Format
All adapters must return DataFrames with:
- **Index**: UTC-aware `pd.DatetimeIndex`
- **Columns**: `open`, `high`, `low`, `close`, `volume` (lowercase, float64)

### Technical Indicators
Always use `ta.trend.IchimokuIndicator` from the `ta` library. Never use raw pandas rolling calculations.

### Type Safety
All function signatures require type hints:
```python
def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ...
```

## Common Pitfalls

1. **yfinance MultiIndex**: Recent versions return MultiIndex columns. The `YFinanceAdapter` flattens and lowercases them.
2. **CCXT timestamps**: Returns milliseconds. Convert with `pd.to_datetime(..., unit='ms', utc=True)`.
3. **Ichimoku NaN rows**: The indicator needs 52+ periods. Always drop NaN before analysis.
4. **HMM training data**: Requires 252+ periods (1 year) for reliable regime detection.
5. **Dashboard read-only**: Observer connects with `mode=ro` URI to prevent locking issues.

## Asset Status Lifecycle

| Status | Scan Interval | Selection |
|--------|--------------|-----------|
| **active** | 5 min | Top 20% by fitness |
| **normal** | 15 min | Middle 60% |
| **dormant** | 1 hour | Bottom 20% |

## Regime Filter Logic

| Regime | Allowed Signals | Blocked Signals |
|--------|----------------|-----------------|
| **bull** | BUY | SELL |
| **bear** | SELL | BUY |
| **chop** | None | ALL (kill zone) |

## Database Schema

```sql
-- Primary tables
assets(symbol, source, status, fitness_score, last_scan, updated_at)
market_snapshots(symbol, close_price, cloud_status, tk_cross, chikou_conf, regime, updated_at)
trades(id, symbol, side, entry_price, exit_price, pnl, exit_reason, entry_time)
```

## Configuration Constants (`src/config/settings.py`)

```python
EVOLUTION_EPOCH = 14400        # 4 hours between fitness recalculations
RISK_PER_TRADE = 0.02          # 2% account risk per trade
PAPER_TRADING_MODE = True      # Currently paper trading only
FITNESS_WEIGHTS = {"win_rate": 0.4, "profit_factor": 0.4, "drawdown": 0.2}
```

## Adding New Assets

Edit `src/config/assets_seed.json`:
```json
[
  {"symbol": "BTC/USDT", "source": "ccxt", "status": "active"},
  {"symbol": "AAPL", "source": "yfinance", "status": "active"}
]
```

The Sentinel loads this on first run if the assets table is empty.

## NumPy Version Constraint

**Important**: NumPy must be `<2.0` due to compatibility issues with FAISS and torch. This is enforced in `requirements.txt`:
```
numpy>=1.24.0,<2
```

## Additional Documentation

Extended documentation is in `/docs`:
- `QUICK_REFERENCE.md` - Command cheatsheet
- `HMM_REGIME_UPGRADE.md` - Regime detection deep-dive
- `RAG_LOCAL_BRAIN.md` - Local RAG architecture
- `QUICK_START_DASHBOARD.md` - Dashboard user guide
