"""
SOLAT Platform Global Configuration Settings

This module defines all global constants and parameters used throughout the SOLAT system.
Centralizing configuration here ensures consistency and simplifies parameter tuning.

All values are immutable and should not be modified at runtime.
"""

# ============================================================================
# SCANNING INTERVALS (in seconds)
# ============================================================================
# Different scan frequencies based on asset status (adaptive resource allocation)

SCAN_INTERVAL_ACTIVE: int = 300  # 5 minutes - High-fitness assets scanned frequently
SCAN_INTERVAL_NORMAL: int = 900  # 15 minutes - Medium-fitness assets
SCAN_INTERVAL_DORMANT: int = 3600  # 1 hour - Low-fitness assets (saves API credits)

# ============================================================================
# EVOLUTION EPOCH (in seconds)
# ============================================================================
# How often the system recalculates fitness scores and reranks assets

EVOLUTION_EPOCH: int = 14400  # 4 hours - Sufficient to collect trade statistics

# ============================================================================
# RISK MANAGEMENT PARAMETERS
# ============================================================================
# Position sizing and risk per trade

RISK_PER_TRADE: float = 0.02  # 2% of account equity risked per trade
MAX_POSITION_SIZE_PERCENT: float = 0.10  # No single position > 10% of account
MAX_DRAWDOWN_ALLOWED: float = 0.20  # Stop trading if account drawdown > 20%

# ============================================================================
# SAFE MODE CONSTRAINTS (HARD RULES)
# ============================================================================
# These are non-negotiable safety limits for the institutional system

MAX_OPEN_TRADES: int = 2  # HARD LIMIT: Maximum concurrent open positions
# If 2 trades are open, the system MUST sleep until one closes

SAFE_MODE_ENABLED: bool = True  # Master switch for safe mode constraints

# ============================================================================
# EVOLUTIONARY ALGORITHM WEIGHTS (SAFE MODE: Win Rate Prioritized)
# ============================================================================
# Fitness score formula: F(x) = (w_wr * win_rate) + (w_pf * profit_factor) - (w_dd * max_drawdown)
# Weights must sum to 1.0 (or be normalized)
#
# SAFE MODE PHILOSOPHY: Prioritize Win Rate (consistency) over Profit Factor.
# A high win rate provides psychological safety and capital preservation.
# Profit factor is secondary - we'd rather have many small wins than
# occasional large wins with frequent losses.

FITNESS_WEIGHTS: dict = {
    "win_rate": 0.55,         # 55% weight: PRIORITIZED - Higher win rate = better fitness
    "profit_factor": 0.25,    # 25% weight: Secondary - More profit per unit risk
    "drawdown": 0.20,         # 20% weight: Penalize large drawdowns
}

# Asset selection thresholds (as percentiles)
ACTIVE_THRESHOLD: float = 0.80  # Top 20% of assets = Active (highest fitness)
DORMANT_THRESHOLD: float = 0.20  # Bottom 20% of assets = Dormant (lowest fitness)

# ============================================================================
# ICHIMOKU STRATEGY PARAMETERS
# ============================================================================
# Technical indicator periods (standard Ichimoku values)

ICHIMOKU_TENKAN: int = 9      # Short-term momentum (9 periods)
ICHIMOKU_KIJUN: int = 26      # Medium-term momentum (26 periods)
ICHIMOKU_SENKOU_B: int = 52   # Long-term trend (52 periods)
ICHIMOKU_DISPLACEMENT: int = 26  # Cloud displacement for forward projection

# ============================================================================
# DATA FETCHING PARAMETERS
# ============================================================================
# How much historical data to fetch for each asset

OHLCV_LOOKBACK: int = 100  # 100 candles for indicator calculation
OHLCV_TIMEFRAME_CRYPTO: str = "1h"   # Hourly data for crypto
OHLCV_TIMEFRAME_STOCKS: str = "1d"   # Daily data for stocks

# ============================================================================
# API RATE LIMITING & RESILIENCE
# ============================================================================
# Retry behavior for transient failures

MAX_API_RETRIES: int = 4  # Maximum retry attempts for API calls
RETRY_BACKOFF_BASE: int = 1  # Base sleep in seconds (exponential: 1, 2, 4, 8)

# ============================================================================
# DATABASE & LOGGING
# ============================================================================
# Paths and logging configuration

DB_PATH: str = "data/db/trading_engine.db"
SENTINEL_LOG_PATH: str = "data/logs/sentinel.log"
OBSERVER_LOG_PATH: str = "data/logs/observer.log"

# ============================================================================
# TRADING MODE
# ============================================================================
# Paper trading = log trades but don't execute. Always True for now.

PAPER_TRADING_MODE: bool = True  # Set to False for live trading (requires approval)

# ============================================================================
# VALIDATION & CONSTRAINTS
# ============================================================================

# ============================================================================
# CONSENSUS VOTING PARAMETERS (Council of 6)
# ============================================================================
# Multi-agent democratic voting system that learns from trade outcomes

CONSENSUS_VOTING_ENABLED: bool = True
CONSENSUS_THRESHOLD_EXECUTE: float = 0.60  # Execute if |score| > 0.60
CONSENSUS_THRESHOLD_DYNAMIC: bool = False  # If True, threshold adjusts with volatility

# Reinforcement Learning parameters
CONSENSUS_LEARNING_RATE: float = 0.02  # Alpha for weight updates (conservative, 2-week half-life)
CONSENSUS_LEARNING_MIN_WEIGHT: float = 0.05  # Minimum agent weight (5%)
CONSENSUS_LEARNING_MAX_WEIGHT: float = 0.35  # Maximum agent weight (35%)
CONSENSUS_LEARNING_RESET_DAYS: int = 7  # Reset agents hitting max weight

# Initial agent weights (domain-expertise-based, not equal)
AGENT_INITIAL_WEIGHTS: dict = {
    'regime': 0.20,           # HMM regime is proven
    'strategy': 0.22,         # Core Ichimoku strategy
    'sniper': 0.15,           # Microstructure analysis
    'news': 0.12,             # Sentiment is weaker signal
    'seasonality': 0.16,      # Time-based patterns
    'institutional': 0.15,    # Portfolio constraints
}

# Position sizing with consensus confidence
POSITION_SIZE_SCALING: bool = True  # Scale position by |consensus_score|
POSITION_SIZE_MIN_RATIO: float = 0.30  # Minimum 30% of Kelly (when consensus=0.3)

# Audit and monitoring
CONSENSUS_VOTING_AUDIT_LOG: bool = True  # Log all votes to DB
CONSENSUS_VOTING_VERBOSE: bool = False  # Detailed vote logging to console


def validate_settings() -> bool:
    """
    Validate that all settings are within acceptable ranges.

    Returns:
        bool: True if all settings are valid, False otherwise
    """
    checks = [
        ("RISK_PER_TRADE", 0 < RISK_PER_TRADE < 0.5, "Risk must be between 0% and 50%"),
        ("EVOLUTION_EPOCH", EVOLUTION_EPOCH > 0, "Evolution epoch must be positive"),
        ("SCAN_INTERVAL_ACTIVE", SCAN_INTERVAL_ACTIVE > 0, "Active scan interval must be positive"),
        ("ICHIMOKU_TENKAN", ICHIMOKU_TENKAN > 0, "Ichimoku tenkan must be positive"),
        ("FITNESS_WEIGHTS sum", abs(sum(FITNESS_WEIGHTS.values()) - 1.0) < 0.01,
         f"Fitness weights must sum to 1.0 (got {sum(FITNESS_WEIGHTS.values())})"),
        # Safe Mode validations
        ("MAX_OPEN_TRADES", 1 <= MAX_OPEN_TRADES <= 10,
         f"MAX_OPEN_TRADES must be between 1 and 10 (got {MAX_OPEN_TRADES})"),
        ("MAX_POSITION_SIZE_PERCENT", 0 < MAX_POSITION_SIZE_PERCENT <= 0.25,
         f"Max position size must be between 0% and 25% (got {MAX_POSITION_SIZE_PERCENT:.0%})"),
        ("WIN_RATE_PRIORITY", FITNESS_WEIGHTS["win_rate"] >= 0.5,
         f"Safe Mode requires win_rate weight >= 50% (got {FITNESS_WEIGHTS['win_rate']:.0%})"),
    ]

    all_valid = True
    for name, condition, message in checks:
        if not condition:
            print(f"✗ Validation Error [{name}]: {message}")
            all_valid = False
        else:
            print(f"✓ Validation OK [{name}]")

    # Print Safe Mode summary
    print("\n" + "=" * 40)
    print("SAFE MODE STATUS")
    print("=" * 40)
    print(f"  Enabled: {SAFE_MODE_ENABLED}")
    print(f"  Max Open Trades: {MAX_OPEN_TRADES}")
    print(f"  Max Position Size: {MAX_POSITION_SIZE_PERCENT:.0%}")
    print(f"  Win Rate Priority: {FITNESS_WEIGHTS['win_rate']:.0%}")
    print("=" * 40)

    return all_valid


if __name__ == "__main__":
    print("=" * 60)
    print("SOLAT PLATFORM SETTINGS VALIDATION")
    print("=" * 60)
    validate_settings()
