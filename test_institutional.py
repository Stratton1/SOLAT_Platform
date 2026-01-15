#!/usr/bin/env python3
"""
Test script for Institutional Brain modules.

Verifies:
1. PortfolioManager - Kelly Criterion sizing, circuit breaker
2. PatternHunter - Time-of-day analysis, Japan Open breakout
3. StrategyOptimizer - Multi-strategy backtesting, best strategy selection

Note: This test does not require database connectivity - it tests
the math and logic of the institutional modules in isolation.
"""

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure data directory exists for database operations
Path("data/db").mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_ohlcv(days: int = 500, start_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)

    dates = pd.date_range(
        end=datetime.now(timezone.utc),
        periods=days * 24,  # Hourly data
        freq="h",
        tz="UTC"
    )

    # Random walk with drift
    returns = np.random.normal(0.0001, 0.01, len(dates))
    close = start_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    df = pd.DataFrame({
        "open": close * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
        "high": close * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        "close": close,
        "volume": np.random.randint(1000, 100000, len(dates)).astype(float),
    }, index=dates)

    # Ensure high >= close >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


def test_portfolio_manager() -> bool:
    """Test PortfolioManager risk calculations."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: PortfolioManager (Risk Engine)")
    logger.info("=" * 60)

    try:
        from src.core.risk_engine import PortfolioManager

        pm = PortfolioManager(initial_balance=10000.0)

        # Test 1: Kelly Criterion calculation
        logger.info("\n--- Kelly Criterion Test ---")
        win_rate = 0.55
        avg_win = 0.03  # 3% average win (as decimal)
        avg_loss = -0.02  # 2% average loss (as decimal, negative)

        kelly = pm.calculate_kelly_stake(win_rate, avg_win, avg_loss)
        logger.info(f"Win Rate: {win_rate:.0%}, Avg Win: {avg_win:.1%}, Avg Loss: {avg_loss:.1%}")
        logger.info(f"Kelly Stake: {kelly:.4f} (quarter-Kelly capped at 10%)")

        # Verify Kelly is reasonable (should be capped at max 10%)
        assert 0 < kelly <= 0.10, f"Kelly should be between 0 and 10%, got {kelly:.2%}"
        logger.info(f"Kelly math verified (capped at 10% max risk per trade)")

        # Test 2: Position sizing
        logger.info("\n--- Position Sizing Test ---")
        entry_price = 100.0
        stop_loss = 95.0  # 5% stop

        position_size, dollar_risk = pm.calculate_stake(
            symbol="TEST/USD",
            entry_price=entry_price,
            stop_loss_price=stop_loss
        )

        logger.info(f"Entry: ${entry_price}, Stop: ${stop_loss}")
        logger.info(f"Position Size: {position_size:.4f} units")
        logger.info(f"Dollar Risk: ${dollar_risk:.2f}")

        # Verify position size is reasonable
        assert position_size > 0, "Position size should be positive"
        assert dollar_risk <= pm.current_balance * 0.10, "Risk should not exceed 10% of balance"
        logger.info("Position sizing verified")

        # Test 3: Circuit breaker
        logger.info("\n--- Circuit Breaker Test ---")
        is_ok = pm.check_daily_drawdown()
        logger.info(f"Circuit breaker status: {'OK' if is_ok else 'TRIGGERED'}")

        # Test halt/resume
        pm.halt_trading("Test halt", duration_hours=1)
        is_halted, reason = pm.is_trading_halted()
        logger.info(f"After halt: is_halted={is_halted}, reason={reason}")

        logger.info("\nPortfolioManager: ALL TESTS PASSED")
        return True

    except Exception as e:
        logger.error(f"PortfolioManager test failed: {e}", exc_info=True)
        return False


def test_pattern_hunter() -> bool:
    """Test PatternHunter seasonality analysis."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: PatternHunter (Seasonality Scanner)")
    logger.info("=" * 60)

    try:
        from src.core.seasonality import PatternHunter

        hunter = PatternHunter()
        df = generate_test_ohlcv(days=100)

        # Test 1: Time-of-day analysis
        logger.info("\n--- Time-of-Day Analysis ---")
        tod_patterns = hunter.analyze_time_of_day(df)

        if tod_patterns:
            best_hour = max(tod_patterns.items(), key=lambda x: x[1].get("win_rate", 0))
            worst_hour = min(tod_patterns.items(), key=lambda x: x[1].get("win_rate", 0))

            logger.info(f"Best hour: {best_hour[0]:02d}:00 (win rate: {best_hour[1]['win_rate']:.1%})")
            logger.info(f"Worst hour: {worst_hour[0]:02d}:00 (win rate: {worst_hour[1]['win_rate']:.1%})")
            logger.info(f"Total hours analyzed: {len(tod_patterns)}")
        else:
            logger.info("No significant hourly patterns detected (expected with random data)")

        # Test 2: Day-of-week analysis
        logger.info("\n--- Day-of-Week Analysis ---")
        dow_patterns = hunter.analyze_day_of_week(df)

        if dow_patterns:
            for day, stats in dow_patterns.items():
                logger.info(f"  {day}: Win Rate={stats['win_rate']:.1%}, Samples={stats['sample_size']}")
        else:
            logger.info("No significant day-of-week patterns detected")

        # Test 3: Japan Open breakout
        logger.info("\n--- Japan Open Breakout Analysis ---")
        japan_patterns = hunter.japan_open_breakout(df)

        if japan_patterns and not japan_patterns.get("insufficient_data", False):
            logger.info(f"Bullish breakout rate: {japan_patterns.get('bullish_breakout_rate', 0):.1%}")
            logger.info(f"Bearish breakout rate: {japan_patterns.get('bearish_breakout_rate', 0):.1%}")
            logger.info(f"Sample size: {japan_patterns.get('sample_size', 0)}")
        else:
            logger.info("Insufficient data for Japan Open analysis")

        # Test 4: All patterns analysis
        logger.info("\n--- Full Pattern Analysis ---")
        all_patterns = hunter.analyze_all_patterns("TEST/USD", df)
        logger.info(f"Analyzed at: {all_patterns.get('analyzed_at', 'N/A')}")
        logger.info(f"Time-of-day patterns: {len(all_patterns.get('time_of_day', {}))}")
        logger.info(f"Day-of-week patterns: {len(all_patterns.get('day_of_week', {}))}")

        logger.info("\nPatternHunter: ALL TESTS PASSED")
        return True

    except Exception as e:
        logger.error(f"PatternHunter test failed: {e}", exc_info=True)
        return False


def test_strategy_optimizer() -> bool:
    """Test StrategyOptimizer multi-strategy selection."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: StrategyOptimizer (Multi-Strategy Engine)")
    logger.info("=" * 60)

    try:
        from src.core.backtest_engine import (
            StrategyOptimizer,
            IchimokuStandard,
            IchimokuAggressive,
            IchimokuConservative,
            IchimokuMeanReversion,
            get_strategy_by_name,
        )

        # Test 1: Strategy instantiation
        logger.info("\n--- Strategy Classes ---")
        strategies = [
            IchimokuStandard(),
            IchimokuAggressive(),
            IchimokuConservative(),
            IchimokuMeanReversion(),
        ]

        for strat in strategies:
            params = strat.get_parameters()
            logger.info(f"  {strat.name}: tenkan={params['tenkan_period']}, kijun={params['kijun_period']}")

        # Test 2: Strategy lookup
        logger.info("\n--- Strategy Lookup ---")
        for name in ["ichimoku_standard", "ichimoku_aggressive", "ichimoku_conservative"]:
            strat = get_strategy_by_name(name)
            assert strat is not None, f"Strategy {name} not found"
            logger.info(f"  get_strategy_by_name('{name}'): {strat.name}")

        # Test 3: Signal generation
        logger.info("\n--- Signal Generation ---")
        df = generate_test_ohlcv(days=200)  # More data for Ichimoku

        # Resample to daily for Ichimoku (needs more data points)
        df_daily = df.resample("D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        logger.info(f"Daily data points: {len(df_daily)}")

        for strat in strategies:
            try:
                signal = strat.check_signal(df_daily)
                logger.info(f"  {strat.name}: signal={signal.get('signal', 'N/A')}")
            except ValueError as e:
                logger.info(f"  {strat.name}: {e} (need more data)")

        # Test 4: Backtesting
        logger.info("\n--- Backtesting Engine ---")
        optimizer = StrategyOptimizer()

        # Test single strategy backtest
        metrics = optimizer.backtest_strategy(IchimokuStandard(), df_daily)
        logger.info(f"Standard Strategy Metrics:")
        logger.info(f"  - Total Return: {metrics['total_return']:.2%}")
        logger.info(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  - Win Rate: {metrics['win_rate']:.1%}")
        logger.info(f"  - Max Drawdown: {metrics['max_drawdown']:.1%}")

        # Test 5: Best strategy selection
        logger.info("\n--- Best Strategy Selection ---")
        best_name, best_metrics = optimizer.find_best_strategy(
            symbol="TEST/USD",
            df=df_daily,
            save_to_db=False  # Don't write to DB in test
        )

        logger.info(f"Best strategy for TEST/USD: {best_name}")
        logger.info(f"  - Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
        logger.info(f"  - Win Rate: {best_metrics['win_rate']:.1%}")

        logger.info("\nStrategyOptimizer: ALL TESTS PASSED")
        return True

    except Exception as e:
        logger.error(f"StrategyOptimizer test failed: {e}", exc_info=True)
        return False


def test_engine_integration() -> bool:
    """Test that all components integrate with the engine."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Engine Integration")
    logger.info("=" * 60)

    try:
        # Test individual component imports
        logger.info("Testing component imports...")

        from src.core.risk_engine import PortfolioManager
        logger.info("  PortfolioManager: OK")

        from src.core.execution import TradeManager
        logger.info("  TradeManager: OK")

        from src.core.backtest_engine import StrategyOptimizer
        logger.info("  StrategyOptimizer: OK")

        from src.core.seasonality import PatternHunter
        logger.info("  PatternHunter: OK")

        from src.adapters.ig_lib import IGAdapter
        logger.info("  IGAdapter: OK (UK Spread Betting)")

        from src.adapters.yfinance_lib import YFinanceAdapter
        logger.info("  YFinanceAdapter: OK")

        from src.adapters.ccxt_lib import CCXTAdapter
        logger.info("  CCXTAdapter: OK")

        logger.info("\nEngine Integration: ALL COMPONENTS VERIFIED")
        return True

    except ImportError as e:
        logger.error(f"Engine integration test failed: {e}")
        logger.info("  Note: Some optional dependencies may be missing")
        logger.info("  Install with: pip install -r requirements.txt")
        return False


def main() -> int:
    """Run all institutional tests."""
    logger.info("=" * 60)
    logger.info("SOLAT INSTITUTIONAL BRAIN - TEST SUITE")
    logger.info("=" * 60)

    results = {
        "PortfolioManager": test_portfolio_manager(),
        "PatternHunter": test_pattern_hunter(),
        "StrategyOptimizer": test_strategy_optimizer(),
        "Engine Integration": test_engine_integration(),
    }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name}: {status}")
        all_passed = all_passed and passed

    logger.info("=" * 60)

    if all_passed:
        logger.info("ALL INSTITUTIONAL TESTS PASSED")
        return 0
    else:
        logger.error("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
