#!/usr/bin/env python3
"""
Test script for SOLAT Safe Mode constraints.

Verifies:
1. MAX_OPEN_TRADES = 2 (hard limit enforced)
2. MAX_POSITION_SIZE = 10% of Equity
3. Win Rate prioritized (55%) over Profit Factor (25%)
4. can_trade() gatekeeper works correctly
5. Settings validation passes

This test uses mock database operations to simulate scenarios
without requiring actual database connectivity.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure data directory exists
Path("data/db").mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_settings_constants() -> bool:
    """Test that Safe Mode constants are correctly defined."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Safe Mode Settings Constants")
    logger.info("=" * 60)

    try:
        from src.config.settings import (
            MAX_OPEN_TRADES,
            MAX_POSITION_SIZE_PERCENT,
            SAFE_MODE_ENABLED,
            FITNESS_WEIGHTS,
        )

        # Test 1: MAX_OPEN_TRADES = 2
        logger.info(f"\n--- MAX_OPEN_TRADES ---")
        logger.info(f"Value: {MAX_OPEN_TRADES}")
        assert MAX_OPEN_TRADES == 2, f"Expected MAX_OPEN_TRADES=2, got {MAX_OPEN_TRADES}"
        logger.info("MAX_OPEN_TRADES = 2 (HARD LIMIT)")

        # Test 2: MAX_POSITION_SIZE_PERCENT = 0.10 (10%)
        logger.info(f"\n--- MAX_POSITION_SIZE_PERCENT ---")
        logger.info(f"Value: {MAX_POSITION_SIZE_PERCENT:.0%}")
        assert MAX_POSITION_SIZE_PERCENT == 0.10, f"Expected 10%, got {MAX_POSITION_SIZE_PERCENT:.0%}"
        logger.info("MAX_POSITION_SIZE = 10% of Equity")

        # Test 3: SAFE_MODE_ENABLED = True
        logger.info(f"\n--- SAFE_MODE_ENABLED ---")
        logger.info(f"Value: {SAFE_MODE_ENABLED}")
        assert SAFE_MODE_ENABLED is True, "Safe Mode should be enabled"
        logger.info("Safe Mode is ENABLED")

        # Test 4: Win Rate Priority >= 50%
        logger.info(f"\n--- FITNESS_WEIGHTS ---")
        logger.info(f"Win Rate: {FITNESS_WEIGHTS['win_rate']:.0%}")
        logger.info(f"Profit Factor: {FITNESS_WEIGHTS['profit_factor']:.0%}")
        logger.info(f"Drawdown: {FITNESS_WEIGHTS['drawdown']:.0%}")

        assert FITNESS_WEIGHTS["win_rate"] >= 0.50, (
            f"Win rate weight should be >= 50%, got {FITNESS_WEIGHTS['win_rate']:.0%}"
        )
        assert FITNESS_WEIGHTS["win_rate"] > FITNESS_WEIGHTS["profit_factor"], (
            "Win rate should be prioritized over profit factor"
        )

        # Verify weights sum to 1.0
        total = sum(FITNESS_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"Weights should sum to 1.0, got {total}"
        logger.info(f"Weights sum: {total:.2f}")

        logger.info("\nSettings Constants: ALL TESTS PASSED")
        return True

    except Exception as e:
        logger.error(f"Settings test failed: {e}", exc_info=True)
        return False


def test_settings_validation() -> bool:
    """Test that settings validation passes."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Settings Validation")
    logger.info("=" * 60)

    try:
        from src.config.settings import validate_settings

        # Run validation
        logger.info("\nRunning validate_settings()...")
        result = validate_settings()

        assert result is True, "Settings validation should pass"
        logger.info("\nSettings Validation: PASSED")
        return True

    except Exception as e:
        logger.error(f"Settings validation failed: {e}", exc_info=True)
        return False


def test_can_trade_gatekeeper() -> bool:
    """Test the can_trade() gatekeeper method."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: can_trade() Gatekeeper Method")
    logger.info("=" * 60)

    try:
        from src.core.risk_engine import PortfolioManager

        # Create PortfolioManager with mock database
        with patch('src.core.risk_engine.get_connection') as mock_conn:
            # Setup mock
            mock_cursor = MagicMock()
            mock_conn.return_value.cursor.return_value = mock_cursor

            # Test scenario 1: 0 open trades - should allow trading
            logger.info("\n--- Scenario 1: 0 Open Trades ---")
            mock_cursor.fetchone.side_effect = [
                None,  # _load_state - no balance record
                (0,),  # get_open_trade_count - 0 trades open
                None,  # is_trading_halted - no halt
            ]

            pm = PortfolioManager(initial_balance=10000.0)
            pm.daily_drawdown = 0.01  # 1% drawdown - safe

            # Reset mock for can_trade calls
            mock_cursor.fetchone.side_effect = [
                None,  # is_trading_halted - no halt
                (0,),  # get_open_trade_count - 0 trades open
            ]

            can, reason = pm.can_trade()
            logger.info(f"can_trade={can}, reason={reason}")
            assert can is True, f"Should allow trading with 0 open trades, got: {reason}"
            logger.info("0 open trades: CAN TRADE")

            # Test scenario 2: 2 open trades - should BLOCK trading
            logger.info("\n--- Scenario 2: 2 Open Trades (AT LIMIT) ---")
            mock_cursor.fetchone.side_effect = [
                None,  # is_trading_halted - no halt
                (2,),  # get_open_trade_count - 2 trades open (AT LIMIT)
            ]

            can, reason = pm.can_trade()
            logger.info(f"can_trade={can}, reason={reason}")
            assert can is False, "Should BLOCK trading at MAX_OPEN_TRADES"
            assert "MAX_OPEN_TRADES" in reason, f"Reason should mention MAX_OPEN_TRADES: {reason}"
            logger.info("2 open trades: BLOCKED (Safe Mode)")

            # Test scenario 3: 1 open trade - should allow trading
            logger.info("\n--- Scenario 3: 1 Open Trade ---")
            mock_cursor.fetchone.side_effect = [
                None,  # is_trading_halted - no halt
                (1,),  # get_open_trade_count - 1 trade open
            ]

            can, reason = pm.can_trade()
            logger.info(f"can_trade={can}, reason={reason}")
            assert can is True, f"Should allow trading with 1 open trade, got: {reason}"
            logger.info("1 open trade: CAN TRADE (1 slot available)")

            # Test scenario 4: Trading halted - should BLOCK
            logger.info("\n--- Scenario 4: Trading Halted ---")
            mock_cursor.fetchone.side_effect = [
                ("Drawdown exceeded", "2025-01-20T00:00:00"),  # is_trading_halted - halted
            ]

            can, reason = pm.can_trade()
            logger.info(f"can_trade={can}, reason={reason}")
            assert can is False, "Should BLOCK trading when halted"
            assert "halted" in reason.lower(), f"Reason should mention halt: {reason}"
            logger.info("Trading halted: BLOCKED (Circuit Breaker)")

        logger.info("\ncan_trade() Gatekeeper: ALL TESTS PASSED")
        return True

    except Exception as e:
        logger.error(f"can_trade() test failed: {e}", exc_info=True)
        return False


def test_position_size_cap() -> bool:
    """Test that position size is capped at 10%."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Position Size Cap (10%)")
    logger.info("=" * 60)

    try:
        from src.core.risk_engine import PortfolioManager

        with patch('src.core.risk_engine.get_connection') as mock_conn:
            # Setup mock
            mock_cursor = MagicMock()
            mock_conn.return_value.cursor.return_value = mock_cursor
            mock_cursor.fetchone.return_value = None
            mock_cursor.fetchall.return_value = []

            pm = PortfolioManager(initial_balance=10000.0)

            # Test Kelly calculation with aggressive stats
            logger.info("\n--- Kelly Criterion with Cap ---")
            # These stats would give a very high Kelly stake
            aggressive_kelly = pm.calculate_kelly_stake(
                win_rate=0.80,  # 80% win rate
                avg_win=0.10,   # 10% average win
                avg_loss=-0.02  # 2% average loss
            )

            logger.info(f"Aggressive stats Kelly stake: {aggressive_kelly:.4f}")
            assert aggressive_kelly <= 0.10, (
                f"Kelly should be capped at 10%, got {aggressive_kelly:.1%}"
            )
            logger.info("Kelly stake capped at 10% (Safe Mode)")

            # Test position sizing
            logger.info("\n--- Position Sizing ---")
            position_size, dollar_risk = pm.calculate_stake(
                symbol="TEST/USD",
                entry_price=100.0,
                stop_loss_price=98.0,  # 2% stop
            )

            max_risk = pm.current_equity * pm.max_position_size
            logger.info(f"Position size: {position_size:.4f} units")
            logger.info(f"Dollar risk: ${dollar_risk:.2f}")
            logger.info(f"Max allowed risk: ${max_risk:.2f}")

            assert dollar_risk <= max_risk, (
                f"Dollar risk (${dollar_risk:.2f}) should not exceed "
                f"max (${max_risk:.2f})"
            )
            logger.info("Position size respects 10% cap")

        logger.info("\nPosition Size Cap: ALL TESTS PASSED")
        return True

    except Exception as e:
        logger.error(f"Position size test failed: {e}", exc_info=True)
        return False


def test_safe_mode_status() -> bool:
    """Test the get_safe_mode_status() method."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Safe Mode Status Dashboard")
    logger.info("=" * 60)

    try:
        from src.core.risk_engine import PortfolioManager
        from src.config.settings import MAX_OPEN_TRADES, SAFE_MODE_ENABLED

        with patch('src.core.risk_engine.get_connection') as mock_conn:
            # Setup mock
            mock_cursor = MagicMock()
            mock_conn.return_value.cursor.return_value = mock_cursor
            mock_cursor.fetchone.side_effect = [
                None,  # _load_state
                None,  # is_trading_halted
                (1,),  # get_open_trade_count
                None,  # is_trading_halted (again for can_trade)
                (1,),  # get_open_trade_count (again for can_trade)
            ]

            pm = PortfolioManager(initial_balance=10000.0)
            pm.daily_drawdown = 0.02  # 2% drawdown

            status = pm.get_safe_mode_status()

            logger.info("\n--- Safe Mode Status ---")
            for key, value in status.items():
                logger.info(f"  {key}: {value}")

            # Verify status fields
            assert status["safe_mode_enabled"] == SAFE_MODE_ENABLED
            assert status["max_open_trades"] == MAX_OPEN_TRADES
            assert "current_open_trades" in status
            assert "slots_available" in status
            assert "can_trade" in status

            logger.info("\nSafe Mode Status: ALL TESTS PASSED")
            return True

    except Exception as e:
        logger.error(f"Safe mode status test failed: {e}", exc_info=True)
        return False


def main() -> int:
    """Run all Safe Mode tests."""
    logger.info("=" * 60)
    logger.info("SOLAT SAFE MODE - TEST SUITE")
    logger.info("=" * 60)
    logger.info("Hard Rules Being Verified:")
    logger.info("  1. MAX_OPEN_TRADES = 2 (system sleeps if exceeded)")
    logger.info("  2. MAX_POSITION_SIZE = 10% of Equity")
    logger.info("  3. Win Rate Priority >= 50%")
    logger.info("=" * 60)

    results = {
        "Settings Constants": test_settings_constants(),
        "Settings Validation": test_settings_validation(),
        "can_trade() Gatekeeper": test_can_trade_gatekeeper(),
        "Position Size Cap": test_position_size_cap(),
        "Safe Mode Status": test_safe_mode_status(),
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
        logger.info("ALL SAFE MODE TESTS PASSED")
        logger.info("")
        logger.info("SAFE MODE ENFORCEMENT VERIFIED:")
        logger.info("  - MAX_OPEN_TRADES = 2 (ENFORCED)")
        logger.info("  - MAX_POSITION_SIZE = 10% (ENFORCED)")
        logger.info("  - Win Rate Priority = 55% (ENFORCED)")
        logger.info("  - can_trade() gatekeeper = ACTIVE")
        return 0
    else:
        logger.error("SOME TESTS FAILED - Safe Mode may not be properly enforced!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
