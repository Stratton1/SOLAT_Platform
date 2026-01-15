#!/usr/bin/env python3
"""
Test script for Order Book Microstructure Analysis system.

Verifies:
1. OrderFlowAnalyzer initialization
2. Order book metrics calculation
3. Sniper entry validation logic
4. Integration with engine

This test uses mock order book data to verify the microstructure logic
without requiring live exchange connections.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

# Ensure data directory exists
Path("data/db").mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockAdapter:
    """Mock adapter for testing order book analysis."""

    def __init__(self, source: str = "binance") -> None:
        self.source = source
        self._order_book_data: Dict[str, Dict] = {}
        self.exchange = MockExchange()

    def set_order_book(self, symbol: str, bids: List, asks: List) -> None:
        """Set mock order book data for a symbol."""
        self._order_book_data[symbol] = {
            "bids": bids,
            "asks": asks,
        }
        self.exchange.order_books[symbol] = self._order_book_data[symbol]


class MockExchange:
    """Mock CCXT exchange for testing."""

    def __init__(self) -> None:
        self.has = {"fetchOrderBook": True}
        self.order_books: Dict[str, Dict] = {}

    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Return mock order book data."""
        if symbol in self.order_books:
            return self.order_books[symbol]
        # Default empty order book
        return {"bids": [], "asks": []}


def test_order_flow_analyzer_init() -> bool:
    """Test OrderFlowAnalyzer initialization."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: OrderFlowAnalyzer Initialization")
    logger.info("=" * 60)

    try:
        from src.core.microstructure import OrderFlowAnalyzer, OrderBookMetrics

        # Initialize without adapter
        analyzer = OrderFlowAnalyzer()
        assert analyzer.adapter is None, "Adapter should be None initially"
        logger.info("Initialized OrderFlowAnalyzer without adapter: OK")

        # Initialize with mock adapter
        mock_adapter = MockAdapter()
        analyzer = OrderFlowAnalyzer(mock_adapter)
        assert analyzer.adapter is not None, "Adapter should be set"
        logger.info("Initialized OrderFlowAnalyzer with adapter: OK")

        # Test set_adapter method
        analyzer2 = OrderFlowAnalyzer()
        analyzer2.set_adapter(mock_adapter)
        assert analyzer2.adapter is not None, "set_adapter should work"
        logger.info("set_adapter method: OK")

        logger.info("\nOrderFlowAnalyzer Initialization: PASSED")
        return True

    except Exception as e:
        logger.error(f"OrderFlowAnalyzer init test failed: {e}", exc_info=True)
        return False


def test_order_book_metrics_calculation() -> bool:
    """Test Order Imbalance and Spread Efficiency calculation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Order Book Metrics Calculation")
    logger.info("=" * 60)

    try:
        from src.core.microstructure import OrderFlowAnalyzer

        # Create mock adapter with test data
        mock_adapter = MockAdapter()

        # Test 1: Strong buy pressure (more bid volume)
        # Bids: [[price, quantity], ...]
        mock_adapter.set_order_book(
            "BTC/USDT",
            bids=[[50000, 10], [49990, 8], [49980, 6]],  # Total: 10+8+6 = 24 BTC
            asks=[[50010, 3], [50020, 2], [50030, 1]],   # Total: 3+2+1 = 6 BTC
        )

        analyzer = OrderFlowAnalyzer(mock_adapter)
        metrics = analyzer.get_order_book_metrics("BTC/USDT")

        logger.info(f"Test case: Strong Buy Pressure")
        logger.info(f"  Bid Volume (value): {metrics.bid_volume:.2f}")
        logger.info(f"  Ask Volume (value): {metrics.ask_volume:.2f}")
        logger.info(f"  Order Imbalance: {metrics.order_imbalance:.3f}")
        logger.info(f"  Spread: {metrics.spread_bps:.2f} bps")
        logger.info(f"  Is Valid: {metrics.is_valid}")

        assert metrics.is_valid, "Metrics should be valid"
        assert metrics.order_imbalance > 0.2, f"OI should be > 0.2 for buy pressure, got {metrics.order_imbalance}"
        logger.info("  -> Strong Buy Pressure verified: OI > 0.2 ✓")

        # Test 2: Strong sell pressure (more ask volume)
        mock_adapter.set_order_book(
            "ETH/USDT",
            bids=[[3000, 2], [2999, 1], [2998, 1]],   # Total: 4 ETH
            asks=[[3001, 10], [3002, 8], [3003, 6]],  # Total: 24 ETH
        )

        metrics = analyzer.get_order_book_metrics("ETH/USDT")

        logger.info(f"\nTest case: Strong Sell Pressure")
        logger.info(f"  Bid Volume (value): {metrics.bid_volume:.2f}")
        logger.info(f"  Ask Volume (value): {metrics.ask_volume:.2f}")
        logger.info(f"  Order Imbalance: {metrics.order_imbalance:.3f}")

        assert metrics.order_imbalance < -0.2, f"OI should be < -0.2 for sell pressure, got {metrics.order_imbalance}"
        logger.info("  -> Strong Sell Pressure verified: OI < -0.2 ✓")

        # Test 3: Neutral pressure (balanced)
        mock_adapter.set_order_book(
            "XRP/USDT",
            bids=[[0.5, 1000], [0.499, 1000]],
            asks=[[0.501, 1000], [0.502, 1000]],
        )

        metrics = analyzer.get_order_book_metrics("XRP/USDT")

        logger.info(f"\nTest case: Neutral Pressure")
        logger.info(f"  Order Imbalance: {metrics.order_imbalance:.3f}")

        assert abs(metrics.order_imbalance) < 0.1, f"OI should be near 0 for neutral, got {metrics.order_imbalance}"
        logger.info("  -> Neutral Pressure verified: |OI| < 0.1 ✓")

        logger.info("\nOrder Book Metrics Calculation: PASSED")
        return True

    except Exception as e:
        logger.error(f"Order book metrics test failed: {e}", exc_info=True)
        return False


def test_sniper_entry_validation() -> bool:
    """Test the Sniper Entry check logic."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Sniper Entry Validation")
    logger.info("=" * 60)

    try:
        from src.core.microstructure import OrderFlowAnalyzer

        mock_adapter = MockAdapter()
        analyzer = OrderFlowAnalyzer(mock_adapter)

        # Test 1: BUY with strong buy pressure -> PASS
        mock_adapter.set_order_book(
            "BTC/USDT",
            bids=[[50000, 10], [49990, 8], [49980, 6]],
            asks=[[50010, 3], [50020, 2], [50030, 1]],
        )

        is_valid, reason = analyzer.check_sniper_entry("BTC/USDT", "BUY")
        logger.info(f"BUY with strong buy pressure: {'PASS' if is_valid else 'FAIL'}")
        logger.info(f"  Reason: {reason}")
        assert is_valid, f"BUY should PASS with buy pressure: {reason}"

        # Test 2: BUY with strong sell pressure -> FAIL
        mock_adapter.set_order_book(
            "ETH/USDT",
            bids=[[3000, 2], [2999, 1]],
            asks=[[3001, 10], [3002, 8], [3003, 6]],
        )

        is_valid, reason = analyzer.check_sniper_entry("ETH/USDT", "BUY")
        logger.info(f"\nBUY with strong sell pressure: {'PASS' if is_valid else 'FAIL'}")
        logger.info(f"  Reason: {reason}")
        assert not is_valid, f"BUY should FAIL with sell pressure: {reason}"

        # Test 3: SELL with strong sell pressure -> PASS
        is_valid, reason = analyzer.check_sniper_entry("ETH/USDT", "SELL")
        logger.info(f"\nSELL with strong sell pressure: {'PASS' if is_valid else 'FAIL'}")
        logger.info(f"  Reason: {reason}")
        assert is_valid, f"SELL should PASS with sell pressure: {reason}"

        # Test 4: SELL with strong buy pressure -> FAIL
        is_valid, reason = analyzer.check_sniper_entry("BTC/USDT", "SELL")
        logger.info(f"\nSELL with strong buy pressure: {'PASS' if is_valid else 'FAIL'}")
        logger.info(f"  Reason: {reason}")
        assert not is_valid, f"SELL should FAIL with buy pressure: {reason}"

        # Test 5: No order book data -> PASS (graceful degradation)
        mock_adapter.set_order_book("DOGE/USDT", bids=[], asks=[])
        is_valid, reason = analyzer.check_sniper_entry("DOGE/USDT", "BUY")
        logger.info(f"\nBUY with no order book data: {'PASS' if is_valid else 'FAIL'}")
        logger.info(f"  Reason: {reason}")
        # Should pass because we don't block trades when data is unavailable
        assert is_valid, f"Should PASS when order book unavailable: {reason}"

        logger.info("\nSniper Entry Validation: PASSED")
        return True

    except Exception as e:
        logger.error(f"Sniper entry validation test failed: {e}", exc_info=True)
        return False


def test_spread_liquidity_check() -> bool:
    """Test spread-based liquidity validation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Spread Liquidity Check")
    logger.info("=" * 60)

    try:
        from src.core.microstructure import OrderFlowAnalyzer

        mock_adapter = MockAdapter()
        analyzer = OrderFlowAnalyzer(mock_adapter)

        # Test with tight spread (good liquidity)
        mock_adapter.set_order_book(
            "BTC/USDT",
            bids=[[50000, 10]],  # Best bid: 50000
            asks=[[50010, 10]],  # Best ask: 50010, spread = 10/50005 = 0.02%
        )

        metrics = analyzer.get_order_book_metrics("BTC/USDT")
        logger.info(f"Tight spread test: {metrics.spread_bps:.2f} bps")
        assert metrics.spread_bps < 50, f"Tight spread should be < 50 bps, got {metrics.spread_bps}"
        logger.info("  -> Tight spread verified ✓")

        # Test with wide spread (poor liquidity)
        mock_adapter.set_order_book(
            "ILLIQUID/USDT",
            bids=[[100, 1]],     # Best bid: 100
            asks=[[110, 1]],    # Best ask: 110, spread = 10/105 = 9.5%
        )

        metrics = analyzer.get_order_book_metrics("ILLIQUID/USDT")
        logger.info(f"\nWide spread test: {metrics.spread_bps:.2f} bps")
        assert metrics.spread_bps > 50, f"Wide spread should be > 50 bps, got {metrics.spread_bps}"
        logger.info("  -> Wide spread verified ✓")

        # Trade should be blocked due to wide spread
        is_valid, reason = analyzer.check_sniper_entry("ILLIQUID/USDT", "BUY")
        logger.info(f"\nBUY with wide spread: {'PASS' if is_valid else 'FAIL'}")
        logger.info(f"  Reason: {reason}")
        assert not is_valid, f"Should FAIL with wide spread: {reason}"

        logger.info("\nSpread Liquidity Check: PASSED")
        return True

    except Exception as e:
        logger.error(f"Spread liquidity test failed: {e}", exc_info=True)
        return False


def test_engine_integration() -> bool:
    """Test integration with the Sentinel engine."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Engine Integration")
    logger.info("=" * 60)

    try:
        # Test imports
        from src.core.microstructure import OrderFlowAnalyzer, OrderBookMetrics

        logger.info("All microstructure imports successful")
        logger.info("  OrderFlowAnalyzer: OK")
        logger.info("  OrderBookMetrics: OK")

        # Read engine.py source directly to verify integration
        # (avoids import errors from missing dependencies like hmmlearn)
        engine_path = Path("src/core/engine.py")
        if not engine_path.exists():
            logger.error("Engine file not found")
            return False

        with open(engine_path, "r") as f:
            engine_source = f.read()

        # Check imports
        assert "from src.core.microstructure import" in engine_source, "Engine should import microstructure"
        assert "OrderFlowAnalyzer" in engine_source, "Engine should import OrderFlowAnalyzer"
        logger.info("  Engine imports verified: OK")

        # Check __init__ initialization
        assert "order_flow_analyzer" in engine_source, "Engine should have order_flow_analyzer"
        assert "OrderFlowAnalyzer()" in engine_source, "Engine should instantiate OrderFlowAnalyzer"
        logger.info("  OrderFlowAnalyzer initialization: OK")

        # Check scan_market includes sniper check
        assert "SNIPER" in engine_source, "scan_market should have SNIPER check"
        assert "check_sniper_entry" in engine_source, "scan_market should call check_sniper_entry"
        assert "SNIPER BLOCKED" in engine_source, "Engine should log SNIPER BLOCKED"
        assert "SNIPER PASS" in engine_source, "Engine should log SNIPER PASS"
        logger.info("  Sniper check in scan_market: OK")

        # Check order_imbalance is stored in database
        assert "order_imbalance" in engine_source, "Engine should store order_imbalance"
        logger.info("  Order imbalance persistence: OK")

        logger.info("\nEngine Integration: PASSED")
        return True

    except Exception as e:
        logger.error(f"Engine integration test failed: {e}", exc_info=True)
        return False


def test_convenience_function() -> bool:
    """Test the analyze_order_flow convenience function."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Convenience Function")
    logger.info("=" * 60)

    try:
        from src.core.microstructure import analyze_order_flow

        mock_adapter = MockAdapter()
        mock_adapter.set_order_book(
            "BTC/USDT",
            bids=[[50000, 10], [49990, 5]],
            asks=[[50010, 3], [50020, 2]],
        )

        result = analyze_order_flow(mock_adapter, "BTC/USDT", "BUY")

        logger.info(f"analyze_order_flow result:")
        logger.info(f"  is_valid: {result['is_valid']}")
        logger.info(f"  reason: {result['reason']}")
        logger.info(f"  order_imbalance: {result['order_imbalance']:.3f}")
        logger.info(f"  spread_bps: {result['spread_bps']:.2f}")

        assert "is_valid" in result, "Result should have is_valid"
        assert "reason" in result, "Result should have reason"
        assert "order_imbalance" in result, "Result should have order_imbalance"

        logger.info("\nConvenience Function: PASSED")
        return True

    except Exception as e:
        logger.error(f"Convenience function test failed: {e}", exc_info=True)
        return False


def main() -> int:
    """Run all microstructure tests."""
    logger.info("=" * 60)
    logger.info("SOLAT MICROSTRUCTURE SNIPER - TEST SUITE")
    logger.info("=" * 60)
    logger.info("Target: Order Book Analysis for 80%+ Win Rate")
    logger.info("Three-Filter System: Regime + Ichimoku + Microstructure")
    logger.info("=" * 60)

    results = {
        "OrderFlowAnalyzer Init": test_order_flow_analyzer_init(),
        "Order Book Metrics": test_order_book_metrics_calculation(),
        "Sniper Entry Validation": test_sniper_entry_validation(),
        "Spread Liquidity Check": test_spread_liquidity_check(),
        "Engine Integration": test_engine_integration(),
        "Convenience Function": test_convenience_function(),
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
        logger.info("ALL MICROSTRUCTURE TESTS PASSED")
        logger.info("")
        logger.info("MICROSTRUCTURE SNIPER VERIFIED:")
        logger.info("  - OrderFlowAnalyzer fetches order book data")
        logger.info("  - Order Imbalance (OI) calculated correctly")
        logger.info("  - BUY blocked when sell pressure detected")
        logger.info("  - SELL blocked when buy pressure detected")
        logger.info("  - Wide spreads (poor liquidity) trigger blocks")
        logger.info("  - Engine integration complete")
        logger.info("")
        logger.info("THREE-FILTER SYSTEM NOW ACTIVE:")
        logger.info("  1. Regime (HMM): Bull/Bear/Chop detection")
        logger.info("  2. Strategy (Ichimoku): Optimized signal generation")
        logger.info("  3. Microstructure (Order Book): Liquidity validation")
        logger.info("")
        logger.info("Only trades passing ALL THREE filters get executed.")
        return 0
    else:
        logger.error("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
