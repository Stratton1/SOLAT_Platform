"""
Order Book Microstructure Analysis Engine.

The OrderFlowAnalyzer provides real-time order book analysis to validate
Ichimoku signals against the actual market liquidity structure.

Theory:
- A BUY signal is only valid if there is Buying Pressure (Bid Vol > Ask Vol)
- A SELL signal is only valid if there is Selling Pressure (Ask Vol > Bid Vol)
- This filters out "Fakeouts" where price breaks resistance but hits hidden sell walls

The Microstructure Sniper represents the third filter in the trading pipeline:
1. Regime (Macro): Is the market safe? (HMM)
2. Strategy (Technical): Is the trend moving? (Optimized Ichimoku)
3. Microstructure (Instant): Is there liquidity to support the move? (Order Book)

Only trades that pass ALL THREE checks get executed.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrderBookMetrics:
    """Container for order book analysis metrics."""

    order_imbalance: float  # -1 (pure sell) to +1 (pure buy)
    spread_efficiency: float  # Higher = worse liquidity
    bid_volume: float
    ask_volume: float
    best_bid: float
    best_ask: float
    mid_price: float
    spread_bps: float  # Spread in basis points
    depth_ratio: float  # Bid depth / Ask depth
    is_valid: bool  # Whether metrics are valid
    error_message: Optional[str] = None


class OrderFlowAnalyzer:
    """
    Order Book Microstructure Analyzer for trade validation.

    Analyzes the limit order book (LOB) to determine market pressure
    and validate Ichimoku signals against actual liquidity.

    Key Metrics:
    - Order Imbalance (OI): Measures buy/sell pressure
    - Spread Efficiency: Measures market liquidity quality

    Usage:
        >>> analyzer = OrderFlowAnalyzer(adapter)
        >>> is_valid = analyzer.check_sniper_entry("BTC/USDT", "BUY")
        >>> if not is_valid:
        ...     print("Trade rejected by Microstructure Sniper")
    """

    # Order Imbalance thresholds
    OI_THRESHOLD_STRONG = 0.20  # Strong pressure threshold
    OI_THRESHOLD_WEAK = 0.10    # Weak pressure threshold

    # Spread thresholds (in basis points)
    MAX_SPREAD_BPS = 50  # Maximum acceptable spread (0.5%)

    # Order book depth (number of levels to analyze)
    ORDER_BOOK_DEPTH = 20

    def __init__(self, adapter: Any = None) -> None:
        """
        Initialize the OrderFlowAnalyzer.

        Args:
            adapter: Market adapter (CCXTAdapter, IGAdapter, etc.)
                    Must have access to order book data.
        """
        self.adapter = adapter
        self._cache: Dict[str, OrderBookMetrics] = {}

    def set_adapter(self, adapter: Any) -> None:
        """Set or update the market adapter."""
        self.adapter = adapter

    def _fetch_order_book_ccxt(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch order book data from CCXT adapter.

        CCXT provides native order book support via exchange.fetch_order_book().

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")

        Returns:
            Order book data dict with 'bids' and 'asks' arrays,
            or None if fetch fails.
        """
        try:
            if not hasattr(self.adapter, 'exchange'):
                logger.warning(f"Adapter does not have exchange attribute")
                return None

            exchange = self.adapter.exchange

            if not exchange.has.get('fetchOrderBook', False):
                logger.warning(f"Exchange {self.adapter.source} does not support fetchOrderBook")
                return None

            order_book = exchange.fetch_order_book(symbol, limit=self.ORDER_BOOK_DEPTH)

            logger.debug(
                f"Fetched order book for {symbol}: "
                f"{len(order_book.get('bids', []))} bids, "
                f"{len(order_book.get('asks', []))} asks"
            )

            return order_book

        except Exception as e:
            logger.error(f"Error fetching CCXT order book for {symbol}: {e}")
            return None

    def _fetch_order_book_ig(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch order book data from IG Markets adapter.

        Note: IG Markets (spread betting) typically does not provide
        Level 2 order book data like exchanges. Instead, we can use
        the bid/ask spread from market prices.

        Args:
            symbol: IG EPIC or symbol alias

        Returns:
            Simulated order book from bid/ask prices, or None if unavailable.
        """
        try:
            if not hasattr(self.adapter, 'ig_service') or not self.adapter._authenticated:
                logger.warning("IG adapter not authenticated")
                return None

            # IG doesn't have traditional order book, but we can get bid/ask
            # from the streaming prices or market details
            ig_service = self.adapter.ig_service

            # Resolve symbol to EPIC if needed
            epic = self.adapter._resolve_epic(symbol)

            # Get market details which include bid/ask
            market_info = ig_service.fetch_market_by_epic(epic)

            if market_info and hasattr(market_info, 'snapshot'):
                snapshot = market_info.snapshot
                bid = float(snapshot.get('bid', 0))
                ask = float(snapshot.get('offer', 0))

                if bid > 0 and ask > 0:
                    # Create synthetic order book with single level
                    return {
                        'bids': [[bid, 1.0]],  # [price, volume]
                        'asks': [[ask, 1.0]],
                        'synthetic': True,
                    }

            logger.debug(f"No order book data available for IG symbol {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error fetching IG order book for {symbol}: {e}")
            return None

    def fetch_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch order book data from the appropriate adapter.

        Automatically detects adapter type and uses the correct method.

        Args:
            symbol: Trading symbol

        Returns:
            Order book data dict with 'bids' and 'asks' arrays
        """
        if self.adapter is None:
            logger.warning("No adapter set for OrderFlowAnalyzer")
            return None

        source = getattr(self.adapter, 'source', 'unknown').lower()

        if source in ['binance', 'coinbase', 'kraken'] or hasattr(self.adapter, 'exchange'):
            return self._fetch_order_book_ccxt(symbol)
        elif source == 'ig':
            return self._fetch_order_book_ig(symbol)
        elif source == 'yfinance':
            # YFinance doesn't provide order book data
            logger.debug(f"Order book not available for YFinance symbol {symbol}")
            return None
        else:
            # Try CCXT as fallback
            return self._fetch_order_book_ccxt(symbol)

    def get_order_book_metrics(self, symbol: str) -> OrderBookMetrics:
        """
        Calculate order book metrics for a symbol.

        Metrics calculated:
        1. Order Imbalance (OI): (Bid_Vol - Ask_Vol) / (Bid_Vol + Ask_Vol)
           - Range: -1 (pure sell pressure) to +1 (pure buy pressure)
        2. Spread Efficiency: (Ask - Bid) / Mid_Price
           - Higher value = worse liquidity (more dangerous)

        Args:
            symbol: Trading symbol

        Returns:
            OrderBookMetrics dataclass with analysis results
        """
        # Try to fetch order book
        order_book = self.fetch_order_book(symbol)

        if order_book is None:
            return OrderBookMetrics(
                order_imbalance=0.0,
                spread_efficiency=0.0,
                bid_volume=0.0,
                ask_volume=0.0,
                best_bid=0.0,
                best_ask=0.0,
                mid_price=0.0,
                spread_bps=0.0,
                depth_ratio=1.0,
                is_valid=False,
                error_message="Order book data unavailable"
            )

        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        if not bids or not asks:
            return OrderBookMetrics(
                order_imbalance=0.0,
                spread_efficiency=0.0,
                bid_volume=0.0,
                ask_volume=0.0,
                best_bid=0.0,
                best_ask=0.0,
                mid_price=0.0,
                spread_bps=0.0,
                depth_ratio=1.0,
                is_valid=False,
                error_message="Empty order book"
            )

        try:
            # Calculate total volumes (price * quantity for each level)
            bid_volume = sum(price * qty for price, qty in bids[:self.ORDER_BOOK_DEPTH])
            ask_volume = sum(price * qty for price, qty in asks[:self.ORDER_BOOK_DEPTH])

            # Best bid/ask (top of book)
            best_bid = bids[0][0]
            best_ask = asks[0][0]

            # Mid price
            mid_price = (best_bid + best_ask) / 2

            # Spread calculations
            spread = best_ask - best_bid
            spread_efficiency = spread / mid_price if mid_price > 0 else 0
            spread_bps = spread_efficiency * 10000  # Convert to basis points

            # Order Imbalance: (Bid - Ask) / (Bid + Ask)
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                order_imbalance = (bid_volume - ask_volume) / total_volume
            else:
                order_imbalance = 0.0

            # Depth ratio (bid depth vs ask depth)
            depth_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0

            metrics = OrderBookMetrics(
                order_imbalance=order_imbalance,
                spread_efficiency=spread_efficiency,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                best_bid=best_bid,
                best_ask=best_ask,
                mid_price=mid_price,
                spread_bps=spread_bps,
                depth_ratio=depth_ratio,
                is_valid=True,
            )

            logger.debug(
                f"Order book metrics for {symbol}: "
                f"OI={order_imbalance:.3f}, Spread={spread_bps:.1f}bps, "
                f"Bid Vol={bid_volume:.2f}, Ask Vol={ask_volume:.2f}"
            )

            # Cache the metrics
            self._cache[symbol] = metrics

            return metrics

        except Exception as e:
            logger.error(f"Error calculating order book metrics for {symbol}: {e}")
            return OrderBookMetrics(
                order_imbalance=0.0,
                spread_efficiency=0.0,
                bid_volume=0.0,
                ask_volume=0.0,
                best_bid=0.0,
                best_ask=0.0,
                mid_price=0.0,
                spread_bps=0.0,
                depth_ratio=1.0,
                is_valid=False,
                error_message=str(e)
            )

    def check_sniper_entry(
        self,
        symbol: str,
        side: str,
        strict: bool = True,
    ) -> Tuple[bool, str]:
        """
        Validate a trade entry against order book microstructure.

        The Sniper Entry Check ensures that the order book pressure
        aligns with the intended trade direction.

        Rules:
        - BUY: Order Imbalance > 0.2 (Strong buy pressure) -> PASS
        - SELL: Order Imbalance < -0.2 (Strong sell pressure) -> PASS
        - Otherwise -> FAIL (Trade rejected by Sniper)

        Args:
            symbol: Trading symbol
            side: Trade side ('BUY' or 'SELL')
            strict: If True, requires strong pressure. If False, accepts weak pressure.

        Returns:
            Tuple[is_valid, reason]:
                - is_valid: True if trade passes Sniper check
                - reason: Explanation of the decision
        """
        metrics = self.get_order_book_metrics(symbol)

        # If order book is unavailable, we can choose to pass or fail
        # Default: PASS (don't block trades when data is unavailable)
        if not metrics.is_valid:
            reason = f"Order book unavailable: {metrics.error_message}"
            logger.warning(f"Sniper PASS (no data): {symbol} {side} - {reason}")
            return True, reason

        oi = metrics.order_imbalance
        threshold = self.OI_THRESHOLD_STRONG if strict else self.OI_THRESHOLD_WEAK

        # Check spread liquidity
        if metrics.spread_bps > self.MAX_SPREAD_BPS:
            reason = f"Spread too wide: {metrics.spread_bps:.1f}bps > {self.MAX_SPREAD_BPS}bps"
            logger.warning(f"Sniper BLOCKED (liquidity): {symbol} {side} - {reason}")
            return False, reason

        side = side.upper()

        if side == "BUY":
            if oi > threshold:
                reason = f"Buy pressure confirmed: OI={oi:.3f} > {threshold}"
                logger.info(f"Sniper PASS: {symbol} BUY - {reason}")
                return True, reason
            elif oi < -threshold:
                reason = f"Opposing sell pressure: OI={oi:.3f} < -{threshold}"
                logger.warning(f"Sniper BLOCKED: {symbol} BUY - {reason}")
                return False, reason
            else:
                reason = f"Weak/neutral pressure: OI={oi:.3f} (threshold: {threshold})"
                logger.info(f"Sniper PASS (neutral): {symbol} BUY - {reason}")
                return True, reason  # Allow neutral pressure

        elif side == "SELL":
            if oi < -threshold:
                reason = f"Sell pressure confirmed: OI={oi:.3f} < -{threshold}"
                logger.info(f"Sniper PASS: {symbol} SELL - {reason}")
                return True, reason
            elif oi > threshold:
                reason = f"Opposing buy pressure: OI={oi:.3f} > {threshold}"
                logger.warning(f"Sniper BLOCKED: {symbol} SELL - {reason}")
                return False, reason
            else:
                reason = f"Weak/neutral pressure: OI={oi:.3f} (threshold: {threshold})"
                logger.info(f"Sniper PASS (neutral): {symbol} SELL - {reason}")
                return True, reason  # Allow neutral pressure

        else:
            reason = f"Invalid side: {side}"
            logger.warning(f"Sniper BLOCKED: {symbol} - {reason}")
            return False, reason

    def get_cached_metrics(self, symbol: str) -> Optional[OrderBookMetrics]:
        """Get cached metrics for a symbol without fetching."""
        return self._cache.get(symbol)

    def clear_cache(self) -> None:
        """Clear the metrics cache."""
        self._cache.clear()


# Convenience function for quick analysis
def analyze_order_flow(
    adapter: Any,
    symbol: str,
    side: str,
) -> Dict[str, Any]:
    """
    Quick order flow analysis for a symbol and side.

    Args:
        adapter: Market adapter
        symbol: Trading symbol
        side: Trade side ('BUY' or 'SELL')

    Returns:
        Dict with analysis results:
            - is_valid: Whether trade passes Sniper check
            - reason: Explanation
            - metrics: OrderBookMetrics (if available)
    """
    analyzer = OrderFlowAnalyzer(adapter)
    metrics = analyzer.get_order_book_metrics(symbol)
    is_valid, reason = analyzer.check_sniper_entry(symbol, side)

    return {
        "is_valid": is_valid,
        "reason": reason,
        "metrics": metrics if metrics.is_valid else None,
        "order_imbalance": metrics.order_imbalance,
        "spread_bps": metrics.spread_bps,
    }
