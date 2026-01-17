"""
Hybrid Trading Strategies for SOLAT Platform.

This module provides advanced strategy implementations that combine
multiple technical analysis approaches for high-frequency day trading
targeting 2-4 trades per day.

Primary Strategy: IchimokuFibonacci
- Uses Ichimoku Cloud for trend direction
- Uses Fibonacci retracements for precise pullback entries
- Designed for intraday timeframes (5m, 15m, 30m, 1h)
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    ta = None

from src.core.backtest_engine import BaseStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================


class TrendDirection(Enum):
    """Market trend direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class FibLevel(Enum):
    """Fibonacci retracement levels."""
    FIB_236 = 0.236
    FIB_382 = 0.382
    FIB_500 = 0.500
    FIB_618 = 0.618
    FIB_786 = 0.786


@dataclass
class SwingPoint:
    """Represents a swing high or swing low point."""
    index: int
    price: float
    is_high: bool
    timestamp: pd.Timestamp


@dataclass
class FibonacciLevels:
    """Fibonacci retracement levels from a swing."""
    swing_high: float
    swing_low: float
    fib_236: float
    fib_382: float
    fib_500: float
    fib_618: float
    fib_786: float
    direction: TrendDirection


# =============================================================================
# ICHIMOKU FIBONACCI HYBRID STRATEGY
# =============================================================================


class IchimokuFibonacci(BaseStrategy):
    """
    Hybrid strategy combining Ichimoku Cloud trend filter with
    Fibonacci retracement entries for high-frequency day trading.

    Strategy Logic:
    ---------------
    TREND FILTER (Ichimoku):
    - Bullish: Price > Cloud AND Cloud is Green (Senkou A > Senkou B)
    - Bearish: Price < Cloud AND Cloud is Red (Senkou A < Senkou B)

    ENTRY SIGNAL (Fibonacci):
    - BUY: In bullish trend, price pulls back to 38.2% or 50% Fib level
           of recent swing, then closes above it (bounce confirmation)
    - SELL: In bearish trend, price rallies to 38.2% or 50% Fib level
            of recent swing, then closes below it (rejection confirmation)

    EXIT STRATEGY:
    - Take Profit: 1.5x Risk (Risk/Reward = 1:1.5)
    - Stop Loss: Below swing low (buys) / Above swing high (sells)
    - Trailing Stop: On Kijun-sen (26-period base line)

    Target: 2-4 trades per day per asset on 15m-1h timeframes

    Parameters:
    -----------
    tenkan_period : int
        Tenkan-sen (conversion line) period. Default: 9
    kijun_period : int
        Kijun-sen (base line) period. Default: 26
    senkou_b_period : int
        Senkou Span B period. Default: 52
    swing_lookback : int
        Periods to look back for swing high/low detection. Default: 20
    fib_tolerance : float
        Tolerance zone around Fib levels (as %). Default: 0.5%
    min_swing_size : float
        Minimum swing size as % of price. Default: 0.5%
    """

    @property
    def name(self) -> str:
        return "ichimoku_fibonacci"

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        swing_lookback: int = 20,
        fib_tolerance: float = 0.005,  # 0.5%
        min_swing_size: float = 0.005,  # 0.5%
        risk_reward_ratio: float = 1.5,
    ) -> None:
        """Initialize the IchimokuFibonacci strategy."""
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = 26
        self.swing_lookback = swing_lookback
        self.fib_tolerance = fib_tolerance
        self.min_swing_size = min_swing_size
        self.risk_reward_ratio = risk_reward_ratio

        logger.info(
            f"Initialized IchimokuFibonacci strategy: "
            f"T={tenkan_period}, K={kijun_period}, S={senkou_b_period}, "
            f"swing_lookback={swing_lookback}"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters for logging and comparison."""
        return {
            "tenkan_period": self.tenkan_period,
            "kijun_period": self.kijun_period,
            "senkou_b_period": self.senkou_b_period,
            "swing_lookback": self.swing_lookback,
            "fib_tolerance": self.fib_tolerance,
            "min_swing_size": self.min_swing_size,
            "risk_reward_ratio": self.risk_reward_ratio,
        }

    # -------------------------------------------------------------------------
    # ICHIMOKU CALCULATIONS
    # -------------------------------------------------------------------------

    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud indicators using pandas_ta.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with Ichimoku columns added
        """
        if ta is None:
            raise ImportError("pandas_ta required. Install: pip install pandas_ta")

        min_periods = self.senkou_b_period + self.displacement
        if len(df) < min_periods:
            raise ValueError(
                f"Insufficient data for Ichimoku. "
                f"Need {min_periods} rows, got {len(df)}"
            )

        result_df = df.copy()

        # Calculate Ichimoku using pandas_ta
        ichimoku_df = ta.ichimoku(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            tenkan=self.tenkan_period,
            kijun=self.kijun_period,
            senkou=self.senkou_b_period,
        )

        if ichimoku_df is not None and len(ichimoku_df) > 0:
            # pandas_ta returns tuple of (ichimoku_df, span_df)
            if isinstance(ichimoku_df, tuple):
                ichi_main = ichimoku_df[0]
            else:
                ichi_main = ichimoku_df

            # Map column names (pandas_ta uses different naming)
            col_mapping = {
                f"ISA_{self.tenkan_period}": "senkou_a",
                f"ISB_{self.kijun_period}": "senkou_b",
                f"ITS_{self.tenkan_period}": "tenkan",
                f"IKS_{self.kijun_period}": "kijun",
                f"ICS_{self.kijun_period}": "chikou",
            }

            for src_col, dst_col in col_mapping.items():
                if src_col in ichi_main.columns:
                    result_df[dst_col] = ichi_main[src_col]

        # Fallback manual calculation if pandas_ta fails
        if "tenkan" not in result_df.columns:
            result_df = self._calculate_ichimoku_manual(df)

        return result_df

    def _calculate_ichimoku_manual(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manual Ichimoku calculation as fallback."""
        result_df = df.copy()

        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        high_t = df["high"].rolling(window=self.tenkan_period).max()
        low_t = df["low"].rolling(window=self.tenkan_period).min()
        result_df["tenkan"] = (high_t + low_t) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        high_k = df["high"].rolling(window=self.kijun_period).max()
        low_k = df["low"].rolling(window=self.kijun_period).min()
        result_df["kijun"] = (high_k + low_k) / 2

        # Senkou Span A: (Tenkan + Kijun) / 2, shifted forward
        result_df["senkou_a"] = ((result_df["tenkan"] + result_df["kijun"]) / 2).shift(
            self.displacement
        )

        # Senkou Span B: (52-period high + 52-period low) / 2, shifted forward
        high_s = df["high"].rolling(window=self.senkou_b_period).max()
        low_s = df["low"].rolling(window=self.senkou_b_period).min()
        result_df["senkou_b"] = ((high_s + low_s) / 2).shift(self.displacement)

        # Chikou Span: Close shifted backward
        result_df["chikou"] = df["close"].shift(-self.displacement)

        return result_df

    def _get_trend_direction(
        self,
        close: float,
        senkou_a: float,
        senkou_b: float,
    ) -> TrendDirection:
        """
        Determine trend direction based on Ichimoku Cloud.

        Bullish: Price > Cloud AND Cloud is green (A > B)
        Bearish: Price < Cloud AND Cloud is red (A < B)
        """
        if pd.isna(senkou_a) or pd.isna(senkou_b):
            return TrendDirection.NEUTRAL

        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        cloud_is_green = senkou_a > senkou_b

        if close > cloud_top and cloud_is_green:
            return TrendDirection.BULLISH
        elif close < cloud_bottom and not cloud_is_green:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL

    # -------------------------------------------------------------------------
    # FIBONACCI CALCULATIONS
    # -------------------------------------------------------------------------

    def _find_swing_points(
        self,
        df: pd.DataFrame,
        lookback: int = None,
    ) -> Tuple[Optional[SwingPoint], Optional[SwingPoint]]:
        """
        Find the most recent swing high and swing low.

        Uses a simple pivot point detection algorithm.

        Args:
            df: OHLCV DataFrame
            lookback: Number of periods to look back

        Returns:
            Tuple of (swing_high, swing_low) or (None, None)
        """
        lookback = lookback or self.swing_lookback

        if len(df) < lookback + 2:
            return None, None

        highs = df["high"].values[-lookback:]
        lows = df["low"].values[-lookback:]
        timestamps = df.index[-lookback:]

        swing_high = None
        swing_low = None

        # Find swing high (local maximum)
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2]:
                if highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
                    idx = len(df) - lookback + i
                    swing_high = SwingPoint(
                        index=idx,
                        price=highs[i],
                        is_high=True,
                        timestamp=timestamps[i],
                    )

        # Find swing low (local minimum)
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2]:
                if lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                    idx = len(df) - lookback + i
                    swing_low = SwingPoint(
                        index=idx,
                        price=lows[i],
                        is_high=False,
                        timestamp=timestamps[i],
                    )

        return swing_high, swing_low

    def _get_fib_levels(
        self,
        high: float,
        low: float,
        direction: TrendDirection,
    ) -> FibonacciLevels:
        """
        Calculate Fibonacci retracement levels.

        For BULLISH trend (upswing retracement):
        - Fib levels are measured down from the high
        - 38.2% = High - (High - Low) * 0.382

        For BEARISH trend (downswing retracement):
        - Fib levels are measured up from the low
        - 38.2% = Low + (High - Low) * 0.382

        Args:
            high: Swing high price
            low: Swing low price
            direction: Current trend direction

        Returns:
            FibonacciLevels with all key levels
        """
        range_size = high - low

        if direction == TrendDirection.BULLISH:
            # Retracement down from high
            fib_236 = high - range_size * FibLevel.FIB_236.value
            fib_382 = high - range_size * FibLevel.FIB_382.value
            fib_500 = high - range_size * FibLevel.FIB_500.value
            fib_618 = high - range_size * FibLevel.FIB_618.value
            fib_786 = high - range_size * FibLevel.FIB_786.value
        else:
            # Retracement up from low
            fib_236 = low + range_size * FibLevel.FIB_236.value
            fib_382 = low + range_size * FibLevel.FIB_382.value
            fib_500 = low + range_size * FibLevel.FIB_500.value
            fib_618 = low + range_size * FibLevel.FIB_618.value
            fib_786 = low + range_size * FibLevel.FIB_786.value

        return FibonacciLevels(
            swing_high=high,
            swing_low=low,
            fib_236=fib_236,
            fib_382=fib_382,
            fib_500=fib_500,
            fib_618=fib_618,
            fib_786=fib_786,
            direction=direction,
        )

    def _is_near_fib_level(
        self,
        price: float,
        fib_level: float,
        tolerance: float = None,
    ) -> bool:
        """
        Check if price is within tolerance of a Fibonacci level.

        Args:
            price: Current price
            fib_level: Target Fibonacci level
            tolerance: Tolerance as decimal (default: self.fib_tolerance)

        Returns:
            True if price is within tolerance zone
        """
        tolerance = tolerance or self.fib_tolerance
        tolerance_amount = fib_level * tolerance

        return abs(price - fib_level) <= tolerance_amount

    def _check_fib_bounce(
        self,
        df: pd.DataFrame,
        fib_levels: FibonacciLevels,
        lookback: int = 3,
    ) -> Tuple[bool, str, float]:
        """
        Check if price has bounced off a Fibonacci level.

        Bounce Confirmation:
        - Price touched 38.2% or 50% Fib level
        - Current candle closes above/below the level (direction dependent)
        - Previous candle was below/above the level

        Args:
            df: OHLCV DataFrame
            fib_levels: Calculated Fibonacci levels
            lookback: Candles to check for bounce

        Returns:
            Tuple of (is_bounce, fib_level_name, fib_level_price)
        """
        if len(df) < lookback + 1:
            return False, "", 0.0

        current_close = df["close"].iloc[-1]
        current_low = df["low"].iloc[-1]
        current_high = df["high"].iloc[-1]
        prev_close = df["close"].iloc[-2]

        # Check each target Fib level (38.2% and 50%)
        target_fibs = [
            ("fib_382", fib_levels.fib_382),
            ("fib_500", fib_levels.fib_500),
        ]

        for fib_name, fib_price in target_fibs:
            if fib_levels.direction == TrendDirection.BULLISH:
                # Bullish bounce: Price touched fib from above, now closing above
                price_touched_fib = current_low <= fib_price * (1 + self.fib_tolerance)
                bounce_confirmed = current_close > fib_price
                was_above = prev_close > fib_price

                if price_touched_fib and bounce_confirmed:
                    return True, fib_name, fib_price

            elif fib_levels.direction == TrendDirection.BEARISH:
                # Bearish bounce: Price touched fib from below, now closing below
                price_touched_fib = current_high >= fib_price * (1 - self.fib_tolerance)
                bounce_confirmed = current_close < fib_price
                was_below = prev_close < fib_price

                if price_touched_fib and bounce_confirmed:
                    return True, fib_name, fib_price

        return False, "", 0.0

    # -------------------------------------------------------------------------
    # SIGNAL GENERATION
    # -------------------------------------------------------------------------

    def check_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal based on Ichimoku trend + Fibonacci entry.

        Signal Logic:
        1. Determine trend direction using Ichimoku Cloud
        2. Find recent swing high/low
        3. Calculate Fibonacci retracement levels
        4. Check if price has bounced off 38.2% or 50% level
        5. Generate BUY/SELL signal with stop loss and take profit

        Args:
            df: OHLCV DataFrame with at least 80 rows

        Returns:
            Dict with signal details:
                - signal: 'BUY', 'SELL', or 'NEUTRAL'
                - reason: Human-readable explanation
                - entry_price: Suggested entry price
                - stop_loss: Calculated stop loss
                - take_profit: Calculated take profit (1.5x risk)
                - fib_level: Which Fib level triggered
                - trend: Current trend direction
                - confidence: Signal confidence (0-1)
        """
        result = {
            "signal": "NEUTRAL",
            "reason": "No setup detected",
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "fib_level": None,
            "trend": "neutral",
            "confidence": 0.0,
            "metadata": {},
        }

        try:
            # Calculate Ichimoku indicators
            df_ichi = self._calculate_ichimoku(df)

            # Get current values
            close = df_ichi["close"].iloc[-1]
            senkou_a = df_ichi["senkou_a"].iloc[-1]
            senkou_b = df_ichi["senkou_b"].iloc[-1]
            kijun = df_ichi["kijun"].iloc[-1]

            # Determine trend direction
            trend = self._get_trend_direction(close, senkou_a, senkou_b)
            result["trend"] = trend.value

            if trend == TrendDirection.NEUTRAL:
                result["reason"] = "No clear trend (price in cloud or mixed signals)"
                return result

            # Find swing points
            swing_high, swing_low = self._find_swing_points(df)

            if swing_high is None or swing_low is None:
                result["reason"] = "Could not identify swing points"
                return result

            # Validate swing size
            swing_size = (swing_high.price - swing_low.price) / swing_low.price
            if swing_size < self.min_swing_size:
                result["reason"] = f"Swing too small ({swing_size:.2%} < {self.min_swing_size:.2%})"
                return result

            # Calculate Fibonacci levels
            fib_levels = self._get_fib_levels(
                high=swing_high.price,
                low=swing_low.price,
                direction=trend,
            )

            # Check for Fib bounce
            is_bounce, fib_name, fib_price = self._check_fib_bounce(df, fib_levels)

            if not is_bounce:
                result["reason"] = f"Awaiting pullback to Fib level (trend: {trend.value})"
                result["metadata"] = {
                    "fib_382": fib_levels.fib_382,
                    "fib_500": fib_levels.fib_500,
                    "swing_high": swing_high.price,
                    "swing_low": swing_low.price,
                }
                return result

            # Generate entry signal
            entry_price = close

            if trend == TrendDirection.BULLISH:
                # BUY setup
                stop_loss = swing_low.price * 0.998  # Slightly below swing low
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * self.risk_reward_ratio)

                result["signal"] = "BUY"
                result["reason"] = (
                    f"Bullish Fib bounce: Price bounced off {fib_name.replace('_', ' ').upper()} "
                    f"({fib_price:.2f}) in uptrend. Entry: {entry_price:.2f}, "
                    f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}"
                )

            else:  # BEARISH
                # SELL setup
                stop_loss = swing_high.price * 1.002  # Slightly above swing high
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * self.risk_reward_ratio)

                result["signal"] = "SELL"
                result["reason"] = (
                    f"Bearish Fib rejection: Price rejected at {fib_name.replace('_', ' ').upper()} "
                    f"({fib_price:.2f}) in downtrend. Entry: {entry_price:.2f}, "
                    f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}"
                )

            # Populate result
            result["entry_price"] = float(entry_price)
            result["stop_loss"] = float(stop_loss)
            result["take_profit"] = float(take_profit)
            result["fib_level"] = fib_name
            result["confidence"] = 0.7 if fib_name == "fib_382" else 0.6  # 38.2% is stronger
            result["metadata"] = {
                "fib_382": fib_levels.fib_382,
                "fib_500": fib_levels.fib_500,
                "swing_high": swing_high.price,
                "swing_low": swing_low.price,
                "kijun": float(kijun) if not pd.isna(kijun) else None,
                "risk_reward": self.risk_reward_ratio,
            }

            logger.info(f"IchimokuFibonacci signal: {result['signal']} | {result['reason']}")
            return result

        except Exception as e:
            logger.error(f"Error in IchimokuFibonacci signal: {e}", exc_info=True)
            result["reason"] = f"Error: {str(e)}"
            return result


# =============================================================================
# STRATEGY FACTORY
# =============================================================================


def get_ichimoku_fibonacci_strategy(
    timeframe: str = "15m",
    aggressive: bool = False,
) -> IchimokuFibonacci:
    """
    Factory function to create IchimokuFibonacci strategy with
    timeframe-optimized parameters.

    Args:
        timeframe: Target timeframe ('5m', '15m', '30m', '1h')
        aggressive: If True, use more sensitive parameters

    Returns:
        Configured IchimokuFibonacci instance
    """
    # Timeframe-specific parameter adjustments
    params = {
        "5m": {
            "tenkan_period": 7,
            "kijun_period": 22,
            "senkou_b_period": 44,
            "swing_lookback": 30,
            "fib_tolerance": 0.003,
        },
        "15m": {
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_b_period": 52,
            "swing_lookback": 24,
            "fib_tolerance": 0.004,
        },
        "30m": {
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_b_period": 52,
            "swing_lookback": 20,
            "fib_tolerance": 0.005,
        },
        "1h": {
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_b_period": 52,
            "swing_lookback": 16,
            "fib_tolerance": 0.006,
        },
    }

    config = params.get(timeframe, params["15m"])

    if aggressive:
        config["fib_tolerance"] *= 1.5
        config["min_swing_size"] = 0.003

    return IchimokuFibonacci(**config)


# =============================================================================
# REGISTER STRATEGY
# =============================================================================

# Add to global strategy registry
try:
    from src.core.backtest_engine import register_strategy, STRATEGY_REGISTRY
    STRATEGY_REGISTRY["ichimoku_fibonacci"] = IchimokuFibonacci()
    logger.info("Registered ichimoku_fibonacci strategy")
except ImportError:
    pass
