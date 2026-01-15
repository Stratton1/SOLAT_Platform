"""
Ichimoku Kinko Hyo strategy implementation.

This module implements a trend-following strategy using the Ichimoku Cloud indicator.
The strategy generates BUY/SELL/NEUTRAL signals based on price position relative to the cloud
and the relationship between Tenkan and Kijun lines.
"""

import logging
from typing import Dict, Any

import pandas as pd
from ta.trend import IchimokuIndicator

logger = logging.getLogger(__name__)


class IchimokuStrategy:
    """
    Ichimoku Cloud trend-following strategy.

    The strategy uses the Ichimoku Cloud indicator to identify trend and
    generate trading signals based on:
    - Price position relative to the cloud (Senkou Span A & B)
    - Tenkan-Kijun (TK) cross signals
    - Cloud color (Green = bullish, Red = bearish)

    Ichimoku components:
    - Tenkan-sen (9-period): (9-period high + 9-period low) / 2
    - Kijun-sen (26-period): (26-period high + 26-period low) / 2
    - Senkou Span A: (Tenkan + Kijun) / 2, shifted forward 26 periods
    - Senkou Span B (52-period): (52-period high + 52-period low) / 2, shifted forward 26 periods
    - Chikou Span: Close price shifted backward 26 periods
    """

    def __init__(self) -> None:
        """Initialize the Ichimoku strategy."""
        self.name = "Ichimoku Kinko Hyo"
        # Standard Ichimoku periods
        self.tenkan_period = 9
        self.kijun_period = 26
        self.senkou_b_period = 52
        self.displacement = 26

        logger.info("Initialized IchimokuStrategy")

    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud indicators.

        Uses the `ta` library IchimokuIndicator to calculate all components.

        Args:
            df (pd.DataFrame): OHLCV DataFrame with DatetimeIndex

        Returns:
            pd.DataFrame: DataFrame with Ichimoku indicators added as columns:
                - tenkan: 9-period momentum line
                - kijun: 26-period momentum line
                - senkou_a: Cloud upper boundary
                - senkou_b: Cloud lower boundary
                - chikou: Lagged close price

        Raises:
            ValueError: If DataFrame is too small for Ichimoku calculation
        """
        if len(df) < self.senkou_b_period + self.displacement:
            raise ValueError(
                f"Insufficient data for Ichimoku. "
                f"Need at least {self.senkou_b_period + self.displacement} rows, "
                f"got {len(df)}"
            )

        # Calculate Ichimoku using ta.trend.IchimokuIndicator
        ichimoku = IchimokuIndicator(
            high=df["high"],
            low=df["low"],
            window1=self.tenkan_period,      # 9-period
            window2=self.kijun_period,       # 26-period
            window3=self.senkou_b_period,    # 52-period
            visual=True,
            fillna=False
        )

        result_df = df.copy()

        # Add Ichimoku components
        result_df["tenkan"] = ichimoku.ichimoku_conversion_line()
        result_df["kijun"] = ichimoku.ichimoku_base_line()
        result_df["senkou_a"] = ichimoku.ichimoku_a()
        result_df["senkou_b"] = ichimoku.ichimoku_b()
        # Chikou Span is the close price shifted backward 26 periods
        # We'll use a simple shift since ta library doesn't provide this directly
        result_df["chikou"] = df["close"].shift(-self.displacement)

        logger.debug(
            f"Calculated Ichimoku for {len(df)} candles. "
            f"Current close={df['close'].iloc[-1]:.2f}, "
            f"tenkan={result_df['tenkan'].iloc[-1]:.2f}, "
            f"kijun={result_df['kijun'].iloc[-1]:.2f}"
        )

        return result_df

    def _determine_cloud_color(self, tenkan: float, kijun: float) -> str:
        """
        Determine cloud color based on Tenkan-Kijun relationship.

        Args:
            tenkan (float): Tenkan-sen value
            kijun (float): Kijun-sen value

        Returns:
            str: "green" if tenkan > kijun, "red" if tenkan < kijun, "neutral" if equal
        """
        if pd.isna(tenkan) or pd.isna(kijun):
            return "neutral"

        if tenkan > kijun:
            return "green"
        elif tenkan < kijun:
            return "red"
        else:
            return "neutral"

    def _is_price_above_cloud(
        self,
        close: float,
        senkou_a: float,
        senkou_b: float
    ) -> bool:
        """
        Check if price is above the cloud.

        Args:
            close (float): Current close price
            senkou_a (float): Senkou Span A
            senkou_b (float): Senkou Span B

        Returns:
            bool: True if price is above both cloud boundaries
        """
        if pd.isna(close) or pd.isna(senkou_a) or pd.isna(senkou_b):
            return False

        cloud_top = max(senkou_a, senkou_b)
        return close > cloud_top

    def _is_price_below_cloud(
        self,
        close: float,
        senkou_a: float,
        senkou_b: float
    ) -> bool:
        """
        Check if price is below the cloud.

        Args:
            close (float): Current close price
            senkou_a (float): Senkou Span A
            senkou_b (float): Senkou Span B

        Returns:
            bool: True if price is below both cloud boundaries
        """
        if pd.isna(close) or pd.isna(senkou_a) or pd.isna(senkou_b):
            return False

        cloud_bottom = min(senkou_a, senkou_b)
        return close < cloud_bottom

    def check_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze OHLCV DataFrame and return a trading signal.

        Signal logic:
        - **BUY**: Close > Cloud (above both Span A & B) AND Tenkan > Kijun
        - **SELL**: Close < Cloud (below both Span A & B) AND Tenkan < Kijun
        - **NEUTRAL**: Cloud breakout or no clear signal

        Args:
            df (pd.DataFrame): OHLCV DataFrame with columns: open, high, low, close, volume

        Returns:
            Dict[str, Any]: Dictionary with keys:
                - signal (str): 'BUY', 'SELL', or 'NEUTRAL'
                - reason (str): Human-readable explanation of the signal
                - close (float): Current close price
                - tenkan (float): Tenkan-sen value
                - kijun (float): Kijun-sen value
                - senkou_a (float): Senkou Span A value
                - senkou_b (float): Senkou Span B value
                - cloud_color (str): 'green', 'red', or 'neutral'

        Raises:
            ValueError: If DataFrame has insufficient data
        """
        try:
            # Calculate Ichimoku indicators
            df_with_ichimoku = self._calculate_ichimoku(df)

            # Get the latest values (last row)
            close = df_with_ichimoku["close"].iloc[-1]
            tenkan = df_with_ichimoku["tenkan"].iloc[-1]
            kijun = df_with_ichimoku["kijun"].iloc[-1]
            senkou_a = df_with_ichimoku["senkou_a"].iloc[-1]
            senkou_b = df_with_ichimoku["senkou_b"].iloc[-1]

            # Determine cloud color
            cloud_color = self._determine_cloud_color(tenkan, kijun)

            # Check signal conditions
            price_above_cloud = self._is_price_above_cloud(close, senkou_a, senkou_b)
            price_below_cloud = self._is_price_below_cloud(close, senkou_a, senkou_b)
            tenkan_above_kijun = (not pd.isna(tenkan) and not pd.isna(kijun) and tenkan > kijun)
            tenkan_below_kijun = (not pd.isna(tenkan) and not pd.isna(kijun) and tenkan < kijun)

            signal = "NEUTRAL"
            reason = "No clear signal"

            # BUY Signal: Price above cloud AND Tenkan > Kijun (bullish)
            if price_above_cloud and tenkan_above_kijun:
                signal = "BUY"
                reason = (
                    f"Price {close:.2f} above cloud (Span A={senkou_a:.2f}, "
                    f"Span B={senkou_b:.2f}) AND Tenkan {tenkan:.2f} > Kijun {kijun:.2f}"
                )

            # SELL Signal: Price below cloud AND Tenkan < Kijun (bearish)
            elif price_below_cloud and tenkan_below_kijun:
                signal = "SELL"
                reason = (
                    f"Price {close:.2f} below cloud (Span A={senkou_a:.2f}, "
                    f"Span B={senkou_b:.2f}) AND Tenkan {tenkan:.2f} < Kijun {kijun:.2f}"
                )

            # Cloud breakout (bullish without TK confirmation)
            elif price_above_cloud and not tenkan_above_kijun:
                reason = (
                    f"Price breakout above cloud (awaiting Tenkan > Kijun confirmation). "
                    f"Close={close:.2f}, Cloud={max(senkou_a, senkou_b):.2f}"
                )

            # Cloud breakout (bearish without TK confirmation)
            elif price_below_cloud and not tenkan_below_kijun:
                reason = (
                    f"Price breakout below cloud (awaiting Tenkan < Kijun confirmation). "
                    f"Close={close:.2f}, Cloud={min(senkou_a, senkou_b):.2f}"
                )

            else:
                reason = f"Price in cloud or consolidation. Close={close:.2f}, Tenkan/Kijun={cloud_color}"

            result = {
                "signal": signal,
                "reason": reason,
                "close": float(close),
                "tenkan": float(tenkan) if not pd.isna(tenkan) else None,
                "kijun": float(kijun) if not pd.isna(kijun) else None,
                "senkou_a": float(senkou_a) if not pd.isna(senkou_a) else None,
                "senkou_b": float(senkou_b) if not pd.isna(senkou_b) else None,
                "cloud_color": cloud_color,
            }

            logger.info(f"Signal: {signal} | Reason: {reason}")
            return result

        except Exception as e:
            logger.error(f"Error checking Ichimoku signal: {e}", exc_info=True)
            return {
                "signal": "NEUTRAL",
                "reason": f"Error: {str(e)}",
                "close": None,
                "tenkan": None,
                "kijun": None,
                "senkou_a": None,
                "senkou_b": None,
                "cloud_color": "error",
            }
