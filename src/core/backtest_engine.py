"""
Multi-Strategy Backtesting Engine for SOLAT Platform.

This module provides:
1. BaseStrategy - Abstract base class for all trading strategies
2. 4 Ichimoku variations (Standard, Aggressive, Mean Reversion, Conservative)
3. StrategyOptimizer - Runs all strategies and identifies optimal per asset

The framework is extensible - new strategies can be added by subclassing BaseStrategy.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from ta.trend import IchimokuIndicator

from src.database.repository import get_connection

logger = logging.getLogger(__name__)


# =============================================================================
# ABSTRACT BASE STRATEGY
# =============================================================================


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategies must implement:
    - name (property): Unique identifier for the strategy
    - check_signal(df): Generate trading signal from OHLCV data
    - get_parameters(): Return strategy parameters for logging

    Usage:
        class MyStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "my_strategy"

            def check_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
                # Your signal logic here
                return {"signal": "BUY", "reason": "..."}

            def get_parameters(self) -> Dict[str, Any]:
                return {"param1": value1}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier (e.g., 'ichimoku_standard')."""
        pass

    @abstractmethod
    def check_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal from OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index should be DatetimeIndex (UTC)

        Returns:
            Dict with keys:
                - signal: 'BUY', 'SELL', or 'NEUTRAL'
                - reason: Human-readable explanation
                - confidence: Optional float 0-1
                - metadata: Optional dict with strategy-specific data
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters for logging and comparison."""
        pass


# =============================================================================
# ICHIMOKU STRATEGY VARIATIONS
# =============================================================================


class IchimokuBase(BaseStrategy):
    """
    Base class for Ichimoku-based strategies.

    Provides common Ichimoku calculation and signal generation logic.
    Subclasses override periods and signal rules.
    """

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26,
    ) -> None:
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement

    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku components using ta library."""
        if len(df) < self.senkou_b_period + self.displacement:
            raise ValueError(
                f"Insufficient data for Ichimoku. "
                f"Need at least {self.senkou_b_period + self.displacement} rows, "
                f"got {len(df)}"
            )

        ichimoku = IchimokuIndicator(
            high=df["high"],
            low=df["low"],
            window1=self.tenkan_period,
            window2=self.kijun_period,
            window3=self.senkou_b_period,
            visual=True,
            fillna=False,
        )

        result_df = df.copy()
        result_df["tenkan"] = ichimoku.ichimoku_conversion_line()
        result_df["kijun"] = ichimoku.ichimoku_base_line()
        result_df["senkou_a"] = ichimoku.ichimoku_a()
        result_df["senkou_b"] = ichimoku.ichimoku_b()
        result_df["chikou"] = df["close"].shift(-self.displacement)

        return result_df

    def _is_price_above_cloud(
        self, close: float, senkou_a: float, senkou_b: float
    ) -> bool:
        """Check if price is above the cloud."""
        if pd.isna(close) or pd.isna(senkou_a) or pd.isna(senkou_b):
            return False
        cloud_top = max(senkou_a, senkou_b)
        return close > cloud_top

    def _is_price_below_cloud(
        self, close: float, senkou_a: float, senkou_b: float
    ) -> bool:
        """Check if price is below the cloud."""
        if pd.isna(close) or pd.isna(senkou_a) or pd.isna(senkou_b):
            return False
        cloud_bottom = min(senkou_a, senkou_b)
        return close < cloud_bottom

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "tenkan_period": self.tenkan_period,
            "kijun_period": self.kijun_period,
            "senkou_b_period": self.senkou_b_period,
            "displacement": self.displacement,
        }


class IchimokuStandard(IchimokuBase):
    """
    Standard Ichimoku strategy with classic 9/26/52 periods.

    Signal Logic:
    - BUY: Price > Cloud AND Tenkan > Kijun
    - SELL: Price < Cloud AND Tenkan < Kijun
    """

    @property
    def name(self) -> str:
        return "ichimoku_standard"

    def __init__(self) -> None:
        super().__init__(tenkan_period=9, kijun_period=26, senkou_b_period=52)

    def check_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            df_ichi = self._calculate_ichimoku(df)

            close = df_ichi["close"].iloc[-1]
            tenkan = df_ichi["tenkan"].iloc[-1]
            kijun = df_ichi["kijun"].iloc[-1]
            senkou_a = df_ichi["senkou_a"].iloc[-1]
            senkou_b = df_ichi["senkou_b"].iloc[-1]

            price_above = self._is_price_above_cloud(close, senkou_a, senkou_b)
            price_below = self._is_price_below_cloud(close, senkou_a, senkou_b)
            tk_bullish = not pd.isna(tenkan) and not pd.isna(kijun) and tenkan > kijun
            tk_bearish = not pd.isna(tenkan) and not pd.isna(kijun) and tenkan < kijun

            signal = "NEUTRAL"
            reason = "No clear signal"

            if price_above and tk_bullish:
                signal = "BUY"
                reason = f"Price {close:.2f} > Cloud, Tenkan {tenkan:.2f} > Kijun {kijun:.2f}"
            elif price_below and tk_bearish:
                signal = "SELL"
                reason = f"Price {close:.2f} < Cloud, Tenkan {tenkan:.2f} < Kijun {kijun:.2f}"

            return {
                "signal": signal,
                "reason": reason,
                "close": float(close),
                "tenkan": float(tenkan) if not pd.isna(tenkan) else None,
                "kijun": float(kijun) if not pd.isna(kijun) else None,
                "cloud_top": float(max(senkou_a, senkou_b)) if not pd.isna(senkou_a) else None,
            }

        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return {"signal": "NEUTRAL", "reason": f"Error: {e}"}


class IchimokuAggressive(IchimokuBase):
    """
    Aggressive Ichimoku with faster periods (7/22/44).

    Generates more signals but may have more false positives.
    Best for trending markets with high volatility.
    """

    @property
    def name(self) -> str:
        return "ichimoku_aggressive"

    def __init__(self) -> None:
        super().__init__(tenkan_period=7, kijun_period=22, senkou_b_period=44)

    def check_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            df_ichi = self._calculate_ichimoku(df)

            close = df_ichi["close"].iloc[-1]
            tenkan = df_ichi["tenkan"].iloc[-1]
            kijun = df_ichi["kijun"].iloc[-1]
            senkou_a = df_ichi["senkou_a"].iloc[-1]
            senkou_b = df_ichi["senkou_b"].iloc[-1]

            # Aggressive: Only requires price vs cloud (no TK confirmation)
            price_above = self._is_price_above_cloud(close, senkou_a, senkou_b)
            price_below = self._is_price_below_cloud(close, senkou_a, senkou_b)

            # Add volume confirmation for aggressive signals
            vol_avg = df["volume"].rolling(20).mean().iloc[-1]
            vol_spike = df["volume"].iloc[-1] > vol_avg * 1.2

            signal = "NEUTRAL"
            reason = "No clear signal"

            if price_above and vol_spike:
                signal = "BUY"
                reason = f"Aggressive BUY: Price {close:.2f} > Cloud with volume spike"
            elif price_below and vol_spike:
                signal = "SELL"
                reason = f"Aggressive SELL: Price {close:.2f} < Cloud with volume spike"

            return {
                "signal": signal,
                "reason": reason,
                "close": float(close),
                "volume_ratio": float(df["volume"].iloc[-1] / vol_avg) if vol_avg > 0 else 1.0,
            }

        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return {"signal": "NEUTRAL", "reason": f"Error: {e}"}


class IchimokuConservative(IchimokuBase):
    """
    Conservative Ichimoku with slower periods (12/30/60).

    Fewer signals but higher quality. Best for sideways/choppy markets.
    Requires Chikou confirmation (Sanyaku Kouten).
    """

    @property
    def name(self) -> str:
        return "ichimoku_conservative"

    def __init__(self) -> None:
        super().__init__(tenkan_period=12, kijun_period=30, senkou_b_period=60)

    def check_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            df_ichi = self._calculate_ichimoku(df)

            close = df_ichi["close"].iloc[-1]
            tenkan = df_ichi["tenkan"].iloc[-1]
            kijun = df_ichi["kijun"].iloc[-1]
            senkou_a = df_ichi["senkou_a"].iloc[-1]
            senkou_b = df_ichi["senkou_b"].iloc[-1]

            # Conservative: Requires ALL confirmations (Sanyaku Kouten)
            price_above = self._is_price_above_cloud(close, senkou_a, senkou_b)
            price_below = self._is_price_below_cloud(close, senkou_a, senkou_b)
            tk_bullish = not pd.isna(tenkan) and not pd.isna(kijun) and tenkan > kijun
            tk_bearish = not pd.isna(tenkan) and not pd.isna(kijun) and tenkan < kijun

            # Chikou confirmation: current close vs close 26 periods ago
            chikou_ref_idx = -self.displacement - 1
            if len(df) > abs(chikou_ref_idx):
                chikou_ref = df["close"].iloc[chikou_ref_idx]
                chikou_bullish = close > chikou_ref
                chikou_bearish = close < chikou_ref
            else:
                chikou_bullish = False
                chikou_bearish = False

            signal = "NEUTRAL"
            reason = "Awaiting full confirmation"

            # Sanyaku Kouten: ALL 3 conditions must align
            if price_above and tk_bullish and chikou_bullish:
                signal = "BUY"
                reason = f"Sanyaku Kouten BUY: Price > Cloud, TK bullish, Chikou confirmed"
            elif price_below and tk_bearish and chikou_bearish:
                signal = "SELL"
                reason = f"Sanyaku Kouten SELL: Price < Cloud, TK bearish, Chikou confirmed"

            return {
                "signal": signal,
                "reason": reason,
                "close": float(close),
                "chikou_confirmed": chikou_bullish or chikou_bearish,
            }

        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return {"signal": "NEUTRAL", "reason": f"Error: {e}"}


class IchimokuMeanReversion(IchimokuBase):
    """
    Mean Reversion Ichimoku - price bouncing off Kijun-sen.

    Looks for price returning to Kijun after extended move.
    Best for ranging markets where trend-following fails.
    """

    @property
    def name(self) -> str:
        return "ichimoku_mean_reversion"

    def __init__(self) -> None:
        super().__init__(tenkan_period=9, kijun_period=26, senkou_b_period=52)

    def check_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            df_ichi = self._calculate_ichimoku(df)

            close = df_ichi["close"].iloc[-1]
            kijun = df_ichi["kijun"].iloc[-1]
            prev_close = df_ichi["close"].iloc[-2]

            if pd.isna(kijun):
                return {"signal": "NEUTRAL", "reason": "Insufficient data"}

            # Calculate distance from Kijun as percentage
            kijun_distance = (close - kijun) / kijun * 100

            # Mean reversion threshold: 2% away from Kijun
            threshold = 2.0

            signal = "NEUTRAL"
            reason = "Price near equilibrium"

            # Buy when price is below Kijun and bouncing up
            if kijun_distance < -threshold and close > prev_close:
                signal = "BUY"
                reason = f"Mean Reversion BUY: Price {kijun_distance:.1f}% below Kijun, bouncing"

            # Sell when price is above Kijun and falling
            elif kijun_distance > threshold and close < prev_close:
                signal = "SELL"
                reason = f"Mean Reversion SELL: Price {kijun_distance:.1f}% above Kijun, falling"

            return {
                "signal": signal,
                "reason": reason,
                "close": float(close),
                "kijun": float(kijun),
                "kijun_distance_pct": float(kijun_distance),
            }

        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return {"signal": "NEUTRAL", "reason": f"Error: {e}"}


# =============================================================================
# STRATEGY OPTIMIZER
# =============================================================================


class StrategyOptimizer:
    """
    Multi-strategy backtesting engine.

    Runs all registered strategies on historical data and identifies
    the optimal strategy per asset based on risk-adjusted returns.

    Usage:
        optimizer = StrategyOptimizer()
        optimizer.register_strategy(MyCustomStrategy())
        best = optimizer.find_best_strategy("BTC/USDT", df)
    """

    def __init__(self) -> None:
        """Initialize with default Ichimoku variations."""
        self.strategies: List[BaseStrategy] = [
            IchimokuStandard(),
            IchimokuAggressive(),
            IchimokuConservative(),
            IchimokuMeanReversion(),
        ]
        logger.info(f"StrategyOptimizer initialized with {len(self.strategies)} strategies")

    def register_strategy(self, strategy: BaseStrategy) -> None:
        """Add a custom strategy to the optimizer."""
        self.strategies.append(strategy)
        logger.info(f"Registered strategy: {strategy.name}")

    def get_strategy_by_name(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy instance by name."""
        for strategy in self.strategies:
            if strategy.name == name:
                return strategy
        return None

    def backtest_strategy(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        initial_capital: float = 10000.0,
    ) -> Dict[str, float]:
        """
        Run backtest for a single strategy on historical data.

        Args:
            strategy: Strategy instance to test
            df: OHLCV DataFrame with at least 100 rows
            initial_capital: Starting capital for simulation

        Returns:
            Dict with performance metrics:
                - total_return: Percentage return
                - sharpe_ratio: Risk-adjusted return
                - max_drawdown: Maximum peak-to-trough decline
                - win_rate: Percentage of winning trades
                - profit_factor: Gross profit / Gross loss
                - total_trades: Number of trades executed
                - avg_win: Average winning trade
                - avg_loss: Average losing trade
        """
        if len(df) < 100:
            logger.warning(f"Insufficient data for backtest: {len(df)} rows")
            return self._empty_metrics()

        trades: List[Dict] = []
        position = None  # Current position: None, 'long', 'short'
        entry_price = 0.0
        equity = initial_capital
        peak_equity = initial_capital
        equity_curve = [initial_capital]

        # Walk through data generating signals
        for i in range(80, len(df)):
            df_slice = df.iloc[: i + 1]

            try:
                result = strategy.check_signal(df_slice)
                signal = result.get("signal", "NEUTRAL")
                current_price = df["close"].iloc[i]

                # Exit logic
                if position == "long" and signal in ["SELL", "NEUTRAL"]:
                    pnl = (current_price - entry_price) / entry_price
                    equity *= 1 + pnl
                    trades.append({"pnl": pnl, "side": "long"})
                    position = None

                elif position == "short" and signal in ["BUY", "NEUTRAL"]:
                    pnl = (entry_price - current_price) / entry_price
                    equity *= 1 + pnl
                    trades.append({"pnl": pnl, "side": "short"})
                    position = None

                # Entry logic
                if position is None:
                    if signal == "BUY":
                        position = "long"
                        entry_price = current_price
                    elif signal == "SELL":
                        position = "short"
                        entry_price = current_price

                # Track equity
                equity_curve.append(equity)
                peak_equity = max(peak_equity, equity)

            except Exception as e:
                logger.debug(f"Backtest signal error at index {i}: {e}")
                continue

        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve, initial_capital)

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics structure."""
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "total_trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    def _calculate_metrics(
        self,
        trades: List[Dict],
        equity_curve: List[float],
        initial_capital: float,
    ) -> Dict[str, float]:
        """Calculate performance metrics from trade list and equity curve."""
        if not trades:
            return self._empty_metrics()

        # Basic metrics
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_trades = len(trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.5

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0

        # Total return
        final_equity = equity_curve[-1] if equity_curve else initial_capital
        total_return = (final_equity - initial_capital) / initial_capital

        # Sharpe ratio (annualized, assuming daily data)
        returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else [0]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "total_trades": total_trades,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
        }

    def find_best_strategy(
        self,
        symbol: str,
        df: pd.DataFrame,
        save_to_db: bool = True,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Run all strategies and return the best performing one.

        Ranking criteria (weighted):
        1. Sharpe ratio (40%) - risk-adjusted returns
        2. Win rate (30%) - consistency
        3. Max drawdown (30%) - risk (inverted)

        Args:
            symbol: Asset symbol
            df: OHLCV DataFrame
            save_to_db: Whether to persist results to database

        Returns:
            Tuple[str, Dict]: (best_strategy_name, metrics)
        """
        results: List[Tuple[str, Dict[str, float], float]] = []

        for strategy in self.strategies:
            logger.info(f"Backtesting {strategy.name} on {symbol}...")
            metrics = self.backtest_strategy(strategy, df)

            # Calculate composite score
            # Normalize: Sharpe (higher=better), WinRate (higher=better), Drawdown (lower=better)
            sharpe_score = min(metrics["sharpe_ratio"] / 3.0, 1.0)  # Cap at 3.0
            wr_score = metrics["win_rate"]
            dd_score = 1.0 - min(metrics["max_drawdown"], 1.0)  # Invert

            composite = 0.4 * sharpe_score + 0.3 * wr_score + 0.3 * dd_score

            results.append((strategy.name, metrics, composite))

            logger.info(
                f"  {strategy.name}: Sharpe={metrics['sharpe_ratio']:.2f}, "
                f"WinRate={metrics['win_rate']:.1%}, MaxDD={metrics['max_drawdown']:.1%}, "
                f"Score={composite:.3f}"
            )

        # Sort by composite score (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        best_name, best_metrics, best_score = results[0]

        logger.info(f"Best strategy for {symbol}: {best_name} (score={best_score:.3f})")

        # Save to database
        if save_to_db:
            self._save_results_to_db(symbol, results)

        return best_name, best_metrics

    def _save_results_to_db(
        self,
        symbol: str,
        results: List[Tuple[str, Dict[str, float], float]],
    ) -> None:
        """Save strategy performance results to database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()

            # First, clear previous optimal flags for this symbol
            cursor.execute(
                "UPDATE strategy_performance SET is_optimal = 0 WHERE symbol = ?",
                (symbol,),
            )

            # Insert/update results for each strategy
            for i, (strategy_name, metrics, score) in enumerate(results):
                is_optimal = 1 if i == 0 else 0  # First is best

                cursor.execute(
                    """
                    INSERT INTO strategy_performance
                    (symbol, strategy_name, win_rate, sharpe_ratio, max_drawdown,
                     total_trades, profit_factor, avg_win, avg_loss, is_optimal, calculated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, strategy_name) DO UPDATE SET
                        win_rate = excluded.win_rate,
                        sharpe_ratio = excluded.sharpe_ratio,
                        max_drawdown = excluded.max_drawdown,
                        total_trades = excluded.total_trades,
                        profit_factor = excluded.profit_factor,
                        avg_win = excluded.avg_win,
                        avg_loss = excluded.avg_loss,
                        is_optimal = excluded.is_optimal,
                        calculated_at = excluded.calculated_at
                    """,
                    (
                        symbol,
                        strategy_name,
                        metrics["win_rate"],
                        metrics["sharpe_ratio"],
                        metrics["max_drawdown"],
                        metrics["total_trades"],
                        metrics["profit_factor"],
                        metrics["avg_win"],
                        metrics["avg_loss"],
                        is_optimal,
                        now,
                    ),
                )

            # Update assets table with optimal strategy
            best_strategy = results[0][0]
            cursor.execute(
                "UPDATE assets SET optimal_strategy = ? WHERE symbol = ?",
                (best_strategy, symbol),
            )

            conn.commit()
            conn.close()
            logger.info(f"Saved strategy performance for {symbol} to database")

        except Exception as e:
            logger.error(f"Error saving strategy results: {e}")

    def get_strategy_matrix(self) -> pd.DataFrame:
        """
        Generate matrix of all assets x all strategies from database.

        Returns:
            DataFrame where rows=assets, columns=strategies, values=win_rate
        """
        try:
            conn = get_connection()
            query = """
                SELECT symbol, strategy_name, win_rate, is_optimal
                FROM strategy_performance
                ORDER BY symbol, strategy_name
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                return pd.DataFrame()

            # Pivot to matrix format
            matrix = df.pivot(index="symbol", columns="strategy_name", values="win_rate")
            return matrix

        except Exception as e:
            logger.error(f"Error loading strategy matrix: {e}")
            return pd.DataFrame()

    def get_optimal_strategies(self) -> Dict[str, str]:
        """
        Get mapping of symbol -> optimal strategy name from database.

        Returns:
            Dict[str, str]: {symbol: strategy_name}
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT symbol, strategy_name FROM strategy_performance
                WHERE is_optimal = 1
                """
            )
            results = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()
            return results

        except Exception as e:
            logger.error(f"Error loading optimal strategies: {e}")
            return {}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Registry of available strategies
STRATEGY_REGISTRY: Dict[str, BaseStrategy] = {
    "ichimoku_standard": IchimokuStandard(),
    "ichimoku_aggressive": IchimokuAggressive(),
    "ichimoku_conservative": IchimokuConservative(),
    "ichimoku_mean_reversion": IchimokuMeanReversion(),
}


def get_strategy_by_name(name: str) -> Optional[BaseStrategy]:
    """
    Get a strategy instance by its name.

    Args:
        name: Strategy name (e.g., 'ichimoku_standard')

    Returns:
        BaseStrategy instance or None if not found
    """
    return STRATEGY_REGISTRY.get(name)


def get_all_strategies() -> List[BaseStrategy]:
    """
    Get all registered strategies.

    Returns:
        List of all BaseStrategy instances
    """
    return list(STRATEGY_REGISTRY.values())


def register_strategy(strategy: BaseStrategy) -> None:
    """
    Register a new strategy in the registry.

    Args:
        strategy: BaseStrategy instance to register
    """
    STRATEGY_REGISTRY[strategy.name] = strategy
    logger.info(f"Registered strategy: {strategy.name}")
