"""
The Gauntlet - Multi-Timeframe Optimization Engine for SOLAT Platform.

This module provides an automated optimization loop that:
1. Iterates through ALL assets in the database
2. Tests multiple timeframes (5m, 15m, 30m, 1h)
3. Runs backtests with the IchimokuFibonacci strategy
4. Selects optimal configuration based on trade frequency and profitability
5. Stores results in the database for live Sentinel use

Target: 2-4 trades per day per asset (high-frequency day trading)

Selection Criteria:
- Frequency: 2-5 trades per day (reject if < 2 or > 8)
- Profitability: Profit Factor > 1.3
- Win Rate: > 45%
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from src.database.repository import get_connection
from src.config.settings import DB_PATH

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class OptimizationResult:
    """Results from a single optimization run."""
    symbol: str
    timeframe: str
    strategy: str
    total_trades: int
    trades_per_day: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    avg_trade_duration: float  # In minutes
    is_valid: bool
    rejection_reason: Optional[str]
    parameters: Dict[str, Any]
    calculated_at: datetime


@dataclass
class GauntletConfig:
    """Configuration for The Gauntlet optimization run."""
    timeframes: List[str] = None
    min_trades_per_day: float = 2.0
    max_trades_per_day: float = 8.0
    min_profit_factor: float = 1.3
    min_win_rate: float = 0.45
    backtest_days: int = 30
    initial_capital: float = 10000.0

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["5m", "15m", "30m", "1h"]


# =============================================================================
# THE GAUNTLET ENGINE
# =============================================================================


class GauntletOptimizer:
    """
    The Gauntlet - Automated Multi-Timeframe Strategy Optimization Engine.

    This engine systematically tests assets across multiple timeframes
    to find the optimal configuration for high-frequency day trading.

    Usage:
        gauntlet = GauntletOptimizer()
        results = gauntlet.run_mass_optimization()
        gauntlet.save_winners_to_db(results)
    """

    def __init__(self, config: GauntletConfig = None) -> None:
        """
        Initialize The Gauntlet.

        Args:
            config: Optional GauntletConfig for custom settings
        """
        self.config = config or GauntletConfig()
        self._adapters = {}
        self._strategy = None

        logger.info(
            f"Gauntlet initialized: timeframes={self.config.timeframes}, "
            f"target={self.config.min_trades_per_day}-{self.config.max_trades_per_day} trades/day"
        )

    def _get_adapter(self, source: str):
        """Get or create market data adapter for source."""
        if source not in self._adapters:
            if source == "ccxt":
                from src.adapters.ccxt_lib import CCXTAdapter
                self._adapters[source] = CCXTAdapter()
            elif source == "yfinance":
                from src.adapters.yfinance_lib import YFinanceAdapter
                self._adapters[source] = YFinanceAdapter()
            elif source == "ig":
                from src.adapters.ig_lib import IGAdapter
                self._adapters[source] = IGAdapter()
            else:
                raise ValueError(f"Unknown data source: {source}")

        return self._adapters[source]

    def _get_strategy(self):
        """Get or create IchimokuFibonacci strategy instance."""
        if self._strategy is None:
            from src.core.strategies import IchimokuFibonacci
            self._strategy = IchimokuFibonacci()
        return self._strategy

    def _load_assets_from_db(self) -> List[Dict[str, str]]:
        """
        Load all assets from the database.

        Returns:
            List of asset dicts with symbol and source
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, source FROM assets")
            assets = [{"symbol": row[0], "source": row[1]} for row in cursor.fetchall()]
            conn.close()

            if not assets:
                # Fallback to seed file
                assets = self._load_assets_from_seed()

            logger.info(f"Loaded {len(assets)} assets for optimization")
            return assets

        except Exception as e:
            logger.error(f"Error loading assets: {e}")
            return self._load_assets_from_seed()

    def _load_assets_from_seed(self) -> List[Dict[str, str]]:
        """Load assets from seed JSON file."""
        try:
            with open("src/config/assets_seed.json", "r") as f:
                assets = json.load(f)
            return [{"symbol": a["symbol"], "source": a["source"]} for a in assets]
        except Exception as e:
            logger.error(f"Error loading seed file: {e}")
            return []

    def _fetch_high_granularity_data(
        self,
        symbol: str,
        source: str,
        days: int = 30,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch high-granularity (5m) data for backtesting.

        Args:
            symbol: Asset symbol
            source: Data source (ccxt, yfinance, ig)
            days: Number of days to fetch

        Returns:
            DataFrame with 5m OHLCV data or None if failed
        """
        try:
            adapter = self._get_adapter(source)

            # Calculate number of 5m candles needed
            # 30 days * 24 hours * 12 candles/hour = 8640 candles
            limit = days * 24 * 12

            # Fetch 5m data (highest granularity we support)
            df = adapter.get_ohlcv(symbol, "5m", limit=limit)

            if df is None or len(df) < 1000:  # Need at least ~3 days
                logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} rows")
                return None

            logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def _resample_data(
        self,
        df: pd.DataFrame,
        target_timeframe: str,
    ) -> pd.DataFrame:
        """
        Resample 5m data to higher timeframe.

        Args:
            df: 5m OHLCV DataFrame
            target_timeframe: Target timeframe (15m, 30m, 1h)

        Returns:
            Resampled DataFrame
        """
        # Map timeframe to pandas resample rule
        resample_map = {
            "5m": "5T",
            "15m": "15T",
            "30m": "30T",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }

        if target_timeframe not in resample_map:
            raise ValueError(f"Unknown timeframe: {target_timeframe}")

        if target_timeframe == "5m":
            return df  # No resampling needed

        rule = resample_map[target_timeframe]

        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        logger.debug(f"Resampled to {target_timeframe}: {len(resampled)} candles")
        return resampled

    def _run_backtest(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Dict[str, Any]:
        """
        Run backtest using IchimokuFibonacci strategy.

        Args:
            df: OHLCV DataFrame
            symbol: Asset symbol
            timeframe: Timeframe being tested

        Returns:
            Dict with backtest metrics
        """
        from src.core.strategies import get_ichimoku_fibonacci_strategy

        # Get timeframe-optimized strategy
        strategy = get_ichimoku_fibonacci_strategy(timeframe)

        if len(df) < 100:
            return self._empty_metrics()

        trades: List[Dict] = []
        position = None
        entry_price = 0.0
        entry_time = None
        equity = self.config.initial_capital
        equity_curve = [self.config.initial_capital]

        # Walk through data generating signals
        min_bars = max(80, strategy.senkou_b_period + strategy.displacement + 10)

        for i in range(min_bars, len(df)):
            df_slice = df.iloc[:i + 1]

            try:
                result = strategy.check_signal(df_slice)
                signal = result.get("signal", "NEUTRAL")
                current_price = df["close"].iloc[i]
                current_time = df.index[i]

                # Exit logic - check stop loss and take profit
                if position is not None:
                    stop_loss = position.get("stop_loss", 0)
                    take_profit = position.get("take_profit", float('inf'))

                    if position["side"] == "long":
                        # Check stop loss
                        if df["low"].iloc[i] <= stop_loss:
                            pnl = (stop_loss - entry_price) / entry_price
                            exit_reason = "stop_loss"
                        # Check take profit
                        elif df["high"].iloc[i] >= take_profit:
                            pnl = (take_profit - entry_price) / entry_price
                            exit_reason = "take_profit"
                        # Check signal exit
                        elif signal in ["SELL", "NEUTRAL"]:
                            pnl = (current_price - entry_price) / entry_price
                            exit_reason = "signal"
                        else:
                            continue  # Hold position

                        equity *= (1 + pnl)
                        duration_mins = (current_time - entry_time).total_seconds() / 60
                        trades.append({
                            "pnl": pnl,
                            "side": "long",
                            "entry_price": entry_price,
                            "exit_price": current_price if exit_reason == "signal" else (stop_loss if exit_reason == "stop_loss" else take_profit),
                            "exit_reason": exit_reason,
                            "duration_mins": duration_mins,
                        })
                        position = None

                    elif position["side"] == "short":
                        # Check stop loss
                        if df["high"].iloc[i] >= stop_loss:
                            pnl = (entry_price - stop_loss) / entry_price
                            exit_reason = "stop_loss"
                        # Check take profit
                        elif df["low"].iloc[i] <= take_profit:
                            pnl = (entry_price - take_profit) / entry_price
                            exit_reason = "take_profit"
                        # Check signal exit
                        elif signal in ["BUY", "NEUTRAL"]:
                            pnl = (entry_price - current_price) / entry_price
                            exit_reason = "signal"
                        else:
                            continue  # Hold position

                        equity *= (1 + pnl)
                        duration_mins = (current_time - entry_time).total_seconds() / 60
                        trades.append({
                            "pnl": pnl,
                            "side": "short",
                            "entry_price": entry_price,
                            "exit_price": current_price if exit_reason == "signal" else (stop_loss if exit_reason == "stop_loss" else take_profit),
                            "exit_reason": exit_reason,
                            "duration_mins": duration_mins,
                        })
                        position = None

                # Entry logic
                if position is None and signal in ["BUY", "SELL"]:
                    entry_price = current_price
                    entry_time = current_time
                    stop_loss = result.get("stop_loss", current_price * 0.98)
                    take_profit = result.get("take_profit", current_price * 1.03)

                    position = {
                        "side": "long" if signal == "BUY" else "short",
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                    }

                equity_curve.append(equity)

            except Exception as e:
                logger.debug(f"Backtest error at index {i}: {e}")
                continue

        return self._calculate_metrics(trades, equity_curve, df)

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics structure."""
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "total_trades": 0,
            "trades_per_day": 0.0,
            "avg_trade_duration": 0.0,
        }

    def _calculate_metrics(
        self,
        trades: List[Dict],
        equity_curve: List[float],
        df: pd.DataFrame,
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

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0

        # Total return
        final_equity = equity_curve[-1] if equity_curve else self.config.initial_capital
        total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital

        # Sharpe ratio
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1]) if len(equity_curve) > 1 else [0]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)  # Hourly annualized
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # Trades per day
        if len(df) > 0 and df.index[-1] > df.index[0]:
            days = (df.index[-1] - df.index[0]).total_seconds() / 86400
            trades_per_day = total_trades / max(days, 1)
        else:
            trades_per_day = 0.0

        # Average trade duration
        durations = [t.get("duration_mins", 0) for t in trades]
        avg_trade_duration = np.mean(durations) if durations else 0.0

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "total_trades": total_trades,
            "trades_per_day": float(trades_per_day),
            "avg_trade_duration": float(avg_trade_duration),
        }

    def _validate_result(
        self,
        metrics: Dict[str, float],
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if results meet selection criteria.

        Criteria:
        - Frequency: 2-5 trades per day (reject if < 2 or > 8)
        - Profitability: Profit Factor > 1.3
        - Win Rate: > 45%

        Args:
            metrics: Backtest metrics dict

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        trades_per_day = metrics.get("trades_per_day", 0)
        profit_factor = metrics.get("profit_factor", 0)
        win_rate = metrics.get("win_rate", 0)
        total_trades = metrics.get("total_trades", 0)

        # Minimum 60 trades (approx 2/day for 30 days)
        if total_trades < 60:
            return False, f"Insufficient trades ({total_trades} < 60)"

        # Check trade frequency
        if trades_per_day < self.config.min_trades_per_day:
            return False, f"Too few trades/day ({trades_per_day:.1f} < {self.config.min_trades_per_day})"

        if trades_per_day > self.config.max_trades_per_day:
            return False, f"Too many trades/day ({trades_per_day:.1f} > {self.config.max_trades_per_day})"

        # Check profitability
        if profit_factor < self.config.min_profit_factor:
            return False, f"Profit factor too low ({profit_factor:.2f} < {self.config.min_profit_factor})"

        # Check win rate
        if win_rate < self.config.min_win_rate:
            return False, f"Win rate too low ({win_rate:.1%} < {self.config.min_win_rate:.0%})"

        return True, None

    def optimize_asset(
        self,
        symbol: str,
        source: str,
        progress_callback=None,
    ) -> List[OptimizationResult]:
        """
        Run optimization for a single asset across all timeframes.

        Args:
            symbol: Asset symbol
            source: Data source
            progress_callback: Optional callback for progress updates

        Returns:
            List of OptimizationResult for each timeframe
        """
        results = []

        # Fetch high-granularity data
        df_5m = self._fetch_high_granularity_data(
            symbol, source, days=self.config.backtest_days
        )

        if df_5m is None:
            logger.warning(f"Skipping {symbol}: No data available")
            return results

        # Test each timeframe
        for timeframe in self.config.timeframes:
            if progress_callback:
                progress_callback(f"Testing {symbol} on {timeframe}...")

            try:
                # Resample data to target timeframe
                df_tf = self._resample_data(df_5m, timeframe)

                if len(df_tf) < 100:
                    logger.warning(f"Insufficient data for {symbol} {timeframe}")
                    continue

                # Run backtest
                metrics = self._run_backtest(df_tf, symbol, timeframe)

                # Validate result
                is_valid, rejection_reason = self._validate_result(metrics)

                # Create result object
                result = OptimizationResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy="ichimoku_fibonacci",
                    total_trades=metrics["total_trades"],
                    trades_per_day=metrics["trades_per_day"],
                    win_rate=metrics["win_rate"],
                    profit_factor=metrics["profit_factor"],
                    sharpe_ratio=metrics["sharpe_ratio"],
                    max_drawdown=metrics["max_drawdown"],
                    total_return=metrics["total_return"],
                    avg_trade_duration=metrics["avg_trade_duration"],
                    is_valid=is_valid,
                    rejection_reason=rejection_reason,
                    parameters=self._get_strategy().get_parameters(),
                    calculated_at=datetime.utcnow(),
                )

                results.append(result)

                logger.info(
                    f"{symbol} {timeframe}: "
                    f"Trades/day={metrics['trades_per_day']:.1f}, "
                    f"WinRate={metrics['win_rate']:.1%}, "
                    f"PF={metrics['profit_factor']:.2f}, "
                    f"Valid={is_valid}"
                )

            except Exception as e:
                logger.error(f"Error optimizing {symbol} {timeframe}: {e}")
                continue

        return results

    def run_mass_optimization(
        self,
        progress_callback=None,
        parallel: bool = False,
    ) -> Dict[str, OptimizationResult]:
        """
        Run optimization across ALL assets and timeframes.

        This is "The Gauntlet" - the main optimization loop.

        Args:
            progress_callback: Optional callback(message) for progress updates
            parallel: If True, run assets in parallel (use with caution)

        Returns:
            Dict mapping symbol -> best OptimizationResult
        """
        logger.info("=" * 60)
        logger.info("THE GAUNTLET - Starting Mass Optimization")
        logger.info("=" * 60)

        assets = self._load_assets_from_db()

        if not assets:
            logger.error("No assets to optimize")
            return {}

        all_results: Dict[str, List[OptimizationResult]] = {}
        winners: Dict[str, OptimizationResult] = {}

        # Process each asset
        for idx, asset in enumerate(assets, 1):
            symbol = asset["symbol"]
            source = asset["source"]

            if progress_callback:
                progress_callback(f"[{idx}/{len(assets)}] Optimizing {symbol}...")

            logger.info(f"\n--- Optimizing {symbol} ({source}) [{idx}/{len(assets)}] ---")

            # Run optimization for this asset
            results = self.optimize_asset(symbol, source, progress_callback)
            all_results[symbol] = results

            # Find the best valid result for this asset
            valid_results = [r for r in results if r.is_valid]

            if valid_results:
                # Sort by composite score: (profit_factor * win_rate) / trades_per_day variance
                # We want high PF, high WR, and trades/day close to 3 (middle of 2-4 range)
                def score(r: OptimizationResult) -> float:
                    pf_score = min(r.profit_factor, 3.0) / 3.0
                    wr_score = r.win_rate
                    # Penalize deviation from target 3 trades/day
                    freq_score = 1.0 - abs(r.trades_per_day - 3.0) / 5.0
                    return pf_score * 0.4 + wr_score * 0.4 + freq_score * 0.2

                best = max(valid_results, key=score)
                winners[symbol] = best

                logger.info(
                    f"  WINNER: {best.timeframe} | "
                    f"Trades/day={best.trades_per_day:.1f}, "
                    f"WR={best.win_rate:.1%}, "
                    f"PF={best.profit_factor:.2f}"
                )
            else:
                logger.warning(f"  No valid configuration found for {symbol}")

        logger.info("\n" + "=" * 60)
        logger.info(f"GAUNTLET COMPLETE: {len(winners)}/{len(assets)} assets optimized")
        logger.info("=" * 60)

        return winners

    def save_winners_to_db(
        self,
        winners: Dict[str, OptimizationResult],
    ) -> int:
        """
        Save optimization winners to the database.

        Updates the assets table with:
        - best_timeframe
        - best_strategy
        - opt_params (JSON)

        Also stores detailed results in gauntlet_results table.

        Args:
            winners: Dict mapping symbol -> OptimizationResult

        Returns:
            Number of records saved
        """
        if not winners:
            return 0

        try:
            conn = get_connection()
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()
            saved = 0

            for symbol, result in winners.items():
                # Prepare optimization parameters JSON
                opt_params = json.dumps({
                    "timeframe": result.timeframe,
                    "strategy": result.strategy,
                    "trades_per_day": result.trades_per_day,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "total_return": result.total_return,
                    "avg_trade_duration": result.avg_trade_duration,
                    "parameters": result.parameters,
                    "calculated_at": result.calculated_at.isoformat(),
                })

                # Update assets table
                cursor.execute(
                    """
                    UPDATE assets SET
                        best_timeframe = ?,
                        best_strategy = ?,
                        opt_params = ?,
                        optimal_strategy = ?,
                        last_updated = ?
                    WHERE symbol = ?
                    """,
                    (
                        result.timeframe,
                        result.strategy,
                        opt_params,
                        result.strategy,
                        now,
                        symbol,
                    ),
                )

                # Insert into gauntlet_results table
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO gauntlet_results
                    (symbol, timeframe, strategy, trades_per_day, win_rate,
                     profit_factor, sharpe_ratio, max_drawdown, total_return,
                     avg_trade_duration, is_valid, parameters, calculated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        result.timeframe,
                        result.strategy,
                        result.trades_per_day,
                        result.win_rate,
                        result.profit_factor,
                        result.sharpe_ratio,
                        result.max_drawdown,
                        result.total_return,
                        result.avg_trade_duration,
                        1 if result.is_valid else 0,
                        json.dumps(result.parameters),
                        result.calculated_at.isoformat(),
                    ),
                )

                saved += 1

            conn.commit()
            conn.close()

            logger.info(f"Saved {saved} optimization results to database")
            return saved

        except Exception as e:
            logger.error(f"Error saving results to database: {e}")
            return 0

    def get_leaderboard(self) -> pd.DataFrame:
        """
        Get optimization leaderboard from database.

        Returns:
            DataFrame with columns: Symbol, Best TF, Win Rate, Trades/Day,
            Profit Factor, Status
        """
        try:
            conn = get_connection()
            df = pd.read_sql_query(
                """
                SELECT
                    symbol as Symbol,
                    timeframe as "Best TF",
                    win_rate as "Win Rate",
                    trades_per_day as "Trades/Day",
                    profit_factor as "Profit Factor",
                    sharpe_ratio as "Sharpe",
                    CASE WHEN is_valid = 1 THEN 'PASS' ELSE 'FAIL' END as Status
                FROM gauntlet_results
                ORDER BY profit_factor DESC, win_rate DESC
                """,
                conn,
            )
            conn.close()
            return df

        except Exception as e:
            logger.error(f"Error loading leaderboard: {e}")
            return pd.DataFrame()


# =============================================================================
# CLI INTERFACE
# =============================================================================


def run_gauntlet_cli():
    """Run The Gauntlet from command line."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("\n" + "=" * 60)
    print("  THE GAUNTLET - SOLAT Multi-Timeframe Optimizer")
    print("=" * 60 + "\n")

    # Initialize and run
    gauntlet = GauntletOptimizer()

    def progress(msg):
        print(f"  {msg}")

    winners = gauntlet.run_mass_optimization(progress_callback=progress)

    # Save results
    if winners:
        saved = gauntlet.save_winners_to_db(winners)
        print(f"\n Saved {saved} optimal configurations to database")

        # Print summary
        print("\n" + "-" * 60)
        print("OPTIMIZATION LEADERBOARD")
        print("-" * 60)
        for symbol, result in sorted(winners.items(), key=lambda x: x[1].profit_factor, reverse=True):
            print(
                f"  {symbol:20s} | {result.timeframe:4s} | "
                f"WR={result.win_rate:.1%} | "
                f"PF={result.profit_factor:.2f} | "
                f"Trades/Day={result.trades_per_day:.1f}"
            )
    else:
        print("\n No valid configurations found. Check your criteria.")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    run_gauntlet_cli()
