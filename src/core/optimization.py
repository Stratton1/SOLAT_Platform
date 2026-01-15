"""
Hyperparameter Optimization Engine for SOLAT Platform.

This module uses Optuna to find optimal Ichimoku parameters that maximize
Win Rate for each specific asset. Key features:

1. Dynamic parameter search (Tenkan: 5-50, Kijun: 15-100, Senkou: 30-200)
2. Walk-Forward Analysis for robustness (Train on first 60%, Test on last 40%)
3. Minimum trade threshold (30 trades) to ensure statistical significance
4. Quality gate (60%+ Win Rate) to reject low-quality configurations
5. Golden Settings saved to JSON for production use

Target: 65%+ Win Rate with robustness validation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from ta.trend import IchimokuIndicator

logger = logging.getLogger(__name__)

# Path for golden settings
GOLDEN_SETTINGS_PATH = Path("data/db/golden_settings.json")


class DynamicIchimokuStrategy:
    """
    Ichimoku strategy with configurable parameters.

    Used by HyperoptEngine to test different parameter combinations.
    """

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        stop_loss_multiplier: float = 2.0,
    ) -> None:
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.stop_loss_multiplier = stop_loss_multiplier
        self.displacement = 26  # Keep standard displacement

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku indicators on DataFrame."""
        min_periods = self.senkou_b_period + self.displacement
        if len(df) < min_periods:
            raise ValueError(f"Need at least {min_periods} rows, got {len(df)}")

        ichimoku = IchimokuIndicator(
            high=df["high"],
            low=df["low"],
            window1=self.tenkan_period,
            window2=self.kijun_period,
            window3=self.senkou_b_period,
            visual=True,
            fillna=False,
        )

        result = df.copy()
        result["tenkan"] = ichimoku.ichimoku_conversion_line()
        result["kijun"] = ichimoku.ichimoku_base_line()
        result["senkou_a"] = ichimoku.ichimoku_a()
        result["senkou_b"] = ichimoku.ichimoku_b()

        # Calculate ATR for stop loss
        result["atr"] = self._calculate_atr(df, period=14)

        return result

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on Ichimoku rules.

        Returns DataFrame with 'signal' column: 1 (buy), -1 (sell), 0 (hold)
        """
        df_ind = self.calculate_indicators(df)

        signals = pd.DataFrame(index=df_ind.index)
        signals["signal"] = 0
        signals["entry_price"] = 0.0
        signals["stop_loss"] = 0.0

        for i in range(1, len(df_ind)):
            close = df_ind["close"].iloc[i]
            senkou_a = df_ind["senkou_a"].iloc[i]
            senkou_b = df_ind["senkou_b"].iloc[i]
            tenkan = df_ind["tenkan"].iloc[i]
            kijun = df_ind["kijun"].iloc[i]
            atr = df_ind["atr"].iloc[i]

            prev_tenkan = df_ind["tenkan"].iloc[i - 1]
            prev_kijun = df_ind["kijun"].iloc[i - 1]

            if pd.isna(senkou_a) or pd.isna(senkou_b) or pd.isna(tenkan) or pd.isna(kijun):
                continue

            cloud_top = max(senkou_a, senkou_b)
            cloud_bottom = min(senkou_a, senkou_b)

            # TK Cross detection
            tk_cross_up = prev_tenkan <= prev_kijun and tenkan > kijun
            tk_cross_down = prev_tenkan >= prev_kijun and tenkan < kijun

            # BUY Signal: Price above cloud + TK cross up
            if close > cloud_top and tk_cross_up:
                signals.iloc[i, signals.columns.get_loc("signal")] = 1
                signals.iloc[i, signals.columns.get_loc("entry_price")] = close
                if not pd.isna(atr):
                    signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - (atr * self.stop_loss_multiplier)

            # SELL Signal: Price below cloud + TK cross down
            elif close < cloud_bottom and tk_cross_down:
                signals.iloc[i, signals.columns.get_loc("signal")] = -1
                signals.iloc[i, signals.columns.get_loc("entry_price")] = close
                if not pd.isna(atr):
                    signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + (atr * self.stop_loss_multiplier)

        return signals


class HyperoptEngine:
    """
    Hyperparameter Optimization Engine using Optuna.

    Searches for optimal Ichimoku parameters that maximize Win Rate
    while ensuring robustness through Walk-Forward Analysis.

    Target: 65%+ Win Rate with validation on holdout data.

    Usage:
        engine = HyperoptEngine()
        settings = engine.find_golden_settings("BTC/USDT", df)
        # Returns: {"tenkan": 14, "kijun": 42, "senkou": 88, "win_rate": 0.68}
    """

    # Parameter search ranges
    TENKAN_RANGE = (5, 50)
    KIJUN_RANGE = (15, 100)
    SENKOU_RANGE = (30, 200)
    STOP_LOSS_RANGE = (1.0, 3.0)

    # Quality thresholds
    MIN_TRADES = 30          # Minimum trades for statistical significance
    MIN_WIN_RATE = 0.60      # Reject configs below 60% win rate
    TARGET_WIN_RATE = 0.65   # Target 65% win rate
    HOLDOUT_MIN_WR = 0.60    # Holdout validation threshold

    def __init__(
        self,
        n_trials: int = 100,
        train_ratio: float = 0.60,  # Walk-Forward: 60% train, 40% test
    ) -> None:
        """
        Initialize the HyperoptEngine.

        Args:
            n_trials: Number of Optuna trials to run
            train_ratio: Ratio of data for training (rest is holdout)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna not installed. Install with: pip install optuna>=3.0.0"
            )

        self.n_trials = n_trials
        self.train_ratio = train_ratio

        # Will be set per optimization run
        self._current_df: Optional[pd.DataFrame] = None
        self._train_df: Optional[pd.DataFrame] = None
        self._holdout_df: Optional[pd.DataFrame] = None

        logger.info(
            f"HyperoptEngine initialized: {n_trials} trials, "
            f"{train_ratio:.0%} train / {1-train_ratio:.0%} holdout"
        )

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and holdout sets (Walk-Forward Analysis).

        Args:
            df: Full OHLCV DataFrame

        Returns:
            Tuple[train_df, holdout_df]
        """
        split_idx = int(len(df) * self.train_ratio)
        train_df = df.iloc[:split_idx].copy()
        holdout_df = df.iloc[split_idx:].copy()

        logger.info(
            f"Data split: Train={len(train_df)} rows ({train_df.index[0]} to {train_df.index[-1]}), "
            f"Holdout={len(holdout_df)} rows ({holdout_df.index[0]} to {holdout_df.index[-1]})"
        )

        return train_df, holdout_df

    def _backtest_params(
        self,
        df: pd.DataFrame,
        tenkan: int,
        kijun: int,
        senkou: int,
        stop_loss_mult: float,
    ) -> Dict[str, float]:
        """
        Run backtest with specific parameters.

        Returns:
            Dict with win_rate, total_trades, profit_factor, etc.
        """
        try:
            strategy = DynamicIchimokuStrategy(
                tenkan_period=tenkan,
                kijun_period=kijun,
                senkou_b_period=senkou,
                stop_loss_multiplier=stop_loss_mult,
            )

            signals = strategy.generate_signals(df)

            # Simulate trades
            trades: List[Dict] = []
            position = None
            entry_price = 0.0
            stop_loss = 0.0

            for i in range(len(signals)):
                signal = signals["signal"].iloc[i]
                current_price = df["close"].iloc[i]

                # Check stop loss hit
                if position == "long" and stop_loss > 0 and current_price <= stop_loss:
                    pnl = (stop_loss - entry_price) / entry_price
                    trades.append({"pnl": pnl, "exit_reason": "stop_loss"})
                    position = None
                elif position == "short" and stop_loss > 0 and current_price >= stop_loss:
                    pnl = (entry_price - stop_loss) / entry_price
                    trades.append({"pnl": pnl, "exit_reason": "stop_loss"})
                    position = None

                # Exit on opposite signal
                if position == "long" and signal == -1:
                    pnl = (current_price - entry_price) / entry_price
                    trades.append({"pnl": pnl, "exit_reason": "signal"})
                    position = None
                elif position == "short" and signal == 1:
                    pnl = (entry_price - current_price) / entry_price
                    trades.append({"pnl": pnl, "exit_reason": "signal"})
                    position = None

                # Enter on signal
                if position is None and signal != 0:
                    if signal == 1:
                        position = "long"
                        entry_price = current_price
                        stop_loss = signals["stop_loss"].iloc[i]
                    elif signal == -1:
                        position = "short"
                        entry_price = current_price
                        stop_loss = signals["stop_loss"].iloc[i]

            # Calculate metrics
            if not trades:
                return {
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "profit_factor": 0.0,
                    "avg_win": 0.0,
                    "avg_loss": 0.0,
                }

            pnls = [t["pnl"] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            win_rate = len(wins) / len(trades) if trades else 0.0

            gross_profit = sum(wins) if wins else 0.0
            gross_loss = abs(sum(losses)) if losses else 0.001
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0

            return {
                "win_rate": win_rate,
                "total_trades": len(trades),
                "profit_factor": profit_factor,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
            }

        except Exception as e:
            logger.debug(f"Backtest error: {e}")
            return {
                "win_rate": 0.0,
                "total_trades": 0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

    def objective(self, trial: "optuna.Trial") -> float:
        """
        Optuna objective function.

        Searches for parameters that maximize Win Rate while meeting
        quality thresholds.

        Args:
            trial: Optuna trial object

        Returns:
            float: Win Rate score (0.0 if rejected)
        """
        # Sample parameters
        tenkan = trial.suggest_int("tenkan", *self.TENKAN_RANGE)
        kijun = trial.suggest_int("kijun", *self.KIJUN_RANGE)
        senkou = trial.suggest_int("senkou", *self.SENKOU_RANGE)
        stop_loss_mult = trial.suggest_float("stop_loss_mult", *self.STOP_LOSS_RANGE)

        # Constraint: kijun must be > tenkan
        if kijun <= tenkan:
            return 0.0

        # Constraint: senkou must be > kijun
        if senkou <= kijun:
            return 0.0

        # Run backtest on training data
        metrics = self._backtest_params(
            self._train_df, tenkan, kijun, senkou, stop_loss_mult
        )

        # Quality gate 1: Minimum trades
        if metrics["total_trades"] < self.MIN_TRADES:
            return 0.0

        # Quality gate 2: Minimum win rate
        if metrics["win_rate"] < self.MIN_WIN_RATE:
            return 0.0

        # Return win rate as score
        return metrics["win_rate"]

    def find_golden_settings(
        self,
        symbol: str,
        df: pd.DataFrame,
        save: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Find optimal Ichimoku parameters for a specific asset.

        Process:
        1. Split data into train (60%) and holdout (40%)
        2. Run Optuna optimization on training data
        3. Validate best params on holdout data (Walk-Forward)
        4. If validation passes, save as "Golden Settings"

        Args:
            symbol: Asset symbol (e.g., "BTC/USDT")
            df: OHLCV DataFrame with at least 200 rows
            save: Whether to save results to golden_settings.json

        Returns:
            Dict with optimal settings or None if no good config found
        """
        if len(df) < 200:
            logger.warning(f"Insufficient data for optimization: {len(df)} rows")
            return None

        logger.info(f"\n{'='*60}")
        logger.info(f"HYPEROPT: Finding Golden Settings for {symbol}")
        logger.info(f"{'='*60}")
        logger.info(f"Data range: {df.index[0]} to {df.index[-1]} ({len(df)} rows)")

        # Split data for Walk-Forward Analysis
        self._train_df, self._holdout_df = self._split_data(df)
        self._current_df = df

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
        )

        # Suppress Optuna logs during optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info(f"\nRunning {self.n_trials} optimization trials...")
        logger.info(f"Target: {self.TARGET_WIN_RATE:.0%} Win Rate")

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        # Check if we found anything good
        if study.best_value < self.MIN_WIN_RATE:
            logger.warning(
                f"No configuration achieved {self.MIN_WIN_RATE:.0%} win rate. "
                f"Best: {study.best_value:.1%}"
            )
            return None

        best_params = study.best_params
        logger.info(f"\n--- Best Training Result ---")
        logger.info(f"Win Rate: {study.best_value:.1%}")
        logger.info(f"Parameters: Tenkan={best_params['tenkan']}, "
                   f"Kijun={best_params['kijun']}, Senkou={best_params['senkou']}, "
                   f"SL Mult={best_params['stop_loss_mult']:.2f}")

        # Walk-Forward Validation on Holdout
        logger.info(f"\n--- Walk-Forward Validation (Holdout) ---")
        holdout_metrics = self._backtest_params(
            self._holdout_df,
            best_params["tenkan"],
            best_params["kijun"],
            best_params["senkou"],
            best_params["stop_loss_mult"],
        )

        logger.info(f"Holdout Win Rate: {holdout_metrics['win_rate']:.1%}")
        logger.info(f"Holdout Trades: {holdout_metrics['total_trades']}")
        logger.info(f"Holdout Profit Factor: {holdout_metrics['profit_factor']:.2f}")

        # Robustness check
        if holdout_metrics["win_rate"] < self.HOLDOUT_MIN_WR:
            logger.warning(
                f"Holdout validation FAILED: {holdout_metrics['win_rate']:.1%} < "
                f"{self.HOLDOUT_MIN_WR:.0%}. Settings may be overfit."
            )
            return None

        if holdout_metrics["total_trades"] < 10:
            logger.warning(
                f"Insufficient holdout trades: {holdout_metrics['total_trades']}. "
                "Need more data."
            )
            return None

        # Success! Create golden settings
        golden_settings = {
            "tenkan": best_params["tenkan"],
            "kijun": best_params["kijun"],
            "senkou": best_params["senkou"],
            "stop_loss_mult": round(best_params["stop_loss_mult"], 2),
            "train_win_rate": round(study.best_value, 4),
            "holdout_win_rate": round(holdout_metrics["win_rate"], 4),
            "holdout_trades": holdout_metrics["total_trades"],
            "profit_factor": round(holdout_metrics["profit_factor"], 2),
            "optimized_at": datetime.utcnow().isoformat(),
            "data_start": str(df.index[0]),
            "data_end": str(df.index[-1]),
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"GOLDEN SETTINGS FOUND for {symbol}")
        logger.info(f"{'='*60}")
        logger.info(f"Tenkan: {golden_settings['tenkan']} (standard: 9)")
        logger.info(f"Kijun: {golden_settings['kijun']} (standard: 26)")
        logger.info(f"Senkou: {golden_settings['senkou']} (standard: 52)")
        logger.info(f"Stop Loss: {golden_settings['stop_loss_mult']}x ATR")
        logger.info(f"Training Win Rate: {golden_settings['train_win_rate']:.1%}")
        logger.info(f"Holdout Win Rate: {golden_settings['holdout_win_rate']:.1%}")
        logger.info(f"{'='*60}\n")

        # Save to file
        if save:
            self._save_golden_settings(symbol, golden_settings)

        return golden_settings

    def _save_golden_settings(
        self, symbol: str, settings: Dict[str, Any]
    ) -> None:
        """Save golden settings to JSON file."""
        try:
            # Ensure directory exists
            GOLDEN_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

            # Load existing settings
            if GOLDEN_SETTINGS_PATH.exists():
                with open(GOLDEN_SETTINGS_PATH, "r") as f:
                    all_settings = json.load(f)
            else:
                all_settings = {}

            # Update with new settings
            all_settings[symbol] = settings

            # Save
            with open(GOLDEN_SETTINGS_PATH, "w") as f:
                json.dump(all_settings, f, indent=2, default=str)

            logger.info(f"Saved golden settings for {symbol} to {GOLDEN_SETTINGS_PATH}")

        except Exception as e:
            logger.error(f"Error saving golden settings: {e}")

    def optimize_multiple(
        self,
        symbols_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Optimize multiple symbols.

        Args:
            symbols_data: Dict mapping symbol to OHLCV DataFrame

        Returns:
            Dict mapping symbol to golden settings (or None if failed)
        """
        results = {}

        for symbol, df in symbols_data.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Optimizing {symbol}...")
            logger.info(f"{'='*60}")

            try:
                settings = self.find_golden_settings(symbol, df, save=True)
                results[symbol] = settings
            except Exception as e:
                logger.error(f"Optimization failed for {symbol}: {e}")
                results[symbol] = None

        return results


# =============================================================================
# GOLDEN SETTINGS MANAGEMENT
# =============================================================================


def load_golden_settings() -> Dict[str, Dict[str, Any]]:
    """
    Load all golden settings from JSON file.

    Returns:
        Dict mapping symbol to settings
    """
    if not GOLDEN_SETTINGS_PATH.exists():
        return {}

    try:
        with open(GOLDEN_SETTINGS_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading golden settings: {e}")
        return {}


def get_golden_settings(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get golden settings for a specific symbol.

    Args:
        symbol: Asset symbol

    Returns:
        Settings dict or None if not found
    """
    all_settings = load_golden_settings()
    return all_settings.get(symbol)


def has_golden_settings(symbol: str) -> bool:
    """Check if golden settings exist for a symbol."""
    return get_golden_settings(symbol) is not None


def get_standard_vs_golden_comparison(symbol: str) -> Dict[str, Any]:
    """
    Compare standard Ichimoku settings vs golden settings.

    Returns:
        Dict with comparison metrics
    """
    golden = get_golden_settings(symbol)

    if golden is None:
        return {
            "has_golden": False,
            "standard_wr": "~50%",  # Typical for standard settings
            "golden_wr": None,
            "improvement": None,
        }

    return {
        "has_golden": True,
        "standard_wr": "~50%",  # Baseline
        "golden_wr": f"{golden['holdout_win_rate']:.0%}",
        "improvement": f"+{(golden['holdout_win_rate'] - 0.5) * 100:.0f}%",
        "settings": {
            "tenkan": golden["tenkan"],
            "kijun": golden["kijun"],
            "senkou": golden["senkou"],
        }
    }
