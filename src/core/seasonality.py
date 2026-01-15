"""
Seasonality pattern detection module for SOLAT.

PatternHunter analyzes historical price data to identify time-based patterns:
- Time-of-day effects (hourly performance)
- Day-of-week effects (weekday performance)
- Japan Open breakout detection (Tokyo session 09:00 JST)

These patterns can inform position sizing and timing decisions.
"""

import logging
import sqlite3
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from src.database.repository import DB_PATH

logger = logging.getLogger(__name__)

# Tokyo timezone for Japan Open analysis
TOKYO_TZ = pytz.timezone("Asia/Tokyo")

# Statistical significance threshold (minimum sample size)
MIN_SAMPLE_SIZE = 30

# Significance threshold for win rate deviation from 50%
SIGNIFICANCE_THRESHOLD = 0.10  # 10% deviation from baseline


class PatternHunter:
    """
    Analyzes historical price data to identify actionable seasonality patterns.

    Patterns detected:
    - Time-of-day: Which hours have historically positive/negative returns
    - Day-of-week: Which weekdays perform better
    - Japan Open breakout: Tokyo session (09:00 JST) breakout success rate
    """

    def __init__(self) -> None:
        """Initialize PatternHunter."""
        self._pattern_cache: Dict[str, Dict[str, Any]] = {}

    def analyze_time_of_day(
        self, df: pd.DataFrame, price_col: str = "close"
    ) -> Dict[int, Dict[str, Any]]:
        """
        Analyze returns by hour of day.

        Groups price returns by the hour they occurred and calculates
        statistics for each hour (0-23).

        Args:
            df: DataFrame with DatetimeIndex and price column
            price_col: Column name for price data

        Returns:
            Dictionary mapping hour (0-23) to stats:
            {
                0: {"avg_return": 0.0012, "win_rate": 0.54, "sample_size": 150, ...},
                1: {...},
                ...
            }
        """
        if df.empty or price_col not in df.columns:
            logger.warning("Empty DataFrame or missing price column for time-of-day analysis")
            return {}

        # Calculate returns
        returns = df[price_col].pct_change().dropna()

        if returns.empty:
            return {}

        # Ensure we have timezone-aware index
        if returns.index.tz is None:
            returns.index = returns.index.tz_localize("UTC")

        # Group by hour
        hourly_stats: Dict[int, Dict[str, Any]] = {}

        for hour in range(24):
            hour_mask = returns.index.hour == hour
            hour_returns = returns[hour_mask]

            if len(hour_returns) < MIN_SAMPLE_SIZE:
                continue

            wins = (hour_returns > 0).sum()
            total = len(hour_returns)

            hourly_stats[hour] = {
                "avg_return": float(hour_returns.mean()),
                "std_return": float(hour_returns.std()),
                "win_rate": float(wins / total) if total > 0 else 0.0,
                "sample_size": total,
                "total_return": float(hour_returns.sum()),
                "max_return": float(hour_returns.max()),
                "min_return": float(hour_returns.min()),
                "is_significant": abs(wins / total - 0.5) > SIGNIFICANCE_THRESHOLD if total > 0 else False,
            }

        logger.debug(f"Time-of-day analysis complete: {len(hourly_stats)} hours with sufficient data")
        return hourly_stats

    def analyze_day_of_week(
        self, df: pd.DataFrame, price_col: str = "close"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze returns by day of week.

        Groups price returns by weekday and calculates statistics.

        Args:
            df: DataFrame with DatetimeIndex and price column
            price_col: Column name for price data

        Returns:
            Dictionary mapping weekday name to stats:
            {
                "Monday": {"avg_return": 0.0015, "win_rate": 0.52, ...},
                "Tuesday": {...},
                ...
            }
        """
        if df.empty or price_col not in df.columns:
            logger.warning("Empty DataFrame or missing price column for day-of-week analysis")
            return {}

        # Calculate daily returns (resample to daily if needed)
        if len(df) > 0:
            # Use daily close-to-close returns
            daily_df = df[price_col].resample("D").last().dropna()
            returns = daily_df.pct_change().dropna()
        else:
            return {}

        if returns.empty:
            return {}

        # Map day numbers to names
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        daily_stats: Dict[str, Dict[str, Any]] = {}

        for day_num, day_name in enumerate(day_names):
            day_mask = returns.index.dayofweek == day_num
            day_returns = returns[day_mask]

            if len(day_returns) < MIN_SAMPLE_SIZE:
                continue

            wins = (day_returns > 0).sum()
            total = len(day_returns)

            daily_stats[day_name] = {
                "avg_return": float(day_returns.mean()),
                "std_return": float(day_returns.std()),
                "win_rate": float(wins / total) if total > 0 else 0.0,
                "sample_size": total,
                "total_return": float(day_returns.sum()),
                "max_return": float(day_returns.max()),
                "min_return": float(day_returns.min()),
                "is_significant": abs(wins / total - 0.5) > SIGNIFICANCE_THRESHOLD if total > 0 else False,
            }

        logger.debug(f"Day-of-week analysis complete: {len(daily_stats)} days with sufficient data")
        return daily_stats

    def japan_open_breakout(
        self, df: pd.DataFrame, breakout_window_hours: int = 1
    ) -> Dict[str, Any]:
        """
        Detect Tokyo session (09:00 JST) breakout patterns.

        Analyzes the first hour of the Tokyo session to identify:
        - High/low of the opening range
        - Breakout success rate (price breaking above high or below low)
        - Average move after breakout

        This is particularly useful for JPY pairs and Asian equity indices.

        Args:
            df: DataFrame with DatetimeIndex and OHLC columns
            breakout_window_hours: Hours to consider for the opening range

        Returns:
            Dictionary with breakout statistics:
            {
                "bullish_breakout_rate": 0.45,
                "bearish_breakout_rate": 0.35,
                "no_breakout_rate": 0.20,
                "avg_bullish_move": 0.0025,
                "avg_bearish_move": -0.0030,
                "sample_size": 200,
                ...
            }
        """
        required_cols = {"open", "high", "low", "close"}
        if df.empty or not required_cols.issubset(df.columns):
            logger.warning("Missing required OHLC columns for Japan Open analysis")
            return {}

        # Ensure timezone-aware index
        if df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC")

        # Convert to Tokyo timezone
        df_tokyo = df.copy()
        df_tokyo.index = df_tokyo.index.tz_convert(TOKYO_TZ)

        # Find Tokyo open (09:00 JST) candles
        tokyo_open_hour = 9

        # Group by date and find the opening range
        breakout_results: List[Dict[str, Any]] = []

        # Get unique dates
        dates = df_tokyo.index.date
        unique_dates = pd.Series(dates).unique()

        for date in unique_dates:
            # Get data for this date
            date_mask = df_tokyo.index.date == date
            day_data = df_tokyo[date_mask]

            if day_data.empty:
                continue

            # Find the opening range (first N hours from 09:00)
            opening_mask = (day_data.index.hour >= tokyo_open_hour) & \
                          (day_data.index.hour < tokyo_open_hour + breakout_window_hours)
            opening_range = day_data[opening_mask]

            if opening_range.empty:
                continue

            # Calculate opening range high/low
            range_high = opening_range["high"].max()
            range_low = opening_range["low"].min()

            # Get rest of day data (after opening range)
            rest_of_day = day_data[day_data.index.hour >= tokyo_open_hour + breakout_window_hours]

            if rest_of_day.empty:
                continue

            # Check for breakouts
            day_high = rest_of_day["high"].max()
            day_low = rest_of_day["low"].min()
            day_close = rest_of_day["close"].iloc[-1]

            bullish_breakout = day_high > range_high
            bearish_breakout = day_low < range_low

            # Calculate move size
            if bullish_breakout:
                move_size = (day_high - range_high) / range_high
            elif bearish_breakout:
                move_size = (day_low - range_low) / range_low
            else:
                move_size = 0.0

            breakout_results.append({
                "date": date,
                "range_high": range_high,
                "range_low": range_low,
                "bullish_breakout": bullish_breakout,
                "bearish_breakout": bearish_breakout,
                "move_size": move_size,
                "close": day_close,
            })

        if len(breakout_results) < MIN_SAMPLE_SIZE:
            logger.warning(f"Insufficient samples for Japan Open analysis: {len(breakout_results)}")
            return {"sample_size": len(breakout_results), "insufficient_data": True}

        # Aggregate statistics
        results_df = pd.DataFrame(breakout_results)
        total = len(results_df)

        bullish_count = results_df["bullish_breakout"].sum()
        bearish_count = results_df["bearish_breakout"].sum()
        no_breakout_count = total - bullish_count - bearish_count

        # Average moves
        bullish_moves = results_df[results_df["bullish_breakout"]]["move_size"]
        bearish_moves = results_df[results_df["bearish_breakout"]]["move_size"]

        return {
            "bullish_breakout_rate": float(bullish_count / total),
            "bearish_breakout_rate": float(bearish_count / total),
            "no_breakout_rate": float(no_breakout_count / total),
            "avg_bullish_move": float(bullish_moves.mean()) if len(bullish_moves) > 0 else 0.0,
            "avg_bearish_move": float(bearish_moves.mean()) if len(bearish_moves) > 0 else 0.0,
            "bullish_move_std": float(bullish_moves.std()) if len(bullish_moves) > 1 else 0.0,
            "bearish_move_std": float(bearish_moves.std()) if len(bearish_moves) > 1 else 0.0,
            "sample_size": total,
            "bullish_samples": int(bullish_count),
            "bearish_samples": int(bearish_count),
            "is_significant": (bullish_count / total > 0.4) or (bearish_count / total > 0.4),
        }

    def analyze_all_patterns(
        self, symbol: str, df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all pattern analyses for a symbol.

        Args:
            symbol: Asset symbol
            df: DataFrame with DatetimeIndex and OHLCV columns

        Returns:
            Dictionary with all pattern results:
            {
                "time_of_day": {...},
                "day_of_week": {...},
                "japan_open": {...},
                "analyzed_at": "2026-01-15T...",
            }
        """
        results = {
            "symbol": symbol,
            "time_of_day": self.analyze_time_of_day(df),
            "day_of_week": self.analyze_day_of_week(df),
            "japan_open": self.japan_open_breakout(df),
            "analyzed_at": datetime.utcnow().isoformat(),
        }

        # Cache results
        self._pattern_cache[symbol] = results

        logger.info(f"Completed all pattern analysis for {symbol}")
        return results

    def save_patterns_to_db(self, symbol: str, patterns: Dict[str, Any]) -> None:
        """
        Save pattern analysis results to the database.

        Stores patterns in the seasonality_patterns table with:
        - pattern_type: "time_of_day", "day_of_week", "japan_open"
        - period_value: hour (0-23), weekday name, or "breakout"
        - Statistical metrics

        Args:
            symbol: Asset symbol
            patterns: Pattern analysis results from analyze_all_patterns()
        """
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        try:
            # Enable WAL mode
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")

            now = datetime.utcnow().isoformat()

            # Save time-of-day patterns
            for hour, stats in patterns.get("time_of_day", {}).items():
                cursor.execute("""
                    INSERT OR REPLACE INTO seasonality_patterns
                    (symbol, pattern_type, period_value, avg_return, win_rate,
                     sample_size, is_significant, calculated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    "time_of_day",
                    str(hour),
                    stats.get("avg_return", 0.0),
                    stats.get("win_rate", 0.0),
                    stats.get("sample_size", 0),
                    1 if stats.get("is_significant", False) else 0,
                    now,
                ))

            # Save day-of-week patterns
            for day_name, stats in patterns.get("day_of_week", {}).items():
                cursor.execute("""
                    INSERT OR REPLACE INTO seasonality_patterns
                    (symbol, pattern_type, period_value, avg_return, win_rate,
                     sample_size, is_significant, calculated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    "day_of_week",
                    day_name,
                    stats.get("avg_return", 0.0),
                    stats.get("win_rate", 0.0),
                    stats.get("sample_size", 0),
                    1 if stats.get("is_significant", False) else 0,
                    now,
                ))

            # Save Japan Open breakout pattern
            japan_open = patterns.get("japan_open", {})
            if japan_open and not japan_open.get("insufficient_data", False):
                # Save bullish breakout stats
                cursor.execute("""
                    INSERT OR REPLACE INTO seasonality_patterns
                    (symbol, pattern_type, period_value, avg_return, win_rate,
                     sample_size, is_significant, calculated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    "japan_open",
                    "bullish_breakout",
                    japan_open.get("avg_bullish_move", 0.0),
                    japan_open.get("bullish_breakout_rate", 0.0),
                    japan_open.get("bullish_samples", 0),
                    1 if japan_open.get("is_significant", False) else 0,
                    now,
                ))

                # Save bearish breakout stats
                cursor.execute("""
                    INSERT OR REPLACE INTO seasonality_patterns
                    (symbol, pattern_type, period_value, avg_return, win_rate,
                     sample_size, is_significant, calculated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    "japan_open",
                    "bearish_breakout",
                    japan_open.get("avg_bearish_move", 0.0),
                    japan_open.get("bearish_breakout_rate", 0.0),
                    japan_open.get("bearish_samples", 0),
                    1 if japan_open.get("is_significant", False) else 0,
                    now,
                ))

            conn.commit()
            logger.info(f"Saved seasonality patterns to database for {symbol}")

        except sqlite3.Error as e:
            logger.error(f"Database error saving patterns for {symbol}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def load_patterns_from_db(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        Load previously calculated patterns from the database.

        Args:
            symbol: Asset symbol

        Returns:
            Dictionary with pattern data organized by type
        """
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        try:
            cursor.execute("PRAGMA journal_mode=WAL;")

            cursor.execute("""
                SELECT pattern_type, period_value, avg_return, win_rate,
                       sample_size, is_significant, calculated_at
                FROM seasonality_patterns
                WHERE symbol = ?
                ORDER BY pattern_type, period_value
            """, (symbol,))

            rows = cursor.fetchall()

            patterns: Dict[str, Dict[str, Any]] = {
                "time_of_day": {},
                "day_of_week": {},
                "japan_open": {},
            }

            for row in rows:
                pattern_type, period_value, avg_return, win_rate, sample_size, is_significant, calculated_at = row

                stat_dict = {
                    "avg_return": avg_return,
                    "win_rate": win_rate,
                    "sample_size": sample_size,
                    "is_significant": bool(is_significant),
                    "calculated_at": calculated_at,
                }

                if pattern_type == "time_of_day":
                    patterns["time_of_day"][int(period_value)] = stat_dict
                elif pattern_type == "day_of_week":
                    patterns["day_of_week"][period_value] = stat_dict
                elif pattern_type == "japan_open":
                    patterns["japan_open"][period_value] = stat_dict

            return patterns

        finally:
            conn.close()

    def get_best_trading_hours(
        self, symbol: str, min_win_rate: float = 0.55
    ) -> List[int]:
        """
        Get the best hours to trade based on historical win rate.

        Args:
            symbol: Asset symbol
            min_win_rate: Minimum win rate threshold

        Returns:
            List of hours (0-23) with win rate above threshold
        """
        patterns = self._pattern_cache.get(symbol)
        if not patterns:
            patterns = self.load_patterns_from_db(symbol)

        best_hours = []
        time_of_day = patterns.get("time_of_day", {})

        for hour, stats in time_of_day.items():
            if stats.get("win_rate", 0) >= min_win_rate and stats.get("is_significant", False):
                best_hours.append(hour)

        return sorted(best_hours)

    def get_best_trading_days(
        self, symbol: str, min_win_rate: float = 0.55
    ) -> List[str]:
        """
        Get the best days to trade based on historical win rate.

        Args:
            symbol: Asset symbol
            min_win_rate: Minimum win rate threshold

        Returns:
            List of weekday names with win rate above threshold
        """
        patterns = self._pattern_cache.get(symbol)
        if not patterns:
            patterns = self.load_patterns_from_db(symbol)

        best_days = []
        day_of_week = patterns.get("day_of_week", {})

        for day_name, stats in day_of_week.items():
            if stats.get("win_rate", 0) >= min_win_rate and stats.get("is_significant", False):
                best_days.append(day_name)

        return best_days

    def should_trade_now(
        self, symbol: str, current_time: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Check if current time is favorable for trading based on patterns.

        Args:
            symbol: Asset symbol
            current_time: Time to check (defaults to now UTC)

        Returns:
            Tuple of (should_trade, reason)
        """
        if current_time is None:
            current_time = datetime.utcnow()

        if current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)

        patterns = self._pattern_cache.get(symbol)
        if not patterns:
            patterns = self.load_patterns_from_db(symbol)

        current_hour = current_time.hour
        current_day = current_time.strftime("%A")

        # Check time-of-day pattern
        time_stats = patterns.get("time_of_day", {}).get(current_hour, {})
        day_stats = patterns.get("day_of_week", {}).get(current_day, {})

        reasons = []

        # Check if current hour is unfavorable
        if time_stats.get("is_significant", False) and time_stats.get("win_rate", 0.5) < 0.45:
            reasons.append(f"Hour {current_hour} has poor win rate ({time_stats['win_rate']:.1%})")

        # Check if current day is unfavorable
        if day_stats.get("is_significant", False) and day_stats.get("win_rate", 0.5) < 0.45:
            reasons.append(f"{current_day} has poor win rate ({day_stats['win_rate']:.1%})")

        if reasons:
            return False, "; ".join(reasons)

        # Check for favorable conditions
        favorable = []
        if time_stats.get("is_significant", False) and time_stats.get("win_rate", 0.5) > 0.55:
            favorable.append(f"Hour {current_hour} is favorable ({time_stats['win_rate']:.1%})")
        if day_stats.get("is_significant", False) and day_stats.get("win_rate", 0.5) > 0.55:
            favorable.append(f"{current_day} is favorable ({day_stats['win_rate']:.1%})")

        if favorable:
            return True, "; ".join(favorable)

        return True, "No significant pattern detected"


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.DEBUG)

    # Create sample data for testing
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="h", tz="UTC")
    np.random.seed(42)

    test_df = pd.DataFrame({
        "open": 100 + np.cumsum(np.random.randn(1000) * 0.5),
        "high": 100 + np.cumsum(np.random.randn(1000) * 0.5) + abs(np.random.randn(1000)) * 0.5,
        "low": 100 + np.cumsum(np.random.randn(1000) * 0.5) - abs(np.random.randn(1000)) * 0.5,
        "close": 100 + np.cumsum(np.random.randn(1000) * 0.5),
        "volume": np.random.randint(1000, 10000, 1000),
    }, index=dates)

    hunter = PatternHunter()

    # Test time-of-day analysis
    print("\n=== Time of Day Analysis ===")
    tod_patterns = hunter.analyze_time_of_day(test_df)
    for hour, stats in list(tod_patterns.items())[:5]:
        print(f"Hour {hour}: Win Rate={stats['win_rate']:.2%}, Samples={stats['sample_size']}")

    # Test day-of-week analysis
    print("\n=== Day of Week Analysis ===")
    dow_patterns = hunter.analyze_day_of_week(test_df)
    for day, stats in dow_patterns.items():
        print(f"{day}: Win Rate={stats['win_rate']:.2%}, Samples={stats['sample_size']}")

    # Test Japan Open analysis
    print("\n=== Japan Open Analysis ===")
    japan_patterns = hunter.japan_open_breakout(test_df)
    print(f"Bullish Breakout Rate: {japan_patterns.get('bullish_breakout_rate', 0):.2%}")
    print(f"Bearish Breakout Rate: {japan_patterns.get('bearish_breakout_rate', 0):.2%}")
    print(f"Sample Size: {japan_patterns.get('sample_size', 0)}")
