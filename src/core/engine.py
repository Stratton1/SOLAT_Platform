"""
Sentinel trading engine orchestrator.

The Sentinel is the core of SOLAT's autonomous trading system. It:
1. Manages the watchlist of assets (the "gene pool")
2. Fetches market data via adapters
3. Runs the Ichimoku strategy on each asset
4. Logs trades and market snapshots to the database
5. Implements the event loop for continuous monitoring
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
from datetime import datetime

from src.adapters.ccxt_lib import CCXTAdapter
from src.adapters.yfinance_lib import YFinanceAdapter
from src.core.ichimoku import IchimokuStrategy
from src.core.evolution import EvolutionaryOptimizer
from src.core.regime import MarketRegimeDetector
from src.config.settings import EVOLUTION_EPOCH
from src.database.repository import get_connection

logger = logging.getLogger(__name__)


class Sentinel:
    """
    Sentinel: Autonomous trading engine orchestrator.

    The Sentinel manages the complete trading workflow:
    - Asset discovery and initialization (gene pool)
    - Market data fetching (via adapters)
    - Strategy signal generation (Ichimoku)
    - Trade logging and market snapshots (persistence)
    - Event loop execution (continuous scanning)

    Architecture:
    - Assets are loaded from JSON seed file on first run
    - Each asset has a status: active, normal, dormant
    - The sentinel scans active assets and logs signals to the database
    - Paper trading mode: All trades are logged but not executed
    """

    def __init__(self, config_dir: str = "src/config") -> None:
        """
        Initialize the Sentinel.

        Args:
            config_dir (str): Directory containing assets_seed.json
        """
        self.config_dir = Path(config_dir)
        self.assets_seed_path = self.config_dir / "assets_seed.json"

        # Initialize adapters
        self.ccxt_adapter = CCXTAdapter()
        self.yfinance_adapter = YFinanceAdapter()

        # Initialize strategy
        self.strategy = IchimokuStrategy()

        # Initialize evolution optimizer
        self.optimizer = EvolutionaryOptimizer()

        # Initialize regime detector (HMM for Bull/Bear/Chop detection)
        self.regime_detector = MarketRegimeDetector(n_components=3, lookback=252)
        self.current_regime = "neutral"  # Track current market regime

        # Track last evolution epoch
        self.last_evolution_time = datetime.utcnow()

        logger.info("Initialized Sentinel engine with evolution optimizer and regime detector")

    def _get_adapter(self, source: str):
        """
        Get the appropriate adapter for a data source.

        Args:
            source (str): Data source ("ccxt" or "yfinance")

        Returns:
            MarketAdapter: The appropriate adapter instance

        Raises:
            ValueError: If source is not supported
        """
        if source.lower() == "ccxt":
            return self.ccxt_adapter
        elif source.lower() == "yfinance":
            return self.yfinance_adapter
        else:
            raise ValueError(f"Unsupported data source: {source}")

    def initialize_assets(self) -> None:
        """
        Load initial assets from seed file into the database.

        This method:
        1. Checks if the assets table is empty
        2. If empty, reads assets_seed.json
        3. Inserts assets into the database with initial status
        4. Sets initial fitness scores to 0.5 (neutral)

        The seed file should contain a list of dicts:
        [
            {"symbol": "BTC/USDT", "source": "ccxt", "status": "active"},
            {"symbol": "AAPL", "source": "yfinance", "status": "active"},
            ...
        ]
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Check if assets table is already populated
            cursor.execute("SELECT COUNT(*) FROM assets")
            count = cursor.fetchone()[0]

            if count > 0:
                logger.info(f"Assets table already populated with {count} assets. Skipping seed initialization.")
                conn.close()
                return

            # Load seed file
            if not self.assets_seed_path.exists():
                logger.warning(f"Assets seed file not found at {self.assets_seed_path}. Skipping initialization.")
                conn.close()
                return

            with open(self.assets_seed_path, "r") as f:
                assets = json.load(f)

            if not assets or not isinstance(assets, list):
                logger.warning("Assets seed file is empty or invalid")
                conn.close()
                return

            # Insert assets into database
            now = datetime.utcnow().isoformat()
            for asset in assets:
                symbol = asset.get("symbol")
                source = asset.get("source")
                status = asset.get("status", "normal")

                if not symbol or not source:
                    logger.warning(f"Skipping invalid asset: {asset}")
                    continue

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO assets
                    (symbol, source, status, fitness_score, last_scan)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (symbol, source, status, 0.5, now)
                )

                logger.info(f"Initialized asset: {symbol} ({source}) with status={status}")

            conn.commit()
            logger.info(f"Successfully initialized {len(assets)} assets from seed file")

        except Exception as e:
            logger.error(f"Error initializing assets: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()

    def scan_market(self) -> Dict[str, int]:
        """
        Scan the market by fetching data and running strategy on active assets.

        This is the core trading loop iteration:
        1. Load active assets from database
        2. For each asset:
           - Fetch OHLCV data (100 candles)
           - Run Ichimoku strategy
           - Update market_snapshots table
           - If signal != NEUTRAL, log to trades table
        3. Return scan statistics

        Returns:
            Dict[str, int]: Statistics about the scan:
                - scanned: Number of assets scanned
                - signals: Number of non-neutral signals
                - errors: Number of errors during scan
        """
        stats = {"scanned": 0, "signals": 0, "errors": 0}

        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Load active assets
            cursor.execute(
                "SELECT symbol, source FROM assets WHERE status = ?",
                ("active",)
            )
            assets = cursor.fetchall()

            if not assets:
                logger.warning("No active assets to scan")
                conn.close()
                return stats

            logger.info(f"Scanning {len(assets)} active assets")

            for symbol, source in assets:
                try:
                    # Step 1: Fetch OHLCV data
                    adapter = self._get_adapter(source)

                    # Determine timeframe based on source
                    timeframe = "1h" if source.lower() == "ccxt" else "1d"

                    logger.debug(f"Fetching {symbol} ({timeframe}) from {source}")
                    df = adapter.get_ohlcv(symbol, timeframe, limit=100)

                    if df is None or df.empty:
                        logger.warning(f"No data returned for {symbol}")
                        stats["errors"] += 1
                        continue

                    # Step 2: Run strategy
                    signal_result = self.strategy.check_signal(df)
                    ichimoku_signal = signal_result.get("signal", "NEUTRAL")

                    # Step 2.5: Detect market regime (HMM)
                    regime = self.regime_detector.predict_regime(df)
                    self.current_regime = regime
                    regime_probs = self.regime_detector.get_regime_probabilities(df)

                    logger.debug(
                        f"Regime for {symbol}: {regime} "
                        f"(Bull:{regime_probs['bull']:.2f}, Bear:{regime_probs['bear']:.2f}, Chop:{regime_probs['chop']:.2f})"
                    )

                    # REGIME FILTER: Apply logic filters based on market regime
                    # Logic:
                    # - Chop: No trades (kill zone)
                    # - Bull: Accept BUY signals, ignore SELL
                    # - Bear: Accept SELL signals, ignore BUY
                    signal = ichimoku_signal
                    regime_filtered = False

                    if regime == "chop":
                        # Never trade in chop regime
                        signal = "NEUTRAL"
                        regime_filtered = True
                        signal_result["reason"] = f"Ichimoku:{ichimoku_signal} | Regime:{regime} (CHOP ZONE - BLOCKED)"

                    elif regime == "bull" and ichimoku_signal == "SELL":
                        # Ignore SELL signals in bull market
                        signal = "NEUTRAL"
                        regime_filtered = True
                        signal_result["reason"] = f"Ichimoku:{ichimoku_signal} | Regime:{regime} (BULL MARKET - SELL BLOCKED)"

                    elif regime == "bear" and ichimoku_signal == "BUY":
                        # Ignore BUY signals in bear market
                        signal = "NEUTRAL"
                        regime_filtered = True
                        signal_result["reason"] = f"Ichimoku:{ichimoku_signal} | Regime:{regime} (BEAR MARKET - BUY BLOCKED)"

                    # Step 3: Update market_snapshots (WITH REGIME)
                    now = datetime.utcnow().isoformat()
                    close_price = df["close"].iloc[-1]

                    cursor.execute(
                        """
                        INSERT INTO market_snapshots
                        (symbol, close_price, cloud_status, tk_cross, chikou_conf, regime, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            symbol,
                            float(close_price),
                            signal_result.get("cloud_color", "unknown"),
                            signal_result.get("cloud_color"),
                            signal,  # Final signal (after regime filter)
                            regime,  # Store regime
                            now
                        )
                    )

                    logger.info(
                        f"Updated snapshot: {symbol} @ {close_price:.2f} "
                        f"| Ichimoku:{ichimoku_signal} | Regime:{regime} | Final Signal:{signal}"
                    )

                    # Step 4: Log trade if signal != NEUTRAL (AFTER regime filtering)
                    if signal != "NEUTRAL":
                        # Determine trade side based on signal
                        side = "BUY" if signal == "BUY" else "SELL"
                        entry_price = close_price

                        cursor.execute(
                            """
                            INSERT INTO trades
                            (symbol, side, entry_price, pnl, exit_reason, entry_time)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                symbol,
                                side,
                                float(entry_price),
                                0.0,  # PnL not calculated yet (paper trading)
                                signal_result.get("reason", "Ichimoku signal"),
                                now
                            )
                        )

                        logger.info(
                            f"Logged {side} signal for {symbol} @ {entry_price:.2f} "
                            f"| Reason: {signal_result.get('reason')}"
                        )

                        stats["signals"] += 1
                    elif regime_filtered:
                        logger.info(
                            f"Filtered out {ichimoku_signal} for {symbol} due to {regime} regime"
                        )

                    stats["scanned"] += 1

                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}", exc_info=True)
                    stats["errors"] += 1
                    continue

            conn.commit()
            logger.info(
                f"Scan complete: scanned={stats['scanned']}, "
                f"signals={stats['signals']}, errors={stats['errors']}"
            )

        except Exception as e:
            logger.error(f"Critical error in scan_market: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()

        return stats

    def run_loop(self, interval: int = 60) -> None:
        """
        Run the infinite event loop.

        Continuously scans the market at the specified interval.
        This is the "heartbeat" of the Sentinel process.

        Args:
            interval (int): Sleep interval between scans in seconds (default: 60)

        Note:
            This function runs indefinitely. Stop via SIGTERM or SIGINT.
        """
        logger.info(f"Starting Sentinel event loop (interval={interval}s)")

        try:
            while True:
                current_time = datetime.utcnow()
                logger.info("=" * 60)
                logger.info(f"Scan start: {current_time.isoformat()}")

                # Execute market scan
                stats = self.scan_market()

                logger.info(f"Scan results: {stats}")

                # Check if evolution epoch has elapsed
                time_since_evolution = (current_time - self.last_evolution_time).total_seconds()
                if time_since_evolution >= EVOLUTION_EPOCH:
                    logger.info(f"Evolution epoch reached ({time_since_evolution:.0f}s >= {EVOLUTION_EPOCH}s)")
                    promoted, demoted = self.optimizer.optimize_pool()
                    self.last_evolution_time = current_time
                    logger.info(f"Evolution complete: {promoted} promoted, {demoted} demoted")
                else:
                    logger.debug(
                        f"Next evolution in {EVOLUTION_EPOCH - time_since_evolution:.0f}s "
                        f"(last: {time_since_evolution:.0f}s ago)"
                    )

                logger.info(f"Next scan in {interval} seconds")
                logger.info("=" * 60)

                # Sleep until next scan
                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Sentinel interrupted by user")
        except Exception as e:
            logger.error(f"Sentinel event loop crashed: {e}", exc_info=True)
        finally:
            logger.info("Sentinel shutdown complete")
