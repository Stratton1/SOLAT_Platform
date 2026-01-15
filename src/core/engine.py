"""
Sentinel trading engine orchestrator.

The Sentinel is the core of SOLAT's autonomous trading system. It:
1. Manages the watchlist of assets (the "gene pool")
2. Fetches market data via adapters
3. Runs the Ichimoku strategy on each asset
4. Logs trades and market snapshots to the database
5. Implements the event loop for continuous monitoring

Institutional Upgrade Features:
- Portfolio-level risk management (Kelly Criterion, drawdown halts)
- Multi-strategy selection per asset
- Split entry execution (TP1 + trailing stop)
- Email and desktop notifications
- Seasonality pattern awareness
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
from datetime import datetime

from src.adapters.ccxt_lib import CCXTAdapter
from src.adapters.yfinance_lib import YFinanceAdapter
from src.adapters.ig_lib import IGAdapter
from src.core.ichimoku import IchimokuStrategy
from src.core.evolution import EvolutionaryOptimizer
from src.core.regime import MarketRegimeDetector
from src.core.risk_engine import PortfolioManager
from src.core.execution import TradeManager
from src.core.backtest_engine import StrategyOptimizer, get_strategy_by_name
from src.core.seasonality import PatternHunter
from src.core.optimization import (
    HyperoptEngine,
    load_golden_settings,
    get_golden_settings,
    has_golden_settings,
    DynamicIchimokuStrategy,
    OPTUNA_AVAILABLE,
)
from src.core.microstructure import OrderFlowAnalyzer, OrderBookMetrics
from src.core.news_sentinel import NewsSentinel, NEWS_SENTINEL_AVAILABLE, SentimentResult
from src.core.consensus import ConsensusEngine
from src.config.settings import EVOLUTION_EPOCH, CONSENSUS_VOTING_ENABLED, MAX_OPEN_TRADES, MAX_POSITION_SIZE_PERCENT
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

    def __init__(
        self,
        config_dir: str = "src/config",
        initial_balance: float = 10000.0,
        enable_notifications: bool = True,
    ) -> None:
        """
        Initialize the Sentinel with institutional-grade components.

        Args:
            config_dir: Directory containing assets_seed.json
            initial_balance: Starting portfolio balance for position sizing
            enable_notifications: Whether to send desktop/email notifications
        """
        self.config_dir = Path(config_dir)
        self.assets_seed_path = self.config_dir / "assets_seed.json"

        # Initialize adapters
        self.ccxt_adapter = CCXTAdapter()
        self.yfinance_adapter = YFinanceAdapter()
        self.ig_adapter = IGAdapter()  # UK Spread Betting (tax-free)

        # Initialize default strategy (fallback)
        self.strategy = IchimokuStrategy()

        # Initialize evolution optimizer
        self.optimizer = EvolutionaryOptimizer()

        # Initialize regime detector (HMM for Bull/Bear/Chop detection)
        self.regime_detector = MarketRegimeDetector(n_components=3, lookback=252)
        self.current_regime = "neutral"  # Track current market regime

        # ============================================================
        # INSTITUTIONAL UPGRADE COMPONENTS
        # ============================================================

        # Portfolio risk management (Kelly Criterion, drawdown halts)
        self.portfolio_manager = PortfolioManager(initial_balance=initial_balance)

        # Trade execution (split entries, TP1/TP2, trailing stops, notifications)
        self.trade_manager = TradeManager(
            enable_desktop_notifications=enable_notifications
        )

        # Multi-strategy backtester and optimizer
        self.strategy_optimizer = StrategyOptimizer()

        # Seasonality pattern analyzer
        self.pattern_hunter = PatternHunter()

        # Cache for optimal strategy per symbol
        self.optimal_strategies: Dict[str, str] = {}
        self._load_optimal_strategies()

        # ============================================================
        # HYPEROPT GOLDEN SETTINGS (AI-Optimized Parameters)
        # ============================================================
        self.golden_settings: Dict[str, Dict] = {}
        self._load_golden_settings()

        # Hyperopt engine (for running optimizations)
        self.hyperopt_engine = HyperoptEngine(n_trials=100) if OPTUNA_AVAILABLE else None

        # ============================================================
        # MICROSTRUCTURE SNIPER (Order Book Analysis)
        # ============================================================
        # Validates trades against order book pressure
        # BUY only when bid pressure > ask pressure
        # SELL only when ask pressure > bid pressure
        self.order_flow_analyzer = OrderFlowAnalyzer()

        # ============================================================
        # NEWS SENTINEL (Global Sentiment Circuit Breaker)
        # ============================================================
        # Monitors financial news and blocks trades against sentiment
        # Fear (< 30) blocks Longs, Greed (> 70) blocks Shorts
        self.news_sentinel = NewsSentinel() if NEWS_SENTINEL_AVAILABLE else None
        self.current_sentiment: Optional[SentimentResult] = None
        self._last_news_check: Optional[datetime] = None

        # ============================================================
        # CONSENSUS ENGINE (Parallel Voting - Council of 6)
        # ============================================================
        # All 6 agents vote independently: Regime, Strategy, Sniper, News, Seasonality, Institutional
        # Votes are aggregated using weighted average (-1.0 to +1.0)
        # Reinforcement learning adjusts weights based on trade outcomes
        if CONSENSUS_VOTING_ENABLED:
            self.consensus_engine = ConsensusEngine()
            logger.info("Consensus voting engine initialized (Council of 6)")
        else:
            self.consensus_engine = None
            logger.warning("Consensus voting disabled - using legacy veto logic")

        # Track last evolution epoch
        self.last_evolution_time = datetime.utcnow()

        logger.info(
            "Initialized Sentinel engine with institutional components: "
            "PortfolioManager, TradeManager, StrategyOptimizer, PatternHunter, "
            "HyperoptEngine, OrderFlowAnalyzer, NewsSentinel"
        )

    def _should_check_news(self) -> bool:
        """
        Check if news sentiment should be refreshed.

        Returns True if:
        - News has never been checked
        - Last check was more than 15 minutes ago
        """
        if self._last_news_check is None:
            return True

        time_since_check = (datetime.utcnow() - self._last_news_check).total_seconds()
        return time_since_check >= 900  # 15 minutes

    def _update_news_sentiment(self) -> None:
        """Update cached news sentiment if needed."""
        if not self.news_sentinel:
            return

        if self._should_check_news():
            logger.info("Refreshing news sentiment...")
            self.current_sentiment = self.news_sentinel.fetch_sentiment()
            self._last_news_check = datetime.utcnow()

            mood_emoji = self.news_sentinel.get_mood_emoji(self.current_sentiment.score)
            logger.info(
                f"News Sentiment: {mood_emoji} {self.current_sentiment.score:.0f}/100 "
                f"({self.current_sentiment.mood})"
            )

    def _load_optimal_strategies(self) -> None:
        """
        Load optimal strategy assignments from the database.

        Populates self.optimal_strategies with symbol -> strategy_name mapping.
        Falls back to 'ichimoku_standard' if no optimal strategy is set.
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT symbol, optimal_strategy FROM assets
                WHERE optimal_strategy IS NOT NULL
            """)

            for row in cursor.fetchall():
                symbol, strategy_name = row
                if strategy_name:
                    self.optimal_strategies[symbol] = strategy_name

            conn.close()
            logger.info(f"Loaded optimal strategies for {len(self.optimal_strategies)} assets")

        except Exception as e:
            logger.warning(f"Could not load optimal strategies: {e}")

    def _load_golden_settings(self) -> None:
        """
        Load AI-optimized Ichimoku parameters from golden_settings.json.

        Golden settings are parameters found by Optuna that achieve
        65%+ win rate for specific assets. When available, these
        override the standard Ichimoku 9/26/52 parameters.
        """
        try:
            self.golden_settings = load_golden_settings()

            if self.golden_settings:
                logger.info(f"Loaded golden settings for {len(self.golden_settings)} assets")
                for symbol, settings in self.golden_settings.items():
                    win_rate = settings.get("holdout_win_rate", 0)
                    logger.info(
                        f"  {symbol}: Tenkan={settings['tenkan']}, "
                        f"Kijun={settings['kijun']}, Senkou={settings['senkou']} "
                        f"(Historical WR: {win_rate:.0%})"
                    )
            else:
                logger.info("No golden settings found. Using standard Ichimoku parameters.")

        except Exception as e:
            logger.warning(f"Could not load golden settings: {e}")
            self.golden_settings = {}

    def _get_signal_with_golden_settings(
        self, symbol: str, df: pd.DataFrame
    ) -> Tuple[Dict[str, Any], str]:
        """
        Get trading signal using golden settings if available.

        If golden settings exist for the symbol, creates a DynamicIchimokuStrategy
        with the optimized parameters. Otherwise falls back to standard strategy.

        Args:
            symbol: Asset symbol
            df: OHLCV DataFrame

        Returns:
            Tuple[signal_result, strategy_name]
        """
        # Check if golden settings exist for this symbol
        if symbol in self.golden_settings:
            settings = self.golden_settings[symbol]

            logger.info(
                f"Using GOLDEN SETTINGS for {symbol} "
                f"(Historical WR: {settings.get('holdout_win_rate', 0):.0%})"
            )

            try:
                # Create dynamic strategy with optimized parameters
                golden_strategy = DynamicIchimokuStrategy(
                    tenkan_period=settings["tenkan"],
                    kijun_period=settings["kijun"],
                    senkou_b_period=settings["senkou"],
                    stop_loss_multiplier=settings.get("stop_loss_mult", 2.0),
                )

                # Generate signals
                signals_df = golden_strategy.generate_signals(df)

                # Get latest signal
                latest_signal = signals_df["signal"].iloc[-1]
                signal_str = "BUY" if latest_signal == 1 else ("SELL" if latest_signal == -1 else "NEUTRAL")

                signal_result = {
                    "signal": signal_str,
                    "reason": f"Golden Settings: T={settings['tenkan']}/K={settings['kijun']}/S={settings['senkou']}",
                    "stop_loss": signals_df["stop_loss"].iloc[-1] if signal_str != "NEUTRAL" else None,
                    "golden_settings": True,
                    "historical_win_rate": settings.get("holdout_win_rate", 0),
                }

                return signal_result, f"golden_{symbol.replace('/', '_')}"

            except Exception as e:
                logger.warning(f"Golden settings signal error for {symbol}: {e}. Falling back to standard.")

        # Fall back to optimal strategy from database
        strategy_name = self.optimal_strategies.get(symbol, "ichimoku_standard")
        strategy = get_strategy_by_name(strategy_name)

        if strategy:
            signal_result = strategy.check_signal(df)
        else:
            signal_result = self.strategy.check_signal(df)
            strategy_name = "ichimoku_standard"

        signal_result["golden_settings"] = False
        return signal_result, strategy_name

    def _update_trailing_stops(self) -> None:
        """
        Update trailing stops for all open Unit 2 trades.

        Called at the end of each scan cycle to adjust trailing stop prices
        based on current market conditions.
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Get all open Unit 2 trades (these have trailing stops)
            cursor.execute("""
                SELECT id, symbol, side, entry_price, trailing_stop_price
                FROM trades
                WHERE is_open = 1 AND unit_number = 2
            """)

            open_trades = cursor.fetchall()
            conn.close()

            for trade_id, symbol, side, entry_price, current_trailing in open_trades:
                try:
                    # Fetch current market data
                    cursor_asset = get_connection().cursor()
                    cursor_asset.execute(
                        "SELECT source FROM assets WHERE symbol = ?", (symbol,)
                    )
                    row = cursor_asset.fetchone()
                    cursor_asset.connection.close()

                    if not row:
                        continue

                    source = row[0]
                    adapter = self._get_adapter(source)
                    timeframe = "1h" if source.lower() == "ccxt" else "1d"
                    df = adapter.get_ohlcv(symbol, timeframe, limit=100)

                    if df is None or df.empty:
                        continue

                    current_price = df["close"].iloc[-1]

                    # Update trailing stop
                    new_trailing = self.trade_manager.update_trailing_stop(
                        trade_id, current_price, df, method="chandelier"
                    )

                    if new_trailing != current_trailing:
                        logger.debug(
                            f"Updated trailing stop for trade {trade_id}: "
                            f"{current_trailing:.4f} -> {new_trailing:.4f}"
                        )

                    # Check exit conditions
                    exit_result = self.trade_manager.check_exit_conditions(
                        trade_id, current_price
                    )

                    if exit_result:
                        exit_reason, exit_price = exit_result
                        pnl = self.trade_manager.close_trade(
                            trade_id, exit_price, exit_reason
                        )
                        logger.info(
                            f"Closed trade {trade_id} ({symbol}): "
                            f"Reason={exit_reason}, PnL={pnl:.2f}"
                        )

                except Exception as e:
                    logger.error(f"Error updating trailing stop for trade {trade_id}: {e}")

        except Exception as e:
            logger.error(f"Error in _update_trailing_stops: {e}")

    def _get_adapter(self, source: str):
        """
        Get the appropriate adapter for a data source.

        Args:
            source: Data source ("ccxt", "yfinance", or "ig")

        Returns:
            MarketAdapter: The appropriate adapter instance

        Raises:
            ValueError: If source is not supported

        Supported Sources:
            - ccxt: Cryptocurrency exchanges (Binance, Coinbase, etc.)
            - yfinance: US equities, ETFs, indices
            - ig: UK Spread Betting via IG Markets (tax-free profits)
        """
        source_lower = source.lower()

        if source_lower == "ccxt":
            return self.ccxt_adapter
        elif source_lower == "yfinance":
            return self.yfinance_adapter
        elif source_lower == "ig":
            return self.ig_adapter
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

    def _collect_agent_votes(self, symbol: str, df: pd.DataFrame, regime: str,
                               ichimoku_signal: str, order_imbalance: float,
                               seasonality_passed: bool) -> Dict[str, Tuple[float, str]]:
        """
        Collect votes from all 6 agents for parallel consensus voting.

        Returns:
            Dict[agent_name: (vote_value, reasoning)]
        """
        votes = {}

        # Agent 1: REGIME (HMM Bull/Bear/Chop detection)
        regime_vote = 1.0 if regime == "bull" else (-1.0 if regime == "bear" else 0.0)
        votes['regime'] = (
            regime_vote,
            f"Regime: {regime.upper()}"
        )

        # Agent 2: STRATEGY (Ichimoku Cloud signals)
        strategy_vote = 1.0 if ichimoku_signal == "BUY" else (-1.0 if ichimoku_signal == "SELL" else 0.0)
        votes['strategy'] = (
            strategy_vote,
            f"Ichimoku: {ichimoku_signal}"
        )

        # Agent 3: SNIPER (Microstructure order flow)
        if abs(order_imbalance) > 0.2:
            sniper_vote = order_imbalance  # Range: -1.0 to +1.0
            reason = "Strong order pressure" if abs(order_imbalance) > 0.5 else "Moderate order pressure"
        else:
            sniper_vote = 0.0
            reason = "Balanced order flow"
        votes['sniper'] = (sniper_vote, reason)

        # Agent 4: NEWS (Sentiment analysis)
        news_vote = 0.0
        news_reason = "No sentiment data"
        if self.current_sentiment:
            sentiment_score = self.current_sentiment.score  # 0-100
            # Convert to vote: -1.0 to +1.0
            news_vote = (sentiment_score - 50) / 50.0
            news_reason = self.current_sentiment.label

        votes['news'] = (news_vote, news_reason)

        # Agent 5: SEASONALITY (Time-based patterns)
        seasonality_vote = 0.5 if seasonality_passed else -0.3
        seasonality_reason = "Seasonally favorable" if seasonality_passed else "Seasonally unfavorable"
        votes['seasonality'] = (seasonality_vote, seasonality_reason)

        # Agent 6: INSTITUTIONAL (Portfolio constraints)
        # Check if we can open more trades (MAX_OPEN_TRADES constraint)
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'open'")
            open_count = cursor.fetchone()[0]
            conn.close()

            if open_count >= MAX_OPEN_TRADES:
                institutional_vote = -1.0
                institutional_reason = f"Max trades reached ({MAX_OPEN_TRADES}/5)"
            else:
                institutional_vote = 0.8  # Portfolio is healthy
                institutional_reason = f"Portfolio ready ({open_count}/5 trades)"
        except:
            institutional_vote = 0.5
            institutional_reason = "Portfolio check failed"

        votes['institutional'] = (institutional_vote, institutional_reason)

        return votes

    def scan_market(self) -> Dict[str, int]:
        """
        Scan the market by fetching data and running strategy on active assets.

        This is the core trading loop iteration with institutional features:
        1. Check if trading is halted (drawdown circuit breaker)
        2. Load active assets from database
        3. For each asset:
           - Check seasonality patterns (optional trade timing)
           - Fetch OHLCV data (100 candles)
           - Run optimal strategy per asset
           - Apply regime filter
           - Calculate position size with Kelly Criterion
           - Execute split entry (TP1 + trailing stop)
           - Send notifications
        4. Update trailing stops for open positions
        5. Check daily drawdown and halt if exceeded
        6. Return scan statistics

        Returns:
            Dict[str, int]: Statistics about the scan:
                - scanned: Number of assets scanned
                - signals: Number of non-neutral signals
                - errors: Number of errors during scan
                - halted: 1 if trading is halted, 0 otherwise
        """
        stats = {"scanned": 0, "signals": 0, "errors": 0, "halted": 0}

        # ============================================================
        # STEP 0: Check if trading is halted (circuit breaker)
        # ============================================================
        is_halted, halt_reason = self.portfolio_manager.is_trading_halted()
        if is_halted:
            logger.warning(f"Trading is HALTED: {halt_reason}")
            stats["halted"] = 1
            # Still update trailing stops for existing positions
            self._update_trailing_stops()
            return stats

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
                    # ============================================================
                    # STEP 1: Check seasonality patterns (optional)
                    # ============================================================
                    should_trade, seasonality_reason = self.pattern_hunter.should_trade_now(symbol)
                    if not should_trade:
                        logger.debug(f"Seasonality filter for {symbol}: {seasonality_reason}")
                        # Continue scanning but log the seasonality concern

                    # ============================================================
                    # STEP 2: Fetch OHLCV data
                    # ============================================================
                    adapter = self._get_adapter(source)
                    timeframe = "1h" if source.lower() == "ccxt" else "1d"

                    logger.debug(f"Fetching {symbol} ({timeframe}) from {source}")
                    df = adapter.get_ohlcv(symbol, timeframe, limit=100)

                    if df is None or df.empty:
                        logger.warning(f"No data returned for {symbol}")
                        stats["errors"] += 1
                        continue

                    # ============================================================
                    # STEP 3: Run optimal strategy for this asset
                    # Uses Golden Settings (AI-optimized) if available
                    # ============================================================
                    signal_result, strategy_name = self._get_signal_with_golden_settings(symbol, df)
                    ichimoku_signal = signal_result.get("signal", "NEUTRAL")

                    # Log if using golden settings
                    if signal_result.get("golden_settings"):
                        logger.info(
                            f"GOLDEN SETTINGS for {symbol}: "
                            f"Historical WR={signal_result.get('historical_win_rate', 0):.0%}"
                        )

                    # ============================================================
                    # STEP 4: Detect market regime (HMM)
                    # ============================================================
                    regime = self.regime_detector.predict_regime(df)
                    self.current_regime = regime
                    regime_probs = self.regime_detector.get_regime_probabilities(df)

                    logger.debug(
                        f"Regime for {symbol}: {regime} "
                        f"(Bull:{regime_probs['bull']:.2f}, Bear:{regime_probs['bear']:.2f}, Chop:{regime_probs['chop']:.2f})"
                    )

                    # ============================================================
                    # STEP 5: PARALLEL CONSENSUS VOTING (Council of 6)
                    # ============================================================
                    # Replace sequential veto with parallel voting if enabled
                    signal = ichimoku_signal
                    regime_filtered = False
                    consensus_score = 0.0
                    agent_votes_dict = {}

                    if self.consensus_engine:
                        # Get order imbalance for sniper agent
                        order_imbalance = 0.0
                        try:
                            self.order_flow_analyzer.set_adapter(adapter)
                            ob_metrics = self.order_flow_analyzer.get_order_book_metrics(symbol)
                            order_imbalance = ob_metrics.order_imbalance
                        except:
                            order_imbalance = 0.0

                        # Collect votes from all 6 agents
                        agent_votes_dict = self._collect_agent_votes(
                            symbol, df, regime, ichimoku_signal,
                            order_imbalance, should_trade
                        )

                        # Extract just the vote values for consensus engine
                        votes_dict = {k: v[0] for k, v in agent_votes_dict.items()}

                        # Get consensus from engine
                        consensus_result = self.consensus_engine.aggregate_consensus(votes_dict)
                        consensus_score = consensus_result.consensus_score
                        consensus_decision = consensus_result.decision

                        # Log the vote breakdown
                        vote_breakdown = " | ".join([
                            f"{k.upper()}:{v[0]:+.2f}({v[1]})"
                            for k, v in agent_votes_dict.items()
                        ])
                        logger.info(
                            f"[COUNCIL_VOTE] {symbol}: {vote_breakdown} => "
                            f"CONSENSUS:{consensus_score:+.2f} DECISION:{consensus_decision}"
                        )

                        # Determine final signal based on consensus
                        if abs(consensus_score) > 0.60:  # CONSENSUS_THRESHOLD_EXECUTE
                            signal = "BUY" if consensus_score > 0 else "SELL"
                            signal_result["reason"] = f"Council Consensus: {consensus_score:+.2f} | {consensus_decision}"
                        else:
                            signal = "NEUTRAL"
                            signal_result["reason"] = f"Consensus too weak: {consensus_score:+.2f} (threshold: 0.60)"
                            regime_filtered = True  # Mark as filtered for logging

                    else:
                        # Legacy veto logic (if consensus voting is disabled)
                        if regime == "chop":
                            signal = "NEUTRAL"
                            regime_filtered = True
                            signal_result["reason"] = f"Strategy:{strategy_name} | Signal:{ichimoku_signal} | Regime:{regime} (CHOP ZONE - BLOCKED)"

                        elif regime == "bull" and ichimoku_signal == "SELL":
                            signal = "NEUTRAL"
                            regime_filtered = True
                            signal_result["reason"] = f"Strategy:{strategy_name} | Signal:{ichimoku_signal} | Regime:{regime} (BULL MARKET - SELL BLOCKED)"

                        elif regime == "bear" and ichimoku_signal == "BUY":
                            signal = "NEUTRAL"
                            regime_filtered = True
                            signal_result["reason"] = f"Strategy:{strategy_name} | Signal:{ichimoku_signal} | Regime:{regime} (BEAR MARKET - BUY BLOCKED)"

                        # Legacy: Microstructure sniper check
                        sniper_passed = True
                        order_imbalance = 0.0

                        if signal != "NEUTRAL":
                            try:
                                self.order_flow_analyzer.set_adapter(adapter)
                                ob_metrics = self.order_flow_analyzer.get_order_book_metrics(symbol)
                                order_imbalance = ob_metrics.order_imbalance

                                sniper_passed, sniper_reason = self.order_flow_analyzer.check_sniper_entry(
                                    symbol, signal, strict=True
                                )

                                if not sniper_passed:
                                    logger.info(f"SNIPER BLOCKED: {symbol} {signal}")
                                    signal = "NEUTRAL"
                                    signal_result["reason"] = f"SNIPER BLOCKED: {sniper_reason}"
                            except:
                                pass

                    # ============================================================
                    # STEP 6: Update market_snapshots (includes microstructure)
                    # ============================================================
                    now = datetime.utcnow().isoformat()
                    close_price = df["close"].iloc[-1]

                    cursor.execute(
                        """
                        INSERT INTO market_snapshots
                        (symbol, close_price, cloud_status, tk_cross, chikou_conf, regime, order_imbalance, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            symbol,
                            float(close_price),
                            signal_result.get("cloud_color", "unknown"),
                            signal_result.get("cloud_color"),
                            signal,
                            regime,
                            order_imbalance,
                            now
                        )
                    )

                    logger.info(
                        f"Updated snapshot: {symbol} @ {close_price:.2f} "
                        f"| Strategy:{strategy_name} | Signal:{ichimoku_signal} | Regime:{regime} "
                        f"| OI:{order_imbalance:.3f} | Final:{signal}"
                    )

                    # ============================================================
                    # STEP 6.5: UPDATE NEWS SENTIMENT (for Council voting)
                    # ============================================================
                    # Keep news sentiment updated for consensus voting
                    if self.news_sentinel and signal != "NEUTRAL":
                        self._update_news_sentiment()

                    # ============================================================
                    # STEP 7: Execute trade if signal is actionable
                    # ============================================================
                    if signal != "NEUTRAL":
                        side = "BUY" if signal == "BUY" else "SELL"
                        entry_price = close_price

                        # Calculate stop loss from ATR
                        stop_loss_price = signal_result.get("stop_loss")
                        if stop_loss_price is None:
                            # Default: 2% from entry
                            stop_loss_price = entry_price * (0.98 if side == "BUY" else 1.02)

                        # Calculate position size using Kelly Criterion
                        position_size, dollar_risk = self.portfolio_manager.calculate_stake(
                            symbol=symbol,
                            entry_price=entry_price,
                            stop_loss_price=stop_loss_price,
                        )

                        if position_size > 0:
                            # Execute split entry (Unit 1 with TP, Unit 2 with trailing)
                            trade_ids = self.trade_manager.execute_split_entry(
                                symbol=symbol,
                                side=side,
                                entry_price=entry_price,
                                stop_loss_price=stop_loss_price,
                                total_position_size=position_size,
                                strategy_name=strategy_name,
                            )

                            # Send notification
                            trade_data = {
                                "symbol": symbol,
                                "side": side,
                                "entry_price": entry_price,
                                "stop_loss": stop_loss_price,
                                "position_size": position_size,
                                "strategy": strategy_name,
                                "regime": regime,
                                "trade_ids": trade_ids,
                            }
                            self.trade_manager.notify_trade_entry(trade_data)

                            logger.info(
                                f"Executed {side} for {symbol} @ {entry_price:.2f} "
                                f"| Size: {position_size:.4f} | Risk: ${dollar_risk:.2f} "
                                f"| Trade IDs: {trade_ids}"
                            )

                            stats["signals"] += 1
                        else:
                            logger.warning(
                                f"Position size is 0 for {symbol} - skipping trade "
                                f"(may be at exposure limit)"
                            )

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

            # ============================================================
            # STEP 8: Update trailing stops for open positions
            # ============================================================
            self._update_trailing_stops()

            # ============================================================
            # STEP 9: Check daily drawdown and halt if exceeded
            # ============================================================
            if not self.portfolio_manager.check_daily_drawdown():
                self.portfolio_manager.halt_trading(
                    reason="Daily drawdown limit exceeded (5%)",
                    duration_hours=24
                )
                logger.warning("TRADING HALTED: Daily drawdown limit exceeded")
                stats["halted"] = 1

            logger.info(
                f"Scan complete: scanned={stats['scanned']}, "
                f"signals={stats['signals']}, errors={stats['errors']}, halted={stats['halted']}"
            )

        except Exception as e:
            logger.error(f"Critical error in scan_market: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()

        return stats

    def optimize_strategies(self, symbols: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Run strategy optimization for specified assets or all active assets.

        Backtests all registered strategies on historical data and selects
        the optimal strategy for each asset based on Sharpe ratio.

        Args:
            symbols: List of symbols to optimize. If None, uses all active assets.

        Returns:
            Dictionary mapping symbol to optimal strategy name.
        """
        results: Dict[str, str] = {}

        try:
            if symbols is None:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT symbol, source FROM assets WHERE status = 'active'")
                assets = cursor.fetchall()
                conn.close()
            else:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT symbol, source FROM assets WHERE symbol IN ({','.join('?' * len(symbols))})",
                    symbols
                )
                assets = cursor.fetchall()
                conn.close()

            logger.info(f"Running strategy optimization for {len(assets)} assets")

            for symbol, source in assets:
                try:
                    adapter = self._get_adapter(source)
                    timeframe = "1d"  # Use daily data for backtesting
                    df = adapter.get_ohlcv(symbol, timeframe, limit=500)

                    if df is None or len(df) < 100:
                        logger.warning(f"Insufficient data for {symbol} optimization")
                        continue

                    optimal_strategy, metrics = self.strategy_optimizer.find_best_strategy(
                        symbol=symbol,
                        df=df,
                        save_to_db=True,
                    )

                    results[symbol] = optimal_strategy
                    self.optimal_strategies[symbol] = optimal_strategy

                    logger.info(
                        f"Optimized {symbol}: {optimal_strategy} "
                        f"(Sharpe: {metrics.get('sharpe_ratio', 0):.2f})"
                    )

                except Exception as e:
                    logger.error(f"Error optimizing {symbol}: {e}")

            return results

        except Exception as e:
            logger.error(f"Error in optimize_strategies: {e}")
            return results

    def analyze_seasonality(self, symbols: Optional[List[str]] = None) -> None:
        """
        Run seasonality analysis for specified assets or all active assets.

        Analyzes time-of-day, day-of-week, and Japan Open patterns.

        Args:
            symbols: List of symbols to analyze. If None, uses all active assets.
        """
        try:
            if symbols is None:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT symbol, source FROM assets WHERE status = 'active'")
                assets = cursor.fetchall()
                conn.close()
            else:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT symbol, source FROM assets WHERE symbol IN ({','.join('?' * len(symbols))})",
                    symbols
                )
                assets = cursor.fetchall()
                conn.close()

            logger.info(f"Running seasonality analysis for {len(assets)} assets")

            for symbol, source in assets:
                try:
                    adapter = self._get_adapter(source)
                    timeframe = "1h"  # Use hourly data for seasonality
                    df = adapter.get_ohlcv(symbol, timeframe, limit=1000)

                    if df is None or len(df) < 200:
                        logger.warning(f"Insufficient data for {symbol} seasonality analysis")
                        continue

                    patterns = self.pattern_hunter.analyze_all_patterns(symbol, df)
                    self.pattern_hunter.save_patterns_to_db(symbol, patterns)

                    logger.info(f"Completed seasonality analysis for {symbol}")

                except Exception as e:
                    logger.error(f"Error analyzing seasonality for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Error in analyze_seasonality: {e}")

    def run_hyperopt(
        self,
        symbols: Optional[List[str]] = None,
        n_trials: int = 100,
    ) -> Dict[str, Dict]:
        """
        Run Optuna hyperparameter optimization to find golden settings.

        Searches for Ichimoku parameters that maximize Win Rate while
        ensuring robustness through Walk-Forward Analysis.

        Target: 65%+ Win Rate with validation on holdout data.

        Args:
            symbols: List of symbols to optimize. If None, uses all active assets.
            n_trials: Number of Optuna trials per symbol

        Returns:
            Dict mapping symbol to golden settings (or None if optimization failed)
        """
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna not installed. Run: pip install optuna>=3.0.0")
            return {}

        results: Dict[str, Dict] = {}

        try:
            if symbols is None:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT symbol, source FROM assets WHERE status = 'active'")
                assets = cursor.fetchall()
                conn.close()
            else:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT symbol, source FROM assets WHERE symbol IN ({','.join('?' * len(symbols))})",
                    symbols
                )
                assets = cursor.fetchall()
                conn.close()

            logger.info(f"Running hyperparameter optimization for {len(assets)} assets")
            logger.info(f"Target: 65%+ Win Rate with Walk-Forward Validation")

            # Create fresh hyperopt engine with specified trials
            hyperopt = HyperoptEngine(n_trials=n_trials)

            for symbol, source in assets:
                try:
                    adapter = self._get_adapter(source)
                    # Use daily data for optimization (more stable patterns)
                    df = adapter.get_ohlcv(symbol, "1d", limit=500)

                    if df is None or len(df) < 200:
                        logger.warning(f"Insufficient data for {symbol} hyperopt (need 200+ rows)")
                        continue

                    logger.info(f"\nOptimizing {symbol}...")
                    golden = hyperopt.find_golden_settings(symbol, df, save=True)

                    if golden:
                        results[symbol] = golden
                        # Update local cache
                        self.golden_settings[symbol] = golden
                        logger.info(
                            f"Found golden settings for {symbol}: "
                            f"WR={golden['holdout_win_rate']:.0%}"
                        )
                    else:
                        logger.warning(f"No golden settings found for {symbol}")

                except Exception as e:
                    logger.error(f"Hyperopt failed for {symbol}: {e}")

            # Summary
            logger.info(f"\n{'='*60}")
            logger.info("HYPEROPT COMPLETE")
            logger.info(f"{'='*60}")
            logger.info(f"Assets optimized: {len(results)}/{len(assets)}")

            for sym, settings in results.items():
                logger.info(
                    f"  {sym}: T={settings['tenkan']}/K={settings['kijun']}/S={settings['senkou']} "
                    f"| WR={settings['holdout_win_rate']:.0%}"
                )

            return results

        except Exception as e:
            logger.error(f"Error in run_hyperopt: {e}")
            return results

    def run_loop(self, interval: int = 60) -> None:
        """
        Run the infinite event loop with institutional features.

        Continuously scans the market at the specified interval.
        This is the "heartbeat" of the Sentinel process.

        Features:
        - Market scanning with regime filtering
        - Evolution optimization every 4 hours
        - Trailing stop management
        - Drawdown circuit breaker

        Args:
            interval: Sleep interval between scans in seconds (default: 60)

        Note:
            This function runs indefinitely. Stop via SIGTERM or SIGINT.
        """
        logger.info(f"Starting Sentinel event loop (interval={interval}s)")
        logger.info("Institutional features: Kelly sizing, split entries, trailing stops, notifications")

        try:
            while True:
                current_time = datetime.utcnow()
                logger.info("=" * 60)
                logger.info(f"Scan start: {current_time.isoformat()}")

                # Execute market scan (includes halt check, trailing stop updates, drawdown check)
                stats = self.scan_market()

                logger.info(f"Scan results: {stats}")

                # Check if evolution epoch has elapsed
                time_since_evolution = (current_time - self.last_evolution_time).total_seconds()
                if time_since_evolution >= EVOLUTION_EPOCH:
                    logger.info(f"Evolution epoch reached ({time_since_evolution:.0f}s >= {EVOLUTION_EPOCH}s)")

                    # Run evolution optimizer
                    promoted, demoted = self.optimizer.optimize_pool()
                    logger.info(f"Evolution complete: {promoted} promoted, {demoted} demoted")

                    # Also re-optimize strategies for active assets (less frequently)
                    # This could be moved to a separate, less frequent epoch
                    self._load_optimal_strategies()

                    self.last_evolution_time = current_time
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
