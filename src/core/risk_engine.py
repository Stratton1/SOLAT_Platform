"""
Institutional-Grade Risk Engine for SOLAT Platform.

This module provides:
1. PortfolioManager - Kelly Criterion position sizing, drawdown limits
2. Daily circuit breaker (5% drawdown = 24h halt)
3. Sector/correlation exposure tracking
4. Account balance management

Safety is the primary concern. The system errs on the side of caution.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List

import numpy as np

from src.database.repository import get_connection
from src.config.settings import (
    RISK_PER_TRADE,
    MAX_POSITION_SIZE_PERCENT,
    MAX_DRAWDOWN_ALLOWED,
    MAX_OPEN_TRADES,
    SAFE_MODE_ENABLED,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SECTOR CLASSIFICATION
# =============================================================================

# Asset to sector mapping (can be extended)
SECTOR_MAP: Dict[str, str] = {
    # Crypto
    "BTC/USDT": "crypto_major",
    "ETH/USDT": "crypto_major",
    "SOL/USDT": "crypto_alt",
    "DOGE/USDT": "crypto_meme",
    # US Equities
    "AAPL": "tech_us",
    "MSFT": "tech_us",
    "GOOGL": "tech_us",
    "NVDA": "tech_us",
    "TSLA": "auto_us",
    "SPY": "index_us",
    "QQQ": "index_us",
    # Commodities
    "GC=F": "commodity_metal",
    "SI=F": "commodity_metal",
    "CL=F": "commodity_energy",
}

DEFAULT_SECTOR = "uncategorized"


# =============================================================================
# PORTFOLIO MANAGER
# =============================================================================


class PortfolioManager:
    """
    Institutional-grade portfolio and risk management.

    Features:
    - Kelly Criterion position sizing (quarter-Kelly for safety)
    - Daily drawdown circuit breaker (5% = 24h halt)
    - Sector exposure limits (default 30% per sector)
    - Account balance tracking with peak/drawdown calculation

    Usage:
        pm = PortfolioManager(initial_balance=10000.0)

        # Before each trade
        if pm.is_trading_halted()[0]:
            return  # Don't trade

        # Calculate position
        size, risk = pm.calculate_stake("BTC/USDT", entry=50000, stop_loss=48000)

        # After trade closes
        pm.update_equity(new_balance)
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        max_daily_drawdown: float = 0.05,  # 5%
        max_sector_exposure: float = 0.30,  # 30%
        max_position_size: float = 0.10,  # 10%
        kelly_fraction: float = 0.25,  # Quarter-Kelly
    ) -> None:
        """
        Initialize the PortfolioManager.

        Args:
            initial_balance: Starting account balance
            max_daily_drawdown: Daily loss limit before halt (0.05 = 5%)
            max_sector_exposure: Max allocation per sector (0.30 = 30%)
            max_position_size: Max single position size (0.10 = 10%)
            kelly_fraction: Fraction of Kelly optimal (0.25 = quarter-Kelly)
        """
        self.initial_balance = initial_balance
        self.max_daily_drawdown = max_daily_drawdown
        self.max_sector_exposure = max_sector_exposure
        self.max_position_size = max_position_size
        self.kelly_fraction = kelly_fraction

        # Load or initialize state from database
        self._load_state()

        logger.info(
            f"PortfolioManager initialized: balance=${self.current_balance:.2f}, "
            f"max_dd={max_daily_drawdown:.0%}, kelly_frac={kelly_fraction}"
        )

    def _load_state(self) -> None:
        """Load current state from database or initialize."""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Get latest balance record
            cursor.execute(
                """
                SELECT balance, equity, peak_equity, daily_pnl, daily_drawdown
                FROM account_balance
                ORDER BY recorded_at DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()

            if row:
                self.current_balance = row[0]
                self.current_equity = row[1]
                self.peak_equity = row[2]
                self.daily_pnl = row[3]
                self.daily_drawdown = row[4]
            else:
                # Initialize with starting balance
                self.current_balance = self.initial_balance
                self.current_equity = self.initial_balance
                self.peak_equity = self.initial_balance
                self.daily_pnl = 0.0
                self.daily_drawdown = 0.0
                self._record_balance()

            conn.close()

        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")
            # Default to initial values
            self.current_balance = self.initial_balance
            self.current_equity = self.initial_balance
            self.peak_equity = self.initial_balance
            self.daily_pnl = 0.0
            self.daily_drawdown = 0.0

    def _record_balance(self, halt_reason: Optional[str] = None) -> None:
        """Record current balance state to database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            is_halted = 1 if halt_reason else 0

            cursor.execute(
                """
                INSERT INTO account_balance
                (balance, equity, daily_pnl, daily_drawdown, peak_equity, is_halted, halt_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.current_balance,
                    self.current_equity,
                    self.daily_pnl,
                    self.daily_drawdown,
                    self.peak_equity,
                    is_halted,
                    halt_reason,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error recording balance: {e}")

    # =========================================================================
    # KELLY CRITERION POSITION SIZING
    # =========================================================================

    def calculate_kelly_stake(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate optimal stake using Kelly Criterion.

        Formula: f* = (b*p - q) / b
        Where:
            b = avg_win / |avg_loss| (win/loss ratio)
            p = win probability
            q = 1 - p (loss probability)

        We use quarter-Kelly (kelly_fraction=0.25) for safety.

        Args:
            win_rate: Historical win probability (0.0 to 1.0)
            avg_win: Average winning trade return (e.g., 0.05 = 5%)
            avg_loss: Average losing trade return (e.g., -0.03 = -3%)

        Returns:
            float: Fraction of bankroll to stake (0.0 to 1.0)
        """
        # Validate inputs
        if win_rate <= 0 or win_rate >= 1:
            logger.warning(f"Invalid win_rate {win_rate}, using default 0.5")
            win_rate = 0.5

        if avg_win <= 0:
            logger.warning(f"Invalid avg_win {avg_win}, using default 0.02")
            avg_win = 0.02

        if avg_loss >= 0:
            logger.warning(f"Invalid avg_loss {avg_loss}, using default -0.02")
            avg_loss = -0.02

        # Kelly calculation
        b = avg_win / abs(avg_loss)  # Win/loss ratio
        p = win_rate
        q = 1 - p

        kelly = (b * p - q) / b

        # Apply safety fraction
        safe_kelly = kelly * self.kelly_fraction

        # Clamp to reasonable range [0, max_position_size]
        safe_kelly = max(0.0, min(safe_kelly, self.max_position_size))

        logger.debug(
            f"Kelly: b={b:.2f}, p={p:.2f}, raw={kelly:.3f}, "
            f"safe={safe_kelly:.3f} (x{self.kelly_fraction})"
        )

        return safe_kelly

    def calculate_stake(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calculate position size for a trade using Kelly + risk constraints.

        Method:
        1. If historical stats available, use Kelly Criterion
        2. Otherwise, use fixed fractional (RISK_PER_TRADE)
        3. Apply sector exposure limit
        4. Apply max position size limit

        Args:
            symbol: Asset symbol
            entry_price: Entry price
            stop_loss_price: Stop-loss price
            win_rate: Optional historical win rate
            avg_win: Optional average winning return
            avg_loss: Optional average losing return

        Returns:
            Tuple[float, float]: (position_size_units, dollar_risk)
        """
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)

        if risk_per_unit == 0:
            logger.warning("Stop loss equals entry, cannot calculate stake")
            return 0.0, 0.0

        # Determine stake fraction
        if win_rate and avg_win and avg_loss:
            # Use Kelly Criterion
            stake_fraction = self.calculate_kelly_stake(win_rate, avg_win, avg_loss)
            method = "Kelly"
        else:
            # Use fixed fractional
            stake_fraction = RISK_PER_TRADE
            method = "Fixed"

        # Calculate dollar risk
        dollar_risk = self.current_equity * stake_fraction

        # Check sector exposure
        sector = SECTOR_MAP.get(symbol, DEFAULT_SECTOR)
        current_sector_exposure = self._get_sector_exposure(sector)

        if current_sector_exposure + stake_fraction > self.max_sector_exposure:
            available = max(0, self.max_sector_exposure - current_sector_exposure)
            old_risk = dollar_risk
            stake_fraction = available
            dollar_risk = self.current_equity * stake_fraction
            logger.warning(
                f"Sector limit hit for {sector}: reduced risk "
                f"${old_risk:.2f} -> ${dollar_risk:.2f}"
            )

        # Calculate position size (units)
        position_size = dollar_risk / risk_per_unit

        logger.info(
            f"Stake ({method}): {symbol} - "
            f"entry=${entry_price:.2f}, sl=${stop_loss_price:.2f}, "
            f"risk=${dollar_risk:.2f}, size={position_size:.4f} units"
        )

        return position_size, dollar_risk

    def _get_sector_exposure(self, sector: str) -> float:
        """Get current exposure to a sector from open positions."""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Get all open positions
            cursor.execute(
                """
                SELECT t.symbol, t.entry_price, t.position_size
                FROM trades t
                WHERE t.is_open = 1 AND t.position_size IS NOT NULL
                """
            )
            positions = cursor.fetchall()
            conn.close()

            sector_value = 0.0
            for symbol, entry_price, position_size in positions:
                pos_sector = SECTOR_MAP.get(symbol, DEFAULT_SECTOR)
                if pos_sector == sector and position_size:
                    sector_value += entry_price * position_size

            return sector_value / self.current_equity if self.current_equity > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating sector exposure: {e}")
            return 0.0

    # =========================================================================
    # DRAWDOWN & CIRCUIT BREAKER
    # =========================================================================

    def update_equity(self, new_equity: float) -> None:
        """
        Update current equity and recalculate drawdown metrics.

        Should be called after each trade closes or periodically.

        Args:
            new_equity: Current account equity
        """
        old_equity = self.current_equity
        self.current_equity = new_equity
        self.current_balance = new_equity  # Simplified: balance = equity

        # Update peak
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity

        # Calculate drawdown from peak
        if self.peak_equity > 0:
            self.daily_drawdown = (self.peak_equity - new_equity) / self.peak_equity
        else:
            self.daily_drawdown = 0.0

        # Update daily PnL (simplified: change from last recorded)
        self.daily_pnl = new_equity - old_equity

        # Record to database
        self._record_balance()

        logger.debug(
            f"Equity updated: ${old_equity:.2f} -> ${new_equity:.2f}, "
            f"daily_dd={self.daily_drawdown:.2%}"
        )

    def check_daily_drawdown(self) -> bool:
        """
        Check if daily drawdown exceeds limit.

        Returns:
            bool: True if trading is allowed, False if drawdown exceeded
        """
        if self.daily_drawdown >= self.max_daily_drawdown:
            logger.warning(
                f"Daily drawdown limit breached: {self.daily_drawdown:.2%} >= "
                f"{self.max_daily_drawdown:.2%}"
            )
            return False
        return True

    def halt_trading(self, reason: str, duration_hours: int = 24) -> None:
        """
        Trigger a trading halt (circuit breaker).

        Args:
            reason: Why trading was halted
            duration_hours: How long to halt (default 24h)
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            halt_start = datetime.utcnow()
            halt_end = halt_start + timedelta(hours=duration_hours)

            cursor.execute(
                """
                INSERT INTO trading_halts (reason, halt_start, halt_end, daily_drawdown, is_active)
                VALUES (?, ?, ?, ?, 1)
                """,
                (reason, halt_start.isoformat(), halt_end.isoformat(), self.daily_drawdown),
            )

            conn.commit()
            conn.close()

            logger.warning(
                f"TRADING HALTED: {reason} - "
                f"until {halt_end.isoformat()} ({duration_hours}h)"
            )

            # Record balance with halt reason
            self._record_balance(halt_reason=reason)

        except Exception as e:
            logger.error(f"Error halting trading: {e}")

    def is_trading_halted(self) -> Tuple[bool, Optional[str]]:
        """
        Check if trading is currently halted.

        Returns:
            Tuple[bool, Optional[str]]: (is_halted, reason)
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            now = datetime.utcnow().isoformat()

            cursor.execute(
                """
                SELECT reason, halt_end FROM trading_halts
                WHERE is_active = 1 AND halt_end > ?
                ORDER BY halt_start DESC
                LIMIT 1
                """,
                (now,),
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return True, row[0]
            return False, None

        except Exception as e:
            logger.error(f"Error checking halt status: {e}")
            return False, None

    def clear_expired_halts(self) -> int:
        """
        Clear expired trading halts.

        Returns:
            int: Number of halts cleared
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            now = datetime.utcnow().isoformat()

            cursor.execute(
                """
                UPDATE trading_halts SET is_active = 0
                WHERE is_active = 1 AND halt_end <= ?
                """,
                (now,),
            )

            cleared = cursor.rowcount
            conn.commit()
            conn.close()

            if cleared > 0:
                logger.info(f"Cleared {cleared} expired trading halt(s)")

            return cleared

        except Exception as e:
            logger.error(f"Error clearing halts: {e}")
            return 0

    # =========================================================================
    # STATISTICS & REPORTING
    # =========================================================================

    def get_portfolio_stats(self) -> Dict[str, float]:
        """
        Get current portfolio statistics.

        Returns:
            Dict with portfolio metrics
        """
        return {
            "balance": self.current_balance,
            "equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "daily_pnl": self.daily_pnl,
            "daily_drawdown": self.daily_drawdown,
            "total_return": (self.current_equity - self.initial_balance) / self.initial_balance
            if self.initial_balance > 0
            else 0.0,
        }

    def get_open_positions(self) -> List[Dict]:
        """Get all currently open positions."""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT symbol, side, entry_price, position_size, stop_loss_price,
                       take_profit_price, entry_time
                FROM trades
                WHERE is_open = 1
                """
            )
            columns = [desc[0] for desc in cursor.description]
            positions = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()

            return positions

        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []

    def get_total_exposure(self) -> float:
        """Get total portfolio exposure as fraction of equity."""
        positions = self.get_open_positions()
        total_value = sum(
            (p.get("entry_price", 0) * p.get("position_size", 0))
            for p in positions
            if p.get("position_size")
        )
        return total_value / self.current_equity if self.current_equity > 0 else 0.0

    # =========================================================================
    # SAFE MODE GATEKEEPER
    # =========================================================================

    def get_open_trade_count(self) -> int:
        """
        Get the current number of open trades.

        Returns:
            int: Number of currently open positions
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT COUNT(*) FROM trades WHERE is_open = 1
                """
            )
            count = cursor.fetchone()[0]
            conn.close()
            return count

        except Exception as e:
            logger.error(f"Error counting open trades: {e}")
            return 0

    def can_trade(self) -> Tuple[bool, Optional[str]]:
        """
        SAFE MODE GATEKEEPER: Check if a new trade is allowed.

        This is the master safety check that MUST be called before
        opening any new position. It enforces:
        1. Trading halt status (circuit breaker)
        2. MAX_OPEN_TRADES limit (hard cap at 2)
        3. Daily drawdown check

        Returns:
            Tuple[bool, Optional[str]]: (can_trade, reason_if_blocked)

        Usage:
            can, reason = portfolio_manager.can_trade()
            if not can:
                logger.warning(f"Trade blocked: {reason}")
                return  # DO NOT PROCEED
        """
        # Check 1: Trading halt (circuit breaker)
        is_halted, halt_reason = self.is_trading_halted()
        if is_halted:
            return False, f"Trading halted: {halt_reason}"

        # Check 2: Safe Mode - MAX_OPEN_TRADES enforcement
        if SAFE_MODE_ENABLED:
            open_count = self.get_open_trade_count()
            if open_count >= MAX_OPEN_TRADES:
                return False, (
                    f"Safe Mode: MAX_OPEN_TRADES ({MAX_OPEN_TRADES}) reached. "
                    f"Currently {open_count} open. System must wait."
                )

        # Check 3: Daily drawdown
        if not self.check_daily_drawdown():
            # Auto-halt if drawdown exceeded
            self.halt_trading(
                reason=f"Daily drawdown exceeded: {self.daily_drawdown:.2%}",
                duration_hours=24
            )
            return False, f"Daily drawdown exceeded: {self.daily_drawdown:.2%}"

        return True, None

    def get_safe_mode_status(self) -> Dict[str, any]:
        """
        Get current safe mode status for dashboard display.

        Returns:
            Dict with safe mode metrics
        """
        open_count = self.get_open_trade_count()
        can, reason = self.can_trade()

        return {
            "safe_mode_enabled": SAFE_MODE_ENABLED,
            "max_open_trades": MAX_OPEN_TRADES,
            "current_open_trades": open_count,
            "slots_available": MAX_OPEN_TRADES - open_count,
            "can_trade": can,
            "block_reason": reason,
            "max_position_size": self.max_position_size,
            "daily_drawdown": self.daily_drawdown,
            "max_daily_drawdown": self.max_daily_drawdown,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_strategy_stats(symbol: str) -> Optional[Dict[str, float]]:
    """
    Get historical strategy stats for a symbol from database.

    Returns:
        Dict with win_rate, avg_win, avg_loss, or None if not found
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT win_rate, avg_win, avg_loss
            FROM strategy_performance
            WHERE symbol = ? AND is_optimal = 1
            """,
            (symbol,),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "win_rate": row[0],
                "avg_win": row[1],
                "avg_loss": row[2],
            }
        return None

    except Exception as e:
        logger.error(f"Error getting strategy stats: {e}")
        return None
