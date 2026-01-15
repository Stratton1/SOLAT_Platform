"""
Trade Execution Manager for SOLAT Platform.

This module provides:
1. TradeManager - Sophisticated order execution with split entries
2. TP1/TP2 - Partial take profits at fixed R:R and trailing stops
3. Trailing stop methods: Chandelier Exit, Kijun-sen based
4. Notifications: Email (smtplib) and Desktop (plyer)

The execution layer separates order management from signal generation,
enabling complex exit strategies and position management.
"""

import logging
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.database.repository import get_connection

logger = logging.getLogger(__name__)


# =============================================================================
# NOTIFICATION SETTINGS (loaded from environment)
# =============================================================================

SMTP_HOST = os.getenv("SOLAT_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SOLAT_SMTP_PORT", "587"))
SMTP_USER = os.getenv("SOLAT_SMTP_USER")
SMTP_PASSWORD = os.getenv("SOLAT_SMTP_PASSWORD")
NOTIFICATION_EMAIL = os.getenv("SOLAT_NOTIFICATION_EMAIL")
ENABLE_DESKTOP = os.getenv("SOLAT_DESKTOP_NOTIFY", "true").lower() == "true"

# Try to import plyer for desktop notifications
try:
    from plyer import notification as desktop_notifier

    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    logger.warning("plyer not installed - desktop notifications disabled")


# =============================================================================
# TRADE MANAGER
# =============================================================================


class TradeManager:
    """
    Sophisticated trade execution with split entries and notifications.

    Features:
    - Split entry: Divide position into 2 units with different exit strategies
    - TP1 (Unit 1): Fixed R:R take-profit (default 1.5R)
    - TP2 (Unit 2): Trailing stop (Chandelier Exit or Kijun-sen)
    - Email and desktop notifications for trade events

    Usage:
        tm = TradeManager()

        # Execute split entry
        trade_ids = tm.execute_split_entry(
            symbol="BTC/USDT",
            side="BUY",
            entry_price=50000,
            stop_loss_price=48000,
            total_position_size=0.1,
            strategy_name="ichimoku_standard"
        )

        # Later, update trailing stops
        for trade_id in trade_ids:
            tm.update_trailing_stop(trade_id, current_price=52000)

        # Check exits
        tm.check_exit_conditions(trade_id, current_price, df)
    """

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        notification_email: Optional[str] = None,
        enable_desktop_notifications: bool = True,
        tp1_rr_ratio: float = 1.5,
        trailing_atr_multiple: float = 3.0,
    ) -> None:
        """
        Initialize TradeManager.

        Args:
            smtp_host: SMTP server host (default from env)
            smtp_port: SMTP server port (default 587)
            smtp_user: SMTP username/email (default from env)
            smtp_password: SMTP password (default from env)
            notification_email: Email to send alerts to (default from env)
            enable_desktop_notifications: Whether to show desktop popups
            tp1_rr_ratio: Risk:Reward ratio for TP1 (default 1.5)
            trailing_atr_multiple: ATR multiple for Chandelier Exit (default 3.0)
        """
        # SMTP settings (prefer args, fall back to env)
        self.smtp_host = smtp_host or SMTP_HOST
        self.smtp_port = smtp_port or SMTP_PORT
        self.smtp_user = smtp_user or SMTP_USER
        self.smtp_password = smtp_password or SMTP_PASSWORD
        self.notification_email = notification_email or NOTIFICATION_EMAIL

        # Desktop notifications
        self.enable_desktop = enable_desktop_notifications and PLYER_AVAILABLE and ENABLE_DESKTOP

        # Trade parameters
        self.tp1_rr_ratio = tp1_rr_ratio
        self.trailing_atr_multiple = trailing_atr_multiple

        # Check email configuration
        self.email_configured = all(
            [self.smtp_host, self.smtp_user, self.smtp_password, self.notification_email]
        )

        logger.info(
            f"TradeManager initialized: email={'configured' if self.email_configured else 'not configured'}, "
            f"desktop={'enabled' if self.enable_desktop else 'disabled'}, "
            f"TP1={tp1_rr_ratio}R, trailing_ATR={trailing_atr_multiple}x"
        )

    # =========================================================================
    # SPLIT ENTRY EXECUTION
    # =========================================================================

    def execute_split_entry(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss_price: float,
        total_position_size: float,
        strategy_name: str,
    ) -> List[int]:
        """
        Execute a split entry: divide position into 2 units with different exits.

        Unit 1 (50%): Fixed TP at tp1_rr_ratio (e.g., 1.5R)
        Unit 2 (50%): Trailing stop (Chandelier Exit)

        Args:
            symbol: Asset symbol
            side: 'BUY' or 'SELL'
            entry_price: Entry price
            stop_loss_price: Initial stop-loss price
            total_position_size: Total position size (units)
            strategy_name: Strategy that generated the signal

        Returns:
            List[int]: Trade IDs for both units [unit1_id, unit2_id]
        """
        # Split position 50/50
        unit1_size = total_position_size * 0.5
        unit2_size = total_position_size * 0.5

        # Calculate TP1 for Unit 1
        tp1_price = self.calculate_tp1(entry_price, stop_loss_price, side)

        # Unit 2 uses trailing stop (no fixed TP)
        trailing_stop = stop_loss_price  # Start at initial SL

        trade_ids = []
        now = datetime.utcnow().isoformat()

        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Insert Unit 1 (fixed TP)
            cursor.execute(
                """
                INSERT INTO trades
                (symbol, side, entry_price, stop_loss_price, take_profit_price,
                 position_size, strategy_name, unit_number, trailing_stop_price,
                 is_open, entry_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    side,
                    entry_price,
                    stop_loss_price,
                    tp1_price,
                    unit1_size,
                    strategy_name,
                    1,  # Unit 1
                    None,  # No trailing for unit 1
                    1,  # is_open = True
                    now,
                ),
            )
            unit1_id = cursor.lastrowid
            trade_ids.append(unit1_id)

            # Insert Unit 2 (trailing stop)
            cursor.execute(
                """
                INSERT INTO trades
                (symbol, side, entry_price, stop_loss_price, take_profit_price,
                 position_size, strategy_name, unit_number, trailing_stop_price,
                 is_open, entry_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    side,
                    entry_price,
                    stop_loss_price,
                    None,  # No fixed TP for unit 2
                    unit2_size,
                    strategy_name,
                    2,  # Unit 2
                    trailing_stop,
                    1,  # is_open = True
                    now,
                ),
            )
            unit2_id = cursor.lastrowid
            trade_ids.append(unit2_id)

            conn.commit()
            conn.close()

            logger.info(
                f"Split entry executed: {symbol} {side} @ {entry_price:.2f}\n"
                f"  Unit 1 (ID={unit1_id}): size={unit1_size:.4f}, TP={tp1_price:.2f}\n"
                f"  Unit 2 (ID={unit2_id}): size={unit2_size:.4f}, trailing SL={trailing_stop:.2f}"
            )

            # Send notification
            self.notify_trade_entry(
                {
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "stop_loss_price": stop_loss_price,
                    "total_size": total_position_size,
                    "tp1_price": tp1_price,
                    "strategy": strategy_name,
                    "trade_ids": trade_ids,
                }
            )

            return trade_ids

        except Exception as e:
            logger.error(f"Error executing split entry: {e}")
            return []

    def calculate_tp1(
        self,
        entry_price: float,
        stop_loss_price: float,
        side: str,
        rr_ratio: Optional[float] = None,
    ) -> float:
        """
        Calculate fixed R:R take-profit price for TP1.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop-loss price
            side: 'BUY' or 'SELL'
            rr_ratio: Risk:Reward ratio (default tp1_rr_ratio)

        Returns:
            float: Take-profit price
        """
        rr = rr_ratio or self.tp1_rr_ratio
        risk = abs(entry_price - stop_loss_price)
        reward = risk * rr

        if side.upper() == "BUY":
            tp_price = entry_price + reward
        else:
            tp_price = entry_price - reward

        return tp_price

    # =========================================================================
    # TRAILING STOP MANAGEMENT
    # =========================================================================

    def update_trailing_stop(
        self,
        trade_id: int,
        current_price: float,
        df: Optional[pd.DataFrame] = None,
        method: str = "chandelier",
    ) -> Optional[float]:
        """
        Update trailing stop for a trade.

        Methods:
        - "chandelier": ATR-based trailing (3x ATR from high/low)
        - "kijun": Trail below/above Kijun-sen line

        Args:
            trade_id: Trade ID to update
            current_price: Current market price
            df: OHLCV DataFrame (required for ATR/Kijun calculation)
            method: Trailing method ("chandelier" or "kijun")

        Returns:
            float: New stop-loss price, or None if no update
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Get trade details
            cursor.execute(
                """
                SELECT symbol, side, entry_price, trailing_stop_price
                FROM trades
                WHERE id = ? AND is_open = 1 AND unit_number = 2
                """,
                (trade_id,),
            )
            row = cursor.fetchone()

            if not row:
                conn.close()
                return None

            symbol, side, entry_price, current_trailing = row
            side = side.upper()

            # Calculate new trailing stop
            if method == "chandelier" and df is not None:
                new_stop = self._calculate_chandelier_stop(df, side)
            elif method == "kijun" and df is not None:
                new_stop = self._calculate_kijun_stop(df, side)
            else:
                # Simple percentage trailing (fallback)
                trail_pct = 0.02  # 2% trail
                if side == "BUY":
                    new_stop = current_price * (1 - trail_pct)
                else:
                    new_stop = current_price * (1 + trail_pct)

            # Only update if it's a tighter stop (favorable direction)
            should_update = False
            if current_trailing is None:
                should_update = True
            elif side == "BUY" and new_stop > current_trailing:
                should_update = True  # Raise stop for long
            elif side == "SELL" and new_stop < current_trailing:
                should_update = True  # Lower stop for short

            if should_update:
                cursor.execute(
                    """
                    UPDATE trades SET trailing_stop_price = ?
                    WHERE id = ?
                    """,
                    (new_stop, trade_id),
                )
                conn.commit()
                logger.debug(
                    f"Trailing stop updated: trade={trade_id}, "
                    f"{current_trailing} -> {new_stop:.2f} ({method})"
                )
            else:
                new_stop = current_trailing

            conn.close()
            return new_stop

        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
            return None

    def _calculate_chandelier_stop(self, df: pd.DataFrame, side: str) -> float:
        """Calculate Chandelier Exit stop (ATR-based trailing)."""
        # Calculate ATR
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=22).mean().iloc[-1]

        if pd.isna(atr):
            atr = (df["high"] - df["low"]).mean()

        atr_multiple = self.trailing_atr_multiple

        if side == "BUY":
            # Long: trail below highest high
            highest_high = df["high"].rolling(22).max().iloc[-1]
            stop = highest_high - (atr * atr_multiple)
        else:
            # Short: trail above lowest low
            lowest_low = df["low"].rolling(22).min().iloc[-1]
            stop = lowest_low + (atr * atr_multiple)

        return stop

    def _calculate_kijun_stop(self, df: pd.DataFrame, side: str) -> float:
        """Calculate Kijun-sen based trailing stop."""
        # Kijun-sen = (26-period high + 26-period low) / 2
        high_26 = df["high"].rolling(26).max().iloc[-1]
        low_26 = df["low"].rolling(26).min().iloc[-1]
        kijun = (high_26 + low_26) / 2

        # Add buffer (1% beyond Kijun)
        buffer = 0.01

        if side == "BUY":
            stop = kijun * (1 - buffer)  # Below Kijun for longs
        else:
            stop = kijun * (1 + buffer)  # Above Kijun for shorts

        return stop

    # =========================================================================
    # EXIT CONDITION CHECKING
    # =========================================================================

    def check_exit_conditions(
        self,
        trade_id: int,
        current_price: float,
    ) -> Optional[Tuple[str, float]]:
        """
        Check if a trade should be closed.

        Checks:
        1. Stop-loss hit (including trailing)
        2. Take-profit hit (Unit 1 only)

        Args:
            trade_id: Trade ID to check
            current_price: Current market price

        Returns:
            Tuple[str, float]: (exit_reason, exit_price) if should close, None otherwise
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT symbol, side, stop_loss_price, take_profit_price,
                       trailing_stop_price, unit_number
                FROM trades
                WHERE id = ? AND is_open = 1
                """,
                (trade_id,),
            )
            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            symbol, side, stop_loss, take_profit, trailing_stop, unit_number = row
            side = side.upper()

            # Determine effective stop-loss (use trailing if set)
            effective_sl = trailing_stop if trailing_stop else stop_loss

            # Check stop-loss
            if side == "BUY" and current_price <= effective_sl:
                return ("Stop-loss hit" if not trailing_stop else "Trailing stop hit", effective_sl)
            elif side == "SELL" and current_price >= effective_sl:
                return ("Stop-loss hit" if not trailing_stop else "Trailing stop hit", effective_sl)

            # Check take-profit (only for Unit 1)
            if take_profit and unit_number == 1:
                if side == "BUY" and current_price >= take_profit:
                    return (f"TP1 hit ({self.tp1_rr_ratio}R)", take_profit)
                elif side == "SELL" and current_price <= take_profit:
                    return (f"TP1 hit ({self.tp1_rr_ratio}R)", take_profit)

            return None

        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return None

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[float]:
        """
        Close a trade and calculate PnL.

        Args:
            trade_id: Trade ID to close
            exit_price: Exit price
            exit_reason: Why the trade was closed

        Returns:
            float: PnL in percentage, or None on error
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Get trade details
            cursor.execute(
                """
                SELECT symbol, side, entry_price, position_size
                FROM trades
                WHERE id = ? AND is_open = 1
                """,
                (trade_id,),
            )
            row = cursor.fetchone()

            if not row:
                conn.close()
                logger.warning(f"Trade {trade_id} not found or already closed")
                return None

            symbol, side, entry_price, position_size = row
            side = side.upper()

            # Calculate PnL
            if side == "BUY":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            pnl_dollar = pnl_pct * entry_price * (position_size or 1)

            # Update trade record
            now = datetime.utcnow().isoformat()
            cursor.execute(
                """
                UPDATE trades
                SET exit_price = ?, pnl = ?, exit_reason = ?, exit_time = ?, is_open = 0
                WHERE id = ?
                """,
                (exit_price, pnl_dollar, exit_reason, now, trade_id),
            )

            conn.commit()
            conn.close()

            logger.info(
                f"Trade {trade_id} closed: {symbol} {side} @ {exit_price:.2f}, "
                f"PnL={pnl_pct:.2%} (${pnl_dollar:.2f}), reason={exit_reason}"
            )

            # Send notification
            self.notify_trade_exit(
                {
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "pnl_dollar": pnl_dollar,
                    "reason": exit_reason,
                }
            )

            return pnl_pct

        except Exception as e:
            logger.error(f"Error closing trade: {e}")
            return None

    # =========================================================================
    # NOTIFICATIONS
    # =========================================================================

    def send_email_notification(
        self,
        subject: str,
        body: str,
        recipient: Optional[str] = None,
    ) -> bool:
        """
        Send email notification via SMTP.

        Args:
            subject: Email subject
            body: Email body (plain text)
            recipient: Override recipient email

        Returns:
            bool: True if sent successfully
        """
        if not self.email_configured:
            logger.debug("Email not configured, skipping notification")
            return False

        try:
            to_email = recipient or self.notification_email

            msg = MIMEMultipart()
            msg["From"] = self.smtp_user
            msg["To"] = to_email
            msg["Subject"] = f"[SOLAT] {subject}"

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_desktop_notification(
        self,
        title: str,
        message: str,
        timeout: int = 10,
    ) -> bool:
        """
        Send desktop notification via plyer.

        Args:
            title: Notification title
            message: Notification message
            timeout: Display duration in seconds

        Returns:
            bool: True if shown successfully
        """
        if not self.enable_desktop:
            logger.debug("Desktop notifications disabled")
            return False

        try:
            desktop_notifier.notify(
                title=f"SOLAT: {title}",
                message=message,
                app_name="SOLAT",
                timeout=timeout,
            )
            logger.debug(f"Desktop notification: {title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send desktop notification: {e}")
            return False

    def notify_trade_entry(self, trade_data: Dict[str, Any]) -> None:
        """Send notifications for new trade entry."""
        symbol = trade_data.get("symbol", "Unknown")
        side = trade_data.get("side", "Unknown")
        entry = trade_data.get("entry_price", 0)
        sl = trade_data.get("stop_loss_price", 0)
        tp1 = trade_data.get("tp1_price", 0)
        size = trade_data.get("total_size", 0)
        strategy = trade_data.get("strategy", "Unknown")

        # Desktop notification
        self.send_desktop_notification(
            title=f"Trade Entry: {symbol}",
            message=f"{side} @ ${entry:.2f}\nSize: {size:.4f}\nSL: ${sl:.2f} | TP1: ${tp1:.2f}",
        )

        # Email notification
        body = f"""
SOLAT Trade Alert - New Position

Symbol: {symbol}
Side: {side}
Entry Price: ${entry:.2f}
Position Size: {size:.4f}
Stop Loss: ${sl:.2f}
Take Profit 1: ${tp1:.2f}
Strategy: {strategy}

Trade IDs: {trade_data.get('trade_ids', [])}

This is an automated alert from SOLAT Trading System.
        """
        self.send_email_notification(f"Trade Entry: {symbol} {side}", body.strip())

    def notify_trade_exit(self, trade_data: Dict[str, Any]) -> None:
        """Send notifications for trade exit."""
        symbol = trade_data.get("symbol", "Unknown")
        side = trade_data.get("side", "Unknown")
        entry = trade_data.get("entry_price", 0)
        exit_price = trade_data.get("exit_price", 0)
        pnl_pct = trade_data.get("pnl_pct", 0)
        pnl_dollar = trade_data.get("pnl_dollar", 0)
        reason = trade_data.get("reason", "Unknown")

        # Determine emoji
        emoji = "✅" if pnl_pct > 0 else "❌"

        # Desktop notification
        self.send_desktop_notification(
            title=f"Trade Exit: {symbol} {emoji}",
            message=f"PnL: {pnl_pct:+.2%} (${pnl_dollar:+.2f})\n{reason}",
        )

        # Email notification
        body = f"""
SOLAT Trade Alert - Position Closed

Symbol: {symbol}
Side: {side}
Entry Price: ${entry:.2f}
Exit Price: ${exit_price:.2f}

P&L: {pnl_pct:+.2%} (${pnl_dollar:+.2f})
Exit Reason: {reason}

Trade ID: {trade_data.get('trade_id', 'N/A')}

This is an automated alert from SOLAT Trading System.
        """
        self.send_email_notification(
            f"Trade Exit: {symbol} {'+' if pnl_pct > 0 else ''}{pnl_pct:.1%}",
            body.strip(),
        )

    def notify_halt(self, reason: str, duration_hours: int) -> None:
        """Send notification for trading halt."""
        self.send_desktop_notification(
            title="⛔ TRADING HALTED",
            message=f"{reason}\nDuration: {duration_hours}h",
            timeout=30,
        )

        body = f"""
SOLAT ALERT - Trading Halted

Reason: {reason}
Duration: {duration_hours} hours

Trading has been automatically suspended due to risk management rules.
The system will resume trading after the halt period expires.

This is an automated alert from SOLAT Trading System.
        """
        self.send_email_notification("⛔ TRADING HALTED", body.strip())
