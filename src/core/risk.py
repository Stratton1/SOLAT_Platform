"""
Risk Management Module for SOLAT

Handles position sizing, stop-loss calculation, and risk-per-trade constraints.
This module ensures that all trades adhere to money management principles and
never exceed the defined risk parameters.
"""

import logging
from typing import Tuple, Optional

from src.config.settings import RISK_PER_TRADE

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk Management Engine for Position Sizing and Stop-Loss Calculation

    This class implements core position sizing and risk management logic:
    - Calculate position size based on account equity and risk tolerance
    - Determine stop-loss levels using ATR (Average True Range)
    - Ensure no single position exceeds risk limits

    All calculations are denominated in base currency (USD for stocks, USDT for crypto).
    """

    def __init__(self, risk_per_trade: float = RISK_PER_TRADE) -> None:
        """
        Initialize the RiskManager.

        Args:
            risk_per_trade (float): Risk as fraction of account (default: 0.02 = 2%)

        Raises:
            ValueError: If risk_per_trade is not between 0 and 1
        """
        if not (0 < risk_per_trade < 1):
            raise ValueError(f"risk_per_trade must be between 0 and 1, got {risk_per_trade}")

        self.risk_per_trade = risk_per_trade
        logger.info(f"Initialized RiskManager (risk_per_trade={risk_per_trade:.2%})")

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Calculate the position size (number of units) for a trade.

        Position sizing formula:
            Position Size = (Account Balance * Risk %) / Risk Per Unit
            Risk Per Unit = abs(Entry Price - Stop Loss Price)

        Example:
            account_balance = 10,000 USD
            entry_price = 100.00 USD
            stop_loss_price = 95.00 USD
            risk_per_trade = 0.02 (2%)

            Risk Per Unit = abs(100.00 - 95.00) = 5.00
            Position Size = (10,000 * 0.02) / 5.00 = 200 / 5.00 = 40 units

            If price moves to stop loss:
            Loss = 40 units * 5.00 = 200 USD = 2% of account ✓

        Args:
            account_balance (float): Total account equity in USD/USDT
            entry_price (float): Entry price for the trade
            stop_loss_price (float): Stop-loss price

        Returns:
            float: Number of units to buy/sell (rounded down for safety)

        Raises:
            ValueError: If parameters are invalid
        """
        if account_balance <= 0:
            raise ValueError(f"account_balance must be positive, got {account_balance}")
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if stop_loss_price <= 0:
            raise ValueError(f"stop_loss_price must be positive, got {stop_loss_price}")
        if entry_price == stop_loss_price:
            raise ValueError("entry_price and stop_loss_price cannot be equal")

        # Calculate risk amount in currency (e.g., USD)
        risk_amount = account_balance * self.risk_per_trade

        # Calculate risk per unit (price distance from entry to stop)
        risk_per_unit = abs(entry_price - stop_loss_price)

        # Calculate position size
        position_size = risk_amount / risk_per_unit

        logger.debug(
            f"Position Size Calculation: "
            f"account={account_balance:.2f}, entry={entry_price:.2f}, "
            f"stop={stop_loss_price:.2f}, risk_amount={risk_amount:.2f}, "
            f"risk_per_unit={risk_per_unit:.2f}, position_size={position_size:.2f}"
        )

        return position_size

    def get_stop_loss(
        self,
        entry_price: float,
        side: str,
        atr_value: float
    ) -> float:
        """
        Calculate stop-loss price using ATR (Average True Range).

        Stop-loss formula:
            For BUY:  Stop Loss = Entry Price - (2 * ATR)
            For SELL: Stop Loss = Entry Price + (2 * ATR)

        The 2x ATR multiplier provides a buffer against normal volatility.
        This prevents whipsaws while keeping losses manageable.

        Example (BUY):
            entry_price = 100.00
            atr_value = 2.50
            stop_loss = 100.00 - (2 * 2.50) = 100.00 - 5.00 = 95.00

        Example (SELL):
            entry_price = 100.00
            atr_value = 2.50
            stop_loss = 100.00 + (2 * 2.50) = 100.00 + 5.00 = 105.00

        Args:
            entry_price (float): Entry price for the trade
            side (str): Trade side ("BUY" or "SELL")
            atr_value (float): Average True Range value (volatility measure)

        Returns:
            float: Stop-loss price (rounded to 2 decimal places for USD/USDT)

        Raises:
            ValueError: If parameters are invalid
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if atr_value <= 0:
            raise ValueError(f"atr_value must be positive, got {atr_value}")

        side_upper = side.upper()
        if side_upper not in ("BUY", "SELL"):
            raise ValueError(f"side must be 'BUY' or 'SELL', got {side}")

        if side_upper == "BUY":
            # For long trades: stop is below entry
            stop_loss = entry_price - (2 * atr_value)
        else:  # SELL
            # For short trades: stop is above entry
            stop_loss = entry_price + (2 * atr_value)

        # Round to 2 decimal places for USD/USDT precision
        stop_loss = round(stop_loss, 2)

        logger.debug(
            f"Stop-Loss Calculation [{side}]: "
            f"entry={entry_price:.2f}, atr={atr_value:.2f}, "
            f"stop_loss={stop_loss:.2f}"
        )

        return stop_loss

    def calculate_risk_reward(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float
    ) -> Tuple[float, float, float]:
        """
        Calculate risk-reward metrics for a trade.

        Returns a tuple of (risk_amount, reward_amount, risk_reward_ratio).

        Args:
            entry_price (float): Entry price
            stop_loss_price (float): Stop-loss price
            take_profit_price (float): Take-profit price

        Returns:
            Tuple[float, float, float]: (risk, reward, ratio)
                - risk: Price distance to stop loss
                - reward: Price distance to take profit
                - ratio: Reward / Risk (should be >= 1.5)
        """
        risk = abs(entry_price - stop_loss_price)
        reward = abs(take_profit_price - entry_price)

        if risk == 0:
            ratio = 0.0
        else:
            ratio = reward / risk

        logger.debug(
            f"Risk-Reward: entry={entry_price:.2f}, risk={risk:.2f}, "
            f"reward={reward:.2f}, ratio={ratio:.2f}"
        )

        return risk, reward, ratio
