"""
Abstract base class and decorators for market data adapters.

All adapters (YFinance, CCXT, etc.) inherit from MarketAdapter and implement
the get_ohlcv() method to fetch OHLCV data from their respective sources.

The resilient_request decorator handles rate limiting and transient failures
with exponential backoff.
"""

import time
import logging
from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, Any

import pandas as pd

logger = logging.getLogger(__name__)


def resilient_request(func: Callable) -> Callable:
    """
    Decorator for handling rate limiting and transient API failures.

    Implements exponential backoff: 1s → 2s → 4s → 8s
    Catches exceptions containing "429" or "Rate Limit" strings.

    Args:
        func: The function to decorate (typically an API call).

    Returns:
        Wrapped function that retries with exponential backoff.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        max_retries = 4
        base_sleep = 1

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).upper()
                is_rate_limit = "429" in error_msg or "RATE LIMIT" in error_msg

                if not is_rate_limit or attempt == max_retries - 1:
                    # Re-raise if not a rate limit error or we've exhausted retries
                    logger.error(f"Request failed: {error_msg}")
                    raise

                # Exponential backoff: 1s, 2s, 4s, 8s
                sleep_time = base_sleep * (2 ** attempt)
                logger.warning(
                    f"Rate limit detected. Retrying in {sleep_time}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_time)

        # Fallback (should not reach here due to raise in the loop)
        raise RuntimeError(f"Failed after {max_retries} attempts")

    return wrapper


class MarketAdapter(ABC):
    """
    Abstract base class for market data adapters.

    All adapters must implement get_ohlcv() and return a normalized DataFrame
    with the following structure:

    - **Index**: UTC-aware DatetimeIndex
    - **Columns**: open, high, low, close, volume (lowercase, float64)

    Example:
        >>> adapter = YFinanceAdapter()
        >>> df = adapter.get_ohlcv("AAPL", "1d", limit=100)
        >>> print(df.head())
        ...
        >>> print(df.dtypes)
        open      float64
        high      float64
        low       float64
        close     float64
        volume    float64
    """

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given symbol.

        Args:
            symbol (str): Market symbol (e.g., "AAPL" for stocks, "BTC/USDT" for crypto)
            timeframe (str): Timeframe (e.g., "1d", "1h", "15m")
            limit (int): Number of candles to fetch

        Returns:
            pd.DataFrame: OHLCV data with UTC DatetimeIndex and columns:
                         open, high, low, close, volume (all lowercase, float64)

        Raises:
            Exception: If data fetch fails or symbol is invalid
        """
        pass
