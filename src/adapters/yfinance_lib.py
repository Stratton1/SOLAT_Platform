"""
YFinance adapter for fetching equity OHLCV data.

Handles the MultiIndex DataFrame issue in recent yfinance versions by
flattening the column hierarchy and normalizing to lowercase column names.

All data is returned with a UTC-aware DatetimeIndex and standard OHLCV columns.
"""

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from src.adapters.interface import MarketAdapter, resilient_request

logger = logging.getLogger(__name__)


class YFinanceAdapter(MarketAdapter):
    """
    Adapter for fetching stock/ETF data from YFinance.

    Handles:
    - MultiIndex DataFrame flattening (recent yfinance versions)
    - Column normalization to lowercase
    - Timezone normalization to UTC
    - Rate limiting via resilient_request decorator

    Example:
        >>> adapter = YFinanceAdapter()
        >>> df = adapter.get_ohlcv("AAPL", "1d", limit=100)
        >>> print(df.head())
        >>> print(df.dtypes)
    """

    def __init__(self) -> None:
        """Initialize the YFinance adapter."""
        self.source = "yfinance"

    @resilient_request
    def _fetch_data(
        self,
        symbol: str,
        period: str,
        interval: str
    ) -> pd.DataFrame:
        """
        Internal method to fetch data from yfinance.

        Args:
            symbol (str): Stock symbol (e.g., "AAPL")
            period (str): Period string (e.g., "1y", "6mo")
            interval (str): Interval (e.g., "1d", "1h", "5m")

        Returns:
            pd.DataFrame: Raw data from yfinance (may have MultiIndex)

        Raises:
            Exception: If fetch fails or symbol is invalid
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                raise ValueError(f"No data returned for symbol {symbol}")

            return df

        except Exception as e:
            logger.error(f"YFinance fetch error for {symbol}: {e}")
            raise

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize yfinance DataFrame to standard format.

        Handles:
        - MultiIndex column flattening
        - Column renaming to lowercase
        - Timezone normalization to UTC
        - Data type conversion to float64

        Args:
            df (pd.DataFrame): Raw DataFrame from yfinance

        Returns:
            pd.DataFrame: Normalized DataFrame with:
                - Index: UTC-aware DatetimeIndex
                - Columns: open, high, low, close, volume (lowercase, float64)
        """
        # Handle MultiIndex columns (recent yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            # Drop the top level (usually ticker symbol) to flatten
            df.columns = df.columns.droplevel(0)

        # Convert column names to lowercase
        df.columns = df.columns.str.lower()

        # Select only OHLCV columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        df = df[required_cols]

        # Ensure data types are float64
        df = df.astype("float64")

        # Ensure index is UTC-aware DatetimeIndex
        if df.index.tz is None:
            # Assume UTC if naive
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            # Convert to UTC if in another timezone
            df.index = df.index.tz_convert("UTC")

        # Remove any NaN rows (edge case during period conversions)
        df = df.dropna(how="all")

        logger.debug(
            f"Normalized DataFrame: {len(df)} rows, "
            f"index tz={df.index.tz}, dtypes={df.dtypes.to_dict()}"
        )

        return df

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from YFinance.

        Args:
            symbol (str): Stock symbol (e.g., "AAPL", "MSFT")
            timeframe (str): Timeframe (e.g., "1d", "1h", "5m")
            limit (int): Number of candles to fetch (approximate)

        Returns:
            pd.DataFrame: Normalized OHLCV data with:
                - Index: UTC-aware DatetimeIndex
                - Columns: open, high, low, close, volume (lowercase, float64)

        Raises:
            ValueError: If symbol is invalid or no data available
            Exception: If API request fails
        """
        # Map timeframe to yfinance interval and period
        timeframe_map = {
            "1m": ("1m", "5d"),
            "5m": ("5m", "30d"),
            "15m": ("15m", "60d"),
            "30m": ("30m", "90d"),
            "1h": ("1h", "730d"),
            "1d": ("1d", "10y"),
            "1w": ("1wk", "10y"),
            "1mo": ("1mo", "10y"),
        }

        if timeframe not in timeframe_map:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {list(timeframe_map.keys())}"
            )

        interval, period = timeframe_map[timeframe]

        logger.info(
            f"Fetching {symbol} {timeframe} (interval={interval}, period={period})"
        )

        # Fetch data from yfinance
        df = self._fetch_data(symbol, period=period, interval=interval)

        # Normalize to standard format
        df = self._normalize_dataframe(df)

        # Limit the number of rows if necessary
        if limit and len(df) > limit:
            df = df.tail(limit)

        logger.info(
            f"Successfully fetched {len(df)} candles for {symbol} {timeframe}"
        )

        return df
