"""
CCXT adapter for fetching cryptocurrency OHLCV data.

Handles normalization of raw CCXT OHLCV data (list of lists) into
a standard DataFrame format with UTC-aware DatetimeIndex.

Supports any CCXT exchange. Default is Binance (free tier).
"""

import logging
from typing import Optional

import pandas as pd
import ccxt

from src.adapters.interface import MarketAdapter, resilient_request

logger = logging.getLogger(__name__)


class CCXTAdapter(MarketAdapter):
    """
    Adapter for fetching cryptocurrency OHLCV data via CCXT.

    Handles:
    - Raw CCXT OHLCV data conversion (list of lists → DataFrame)
    - Timestamp normalization (milliseconds → UTC DatetimeIndex)
    - Column normalization to lowercase
    - Support for any CCXT exchange

    Default exchange: Binance (free tier)

    Example:
        >>> adapter = CCXTAdapter()
        >>> df = adapter.get_ohlcv("BTC/USDT", "1h", limit=100)
        >>> print(df.head())
        >>> print(df.dtypes)
    """

    def __init__(self, exchange_name: str = "binance") -> None:
        """
        Initialize the CCXT adapter.

        Args:
            exchange_name (str): CCXT exchange name (default: "binance")
                                 Must be a valid CCXT exchange ID.

        Raises:
            ValueError: If exchange is not supported by CCXT
        """
        # Validate and initialize exchange
        if exchange_name.lower() not in ccxt.exchanges:
            raise ValueError(
                f"Exchange '{exchange_name}' not supported. "
                f"Available: {ccxt.exchanges[:10]}... ({len(ccxt.exchanges)} total)"
            )

        # Initialize the exchange
        exchange_class = getattr(ccxt, exchange_name.lower())
        self.exchange = exchange_class()
        self.source = exchange_name

        logger.info(f"Initialized CCXT adapter for {exchange_name}")

    @resilient_request
    def _fetch_raw_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> list:
        """
        Internal method to fetch raw OHLCV data from CCXT.

        CCXT returns OHLCV as: [timestamp_ms, open, high, low, close, volume, ...]

        Args:
            symbol (str): Trading pair (e.g., "BTC/USDT")
            timeframe (str): Timeframe (e.g., "1h", "15m", "1d")
            limit (int): Number of candles to fetch

        Returns:
            list: List of OHLCV candles from CCXT

        Raises:
            Exception: If symbol is invalid or network error
        """
        try:
            if not self.exchange.has["fetchOHLCV"]:
                raise ValueError(
                    f"Exchange {self.source} does not support fetchOHLCV"
                )

            # Fetch OHLCV data from exchange
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv:
                raise ValueError(f"No data returned for {symbol} {timeframe}")

            logger.debug(f"Fetched {len(ohlcv)} candles for {symbol} {timeframe}")
            return ohlcv

        except Exception as e:
            logger.error(f"CCXT fetch error for {symbol}: {e}")
            raise

    def _convert_to_dataframe(
        self,
        ohlcv: list,
        symbol: str
    ) -> pd.DataFrame:
        """
        Convert raw CCXT OHLCV data to normalized DataFrame.

        CCXT format: [[timestamp_ms, open, high, low, close, volume], ...]

        Returns DataFrame with:
        - Index: UTC-aware DatetimeIndex (converted from milliseconds)
        - Columns: open, high, low, close, volume (lowercase, float64)

        Args:
            ohlcv (list): Raw OHLCV data from CCXT
            symbol (str): Trading pair (for logging)

        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        if not ohlcv:
            raise ValueError(f"Empty OHLCV data for {symbol}")

        # Extract timestamp (first column) and OHLCV data (next 5 columns)
        timestamps = [candle[0] for candle in ohlcv]
        data = {
            "open": [candle[1] for candle in ohlcv],
            "high": [candle[2] for candle in ohlcv],
            "low": [candle[3] for candle in ohlcv],
            "close": [candle[4] for candle in ohlcv],
            "volume": [candle[5] for candle in ohlcv],
        }

        # Create DataFrame
        df = pd.DataFrame(data)

        # Convert timestamps from milliseconds to UTC DatetimeIndex
        df.index = pd.to_datetime(timestamps, unit="ms", utc=True)

        # Ensure data types are float64
        df = df.astype("float64")

        logger.debug(
            f"Converted {len(df)} candles: "
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
        Fetch and normalize OHLCV data from CCXT.

        Args:
            symbol (str): Trading pair (e.g., "BTC/USDT", "ETH/USDT")
            timeframe (str): Timeframe (e.g., "1h", "15m", "1d")
            limit (int): Number of candles to fetch

        Returns:
            pd.DataFrame: Normalized OHLCV data with:
                - Index: UTC-aware DatetimeIndex
                - Columns: open, high, low, close, volume (lowercase, float64)

        Raises:
            ValueError: If symbol is invalid or timeframe not supported
            Exception: If API request fails
        """
        logger.info(
            f"Fetching {symbol} {timeframe} from {self.source} (limit={limit})"
        )

        # Fetch raw OHLCV data
        ohlcv = self._fetch_raw_ohlcv(symbol, timeframe, limit)

        # Convert to DataFrame
        df = self._convert_to_dataframe(ohlcv, symbol)

        logger.info(
            f"Successfully fetched {len(df)} candles for {symbol} {timeframe}"
        )

        return df
