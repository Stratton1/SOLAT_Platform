"""
IG Markets adapter for fetching spread betting OHLCV data.

IG Markets is a UK-based spread betting and CFD platform. Spread betting
profits are tax-free in the UK, making this an attractive option for
UK-based traders.

This adapter uses the trading-ig library to authenticate and fetch
historical price data via IG's REST API.

Important Notes:
- IG uses "EPICs" as symbol identifiers (e.g., CS.D.GBPUSD.TODAY.IP)
- Volume data may not be available for all instruments
- Rate limits apply - use resilient_request decorator
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11 fallback

try:
    from trading_ig import IGService
    from trading_ig.config import config as ig_config
    IG_AVAILABLE = True
except ImportError:
    IG_AVAILABLE = False
    IGService = None

from src.adapters.interface import MarketAdapter, resilient_request

logger = logging.getLogger(__name__)

# Path to secrets file
SECRETS_PATH = Path("src/config/secrets.toml")

# IG timeframe mapping: SOLAT timeframe -> IG resolution
TIMEFRAME_MAP: Dict[str, str] = {
    "1m": "MINUTE",
    "5m": "MINUTE_5",
    "15m": "MINUTE_15",
    "30m": "MINUTE_30",
    "1h": "HOUR",
    "2h": "HOUR_2",
    "4h": "HOUR_4",
    "1d": "DAY",
    "1w": "WEEK",
    "1mo": "MONTH",
}

# Common IG EPICs for reference
# These can be used in assets_seed.json with source="ig"
COMMON_EPICS: Dict[str, str] = {
    # Forex Majors
    "GBP/USD": "CS.D.GBPUSD.TODAY.IP",
    "EUR/USD": "CS.D.EURUSD.TODAY.IP",
    "USD/JPY": "CS.D.USDJPY.TODAY.IP",
    "GBP/JPY": "CS.D.GBPJPY.TODAY.IP",
    "EUR/GBP": "CS.D.EURGBP.TODAY.IP",
    # Indices
    "FTSE100": "IX.D.FTSE.DAILY.IP",
    "DAX": "IX.D.DAX.DAILY.IP",
    "S&P500": "IX.D.SPTRD.DAILY.IP",
    "DOW": "IX.D.DOW.DAILY.IP",
    "NASDAQ": "IX.D.NASDAQ.DAILY.IP",
    # Commodities
    "GOLD": "CS.D.USCGC.TODAY.IP",
    "SILVER": "CS.D.USCSI.TODAY.IP",
    "CRUDE_OIL": "CC.D.CL.UNC.IP",
    # Crypto (where available)
    "BTC/USD": "CS.D.BITCOIN.TODAY.IP",
    "ETH/USD": "CS.D.ETHUSD.TODAY.IP",
}


class IGAdapter(MarketAdapter):
    """
    Adapter for fetching OHLCV data from IG Markets (UK Spread Betting).

    Features:
    - Authenticates via IG REST API using trading-ig library
    - Maps SOLAT timeframes to IG resolutions
    - Supports both DEMO and LIVE accounts
    - Normalizes data to standard OHLCV format

    Tax Note:
        Spread betting profits in the UK are currently tax-free
        (no Capital Gains Tax or Stamp Duty). This makes IG an
        attractive option for UK-based algorithmic traders.

    Example:
        >>> adapter = IGAdapter()
        >>> df = adapter.get_ohlcv("CS.D.GBPUSD.TODAY.IP", "1h", limit=100)
        >>> print(df.head())

        # Or use common symbol aliases:
        >>> df = adapter.get_ohlcv("GBP/USD", "1d", limit=100)
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """
        Initialize the IG Markets adapter.

        Loads credentials from secrets.toml and establishes
        an authenticated session with IG's API.

        Args:
            config_path: Path to secrets.toml. Defaults to src/config/secrets.toml
        """
        self.source = "ig"
        self.ig_service: Optional[IGService] = None
        self._authenticated = False

        if not IG_AVAILABLE:
            logger.warning(
                "trading-ig not installed. Install with: pip install trading-ig"
            )
            return

        # Load configuration
        config_path = config_path or SECRETS_PATH
        self.config = self._load_config(config_path)

        if self.config:
            self._authenticate()

    def _load_config(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load IG credentials from secrets.toml.

        Args:
            config_path: Path to the secrets file

        Returns:
            Dict with IG credentials or None if not found
        """
        if not config_path.exists():
            logger.warning(
                f"Secrets file not found at {config_path}. "
                f"Copy secrets.toml.example to secrets.toml and add credentials."
            )
            return None

        try:
            with open(config_path, "rb") as f:
                secrets = tomllib.load(f)

            ig_config_data = secrets.get("ig", {})

            if not all(k in ig_config_data for k in ["username", "password", "api_key"]):
                logger.warning(
                    "Missing required IG credentials in secrets.toml. "
                    "Required: username, password, api_key"
                )
                return None

            return ig_config_data

        except Exception as e:
            logger.error(f"Error loading secrets.toml: {e}")
            return None

    def _authenticate(self) -> None:
        """
        Authenticate with IG Markets API.

        Creates an IGService instance and establishes a session.
        """
        if not self.config or not IG_AVAILABLE:
            return

        try:
            # Determine account type (DEMO or LIVE)
            acc_type = self.config.get("acc_type", "DEMO").upper()

            self.ig_service = IGService(
                username=self.config["username"],
                password=self.config["password"],
                api_key=self.config["api_key"],
                acc_type=acc_type,
            )

            # Create session (authenticate)
            self.ig_service.create_session()
            self._authenticated = True

            logger.info(f"Successfully authenticated with IG Markets ({acc_type})")

        except Exception as e:
            logger.error(f"IG authentication failed: {e}")
            self._authenticated = False

    def _resolve_epic(self, symbol: str) -> str:
        """
        Resolve a symbol to an IG EPIC.

        If the symbol is already an EPIC (contains periods), returns it directly.
        Otherwise, looks up the symbol in the COMMON_EPICS mapping.

        Args:
            symbol: Either an EPIC or a common symbol alias

        Returns:
            The IG EPIC string
        """
        # If it looks like an EPIC already (e.g., CS.D.GBPUSD.TODAY.IP)
        if "." in symbol and len(symbol.split(".")) >= 3:
            return symbol

        # Look up in common EPICs
        if symbol.upper() in COMMON_EPICS:
            return COMMON_EPICS[symbol.upper()]

        # Return as-is and let IG API handle the error
        return symbol

    @resilient_request
    def _fetch_historical_prices(
        self,
        epic: str,
        resolution: str,
        num_points: int
    ) -> pd.DataFrame:
        """
        Fetch historical price data from IG Markets.

        Args:
            epic: IG EPIC identifier
            resolution: IG resolution (e.g., 'HOUR', 'DAY')
            num_points: Number of data points to fetch

        Returns:
            Raw DataFrame from IG API
        """
        if not self._authenticated or not self.ig_service:
            raise RuntimeError(
                "Not authenticated with IG Markets. "
                "Check your credentials in secrets.toml."
            )

        try:
            response = self.ig_service.fetch_historical_prices_by_epic_and_num_points(
                epic=epic,
                resolution=resolution,
                numpoints=num_points,
            )

            # IG returns a nested structure with 'prices' DataFrame
            if hasattr(response, "prices"):
                return response.prices
            elif isinstance(response, dict) and "prices" in response:
                return pd.DataFrame(response["prices"])
            else:
                # Try to convert response directly
                return pd.DataFrame(response)

        except Exception as e:
            logger.error(f"IG fetch error for {epic}: {e}")
            raise

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize IG Markets DataFrame to standard OHLCV format.

        IG returns data in a nested structure with bid/ask prices.
        This method extracts the mid prices and normalizes columns.

        Args:
            df: Raw DataFrame from IG API

        Returns:
            Normalized DataFrame with standard OHLCV columns
        """
        if df.empty:
            return df

        # IG uses nested columns: bid.Open, ask.Open, etc.
        # We'll use the mid price for our OHLCV
        result_df = pd.DataFrame()

        # Try different column patterns IG might use
        try:
            # Pattern 1: Nested bid/ask structure
            if "bid" in df.columns or "openPrice" in df.columns:
                if "openPrice" in df.columns:
                    # Direct structure
                    result_df["open"] = df["openPrice"]["bid"]
                    result_df["high"] = df["highPrice"]["bid"]
                    result_df["low"] = df["lowPrice"]["bid"]
                    result_df["close"] = df["closePrice"]["bid"]
                elif "bid" in str(df.columns):
                    # Multi-level columns
                    result_df["open"] = (df[("bid", "Open")] + df[("ask", "Open")]) / 2
                    result_df["high"] = (df[("bid", "High")] + df[("ask", "High")]) / 2
                    result_df["low"] = (df[("bid", "Low")] + df[("ask", "Low")]) / 2
                    result_df["close"] = (df[("bid", "Close")] + df[("ask", "Close")]) / 2
            else:
                # Pattern 2: Simple column names
                col_mapping = {
                    "Open": "open", "open": "open",
                    "High": "high", "high": "high",
                    "Low": "low", "low": "low",
                    "Close": "close", "close": "close",
                    "Volume": "volume", "volume": "volume",
                }

                for orig, new in col_mapping.items():
                    if orig in df.columns:
                        result_df[new] = df[orig]

        except Exception as e:
            logger.warning(f"Column extraction error, trying fallback: {e}")
            # Fallback: try to extract any OHLCV-like columns
            df.columns = [str(c).lower() for c in df.columns]
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    result_df[col] = df[col]

        # Handle volume (may not be available for all IG instruments)
        if "volume" not in result_df.columns:
            if "lastTradedVolume" in df.columns:
                result_df["volume"] = df["lastTradedVolume"]
            else:
                # Set volume to 0 if not available
                result_df["volume"] = 0.0

        # Ensure all required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [c for c in required_cols if c not in result_df.columns]
        if missing_cols:
            raise ValueError(
                f"Could not extract required columns: {missing_cols}. "
                f"Available: {list(df.columns)}"
            )

        # Set index from timestamp
        if "snapshotTime" in df.columns:
            result_df.index = pd.to_datetime(df["snapshotTime"], utc=True)
        elif "snapshotTimeUTC" in df.columns:
            result_df.index = pd.to_datetime(df["snapshotTimeUTC"], utc=True)
        elif df.index.name in ["DateTime", "date", "timestamp", None]:
            result_df.index = pd.to_datetime(df.index, utc=True)

        # Ensure index is UTC-aware
        if result_df.index.tz is None:
            result_df.index = result_df.index.tz_localize("UTC")
        else:
            result_df.index = result_df.index.tz_convert("UTC")

        # Convert to float64
        result_df = result_df.astype("float64")

        # Sort by index (oldest first)
        result_df = result_df.sort_index()

        # Remove NaN rows
        result_df = result_df.dropna(how="all")

        logger.debug(
            f"Normalized IG DataFrame: {len(result_df)} rows, "
            f"index tz={result_df.index.tz}"
        )

        return result_df

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from IG Markets.

        Args:
            symbol: IG EPIC (e.g., "CS.D.GBPUSD.TODAY.IP") or common alias
                   (e.g., "GBP/USD", "GOLD", "FTSE100")
            timeframe: Timeframe (e.g., "1d", "1h", "4h")
            limit: Number of candles to fetch

        Returns:
            pd.DataFrame: Normalized OHLCV data with:
                - Index: UTC-aware DatetimeIndex
                - Columns: open, high, low, close, volume (lowercase, float64)

        Raises:
            ValueError: If timeframe is unsupported
            RuntimeError: If not authenticated
            Exception: If API request fails
        """
        if not IG_AVAILABLE:
            raise RuntimeError(
                "trading-ig not installed. Install with: pip install trading-ig"
            )

        if not self._authenticated:
            raise RuntimeError(
                "Not authenticated with IG Markets. "
                "Check your credentials in secrets.toml."
            )

        # Resolve symbol to EPIC
        epic = self._resolve_epic(symbol)

        # Map timeframe to IG resolution
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {list(TIMEFRAME_MAP.keys())}"
            )

        resolution = TIMEFRAME_MAP[timeframe]

        logger.info(
            f"Fetching {symbol} ({epic}) {timeframe} "
            f"(resolution={resolution}, limit={limit})"
        )

        # Fetch data from IG
        df = self._fetch_historical_prices(epic, resolution, limit)

        if df is None or df.empty:
            raise ValueError(f"No data returned for {epic}")

        # Normalize to standard format
        df = self._normalize_dataframe(df)

        # Limit rows if necessary
        if limit and len(df) > limit:
            df = df.tail(limit)

        logger.info(
            f"Successfully fetched {len(df)} candles for {symbol} {timeframe}"
        )

        return df

    def search_markets(self, search_term: str) -> pd.DataFrame:
        """
        Search IG Markets for instruments matching a term.

        Useful for finding EPICs for new instruments.

        Args:
            search_term: Search query (e.g., "GBPUSD", "Bitcoin")

        Returns:
            DataFrame with matching instruments and their EPICs
        """
        if not self._authenticated or not self.ig_service:
            raise RuntimeError("Not authenticated with IG Markets")

        try:
            response = self.ig_service.search_markets(search_term)

            if hasattr(response, "markets"):
                return response.markets
            elif isinstance(response, dict):
                return pd.DataFrame(response.get("markets", []))
            else:
                return pd.DataFrame(response)

        except Exception as e:
            logger.error(f"IG search error: {e}")
            raise

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get IG account information.

        Returns:
            Dict with account balance, equity, margin info, etc.
        """
        if not self._authenticated or not self.ig_service:
            raise RuntimeError("Not authenticated with IG Markets")

        try:
            accounts = self.ig_service.fetch_accounts()
            return accounts

        except Exception as e:
            logger.error(f"Error fetching IG account info: {e}")
            raise


# Convenience function for quick testing
def test_ig_connection() -> bool:
    """
    Test IG Markets connection and authentication.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        adapter = IGAdapter()
        if adapter._authenticated:
            logger.info("IG Markets connection test: SUCCESS")
            return True
        else:
            logger.warning("IG Markets connection test: FAILED (not authenticated)")
            return False
    except Exception as e:
        logger.error(f"IG Markets connection test: FAILED ({e})")
        return False
