"""
SOLAT Dashboard - Mission Control Center

Traffic Light regime display with sidebar-based multi-page navigation.
- 🏠 Mission Control: Home with traffic light regime header
- 🔬 Market Analyzer: Detailed surveillance and charting
- 🧠 The Brain: HMM regime visualization
- ⚙️ Settings: Configuration and risk management

WARNING: This dashboard is READ-ONLY and connects to the database in read-only mode.
"""

import logging
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dashboard components
from dashboard.views import (
    load_custom_css,
    render_traffic_light_header,
    render_ticker_header,
    render_metric_cards,
    render_surveillance_table,
    render_strategy_explanation,
    render_evolution_metrics,
    render_trades_summary,
    render_debug_info,
)
from dashboard.charts import render_ichimoku_chart
from src.adapters.ccxt_lib import CCXTAdapter
from src.adapters.yfinance_lib import YFinanceAdapter

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SOLAT Mission Control",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
load_custom_css()

# ============================================================================
# AUTO-REFRESH
# ============================================================================

# Auto-refresh every 30 seconds
count = st_autorefresh(interval=30000)  # 30000 ms = 30 seconds

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================


@st.cache_resource
def get_db_connection():
    """
    Get a read-only SQLite connection.

    Uses URI mode with mode=ro to prevent locking issues.
    """
    try:
        conn = sqlite3.connect(
            "file:data/db/trading_engine.db?mode=ro",
            uri=True,
            timeout=5,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None


def query_market_snapshots():
    """
    Query market snapshots from the database with regime data.

    Returns:
        pd.DataFrame: Market snapshots joined with asset info and regime
    """
    try:
        conn = get_db_connection()
        if not conn:
            return None

        query = """
            SELECT
                m.symbol,
                a.source,
                a.status,
                a.fitness_score,
                m.close_price,
                m.cloud_status,
                m.chikou_conf,
                m.regime,
                m.updated_at
            FROM market_snapshots m
            JOIN assets a ON m.symbol = a.symbol
            ORDER BY m.updated_at DESC
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df is not None and not df.empty:
            df["updated_at"] = pd.to_datetime(df["updated_at"])
        return df

    except Exception as e:
        logger.error(f"Error querying market snapshots: {e}")
        return None


def query_assets():
    """
    Query all assets from the database.

    Returns:
        pd.DataFrame: All assets with their current status and fitness
    """
    try:
        conn = get_db_connection()
        if not conn:
            return None

        query = """
            SELECT symbol, source, status, fitness_score, last_scan
            FROM assets
            ORDER BY fitness_score DESC
        """

        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    except Exception as e:
        logger.error(f"Error querying assets: {e}")
        return None


def query_trades(limit: int = 20):
    """
    Query recent trades from the database.

    Args:
        limit (int): Maximum number of trades to return

    Returns:
        pd.DataFrame: Recent trades
    """
    try:
        conn = get_db_connection()
        if not conn:
            return None

        query = f"""
            SELECT symbol, side, entry_price, pnl, exit_reason, entry_time
            FROM trades
            ORDER BY entry_time DESC
            LIMIT {limit}
        """

        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    except Exception as e:
        logger.error(f"Error querying trades: {e}")
        return None


def get_current_regime():
    """
    Get the most recent market regime from database.

    Returns:
        str: Current regime ('bull', 'bear', 'chop', or 'neutral')
    """
    try:
        conn = get_db_connection()
        if not conn:
            return "neutral"

        query = """
            SELECT regime FROM market_snapshots
            ORDER BY updated_at DESC
            LIMIT 1
        """

        result = conn.execute(query).fetchone()
        conn.close()

        return result[0] if result else "neutral"

    except Exception as e:
        logger.error(f"Error fetching regime: {e}")
        return "neutral"


def fetch_ohlcv_for_chart(symbol: str, source: str, timeframe: str = "1h"):
    """
    Fetch OHLCV data for charting only.

    This is the ONE exception to the "no API calls" rule.
    We fetch historical candles ONLY when user selects a specific asset to chart.

    Args:
        symbol (str): Asset symbol
        source (str): Data source (ccxt or yfinance)
        timeframe (str): Timeframe

    Returns:
        pd.DataFrame: OHLCV data with Ichimoku indicators
    """
    try:
        if source.lower() == "ccxt":
            adapter = CCXTAdapter()
            df = adapter.get_ohlcv(symbol, timeframe, limit=100)
        elif source.lower() == "yfinance":
            adapter = YFinanceAdapter()
            df = adapter.get_ohlcv(symbol, timeframe, limit=100)
        else:
            logger.error(f"Unknown source: {source}")
            return None

        # Calculate Ichimoku indicators
        from src.core.ichimoku import IchimokuStrategy

        strategy = IchimokuStrategy()
        df_with_ichimoku = strategy._calculate_ichimoku(df)

        return df_with_ichimoku

    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        return None


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================


def render_navigation():
    """Render the sidebar navigation menu."""
    with st.sidebar:
        st.title("🤖 SOLAT Command Center")

        # Navigation pages
        page = st.radio(
            "Navigation",
            options=[
                "🏠 Mission Control",
                "🔬 Market Analyzer",
                "🧠 The Brain",
                "⚙️ Settings"
            ],
            label_visibility="collapsed"
        )

        st.divider()

        # Market status
        st.subheader("Market Status")
        current_regime = get_current_regime()

        if current_regime == "bull":
            st.success("🐂 BULLISH TREND - Long trades enabled", icon="✅")
        elif current_regime == "bear":
            st.error("🐻 BEARISH TREND - Short trades enabled", icon="⚠️")
        elif current_regime == "chop":
            st.warning("🦀 CHOPPY/VOLATILE - Trading paused", icon="⛔")
        else:
            st.info("⚪ NEUTRAL - Awaiting training data", icon="ℹ️")

        st.divider()

        # System info
        st.subheader("System")
        st.caption("""
        - **Status**: 🟢 Running
        - **Mode**: Read-Only
        - **Database**: SQLite (WAL)
        - **Refresh**: 30 sec
        """)

        return page


# ============================================================================
# PAGE FUNCTIONS
# ============================================================================


def page_mission_control():
    """🏠 Mission Control - Home page with traffic light regime header."""

    # Traffic light header
    current_regime = get_current_regime()
    render_traffic_light_header(current_regime)

    st.divider()

    # Query data
    snapshots_df = query_market_snapshots()

    if snapshots_df is None or snapshots_df.empty:
        st.warning("⏳ Waiting for first market scan... (Sentinel running in background)")
        st.info("The dashboard will update automatically every 30 seconds")
    else:
        # Show only signals
        active_signals = snapshots_df[snapshots_df["chikou_conf"] != "NEUTRAL"]

        # Metric cards
        st.subheader("Key Performance Indicators")
        render_metric_cards(snapshots_df)

        st.divider()

        # Active signals section
        if not active_signals.empty:
            st.subheader("🎯 Active Trading Signals")
            st.caption(f"Showing {len(active_signals)} active signal(s)")

            # Color-code by signal
            display_df = active_signals.copy()
            display_df["signal_type"] = display_df["chikou_conf"].apply(
                lambda x: "🟢 BUY" if x == "BUY" else ("🔴 SELL" if x == "SELL" else "⚪ N/A")
            )
            display_df["regime_tag"] = display_df["regime"].apply(
                lambda x: f"🐂 {x.upper()}" if x == "bull" else (
                    f"🐻 {x.upper()}" if x == "bear" else (
                        f"🦀 {x.upper()}" if x == "chop" else f"⚪ {x.upper()}"
                    )
                )
            )

            cols = st.columns(len(active_signals))
            for idx, (col, (_, row)) in enumerate(zip(cols, active_signals.iterrows())):
                with col:
                    st.metric(
                        f"{row['symbol']}",
                        f"${row['close_price']:,.2f}",
                        delta=f"{row['signal_type']}"
                    )
        else:
            st.info("📭 No active signals at this time")

        st.divider()

        # Strategy explanation
        render_strategy_explanation()


def page_market_analyzer():
    """🔬 Market Analyzer - Detailed surveillance and charting."""

    st.header("Market Analyzer")

    # Query data
    snapshots_df = query_market_snapshots()

    if snapshots_df is None or snapshots_df.empty:
        st.warning("⏳ Waiting for first market scan...")
    else:
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            show_only_active = st.checkbox("Show only active assets", value=False)
        with col2:
            show_only_signals = st.checkbox("Show only signals", value=False)

        # Apply filters
        filtered_df = snapshots_df.copy()
        if show_only_active:
            filtered_df = filtered_df[filtered_df["status"] == "active"]
        if show_only_signals:
            filtered_df = filtered_df[filtered_df["chikou_conf"] != "NEUTRAL"]

        # Ticker header
        render_ticker_header(filtered_df)

        st.divider()

        # Main surveillance table
        st.subheader("Market Surveillance Table")
        render_surveillance_table(filtered_df)

        st.divider()

        # Asset selection for detailed chart
        st.subheader("Detailed Chart View")
        st.caption("Select an asset to view its Ichimoku Cloud chart with technical indicators")

        selected_symbol = st.selectbox(
            "Select asset:",
            options=snapshots_df["symbol"].unique(),
            key="chart_selector"
        )

        if selected_symbol:
            # Get source for this asset
            source = snapshots_df[snapshots_df["symbol"] == selected_symbol]["source"].iloc[0]

            with st.spinner(f"Loading chart for {selected_symbol}..."):
                df_chart = fetch_ohlcv_for_chart(selected_symbol, source)

            if df_chart is not None and not df_chart.empty:
                fig = render_ichimoku_chart(df_chart, selected_symbol)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Failed to load chart for {selected_symbol}")


def page_the_brain():
    """🧠 The Brain - HMM regime visualization."""

    st.header("The Brain - Regime Detector (HMM)")

    st.info(
        "The Brain uses Hidden Markov Models to detect market regimes. "
        "It learns from 252 periods (1 year) of price data and predicts "
        "Bull 🐂, Bear 🐻, or Chop 🦀 conditions."
    )

    # Get current regime
    current_regime = get_current_regime()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Regime", current_regime.upper(), delta="Live")

    with col2:
        icon = "🐂" if current_regime == "bull" else (
            "🐻" if current_regime == "bear" else (
                "🦀" if current_regime == "chop" else "⚪"
            )
        )
        st.metric("Status", icon, delta="Updated", delta_color="off")

    with col3:
        regime_probs = {
            "bull": 0.0,
            "bear": 0.0,
            "chop": 0.0,
            "neutral": 1.0
        }
        # TODO: Update with actual probabilities from regime detector
        max_prob = max(regime_probs.values())
        st.metric("Confidence", f"{max_prob:.1%}", delta="HMM Model")

    st.divider()

    st.subheader("Regime Probability Distribution")
    st.info("Coming soon: Probability timeline chart showing regime confidence over time")

    st.divider()

    st.subheader("Regime Definitions")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("🐂 BULLISH TREND", icon="✅")
        st.caption("""
        - High mean return
        - Low volatility
        - Uptrend in progress
        - Long trades enabled
        - SELL signals blocked
        """)

    with col2:
        st.error("🐻 BEARISH TREND", icon="⚠️")
        st.caption("""
        - Low/negative return
        - Higher volatility
        - Downtrend in progress
        - Short trades enabled
        - BUY signals blocked
        """)

    with col3:
        st.warning("🦀 CHOPPY/VOLATILE", icon="⛔")
        st.caption("""
        - Low return
        - Very high volatility
        - No clear direction
        - Trading paused
        - All signals blocked
        """)


def page_settings():
    """⚙️ Settings - Configuration and risk management."""

    st.header("Settings & Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentinel Configuration")
        st.code("""
SCAN_INTERVAL_ACTIVE = 300s (5 min)
SCAN_INTERVAL_NORMAL = 600s (10 min)
SCAN_INTERVAL_DORMANT = 3600s (1 hour)
EVOLUTION_EPOCH = 14400s (4 hours)
RISK_PER_TRADE = 2%
PAPER_TRADING = True
        """)

    with col2:
        st.subheader("Ichimoku Parameters")
        st.code("""
Tenkan-sen: 9-period
Kijun-sen: 26-period
Senkou Span B: 52-period
Cloud shift: 26 forward
Chikou shift: 26 backward

BUY: Price > Cloud
     Tenkan > Kijun
     Chikou > Price

SELL: Price < Cloud
      Tenkan < Kijun
      Chikou < Price
        """)

    st.divider()

    st.subheader("HMM Regime Detector")
    st.code("""
Model: Gaussian Hidden Markov Model
States: 3 (Bull, Bear, Chop)
Lookback: 252 periods (1 year)
Features: Returns + Rolling Volatility
Window: 20-period majority vote
Training: Every Sentinel init

Scaler: StandardScaler (fit_transform)
Covariance: Full
Iterations: 1000
Random State: 42
    """)

    st.divider()

    # System debug
    st.subheader("System Debug Information")
    render_debug_info()

    st.divider()

    # Raw data viewer
    with st.expander("🔍 Raw Data Viewer (Advanced)", expanded=False):
        st.subheader("Query Raw Database")

        data_source = st.radio(
            "Select data source:",
            options=["Market Snapshots", "Assets", "Trades"],
            horizontal=True
        )

        if data_source == "Market Snapshots":
            df = query_market_snapshots()
            st.write(f"Total records: {len(df) if df is not None else 0}")
            if df is not None:
                st.dataframe(df, use_container_width=True)

        elif data_source == "Assets":
            df = query_assets()
            st.write(f"Total records: {len(df) if df is not None else 0}")
            if df is not None:
                st.dataframe(df, use_container_width=True)

        elif data_source == "Trades":
            df = query_trades(limit=100)
            st.write(f"Total records: {len(df) if df is not None else 0}")
            if df is not None:
                st.dataframe(df, use_container_width=True)


# ============================================================================
# MAIN APP LOGIC
# ============================================================================


def main():
    """Main application entry point."""

    # Render navigation sidebar
    page = render_navigation()

    # Route to appropriate page
    if page == "🏠 Mission Control":
        page_mission_control()
    elif page == "🔬 Market Analyzer":
        page_market_analyzer()
    elif page == "🧠 The Brain":
        page_the_brain()
    elif page == "⚙️ Settings":
        page_settings()


if __name__ == "__main__":
    main()
