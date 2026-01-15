"""
Enhanced UI Components for SOLAT Dashboard

Professional trading terminal styling with information density and clarity.
"""

import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def load_custom_css() -> None:
    """Load custom CSS styling for professional dashboard appearance."""
    try:
        with open("dashboard/assets/style.css", "r") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        logger.warning("Custom CSS file not found, using default styling")


def render_traffic_light_header(regime: str) -> None:
    """
    Render a massive traffic light status header showing market regime.

    Args:
        regime (str): Current regime ('bull', 'bear', 'chop', or 'neutral')
    """
    if regime == "bull":
        st.success(
            "🐂 MARKET STATUS: BULLISH TREND\n\n"
            "High mean return | Low volatility | Uptrend in progress\n\n"
            "✅ **LONG TRADES ENABLED** - BUY signals active | SELL signals blocked",
            icon="📈"
        )
    elif regime == "bear":
        st.error(
            "🐻 MARKET STATUS: BEARISH TREND\n\n"
            "Low/negative return | Higher volatility | Downtrend in progress\n\n"
            "⚠️ **SHORT TRADES ENABLED** - SELL signals active | BUY signals blocked",
            icon="📉"
        )
    elif regime == "chop":
        st.warning(
            "🦀 MARKET STATUS: CHOPPY/VOLATILE\n\n"
            "Low return | Very high volatility | No clear direction\n\n"
            "⛔ **TRADING PAUSED** - All signals blocked (kill zone)",
            icon="⚠️"
        )
    else:
        st.info(
            "⚪ MARKET STATUS: AWAITING DATA\n\n"
            "Sentinel is collecting data for regime detection\n\n"
            "ℹ️ **TRAINING IN PROGRESS** - First predictions coming soon",
            icon="⏳"
        )


def render_ticker_header(df: pd.DataFrame) -> None:
    """
    Render a live ticker of top assets at the very top of the page.

    Shows: Symbol, Last Price, 24h Change %
    """
    if df is None or df.empty:
        return

    st.markdown("---")

    # Get top 3 assets by fitness
    top_assets = df.nlargest(3, "fitness_score")

    cols = st.columns(3)
    for idx, (col, (_, row)) in enumerate(zip(cols, top_assets.iterrows())):
        with col:
            symbol = row["symbol"]
            price = row["close_price"]
            fitness = row["fitness_score"]

            # Create a simple "ticker" display
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.metric(
                    label=symbol,
                    value=f"${price:,.2f}",
                    delta=f"{fitness:.3f}",
                    delta_color="off"
                )
            with col2:
                signal = row.get("chikou_conf", "NEUTRAL")
                if signal == "BUY":
                    st.write("🟢 BUY")
                elif signal == "SELL":
                    st.write("🔴 SELL")
                else:
                    st.write("⚪ NEUTRAL")


def render_news_sentiment_gauge(sentiment_score: float = 50.0) -> None:
    """
    Render the News Sentiment gauge showing market mood.

    Args:
        sentiment_score: Sentiment score from 0 (Extreme Fear) to 100 (Extreme Greed)
    """
    # Determine mood and styling
    if sentiment_score < 20:
        mood = "EXTREME FEAR"
        emoji = "😱"
        color = "#ff0000"
        restriction = "All Longs BLOCKED"
    elif sentiment_score < 30:
        mood = "Fear"
        emoji = "😰"
        color = "#ff6b6b"
        restriction = "Longs Restricted"
    elif sentiment_score > 80:
        mood = "EXTREME GREED"
        emoji = "🤑"
        color = "#00ff00"
        restriction = "All Shorts BLOCKED"
    elif sentiment_score > 70:
        mood = "Greed"
        emoji = "😄"
        color = "#31a24c"
        restriction = "Shorts Restricted"
    else:
        mood = "Neutral"
        emoji = "😐"
        color = "#f0883e"
        restriction = "No Restrictions"

    # Render gauge-like display
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg,
            #ff0000 0%,
            #ff6b6b 20%,
            #f0883e 40%,
            #f0883e 60%,
            #31a24c 80%,
            #00ff00 100%
        );
        height: 8px;
        border-radius: 4px;
        margin: 10px 0;
        position: relative;
    ">
        <div style="
            position: absolute;
            left: {sentiment_score}%;
            top: -4px;
            width: 4px;
            height: 16px;
            background: white;
            border: 2px solid #333;
            border-radius: 2px;
            transform: translateX(-50%);
        "></div>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 11px; color: #8b949e;">
        <span>Fear</span>
        <span>Neutral</span>
        <span>Greed</span>
    </div>
    """, unsafe_allow_html=True)

    # Display mood text
    st.markdown(f"""
    <div style="text-align: center; margin-top: 5px;">
        <span style="font-size: 24px;">{emoji}</span>
        <span style="font-size: 18px; font-weight: bold; color: {color};">{mood}</span>
        <span style="font-size: 14px; color: #8b949e;">({sentiment_score:.0f}/100)</span>
    </div>
    <div style="text-align: center; font-size: 12px; color: #8b949e; margin-top: 5px;">
        {restriction}
    </div>
    """, unsafe_allow_html=True)


def render_metric_cards(df: pd.DataFrame, news_sentiment: float = 50.0) -> None:
    """
    Render a 5-column metric card row showing key KPIs.

    Shows: Total Assets, Active Signals, Portfolio Fitness, News Sentiment, Last Heartbeat
    """
    if df is None or df.empty:
        st.warning("⏳ No market data available yet")
        return

    # Calculate metrics
    total_assets = len(df)
    buy_signals = len(df[df.get("chikou_conf", "") == "BUY"])
    sell_signals = len(df[df.get("chikou_conf", "") == "SELL"])
    active_count = len(df[df.get("status", "") == "active"])
    avg_fitness = df["fitness_score"].mean() if "fitness_score" in df.columns else 0.0

    # Render 5-column layout
    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")

    with col1:
        st.metric(
            label="📊 Total Assets",
            value=int(total_assets),
            delta=f"{active_count} Active",
            delta_color="off"
        )

    with col2:
        total_signals = buy_signals + sell_signals
        st.metric(
            label="🎯 Active Signals",
            value=int(total_signals),
            delta=f"🟢 {buy_signals} / 🔴 {sell_signals}",
            delta_color="off"
        )

    with col3:
        st.metric(
            label="💪 Portfolio Fitness",
            value=f"{avg_fitness:.3f}",
            delta="Self-Optimizing",
            delta_color="off"
        )

    with col4:
        # News Sentiment
        if news_sentiment < 30:
            mood_emoji = "😰"
            mood_text = "Fear"
            restriction = "Longs Blocked"
        elif news_sentiment > 70:
            mood_emoji = "😄"
            mood_text = "Greed"
            restriction = "Shorts Blocked"
        else:
            mood_emoji = "😐"
            mood_text = "Neutral"
            restriction = "No Blocks"

        st.metric(
            label="📰 Market Mood",
            value=f"{mood_emoji} {news_sentiment:.0f}",
            delta=f"{mood_text} - {restriction}",
            delta_color="off"
        )

    with col5:
        # Heartbeat timestamp
        latest_update = df["updated_at"].max() if "updated_at" in df.columns else datetime.now()
        time_ago = (datetime.now() - pd.to_datetime(latest_update)).total_seconds()
        if time_ago < 120:
            status = "🟢 Live"
        elif time_ago < 600:
            status = "🟡 Recent"
        else:
            status = "🔴 Stale"

        st.metric(
            label="💓 Heartbeat",
            value=status,
            delta=f"{int(time_ago)}s ago",
            delta_color="off"
        )


def render_sidebar() -> Tuple[bool, bool]:
    """
    Render the sidebar with system status and filter options.

    Returns:
        Tuple[bool, bool]: (show_only_active, show_only_signals)
    """
    with st.sidebar:
        st.title("⚙️ System Control")

        # System Status Section
        st.subheader("System Status")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Status", "🟢 Running", delta=None)
        with col2:
            st.metric("Last Update", "Live", delta=None)

        st.divider()

        # Filters Section
        st.subheader("📋 Filters")

        show_only_active = st.checkbox(
            "Show Only Active Assets",
            value=False,
            help="Display only assets with 'active' status (high fitness)"
        )

        show_only_signals = st.checkbox(
            "Show Only Buy/Sell Signals",
            value=False,
            help="Hide assets with 'NEUTRAL' signals"
        )

        st.divider()

        # Legend Section
        st.subheader("🎨 Signal Legend")
        st.markdown("""
        **Signal Types:**
        - 🟢 **BUY**: Price > Cloud + Tenkan > Kijun
        - 🔴 **SELL**: Price < Cloud + Tenkan < Kijun
        - ⚪ **NEUTRAL**: No clear signal

        **Cloud Colors:**
        - 🟩 **Green Cloud**: Bullish (Senkou A > B)
        - 🟥 **Red Cloud**: Bearish (Senkou A < B)

        **Order Flow (Microstructure):**
        - 🟢🟢 **Strong Buy Pressure**: OI > +0.20
        - 🟢 **Mild Buy Pressure**: OI > 0
        - 🔴 **Mild Sell Pressure**: OI < 0
        - 🔴🔴 **Strong Sell Pressure**: OI < -0.20

        **News Sentiment (0-100):**
        - 😱 **Extreme Fear** (0-20): All Longs blocked
        - 😰 **Fear** (20-30): Longs restricted
        - 😐 **Neutral** (30-70): No restrictions
        - 😄 **Greed** (70-80): Shorts restricted
        - 🤑 **Extreme Greed** (80-100): All Shorts blocked
        """)

        st.divider()

        # Info Section
        with st.expander("ℹ️ About SOLAT"):
            st.markdown("""
            **SOLAT** - Self-Optimizing Local Algorithmic Trading Platform

            An autonomous trading system that:
            - **Scans** 4+ assets using Ichimoku Cloud strategy
            - **Evolves** by ranking assets by fitness every 4 hours
            - **Manages** risk with 2% per-trade limits
            - **Operates** 24/7 with zero infrastructure costs

            **Paper trading mode enabled** - no real orders executed.

            **Learn More:**
            - [Ichimoku Cloud](https://en.wikipedia.org/wiki/Ichimoku_Kink%C5%8D_Hy%C5%8D)
            - [Fitness Algorithm](CLAUDE.md)
            """)

    return show_only_active, show_only_signals


def render_surveillance_table(df: pd.DataFrame) -> None:
    """
    Render the main surveillance table with colored indicators.

    Uses styled dataframe with conditional formatting for signals.
    Includes Microstructure Order Imbalance (OI) indicator.
    """
    if df is None or df.empty:
        st.warning("No market data available yet. Waiting for first scan...")
        return

    # Prepare display columns
    base_cols = ["symbol", "source", "status", "fitness_score", "close_price",
                 "cloud_status", "chikou_conf"]

    # Add order_imbalance if available
    if "order_imbalance" in df.columns:
        base_cols.append("order_imbalance")

    display_df = df[base_cols].copy()

    # Rename for display
    col_names = ["Asset", "Source", "Status", "Fitness", "Price", "Cloud", "Signal"]
    if "order_imbalance" in df.columns:
        col_names.append("Order Flow")

    display_df.columns = col_names

    # Format numeric columns
    display_df["Fitness"] = display_df["Fitness"].apply(lambda x: f"{x:.4f}")
    display_df["Price"] = display_df["Price"].apply(lambda x: f"${x:,.2f}")

    # Format Order Flow as visual indicator
    if "Order Flow" in display_df.columns:
        def format_order_imbalance(val):
            """Format OI as visual bar indicator."""
            if pd.isna(val) or val == 0:
                return "⚪ 0.00"
            elif val > 0.2:
                return f"🟢🟢 +{val:.2f}"
            elif val > 0:
                return f"🟢 +{val:.2f}"
            elif val < -0.2:
                return f"🔴🔴 {val:.2f}"
            else:
                return f"🔴 {val:.2f}"

        display_df["Order Flow"] = display_df["Order Flow"].apply(format_order_imbalance)

    # Create styled dataframe with conditional formatting
    def color_signal(val):
        """Color code the signal column."""
        if val == "BUY":
            return "color: white; background-color: #31a24c; font-weight: bold;"
        elif val == "SELL":
            return "color: white; background-color: #ff6b6b; font-weight: bold;"
        else:
            return "color: #8b949e; background-color: transparent;"

    def color_status(val):
        """Color code the status column."""
        if val == "active":
            return "background-color: rgba(49, 162, 76, 0.2);"
        elif val == "normal":
            return "background-color: rgba(240, 136, 62, 0.2);"
        else:  # dormant
            return "background-color: rgba(255, 107, 107, 0.2);"

    def color_cloud(val):
        """Color code the cloud status."""
        if val == "green":
            return "color: white; background-color: #31a24c;"
        elif val == "red":
            return "color: white; background-color: #ff6b6b;"
        else:
            return "color: #8b949e;"

    # Apply styling
    styled_df = display_df.style.applymap(
        color_signal,
        subset=["Signal"]
    ).applymap(
        color_status,
        subset=["Status"]
    ).applymap(
        color_cloud,
        subset=["Cloud"]
    )

    # Build column config dynamically
    column_config = {
        "Asset": st.column_config.TextColumn(width=80),
        "Source": st.column_config.TextColumn(width=70),
        "Status": st.column_config.TextColumn(width=80),
        "Fitness": st.column_config.TextColumn(width=90),
        "Price": st.column_config.TextColumn(width=100),
        "Cloud": st.column_config.TextColumn(width=80),
        "Signal": st.column_config.TextColumn(width=90),
    }

    # Add Order Flow column config if present
    if "Order Flow" in display_df.columns:
        column_config["Order Flow"] = st.column_config.TextColumn(
            width=100,
            help="Order Imbalance: 🟢 = Buy Pressure, 🔴 = Sell Pressure"
        )

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400,
        column_config=column_config
    )

    # Add helpful footer
    st.caption(
        "💡 **Tip:** Hover over any value for details. "
        "Strong Buy = Green Cloud + Positive Tenkan/Kijun Cross + 🟢🟢 Order Flow"
    )


def render_strategy_explanation() -> None:
    """
    Render an expandable section explaining the Ichimoku strategy.
    """
    with st.expander("📘 Strategy Guide & Signal Explanation", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ichimoku Components")
            st.markdown("""
            **Technical Indicators:**

            1. **Tenkan-sen (Blue Line)**
               - 9-period momentum
               - Short-term trend indicator
               - Crosses above Kijun = bullish

            2. **Kijun-sen (Red Line)**
               - 26-period momentum
               - Medium-term support/resistance
               - Crosses below Tenkan = bearish

            3. **Senkou Spans (Cloud)**
               - Upper boundary: (Tenkan + Kijun) / 2
               - Lower boundary: 52-period high/low
               - **Green Cloud** = Bullish bias
               - **Red Cloud** = Bearish bias

            4. **Chikou Span (Purple Dashed)**
               - Current close shifted 26 periods back
               - Confirms trend strength
            """)

        with col2:
            st.subheader("Signal Definitions")
            st.markdown("""
            **🟢 STRONG BUY Signal**
            - Price closes ABOVE the cloud
            - Tenkan (blue) > Kijun (red)
            - Chikou in uptrend
            - Cloud is GREEN (bullish)

            **🔴 STRONG SELL Signal**
            - Price closes BELOW the cloud
            - Tenkan (blue) < Kijun (red)
            - Chikou in downtrend
            - Cloud is RED (bearish)

            **⚪ NEUTRAL Signal**
            - No clear Tenkan/Kijun cross
            - Price within cloud (indecision)
            - Conflicting indicators
            """)

        st.divider()

        st.subheader("Fitness Score Calculation")
        st.markdown("""
        **How SOLAT Ranks Assets (Self-Optimization)**

        ```
        Fitness = (0.4 × Win Rate) + (0.4 × Profit Factor) - (0.2 × Max Drawdown)
        ```

        - **Win Rate**: % of profitable trades
        - **Profit Factor**: Total Wins / Total Losses
        - **Max Drawdown**: Largest peak-to-trough decline

        **Asset Status Based on Fitness:**
        - 🟢 **Active** (Top 20%): Scanned every 5 min
        - 🟡 **Normal** (Middle 60%): Scanned every 1 hour
        - 🔴 **Dormant** (Bottom 20%): Scanned every 1 hour (saves API credits)
        """)


def render_evolution_metrics(assets_df: pd.DataFrame) -> None:
    """
    Render evolution and fitness metrics with charts and rankings.
    """
    if assets_df is None or assets_df.empty:
        st.info("No asset data available")
        return

    from dashboard.charts import render_fitness_bar_chart, render_status_pie_chart

    col1, col2 = st.columns(2)

    with col1:
        # Fitness bar chart (top 5)
        fitness_data = dict(zip(assets_df["symbol"], assets_df["fitness_score"]))
        fig_fitness = render_fitness_bar_chart(fitness_data)
        st.plotly_chart(fig_fitness, use_container_width=True)

    with col2:
        # Status pie chart
        status_counts = assets_df["status"].value_counts().to_dict()
        fig_status = render_status_pie_chart(status_counts)
        st.plotly_chart(fig_status, use_container_width=True)

    st.divider()

    # Status metrics
    st.subheader("Asset Status Distribution")

    col1, col2, col3 = st.columns(3)

    with col1:
        active_count = len(assets_df[assets_df["status"] == "active"])
        st.metric("🟢 Active (High Fitness)", active_count, border=True)

    with col2:
        normal_count = len(assets_df[assets_df["status"] == "normal"])
        st.metric("🟡 Normal (Medium Fitness)", normal_count, border=True)

    with col3:
        dormant_count = len(assets_df[assets_df["status"] == "dormant"])
        st.metric("🔴 Dormant (Low Fitness)", dormant_count, border=True)

    st.divider()

    # Full asset table
    st.subheader("All Assets with Fitness Scores")

    asset_display = assets_df.copy()
    asset_display["fitness_score"] = asset_display["fitness_score"].apply(
        lambda x: f"{x:.4f}"
    )
    asset_display["last_scan"] = pd.to_datetime(
        asset_display["last_scan"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    st.dataframe(asset_display, use_container_width=True, height=300)


def render_trades_summary() -> None:
    """
    Render paper trading info and placeholders for trade history.
    """
    st.subheader("📝 Recent Trades (Paper Trading)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Trades", 0, delta="Paper Mode", delta_color="off")

    with col2:
        st.metric("Win Rate", "N/A", delta="Waiting for trades...", delta_color="off")

    with col3:
        st.metric("Profit Factor", "N/A", delta="Pending", delta_color="off")

    st.info(
        "📋 **Paper Trading Mode** - All trades are logged to the database "
        "but NOT executed. This is a safe testing environment with zero capital at risk."
    )

    st.caption("Trades will appear here once the Sentinel generates and logs signals.")


def render_debug_info() -> None:
    """
    Render debug information (timestamp, DB status, etc.).
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with col2:
        st.caption("Mode: Read-Only (Dashboard)")

    with col3:
        st.caption("Database: WAL-enabled SQLite")
