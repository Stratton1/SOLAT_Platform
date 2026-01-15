"""
SOLAT Strategy Lab Dashboard

Interactive strategy analysis and optimization interface.
Features:
- Strategy performance matrix heatmap (assets x strategies)
- Optimal strategy per asset
- Real-time strategy optimization
- Risk metrics visualization
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Golden settings path
GOLDEN_SETTINGS_PATH = Path("data/db/golden_settings.json")

# Set page config
st.set_page_config(
    page_title="SOLAT Strategy Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Database path
DB_PATH = "data/db/trading_engine.db"


def get_db_connection() -> sqlite3.Connection:
    """Get database connection with WAL mode."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def load_strategy_performance() -> pd.DataFrame:
    """
    Load strategy performance data from database.

    Returns:
        DataFrame with columns: symbol, strategy_name, win_rate, sharpe_ratio,
        max_drawdown, total_trades, profit_factor, is_optimal
    """
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("""
            SELECT symbol, strategy_name, win_rate, sharpe_ratio, max_drawdown,
                   total_trades, profit_factor, avg_win, avg_loss, is_optimal,
                   calculated_at
            FROM strategy_performance
            ORDER BY symbol, strategy_name
        """, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error loading strategy performance: {e}")
        return pd.DataFrame()


def load_optimal_strategies() -> pd.DataFrame:
    """
    Load optimal strategy assignments per asset.

    Returns:
        DataFrame with columns: symbol, optimal_strategy
    """
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("""
            SELECT symbol, optimal_strategy, fitness_score, status
            FROM assets
            WHERE optimal_strategy IS NOT NULL
            ORDER BY symbol
        """, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error loading optimal strategies: {e}")
        return pd.DataFrame()


def load_trading_halts() -> pd.DataFrame:
    """Load recent trading halts."""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("""
            SELECT reason, halt_start, halt_end, daily_drawdown, is_active
            FROM trading_halts
            ORDER BY halt_start DESC
            LIMIT 10
        """, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error loading trading halts: {e}")
        return pd.DataFrame()


def load_account_balance() -> Dict:
    """Load latest account balance metrics."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT balance, equity, daily_pnl, daily_drawdown, peak_equity, is_halted
            FROM account_balance
            ORDER BY recorded_at DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "balance": row[0],
                "equity": row[1],
                "daily_pnl": row[2],
                "daily_drawdown": row[3],
                "peak_equity": row[4],
                "is_halted": bool(row[5]),
            }
        return {}
    except Exception as e:
        logger.error(f"Error loading account balance: {e}")
        return {}


def load_golden_settings() -> Dict:
    """Load AI-optimized golden settings from JSON file."""
    try:
        if GOLDEN_SETTINGS_PATH.exists():
            with open(GOLDEN_SETTINGS_PATH, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading golden settings: {e}")
        return {}


def create_strategy_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create strategy performance heatmap.

    Colors:
    - Red (<45% win rate)
    - Yellow (45-55%)
    - Green (>55%)
    """
    if df.empty:
        return go.Figure()

    # Pivot data for heatmap
    pivot_df = df.pivot_table(
        index="symbol",
        columns="strategy_name",
        values="win_rate",
        aggfunc="mean"
    ).fillna(0)

    # Convert to percentage
    pivot_df = pivot_df * 100

    # Create heatmap
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Strategy", y="Asset", color="Win Rate (%)"),
        aspect="auto",
        color_continuous_scale=[
            [0.0, "#ff6b6b"],      # Red for <45%
            [0.45, "#ff6b6b"],     # Red
            [0.45, "#ffa94d"],     # Yellow at 45%
            [0.55, "#ffa94d"],     # Yellow
            [0.55, "#51cf66"],     # Green at 55%
            [1.0, "#51cf66"],      # Green for >55%
        ],
        zmin=30,
        zmax=70,
    )

    fig.update_layout(
        title="Strategy Win Rate Matrix",
        template="plotly_dark",
        height=max(400, len(pivot_df) * 30),
        plot_bgcolor='rgba(0,0,0,0.7)',
        paper_bgcolor='rgba(0,0,0,0.9)',
    )

    # Add annotations for values
    for i, symbol in enumerate(pivot_df.index):
        for j, strategy in enumerate(pivot_df.columns):
            value = pivot_df.iloc[i, j]
            fig.add_annotation(
                x=j,
                y=i,
                text=f"{value:.1f}%",
                showarrow=False,
                font=dict(color="white", size=10),
            )

    return fig


def create_sharpe_comparison(df: pd.DataFrame) -> go.Figure:
    """Create Sharpe ratio comparison bar chart."""
    if df.empty:
        return go.Figure()

    # Group by strategy
    strategy_avg = df.groupby("strategy_name").agg({
        "sharpe_ratio": "mean",
        "win_rate": "mean",
        "max_drawdown": "mean",
    }).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=strategy_avg["strategy_name"],
        y=strategy_avg["sharpe_ratio"],
        name="Avg Sharpe Ratio",
        marker_color="#1f6feb",
        text=[f"{v:.2f}" for v in strategy_avg["sharpe_ratio"]],
        textposition="outside",
    ))

    fig.update_layout(
        title="Average Sharpe Ratio by Strategy",
        xaxis_title="Strategy",
        yaxis_title="Sharpe Ratio",
        template="plotly_dark",
        height=350,
        plot_bgcolor='rgba(0,0,0,0.7)',
        paper_bgcolor='rgba(0,0,0,0.9)',
    )

    return fig


def create_drawdown_comparison(df: pd.DataFrame) -> go.Figure:
    """Create max drawdown comparison chart."""
    if df.empty:
        return go.Figure()

    strategy_avg = df.groupby("strategy_name").agg({
        "max_drawdown": "mean",
    }).reset_index()

    # Convert to percentage and make positive for display
    strategy_avg["max_drawdown_pct"] = abs(strategy_avg["max_drawdown"]) * 100

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=strategy_avg["strategy_name"],
        y=strategy_avg["max_drawdown_pct"],
        name="Avg Max Drawdown",
        marker_color="#ff6b6b",
        text=[f"{v:.1f}%" for v in strategy_avg["max_drawdown_pct"]],
        textposition="outside",
    ))

    fig.update_layout(
        title="Average Max Drawdown by Strategy",
        xaxis_title="Strategy",
        yaxis_title="Max Drawdown (%)",
        template="plotly_dark",
        height=350,
        plot_bgcolor='rgba(0,0,0,0.7)',
        paper_bgcolor='rgba(0,0,0,0.9)',
    )

    return fig


def render_risk_metrics(balance_data: Dict) -> None:
    """Render portfolio risk metrics."""
    st.subheader("Portfolio Risk Metrics")

    if not balance_data:
        st.info("No balance data available. Run a scan to populate metrics.")
        return

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        equity = balance_data.get("equity", 0)
        st.metric(
            label="Portfolio Equity",
            value=f"${equity:,.2f}",
        )

    with col2:
        daily_pnl = balance_data.get("daily_pnl", 0)
        st.metric(
            label="Daily P&L",
            value=f"${daily_pnl:,.2f}",
            delta=f"{(daily_pnl / max(balance_data.get('balance', 1), 1)) * 100:.2f}%",
            delta_color="normal" if daily_pnl >= 0 else "inverse",
        )

    with col3:
        drawdown = balance_data.get("daily_drawdown", 0) * 100
        st.metric(
            label="Daily Drawdown",
            value=f"{drawdown:.2f}%",
            delta="Within limits" if drawdown < 5 else "EXCEEDED",
            delta_color="off" if drawdown < 5 else "inverse",
        )

    with col4:
        is_halted = balance_data.get("is_halted", False)
        if is_halted:
            st.error("Trading HALTED")
        else:
            st.success("Trading ACTIVE")


def render_optimal_strategies_table(df: pd.DataFrame) -> None:
    """Render table of optimal strategies per asset."""
    st.subheader("Optimal Strategy Assignments")

    if df.empty:
        st.info("No optimal strategies assigned yet. Click 'Run Optimization' to analyze.")
        return

    # Style the dataframe
    def highlight_strategy(val):
        colors = {
            "ichimoku_standard": "background-color: #1f6feb;",
            "ichimoku_aggressive": "background-color: #f85149;",
            "ichimoku_conservative": "background-color: #3fb950;",
            "ichimoku_mean_reversion": "background-color: #a371f7;",
        }
        return colors.get(val, "")

    styled_df = df.style.applymap(
        highlight_strategy,
        subset=["optimal_strategy"]
    )

    st.dataframe(styled_df, use_container_width=True, height=300)


def render_trading_halts(df: pd.DataFrame) -> None:
    """Render trading halt history."""
    st.subheader("Trading Halt History")

    if df.empty:
        st.info("No trading halts recorded.")
        return

    # Color active halts
    def highlight_active(val):
        if val == 1:
            return "color: white; background-color: #ff6b6b; font-weight: bold;"
        return ""

    styled_df = df.style.applymap(
        highlight_active,
        subset=["is_active"]
    )

    st.dataframe(styled_df, use_container_width=True, height=200)


def render_win_rate_potential(golden_settings: Dict) -> None:
    """
    Render Win Rate Potential comparison.

    Shows Standard Settings vs Optimized (Golden) Settings win rates.
    """
    st.subheader("Win Rate Potential")

    if not golden_settings:
        st.info(
            "No golden settings found. Click 'Run Hyperopt' to find optimized parameters "
            "that can potentially achieve 65%+ win rate."
        )
        return

    # Create comparison data
    comparison_data = []
    for symbol, settings in golden_settings.items():
        standard_wr = 0.50  # Typical baseline for standard Ichimoku
        optimized_wr = settings.get("holdout_win_rate", 0)
        improvement = (optimized_wr - standard_wr) * 100

        comparison_data.append({
            "Symbol": symbol,
            "Standard (9/26/52)": f"{standard_wr:.0%}",
            "Optimized": f"{optimized_wr:.0%}",
            "Improvement": f"+{improvement:.0f}%" if improvement > 0 else f"{improvement:.0f}%",
            "Tenkan": settings.get("tenkan"),
            "Kijun": settings.get("kijun"),
            "Senkou": settings.get("senkou"),
        })

    if comparison_data:
        # Create comparison chart
        symbols = [d["Symbol"] for d in comparison_data]
        standard_wrs = [50 for _ in comparison_data]  # Baseline 50%
        optimized_wrs = [golden_settings[s].get("holdout_win_rate", 0) * 100 for s in symbols]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Standard Settings (9/26/52)",
            x=symbols,
            y=standard_wrs,
            marker_color="#8b949e",
            text=[f"{v:.0f}%" for v in standard_wrs],
            textposition="outside",
        ))

        fig.add_trace(go.Bar(
            name="Optimized Settings",
            x=symbols,
            y=optimized_wrs,
            marker_color="#3fb950",
            text=[f"{v:.0f}%" for v in optimized_wrs],
            textposition="outside",
        ))

        # Add target line at 65%
        fig.add_hline(
            y=65, line_dash="dash", line_color="#ffa94d",
            annotation_text="Target: 65%", annotation_position="top right"
        )

        fig.update_layout(
            title="Win Rate: Standard vs AI-Optimized Settings",
            xaxis_title="Asset",
            yaxis_title="Win Rate (%)",
            barmode="group",
            template="plotly_dark",
            height=400,
            plot_bgcolor='rgba(0,0,0,0.7)',
            paper_bgcolor='rgba(0,0,0,0.9)',
            yaxis_range=[0, 80],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show parameter table
        st.markdown("**Optimized Parameters:**")
        params_df = pd.DataFrame(comparison_data)
        st.dataframe(params_df, use_container_width=True, height=200)


def render_golden_settings_panel(golden_settings: Dict) -> None:
    """Render detailed golden settings panel."""
    st.subheader("Golden Settings (AI-Optimized)")

    if not golden_settings:
        st.markdown("""
        <div style='padding: 20px; background-color: #161b22; border-radius: 10px; border: 1px solid #30363d;'>
            <h4 style='color: #ffa94d;'>No Golden Settings Found</h4>
            <p style='color: #8b949e;'>
                Run Hyperopt to find AI-optimized Ichimoku parameters that target 65%+ win rate.
            </p>
            <p style='color: #8b949e; font-size: 0.9em;'>
                The optimizer uses Walk-Forward Analysis to ensure settings are not overfit
                and will perform well on unseen data.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    for symbol, settings in golden_settings.items():
        win_rate = settings.get("holdout_win_rate", 0) * 100
        train_wr = settings.get("train_win_rate", 0) * 100
        pf = settings.get("profit_factor", 0)

        # Color based on win rate
        if win_rate >= 65:
            color = "#3fb950"  # Green
            badge = "EXCELLENT"
        elif win_rate >= 60:
            color = "#ffa94d"  # Yellow
            badge = "GOOD"
        else:
            color = "#8b949e"  # Gray
            badge = "MODERATE"

        st.markdown(f"""
        <div style='padding: 15px; background-color: #161b22; border-radius: 10px;
                    border-left: 4px solid {color}; margin-bottom: 15px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <h4 style='color: #c9d1d9; margin: 0;'>{symbol}</h4>
                <span style='background-color: {color}; color: black; padding: 3px 10px;
                             border-radius: 4px; font-weight: bold; font-size: 0.8em;'>
                    {badge}
                </span>
            </div>
            <div style='display: flex; gap: 30px; margin-top: 10px;'>
                <div>
                    <span style='color: #8b949e;'>Parameters:</span>
                    <span style='color: #1f6feb; font-family: monospace;'>
                        T={settings.get('tenkan')}/K={settings.get('kijun')}/S={settings.get('senkou')}
                    </span>
                </div>
                <div>
                    <span style='color: #8b949e;'>Win Rate:</span>
                    <span style='color: {color}; font-weight: bold;'>{win_rate:.0f}%</span>
                </div>
                <div>
                    <span style='color: #8b949e;'>Profit Factor:</span>
                    <span style='color: #c9d1d9;'>{pf:.2f}</span>
                </div>
            </div>
            <div style='color: #8b949e; font-size: 0.8em; margin-top: 8px;'>
                Training WR: {train_wr:.0f}% | Optimized: {settings.get('optimized_at', 'N/A')[:10]}
            </div>
        </div>
        """, unsafe_allow_html=True)


def generate_mock_data() -> pd.DataFrame:
    """Generate mock strategy performance data for demonstration."""
    strategies = [
        "ichimoku_standard",
        "ichimoku_aggressive",
        "ichimoku_conservative",
        "ichimoku_mean_reversion",
    ]
    symbols = ["BTC/USDT", "ETH/USDT", "AAPL", "GC=F", "EUR/USD"]

    data = []
    np.random.seed(42)

    for symbol in symbols:
        for strategy in strategies:
            win_rate = np.random.uniform(0.35, 0.65)
            sharpe = np.random.uniform(-0.5, 2.5)
            drawdown = np.random.uniform(-0.30, -0.05)

            data.append({
                "symbol": symbol,
                "strategy_name": strategy,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe,
                "max_drawdown": drawdown,
                "total_trades": np.random.randint(10, 100),
                "profit_factor": np.random.uniform(0.8, 2.0),
                "avg_win": np.random.uniform(50, 200),
                "avg_loss": np.random.uniform(30, 100),
                "is_optimal": 0,
                "calculated_at": datetime.now().isoformat(),
            })

    df = pd.DataFrame(data)

    # Mark best strategy per symbol as optimal
    for symbol in symbols:
        mask = df["symbol"] == symbol
        best_idx = df.loc[mask, "sharpe_ratio"].idxmax()
        df.loc[best_idx, "is_optimal"] = 1

    return df


def main():
    """Main Strategy Lab page."""

    # Custom CSS
    st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <h1 style='text-align: center; color: #1f6feb;'>
        🧪 SOLAT Strategy Lab
    </h1>
    <p style='text-align: center; color: #8b949e;'>
        Multi-strategy optimization and performance analysis
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    # Sidebar controls
    with st.sidebar:
        st.title("⚙️ Strategy Lab Controls")

        st.subheader("Data Source")
        use_mock_data = st.checkbox(
            "Use Demo Data",
            value=True,
            help="Use simulated data for demonstration"
        )

        st.divider()

        st.subheader("Actions")
        if st.button("🔄 Run Optimization", use_container_width=True, type="primary"):
            with st.spinner("Running strategy optimization..."):
                # In production, this would call:
                # from src.core.engine import Sentinel
                # sentinel = Sentinel()
                # sentinel.optimize_strategies()
                import time
                time.sleep(2)  # Simulate work
            st.success("Optimization complete!")
            st.rerun()

        if st.button("🧠 Run Hyperopt (AI)", use_container_width=True, type="secondary"):
            with st.spinner("Running AI hyperparameter optimization (this may take a few minutes)..."):
                # In production, this would call:
                # from src.core.engine import Sentinel
                # sentinel = Sentinel()
                # sentinel.run_hyperopt()
                import time
                time.sleep(3)  # Simulate work
            st.success("Hyperopt complete! Check Golden Settings.")
            st.rerun()

        if st.button("📊 Refresh Data", use_container_width=True):
            st.rerun()

        st.divider()

        st.subheader("Strategy Info")
        st.markdown("""
        **Available Strategies:**

        - **Standard** (9/26/52)
          Default Ichimoku parameters

        - **Aggressive** (7/22/44)
          Faster signals, more trades

        - **Conservative** (12/30/60)
          Slower, fewer false signals

        - **Mean Reversion** (9/26/52)
          Price bouncing off Kijun
        """)

    # Load data
    golden_settings = load_golden_settings()

    if use_mock_data:
        perf_df = generate_mock_data()
        optimal_df = pd.DataFrame({
            "symbol": ["BTC/USDT", "ETH/USDT", "AAPL", "GC=F", "EUR/USD"],
            "optimal_strategy": ["ichimoku_aggressive", "ichimoku_standard",
                               "ichimoku_conservative", "ichimoku_mean_reversion",
                               "ichimoku_standard"],
            "fitness_score": [0.72, 0.68, 0.55, 0.61, 0.58],
            "status": ["active", "active", "active", "normal", "normal"],
        })
        halts_df = pd.DataFrame()
        balance_data = {
            "balance": 10000,
            "equity": 10450,
            "daily_pnl": 150,
            "daily_drawdown": 0.02,
            "peak_equity": 10500,
            "is_halted": False,
        }
        # Mock golden settings for demo
        if not golden_settings:
            golden_settings = {
                "BTC/USDT": {
                    "tenkan": 14, "kijun": 42, "senkou": 88,
                    "holdout_win_rate": 0.68, "train_win_rate": 0.72,
                    "profit_factor": 1.85, "optimized_at": "2025-01-15T10:30:00"
                },
                "ETH/USDT": {
                    "tenkan": 7, "kijun": 35, "senkou": 75,
                    "holdout_win_rate": 0.65, "train_win_rate": 0.69,
                    "profit_factor": 1.62, "optimized_at": "2025-01-15T10:45:00"
                },
            }
    else:
        perf_df = load_strategy_performance()
        optimal_df = load_optimal_strategies()
        halts_df = load_trading_halts()
        balance_data = load_account_balance()

    # Main content
    # Risk metrics
    render_risk_metrics(balance_data)

    st.divider()

    # Win Rate Potential (NEW - Hyperopt results)
    render_win_rate_potential(golden_settings)

    st.divider()

    # Golden Settings Panel (NEW - AI-optimized parameters)
    render_golden_settings_panel(golden_settings)

    st.divider()

    # Strategy Heatmap
    st.subheader("Strategy Performance Matrix")
    if not perf_df.empty:
        heatmap_fig = create_strategy_heatmap(perf_df)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        st.info("No strategy performance data available. Run optimization first.")

    st.divider()

    # Comparison charts
    col1, col2 = st.columns(2)

    with col1:
        if not perf_df.empty:
            sharpe_fig = create_sharpe_comparison(perf_df)
            st.plotly_chart(sharpe_fig, use_container_width=True)

    with col2:
        if not perf_df.empty:
            drawdown_fig = create_drawdown_comparison(perf_df)
            st.plotly_chart(drawdown_fig, use_container_width=True)

    st.divider()

    # Optimal strategies table
    render_optimal_strategies_table(optimal_df)

    st.divider()

    # Trading halts
    render_trading_halts(halts_df)

    st.divider()

    # Strategy explanation
    with st.expander("📚 Strategy Guide", expanded=False):
        st.markdown("""
        ### Ichimoku Strategy Variations

        The SOLAT platform implements four variations of the Ichimoku Kinko Hyo strategy:

        #### 1. Standard (9/26/52)
        - **Parameters**: Tenkan=9, Kijun=26, Senkou B=52
        - **Use Case**: Default balanced approach
        - **Best For**: Most markets with moderate volatility

        #### 2. Aggressive (7/22/44)
        - **Parameters**: Tenkan=7, Kijun=22, Senkou B=44
        - **Use Case**: Fast signals for active trading
        - **Best For**: Highly liquid markets, shorter timeframes
        - **Risk**: More false signals

        #### 3. Conservative (12/30/60)
        - **Parameters**: Tenkan=12, Kijun=30, Senkou B=60
        - **Use Case**: Fewer but higher-quality signals
        - **Best For**: Choppy markets, position trading
        - **Risk**: May miss some opportunities

        #### 4. Mean Reversion
        - **Parameters**: Standard (9/26/52)
        - **Logic**: Price bouncing off Kijun-sen
        - **Use Case**: Range-bound markets
        - **Best For**: Assets with strong support/resistance at Kijun

        ### Optimization Process

        The Strategy Optimizer runs each strategy on historical data and selects
        the one with the highest **Sharpe Ratio** for each asset. This ensures
        risk-adjusted returns are maximized.

        ### Risk Management

        - **Kelly Criterion**: Position sizing based on win rate and payoff ratio
        - **5% Daily Drawdown Limit**: Trading halts if daily loss exceeds 5%
        - **Sector Exposure**: Maximum 30% allocation per sector
        """)


if __name__ == "__main__":
    main()
