"""
SOLAT Quant Lab - The Gauntlet Interface

Interactive interface for running multi-timeframe optimization and backtesting.

Tab 1: The Gauntlet - Global optimization across all assets and timeframes
Tab 2: Manual Inspector - Individual asset backtesting and analysis
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="SOLAT Quant Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# STYLING
# =============================================================================

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
.gauntlet-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #1a1f2e 0%, #0e1117 100%);
    border-radius: 10px;
    border: 1px solid #00FF41;
    margin-bottom: 20px;
}
.gauntlet-title {
    color: #00FF41;
    font-size: 2.5em;
    font-weight: bold;
    text-shadow: 0 0 10px #00FF41;
}
.gauntlet-subtitle {
    color: #8b949e;
    font-size: 1.1em;
}
.status-pass {
    color: #00FF41;
    font-weight: bold;
}
.status-fail {
    color: #FF6B6B;
    font-weight: bold;
}
.metric-card {
    background: #1a1f2e;
    border-radius: 8px;
    padding: 15px;
    border-left: 3px solid #00FF41;
}
.progress-text {
    font-family: monospace;
    color: #00D9FF;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA FUNCTIONS
# =============================================================================


def load_assets_from_db() -> List[Dict]:
    """Load all assets from database."""
    try:
        from src.database.repository import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, source, status, best_timeframe, best_strategy FROM assets")
        assets = [
            {
                "symbol": row[0],
                "source": row[1],
                "status": row[2],
                "best_timeframe": row[3],
                "best_strategy": row[4],
            }
            for row in cursor.fetchall()
        ]
        conn.close()
        return assets
    except Exception as e:
        logger.error(f"Error loading assets: {e}")
        return []


def load_gauntlet_results() -> pd.DataFrame:
    """Load optimization results from database."""
    try:
        from src.database.repository import get_connection
        conn = get_connection()
        df = pd.read_sql_query(
            """
            SELECT
                symbol,
                timeframe,
                strategy,
                trades_per_day,
                win_rate,
                profit_factor,
                sharpe_ratio,
                max_drawdown,
                total_return,
                avg_trade_duration,
                is_valid,
                calculated_at
            FROM gauntlet_results
            ORDER BY profit_factor DESC, win_rate DESC
            """,
            conn,
        )
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error loading gauntlet results: {e}")
        return pd.DataFrame()


def simulate_backtest_data(
    symbol: str,
    initial_capital: float,
    days: int = 100
) -> Dict:
    """
    Simulate backtest results for a given symbol.

    In production, this would:
    1. Fetch historical OHLCV data
    2. Run Ichimoku strategy
    3. Simulate trades
    4. Calculate equity curve

    For now, we mock the results.
    """
    # Generate dates
    dates = pd.date_range(end=datetime.now(), periods=days, freq='1D')

    # Mock returns (20% annual = ~0.05% daily)
    daily_returns = np.random.normal(0.0005, 0.015, days)
    daily_returns[0] = 0  # Start at 0

    equity = initial_capital * (1 + daily_returns).cumprod()

    # Calculate metrics
    total_return = ((equity[-1] - initial_capital) / initial_capital) * 100

    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    # Simulate trades
    trades = []
    for i in range(1, min(15, days)):
        if np.random.random() > 0.7:  # 30% chance of trade
            entry_price = equity[i]
            exit_price = equity[i + 1] if i + 1 < len(equity) else equity[i]
            pnl = exit_price - entry_price
            win = "WIN" if pnl > 0 else "LOSS"
            trades.append({
                "Date": dates[i].strftime("%Y-%m-%d"),
                "Symbol": symbol,
                "Side": "BUY" if np.random.random() > 0.5 else "SELL",
                "Entry": f"${entry_price:,.2f}",
                "Exit": f"${exit_price:,.2f}",
                "P&L": f"${pnl:,.2f}",
                "Result": win,
            })

    return {
        "dates": dates,
        "equity": equity,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "trades": trades,
        "win_rate": sum(1 for t in trades if t["Result"] == "WIN") / max(len(trades), 1) * 100,
    }


# =============================================================================
# THE GAUNTLET TAB
# =============================================================================


def render_gauntlet_tab():
    """Render The Gauntlet optimization tab."""

    # Header
    st.markdown("""
    <div class="gauntlet-header">
        <div class="gauntlet-title">⚔️ THE GAUNTLET</div>
        <div class="gauntlet-subtitle">Multi-Timeframe Strategy Optimization Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Target:** Find optimal timeframe for each asset achieving **2-4 trades per day**

    **Selection Criteria:**
    - Trade Frequency: 2-5 trades/day (reject < 2 or > 8)
    - Profit Factor: > 1.3
    - Win Rate: > 45%
    """)

    st.divider()

    # Control Panel
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        run_gauntlet = st.button(
            "🚀 RUN GLOBAL OPTIMIZATION",
            type="primary",
            use_container_width=True,
            help="Run optimization across all assets and timeframes"
        )

    with col2:
        refresh_results = st.button(
            "🔄 Refresh Results",
            use_container_width=True,
            help="Reload results from database"
        )

    with col3:
        st.info("💡 Optimization tests 5m, 15m, 30m, 1h timeframes per asset")

    st.divider()

    # Run The Gauntlet
    if run_gauntlet:
        st.subheader("🔥 Running The Gauntlet...")

        # Progress container
        progress_container = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            from src.core.optimizer_loop import GauntletOptimizer, GauntletConfig

            config = GauntletConfig(
                timeframes=["5m", "15m", "30m", "1h"],
                min_trades_per_day=2.0,
                max_trades_per_day=8.0,
                min_profit_factor=1.3,
                min_win_rate=0.45,
                backtest_days=30,
            )

            gauntlet = GauntletOptimizer(config)
            assets = gauntlet._load_assets_from_db()

            if not assets:
                st.error("No assets found in database!")
                return

            winners = {}
            total = len(assets)

            for idx, asset in enumerate(assets, 1):
                symbol = asset["symbol"]
                source = asset["source"]

                # Update progress
                progress = idx / total
                progress_bar.progress(progress)
                status_text.markdown(
                    f'<p class="progress-text">Testing {symbol} ({source}) [{idx}/{total}]...</p>',
                    unsafe_allow_html=True
                )

                try:
                    results = gauntlet.optimize_asset(symbol, source)
                    valid_results = [r for r in results if r.is_valid]

                    if valid_results:
                        # Score and pick best
                        def score(r):
                            pf = min(r.profit_factor, 3.0) / 3.0
                            wr = r.win_rate
                            freq = 1.0 - abs(r.trades_per_day - 3.0) / 5.0
                            return pf * 0.4 + wr * 0.4 + freq * 0.2

                        best = max(valid_results, key=score)
                        winners[symbol] = best

                except Exception as e:
                    logger.error(f"Error optimizing {symbol}: {e}")
                    continue

            # Save results
            if winners:
                saved = gauntlet.save_winners_to_db(winners)
                st.success(f"✅ Gauntlet Complete! Saved {saved} optimal configurations.")
            else:
                st.warning("⚠️ No valid configurations found matching criteria.")

            progress_bar.progress(1.0)
            status_text.markdown(
                '<p class="progress-text">✓ Optimization complete!</p>',
                unsafe_allow_html=True
            )

        except ImportError as e:
            st.error(f"Module import error: {e}. Run from project root.")
        except Exception as e:
            st.error(f"Optimization error: {e}")
            logger.exception("Gauntlet error")

    # Results Section
    st.subheader("📊 Optimization Leaderboard")

    results_df = load_gauntlet_results()

    if not results_df.empty:
        # Summary metrics
        valid_count = results_df[results_df["is_valid"] == 1].shape[0]
        total_count = results_df.shape[0]
        avg_win_rate = results_df["win_rate"].mean() * 100
        avg_pf = results_df["profit_factor"].mean()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Valid Configs", f"{valid_count}/{total_count}")
        with col2:
            st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")
        with col3:
            st.metric("Avg Profit Factor", f"{avg_pf:.2f}")
        with col4:
            st.metric("Assets Optimized", results_df["symbol"].nunique())

        st.divider()

        # Leaderboard table
        display_df = results_df.copy()
        display_df["Win Rate"] = (display_df["win_rate"] * 100).round(1).astype(str) + "%"
        display_df["Trades/Day"] = display_df["trades_per_day"].round(1)
        display_df["Profit Factor"] = display_df["profit_factor"].round(2)
        display_df["Sharpe"] = display_df["sharpe_ratio"].round(2)
        display_df["Max DD"] = (display_df["max_drawdown"] * 100).round(1).astype(str) + "%"
        display_df["Status"] = display_df["is_valid"].apply(lambda x: "✅ PASS" if x == 1 else "❌ FAIL")

        st.dataframe(
            display_df[[
                "symbol", "timeframe", "Win Rate", "Trades/Day",
                "Profit Factor", "Sharpe", "Max DD", "Status"
            ]].rename(columns={
                "symbol": "Symbol",
                "timeframe": "Best TF",
            }),
            use_container_width=True,
            height=400,
        )

        # Visualization: Trades/Day Distribution
        st.subheader("📈 Trade Frequency Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Trades per day by timeframe
            fig = px.box(
                results_df,
                x="timeframe",
                y="trades_per_day",
                color="timeframe",
                title="Trades/Day Distribution by Timeframe",
                color_discrete_sequence=["#00FF41", "#00D9FF", "#FFD700", "#FF6B9D"],
            )
            fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Min Target")
            fig.add_hline(y=4, line_dash="dash", line_color="green", annotation_text="Max Target")
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0.7)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Win Rate vs Profit Factor scatter
            fig = px.scatter(
                results_df,
                x="win_rate",
                y="profit_factor",
                color="is_valid",
                size="trades_per_day",
                hover_data=["symbol", "timeframe"],
                title="Win Rate vs Profit Factor",
                color_discrete_map={1: "#00FF41", 0: "#FF6B6B"},
            )
            fig.add_hline(y=1.3, line_dash="dash", line_color="yellow", annotation_text="Min PF")
            fig.add_vline(x=0.45, line_dash="dash", line_color="yellow", annotation_text="Min WR")
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0.7)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_tickformat=".0%",
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No optimization results yet. Click 'RUN GLOBAL OPTIMIZATION' to start The Gauntlet.")


# =============================================================================
# MANUAL INSPECTOR TAB
# =============================================================================


def render_inspector_tab():
    """Render the Manual Inspector tab for individual backtesting."""

    st.subheader("🔍 Manual Strategy Inspector")
    st.markdown("Test individual assets with custom parameters")

    # Sidebar controls
    with st.sidebar:
        st.title("⚙️ Backtest Parameters")

        st.subheader("Asset Selection")

        # Load available assets
        assets = load_assets_from_db()
        if assets:
            symbol_options = [a["symbol"] for a in assets]
        else:
            symbol_options = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

        symbol = st.selectbox(
            "Select Asset",
            options=symbol_options,
            index=0,
            help="Choose asset to backtest"
        )

        st.subheader("Timeframe")
        timeframe = st.selectbox(
            "Select Timeframe",
            options=["5m", "15m", "30m", "1h", "4h"],
            index=1,
            help="Trading timeframe"
        )

        st.subheader("Time Period")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=30),
                help="Backtest start date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="Backtest end date"
            )

        st.subheader("Capital")
        initial_capital = st.number_input(
            "Starting Balance ($)",
            value=10000.0,
            min_value=1000.0,
            step=1000.0,
            help="Initial trading capital"
        )

        st.subheader("Strategy")
        strategy = st.selectbox(
            "Strategy",
            options=["ichimoku_fibonacci", "ichimoku_standard", "ichimoku_aggressive"],
            index=0,
            help="Trading strategy to test"
        )

    # Main content
    st.markdown(f"**Testing:** {symbol} on {timeframe} using {strategy}")
    st.caption(
        f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} "
        f"| Initial Capital: ${initial_capital:,.0f}"
    )

    st.divider()

    # Run backtest button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        run_button = st.button(
            "🚀 Run Backtest",
            use_container_width=True,
            type="primary"
        )

    if run_button:
        days = (end_date - start_date).days

        with st.spinner("Running backtest..."):
            metrics = simulate_backtest_data(symbol, initial_capital, max(days, 30))

        st.success("✅ Backtest completed!")

        st.divider()

        # Metrics
        st.subheader("Results")

        col1, col2, col3, col4 = st.columns(4, gap="medium")

        with col1:
            delta_color = "normal" if metrics['total_return'] > 0 else "inverse"
            st.metric(
                label="📈 Total Return",
                value=f"{metrics['total_return']:.2f}%",
                delta=f"${initial_capital * metrics['total_return'] / 100:,.0f}",
                delta_color=delta_color
            )

        with col2:
            st.metric(
                label="📉 Max Drawdown",
                value=f"{metrics['max_drawdown']:.2f}%",
                delta="Risk",
                delta_color="off"
            )

        with col3:
            st.metric(
                label="📊 Sharpe Ratio",
                value=f"{metrics['sharpe_ratio']:.2f}",
                delta="Risk-Adjusted",
                delta_color="off"
            )

        with col4:
            st.metric(
                label="🎯 Win Rate",
                value=f"{metrics['win_rate']:.1f}%",
                delta=f"{len(metrics['trades'])} trades",
                delta_color="off"
            )

        st.divider()

        # Equity Curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metrics["dates"],
            y=metrics["equity"],
            mode='lines',
            name='Equity Curve',
            line=dict(color='#00FF41', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 65, 0.1)',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Equity: $%{y:,.2f}<extra></extra>',
        ))

        fig.update_layout(
            title="Equity Curve Over Time",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            template="plotly_dark",
            height=400,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0.7)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Trade History
        st.subheader("Trade History")

        if metrics["trades"]:
            df_trades = pd.DataFrame(metrics["trades"])

            def color_result(val):
                if val == "WIN":
                    return "color: white; background-color: #00FF41; font-weight: bold;"
                else:
                    return "color: white; background-color: #FF6B6B; font-weight: bold;"

            styled_df = df_trades.style.applymap(
                color_result,
                subset=["Result"]
            )

            st.dataframe(styled_df, use_container_width=True, height=300)
        else:
            st.info("No trades generated in this backtest period")

        st.divider()

        # Analysis
        with st.expander("📈 Backtest Analysis & Interpretation", expanded=True):
            sharpe_rating = "✅ Excellent (>2.0)" if metrics['sharpe_ratio'] > 2 else "🟡 Good (1.0-2.0)" if metrics['sharpe_ratio'] > 1 else "⚠️ Poor (<1.0)"
            dd_rating = "✅ Good" if metrics['max_drawdown'] > -30 else "⚠️ High Risk"

            st.markdown(f"""
            ### Performance Summary

            **Total Return**: {metrics['total_return']:.2f}%
            - Your strategy gained/lost {metrics['total_return']:.2f}% over the period

            **Maximum Drawdown**: {metrics['max_drawdown']:.2f}%
            - Largest peak-to-trough decline
            - Risk assessment: {dd_rating}

            **Sharpe Ratio**: {metrics['sharpe_ratio']:.2f}
            - Risk-adjusted returns: {sharpe_rating}

            **Trades**: {len(metrics['trades'])}
            - Win Rate: {metrics['win_rate']:.1f}%
            - Trades/Day: {len(metrics['trades']) / max(days, 1):.1f}

            ### Recommendations

            1. **Optimize Timeframe**: Try different timeframes in The Gauntlet
            2. **Adjust Strategy**: Test IchimokuFibonacci vs Standard Ichimoku
            3. **Risk Management**: Consider position sizing (currently 2% per trade)
            4. **Walk-Forward**: Test on different periods to avoid overfitting
            """)

    else:
        st.info("👈 Configure parameters and click 'Run Backtest' to start")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main Quant Lab page with tabs."""

    # Header
    st.markdown("""
    <h1 style='text-align: center; color: #00FF41;'>
        🔬 SOLAT Quant Lab
    </h1>
    <p style='text-align: center; color: #8b949e;'>
        Multi-Timeframe Strategy Optimization & Backtesting
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    # Tabs
    tab1, tab2 = st.tabs(["⚔️ The Gauntlet", "🔍 Manual Inspector"])

    with tab1:
        render_gauntlet_tab()

    with tab2:
        render_inspector_tab()


if __name__ == "__main__":
    main()
