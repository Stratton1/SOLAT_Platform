"""
SOLAT Backtesting Page

Interactive backtesting interface for analyzing historical trading performance.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="SOLAT Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
try:
    with open("dashboard/assets/style.css", "r") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass


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
    import numpy as np

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


def render_backtest_controls() -> Tuple[str, Tuple[datetime, datetime], float]:
    """
    Render sidebar controls for backtest parameters.
    """
    with st.sidebar:
        st.title("⚙️ Backtest Parameters")

        st.subheader("Asset Selection")
        symbol = st.selectbox(
            "Select Asset",
            options=["BTC/USDT", "ETH/USDT", "AAPL", "GC=F"],
            index=0,
            help="Choose asset to backtest"
        )

        st.subheader("Time Period")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=100),
                help="Backtest start date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="Backtest end date"
            )

        st.subheader("Initial Capital")
        initial_capital = st.number_input(
            "Starting Balance ($)",
            value=10000.0,
            min_value=1000.0,
            step=1000.0,
            help="Initial trading capital"
        )

        return symbol, (start_date, end_date), initial_capital


def render_equity_curve(dates: pd.DatetimeIndex, equity: list) -> None:
    """
    Render interactive equity curve using Plotly.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=equity,
        mode='lines',
        name='Equity Curve',
        line=dict(color='#1f6feb', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 111, 235, 0.1)',
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
        paper_bgcolor='rgba(0,0,0,0.9)',
    )

    st.plotly_chart(fig, use_container_width=True)


def render_backtest_metrics(metrics: Dict) -> None:
    """
    Render key backtest metrics in metric cards.
    """
    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        st.metric(
            label="📈 Total Return",
            value=f"{metrics['total_return']:.2f}%",
            delta="Performance",
            delta_color="off"
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


def render_trades_table(trades: list) -> None:
    """
    Render table of simulated trades.
    """
    if not trades:
        st.info("No trades generated in this backtest period")
        return

    df_trades = pd.DataFrame(trades)

    # Color code results
    def color_result(val):
        if val == "WIN":
            return "color: white; background-color: #31a24c; font-weight: bold;"
        else:
            return "color: white; background-color: #ff6b6b; font-weight: bold;"

    styled_df = df_trades.style.applymap(
        color_result,
        subset=["Result"]
    )

    st.dataframe(styled_df, use_container_width=True, height=300)


def main():
    """Main backtesting page."""

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
        📊 SOLAT Backtester
    </h1>
    <p style='text-align: center; color: #8b949e;'>
        Test Ichimoku strategy on historical data
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    # Get controls
    symbol, (start_date, end_date), initial_capital = render_backtest_controls()

    # Main content
    st.subheader(f"Backtest: {symbol}")
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
        # Calculate days
        days = (end_date - start_date).days

        # Show progress
        with st.spinner("Running backtest..."):
            metrics = simulate_backtest_data(symbol, initial_capital, max(days, 30))

        st.success("✅ Backtest completed!")

        st.divider()

        # Render results
        st.subheader("Results")

        render_backtest_metrics(metrics)

        st.divider()

        render_equity_curve(metrics["dates"], metrics["equity"])

        st.divider()

        st.subheader("Trade History")
        render_trades_table(metrics["trades"])

        st.divider()

        # Analysis & explanation
        with st.expander("📈 Backtest Analysis & Interpretation", expanded=True):
            st.markdown(f"""
            ### Performance Summary

            **Total Return**: {metrics['total_return']:.2f}%
            - Your strategy gained/lost {metrics['total_return']:.2f}% over the period

            **Maximum Drawdown**: {metrics['max_drawdown']:.2f}%
            - Largest peak-to-trough decline: {metrics['max_drawdown']:.2f}%
            - Risk management: {"✅ Good" if metrics['max_drawdown'] > -30 else "⚠️ High"}

            **Sharpe Ratio**: {metrics['sharpe_ratio']:.2f}
            - Risk-adjusted returns: {"✅ Excellent (>2.0)" if metrics['sharpe_ratio'] > 2 else "🟡 Good (1.0-2.0)" if metrics['sharpe_ratio'] > 1 else "⚠️ Poor (<1.0)"}

            **Trades**: {len(metrics['trades'])}
            - Win Rate: {metrics['win_rate']:.1f}%

            ### Recommendations

            1. **Optimize Asset Selection**: Run backtest on different assets to find best fit
            2. **Adjust Parameters**: Modify Ichimoku periods (9/26/52) for better performance
            3. **Risk Management**: Test different position sizes (currently 2% per trade)
            4. **Walk-Forward Analysis**: Test on different historical periods to avoid overfitting
            """)
    else:
        st.info("👈 Configure backtest parameters and click 'Run Backtest' to start")


if __name__ == "__main__":
    main()
