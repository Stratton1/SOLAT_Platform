"""
Ichimoku Cloud Chart Rendering Module

This module handles all chart generation for the SOLAT dashboard.
Uses Plotly for interactive, production-grade visualizations.
"""

import logging
from typing import Dict, Any

import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def render_ichimoku_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Render a comprehensive Ichimoku Cloud chart with regime-based background shading.

    Creates an interactive candlestick chart with:
    - OHLCV candlesticks (main price action)
    - Tenkan-sen (9-period momentum, blue line)
    - Kijun-sen (26-period momentum, red line)
    - Senkou Span A (cloud upper boundary)
    - Senkou Span B (cloud lower boundary, filled)
    - Chikou Span (lagging line, purple)
    - Background shading showing historical market regimes

    Cloud color:
    - GREEN: Senkou A > B (bullish cloud)
    - RED: Senkou A < B (bearish cloud)

    Background shading:
    - GREEN: Bull regime periods
    - RED: Bear regime periods
    - GREY: Chop regime periods

    Args:
        df (pd.DataFrame): DataFrame with columns:
            - open, high, low, close, volume (OHLCV)
            - tenkan, kijun (momentum lines)
            - senkou_a, senkou_b (cloud boundaries)
            - chikou (lagging line)
            - regime (optional, for background shading)
            Must have DatetimeIndex
        symbol (str): Asset symbol for title

    Returns:
        go.Figure: Interactive Plotly figure
    """
    if df is None or df.empty:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        return fig

    # Ensure we have required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    ichimoku_cols = ["tenkan", "kijun", "senkou_a", "senkou_b", "chikou"]

    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Missing column: {col}")

    # Create figure
    fig = go.Figure()

    # Add regime-based background shading if regime column exists
    if "regime" in df.columns:
        # Get price range for background rectangles
        price_min = df["low"].min()
        price_max = df["high"].max()

        # Identify regime change points
        regime_shifts = df["regime"].ne(df["regime"].shift()).cumsum()

        # Add background rectangles for each regime zone
        for regime_group, group_df in df.groupby(regime_shifts):
            regime = group_df["regime"].iloc[0]
            start_date = group_df.index[0]
            end_date = group_df.index[-1]

            # Map regime to color
            if regime == "bull":
                color = "rgba(0, 255, 0, 0.05)"  # Very transparent green
            elif regime == "bear":
                color = "rgba(255, 0, 0, 0.05)"  # Very transparent red
            elif regime == "chop":
                color = "rgba(128, 128, 128, 0.05)"  # Very transparent grey
            else:
                color = "rgba(200, 200, 200, 0.02)"  # Barely visible for neutral
                continue  # Skip neutral regime

            # Add background rectangle
            fig.add_vrect(
                x0=start_date,
                x1=end_date,
                fillcolor=color,
                layer="below",
                line_width=0,
                annotation_text=regime.upper() if len(regime) < 10 else "",
                annotation_position="top left"
            )

    # Add candlestick trace (OHLCV)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=f"{symbol} Price",
        increasing_line_color="green",
        decreasing_line_color="red",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>" +
                      "Open: $%{open:.2f}<br>" +
                      "High: $%{high:.2f}<br>" +
                      "Low: $%{low:.2f}<br>" +
                      "Close: $%{close:.2f}<br>" +
                      "Volume: %{customdata}<extra></extra>",
        customdata=df["volume"]
    ))

    # Add Ichimoku components if available
    if "tenkan" in df.columns and not df["tenkan"].isna().all():
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["tenkan"],
            mode="lines",
            name="Tenkan-sen (9)",
            line=dict(color="blue", width=1),
            hovertemplate="Tenkan: $%{y:.2f}<extra></extra>"
        ))

    if "kijun" in df.columns and not df["kijun"].isna().all():
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["kijun"],
            mode="lines",
            name="Kijun-sen (26)",
            line=dict(color="red", width=1),
            hovertemplate="Kijun: $%{y:.2f}<extra></extra>"
        ))

    # Add cloud (Senkou Span A and B)
    if "senkou_a" in df.columns and "senkou_b" in df.columns:
        # Determine cloud color based on Senkou relationship
        # We'll add cloud boundaries but color based on logic

        # Senkou Span B (bottom line of cloud)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["senkou_b"],
            mode="lines",
            name="Senkou Span B (52)",
            line=dict(color="rgba(0,0,0,0)", width=0),  # Invisible line
            hovertemplate="Senkou B: $%{y:.2f}<extra></extra>",
            showlegend=False
        ))

        # Senkou Span A (top line of cloud) - filled area between A and B
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["senkou_a"],
            mode="lines",
            name="Senkou Span A (26)",
            line=dict(color="rgba(0,0,0,0)", width=0),  # Invisible line
            fill="tonexty",  # Fill to previous trace (Senkou B)
            fillcolor="rgba(0, 255, 0, 0.2)",  # Light green (default)
            hovertemplate="Senkou A: $%{y:.2f}<extra></extra>",
            showlegend=False
        ))

        # Determine cloud color and recolor if needed
        # Check if majority of cloud is red (A < B)
        cloud_is_red = (df["senkou_a"] < df["senkou_b"]).sum() > (len(df) * 0.6)

        if cloud_is_red:
            # Update fillcolor to red
            fig.data[-1].fillcolor = "rgba(255, 0, 0, 0.2)"  # Light red

        # Add cloud legend entry (manually)
        cloud_color = "Red Cloud" if cloud_is_red else "Green Cloud"
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=cloud_color,
            marker=dict(
                size=10,
                color="red" if cloud_is_red else "green",
                opacity=0.4
            ),
            showlegend=True
        ))

    # Add Chikou Span if available
    if "chikou" in df.columns and not df["chikou"].isna().all():
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["chikou"],
            mode="lines",
            name="Chikou Span (26 lag)",
            line=dict(color="purple", width=1, dash="dash"),
            hovertemplate="Chikou: $%{y:.2f}<extra></extra>"
        ))

    # Update layout
    fig.update_layout(
        title={
            "text": f"<b>{symbol} - Ichimoku Cloud Chart</b><br><sub>Real-time market monitoring</sub>",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "color": "white"}
        },
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=False),  # Remove rangeslider
            gridcolor="rgba(255,255,255,0.1)",
            showgrid=True
        ),
        yaxis=dict(
            title="Price (USD/USDT)",
            gridcolor="rgba(255,255,255,0.1)",
            showgrid=True
        ),
        template="plotly_dark",  # Dark mode
        hovermode="x unified",
        height=600,
        font=dict(size=11, color="white"),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1
        ),
        plot_bgcolor="rgba(0,0,0,0.7)",
        paper_bgcolor="rgba(0,0,0,0.9)",
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig


def render_fitness_bar_chart(fitness_data: Dict[str, float]) -> go.Figure:
    """
    Render a bar chart of asset fitness scores.

    Args:
        fitness_data (Dict[str, float]): Mapping of symbol to fitness score

    Returns:
        go.Figure: Plotly bar chart
    """
    if not fitness_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No fitness data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        return fig

    # Sort by fitness (top 5)
    sorted_data = sorted(fitness_data.items(), key=lambda x: x[1], reverse=True)[:5]
    symbols = [item[0] for item in sorted_data]
    scores = [item[1] for item in sorted_data]

    fig = go.Figure(data=[
        go.Bar(
            x=symbols,
            y=scores,
            name="Fitness Score",
            marker=dict(
                color=scores,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Fitness")
            ),
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Fitness: %{y:.3f}<extra></extra>"
        )
    ])

    fig.update_layout(
        title="<b>Top 5 Fittest Assets</b>",
        xaxis_title="Asset",
        yaxis_title="Fitness Score",
        template="plotly_dark",
        height=400,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0.7)",
        paper_bgcolor="rgba(0,0,0,0.9)"
    )

    return fig


def render_status_pie_chart(status_counts: Dict[str, int]) -> go.Figure:
    """
    Render a pie chart of asset status distribution.

    Args:
        status_counts (Dict[str, int]): Count by status (active, normal, dormant)

    Returns:
        go.Figure: Plotly pie chart
    """
    if not status_counts:
        fig = go.Figure()
        fig.add_annotation(text="No data available")
        return fig

    labels = list(status_counts.keys())
    values = list(status_counts.values())

    # Color map
    color_map = {
        "active": "rgb(0, 255, 0)",
        "normal": "rgb(255, 165, 0)",
        "dormant": "rgb(255, 0, 0)"
    }
    colors = [color_map.get(label, "gray") for label in labels]

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
        )
    ])

    fig.update_layout(
        title="<b>Asset Status Distribution</b>",
        template="plotly_dark",
        height=400,
        paper_bgcolor="rgba(0,0,0,0.9)"
    )

    return fig
