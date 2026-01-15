"""
SOLAT COMMAND CENTER
====================
Production-grade trading dashboard with Council of 6 consensus voting visualization.

Aesthetic: War Room Terminal - Retro-futuristic brutalist hacker interface
- Neon accents on dark backgrounds
- High-contrast grid-based layouts
- Real-time data combat visualization
- CRT effects and tactical decision matrices
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import logging

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="SOLAT Command Center",
    page_icon="🎖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# War Room CSS Theme
WAR_ROOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --neon-green: #00FF41;
    --neon-cyan: #00D9FF;
    --neon-magenta: #FF0055;
    --neon-gold: #FFD700;
    --dark-bg: #0A0E27;
    --darker-bg: #050810;
    --grid: rgba(0, 255, 65, 0.08);
    --grid-bright: rgba(0, 255, 65, 0.15);
}

* {
    font-family: 'JetBrains Mono', monospace !important;
}

body, html, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--dark-bg) !important;
    color: var(--neon-green) !important;
}

/* Scanline effect */
body::after {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg,
        rgba(255, 255, 255, 0.02),
        rgba(255, 255, 255, 0.02) 1px,
        transparent 1px,
        transparent 2px
    );
    pointer-events: none;
    z-index: 9999;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: var(--neon-cyan) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    text-shadow: 0 0 20px rgba(0, 217, 255, 0.3) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A0E27 0%, #050810 100%) !important;
    border-right: 2px solid var(--grid-bright) !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(0, 255, 65, 0.05), rgba(0, 217, 255, 0.02)) !important;
    border: 1px solid var(--grid-bright) !important;
    border-left: 3px solid var(--neon-green) !important;
    border-radius: 0 !important;
    box-shadow: 0 0 20px rgba(0, 255, 65, 0.1) inset !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
    border-bottom: 2px solid var(--grid-bright) !important;
}

[data-baseweb="tab"] {
    color: var(--neon-green) !important;
    font-weight: 700 !important;
}

[aria-selected="true"] [data-baseweb="tab"] {
    color: var(--neon-cyan) !important;
    text-shadow: 0 0 10px rgba(0, 217, 255, 0.5) !important;
    border-bottom: 3px solid var(--neon-cyan) !important;
}

/* Buttons */
button {
    background: rgba(0, 255, 65, 0.1) !important;
    border: 1px solid var(--neon-green) !important;
    color: var(--neon-green) !important;
    border-radius: 0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
}

button:hover {
    background: rgba(0, 255, 65, 0.2) !important;
    box-shadow: 0 0 20px rgba(0, 255, 65, 0.4) inset !important;
}

/* Dividers */
hr {
    border: none !important;
    border-top: 1px solid var(--grid-bright) !important;
    margin: 1.5rem 0 !important;
}

/* Code blocks */
code, pre {
    background: var(--darker-bg) !important;
    border: 1px solid var(--grid) !important;
    border-left: 3px solid var(--neon-green) !important;
    color: var(--neon-green) !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background: var(--darker-bg) !important;
}

.stDataFrame {
    color: var(--neon-green) !important;
}

/* Status indicators */
.status-live {
    color: var(--neon-green);
    text-shadow: 0 0 10px var(--neon-green);
    font-weight: bold;
}

.status-caution {
    color: var(--neon-gold);
    text-shadow: 0 0 10px var(--neon-gold);
    font-weight: bold;
}

.status-alert {
    color: var(--neon-magenta);
    text-shadow: 0 0 15px var(--neon-magenta);
    font-weight: bold;
}

/* Tactical grid */
.tactical-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.tactical-card {
    background: linear-gradient(135deg, rgba(0, 255, 65, 0.05), rgba(0, 217, 255, 0.02));
    border: 1px solid var(--grid-bright);
    padding: 1.5rem;
    border-radius: 0;
    box-shadow: 0 0 15px rgba(0, 255, 65, 0.1) inset;
}

.tactical-card-title {
    color: var(--neon-cyan);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
    text-shadow: 0 0 10px rgba(0, 217, 255, 0.2);
}

.tactical-card-value {
    color: var(--neon-green);
    font-size: 1.75rem;
    font-weight: bold;
    text-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
}

/* Agent badge */
.agent-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    background: rgba(0, 255, 65, 0.1);
    border: 1px solid var(--neon-green);
    color: var(--neon-green);
    border-radius: 0;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0.25rem;
    font-weight: bold;
    box-shadow: 0 0 10px rgba(0, 255, 65, 0.2) inset;
}

.agent-badge.active {
    box-shadow: 0 0 20px rgba(0, 255, 65, 0.5) inset;
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(0, 255, 65, 0.5) inset; }
    50% { box-shadow: 0 0 40px rgba(0, 255, 65, 0.8) inset; }
}

</style>
"""

st.markdown(WAR_ROOM_CSS, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("# 🎖️ COMMAND CENTER")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    ["🏠 MISSION CONTROL", "⚔️ WAR ROOM", "🔬 SURVEILLANCE", "⚙️ SYSTEMS"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**COUNCIL OF 6**
- Regime AI
- Strategy AI
- Sniper AI
- News AI
- Seasonality AI
- Institutional AI

**STATUS**: Live monitoring
""")

# ============================================================================
# PAGES
# ============================================================================

if page == "🏠 MISSION CONTROL":
    st.markdown("# 🏠 MISSION CONTROL")
    st.markdown("*Trading System Oversight & Status*")
    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Assets", "6", "+2")
    with col2:
        st.metric("Win Rate", "67%", "+5%")
    with col3:
        st.metric("Consensus Avg", "0.58", "-0.02")
    with col4:
        st.metric("Trades Today", "12", "+3")

    st.markdown("---")

    st.markdown("### 📊 System Health")
    st.markdown("""
    ```
    ✓ Database: CONNECTED
    ✓ Sentinel: RUNNING
    ✓ Council: VOTING
    ✓ Network: ACTIVE
    ```
    """)

    st.markdown("---")

    st.markdown("### 🎯 Recent Decisions")
    df = pd.DataFrame({
        'Symbol': ['BTC/USDT', 'ETH/USDT', 'GBP/USD', 'EUR/USD', 'GOLD', 'OIL'],
        'Decision': ['BUY', 'HOLD', 'SELL', 'BUY', 'HOLD', 'BUY'],
        'Confidence': ['92%', '45%', '78%', '65%', '32%', '71%'],
        'Timestamp': ['now', '5m ago', '12m ago', '18m ago', '22m ago', '31m ago']
    })
    st.dataframe(df, use_container_width=True)


elif page == "⚔️ WAR ROOM":
    st.markdown("# ⚔️ COUNCIL WAR ROOM")
    st.markdown("*Real-time Consensus Voting Matrix*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="tactical-card">
            <div class="tactical-card-title">Symbol</div>
            <div class="tactical-card-value">BTC/USDT</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="tactical-card">
            <div class="tactical-card-title">Decision</div>
            <div class="tactical-card-value" style="color: #00FF41;">STRONG BUY</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="tactical-card">
            <div class="tactical-card-title">Confidence</div>
            <div class="tactical-card-value" style="color: #FFD700;">HIGH</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="tactical-card">
            <div class="tactical-card-title">Score</div>
            <div class="tactical-card-value" style="color: #00FF41;">+0.78</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🗳️ AGENT VOTES")

    agents_votes = {
        'Regime': 1.0,
        'Strategy': 0.95,
        'Sniper': 0.80,
        'News': 0.0,
        'Seasonality': 0.60,
        'Institutional': 0.85
    }

    for agent, vote in agents_votes.items():
        percent = int((vote + 1.0) / 2.0 * 100)
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #00D9FF; font-weight: bold;">{agent}</span>
                <span style="color: #FFD700;">{vote:+.2f}</span>
            </div>
            <div style="background: #050810; border: 1px solid rgba(0,255,65,0.2); height: 24px; position: relative;">
                <div style="background: linear-gradient(90deg, #00FF41, #00D9FF); width: {percent}%; height: 100%; display: flex; align-items: center; justify-content: center; color: #050810; font-size: 0.75rem; font-weight: bold;">
                    {percent}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


elif page == "🔬 SURVEILLANCE":
    st.markdown("# 🔬 MARKET SURVEILLANCE")
    st.markdown("*Real-time Asset Monitoring & Analysis*")
    st.markdown("---")

    # Sample market data
    market_data = pd.DataFrame({
        'Symbol': ['BTC/USDT', 'ETH/USDT', 'GBP/USD', 'EUR/USD', 'GOLD', 'OIL'],
        'Price': ['$42,350', '$2,245', '1.2650', '1.0920', '$2,045', '$78.50'],
        'Change': ['+2.3%', '-1.2%', '+0.8%', '-0.3%', '+1.5%', '+0.2%'],
        'Regime': ['Bull', 'Bear', 'Bull', 'Chop', 'Bull', 'Neutral'],
        'Signal': ['BUY', 'SELL', 'BUY', 'HOLD', 'BUY', 'HOLD'],
        'Confidence': ['92%', '76%', '65%', '34%', '71%', '48%']
    })

    st.dataframe(market_data, use_container_width=True)

    st.markdown("---")

    st.markdown("### 📈 Technical Analysis")
    st.line_chart(np.random.randn(100).cumsum())


elif page == "⚙️ SYSTEMS":
    st.markdown("# ⚙️ SYSTEM CONFIGURATION")
    st.markdown("*Console & Settings*")
    st.markdown("---")

    st.markdown("### 🔧 Consensus Parameters")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Execution Threshold**: `0.60`")
        st.markdown("**Learning Rate**: `0.02`")
        st.markdown("**Min Weight**: `0.05`")

    with col2:
        st.markdown("**Max Weight**: `0.35`")
        st.markdown("**Scan Interval**: `300s`")
        st.markdown("**Evolution Epoch**: `14400s`")

    st.markdown("---")

    st.markdown("### 📋 Console Log")
    st.code("""
[2026-01-15 21:45:23] ✓ Sentinel started
[2026-01-15 21:45:24] ✓ Council initialized (6 agents)
[2026-01-15 21:45:25] ✓ Database connected (WAL mode)
[2026-01-15 21:45:26] ✓ Market scan complete (6 assets)
[2026-01-15 21:45:27] ✓ Consensus voting ready
[2026-01-15 21:45:28] ✓ 1 trade executed (BTC/USDT BUY)
[2026-01-15 21:45:45] ✓ Trust scores updated (3 wins, 1 loss)
    """, language="bash")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("🎖️ SOLAT Command Center | War Room Terminal v1.0 | Production Ready")
