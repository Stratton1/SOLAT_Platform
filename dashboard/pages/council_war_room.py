"""
COUNCIL WAR ROOM
================
Real-time visualization of the Council of 6 voting on trades.

Aesthetic: Retro-futuristic brutalist hacker terminal
- High contrast neon on dark
- CRT effects and scanlines
- Grid-based combat visualization
- Real-time voting mechanics displayed as tactical decisions
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Tuple

# ============================================================================
# CUSTOM CSS - War Room Terminal Aesthetic
# ============================================================================

WAR_ROOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --primary-neon: #00FF41;
    --secondary-neon: #00D9FF;
    --danger-neon: #FF0055;
    --warning-neon: #FFD700;
    --dark-bg: #0A0E27;
    --darker-bg: #050810;
    --grid-line: rgba(0, 255, 65, 0.1);
    --scanline: rgba(255, 255, 255, 0.03);
}

/* Global */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--dark-bg) !important;
    color: var(--primary-neon) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

/* Scanline effect overlay */
body::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: repeating-linear-gradient(
        0deg,
        var(--scanline),
        var(--scanline) 1px,
        transparent 1px,
        transparent 2px
    );
    pointer-events: none;
    z-index: 999;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Mono', monospace !important;
    color: var(--secondary-neon) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    font-weight: 700 !important;
}

/* Containers with grid border */
[data-testid="stVerticalBlock"] {
    border-left: 2px solid var(--grid-line) !important;
    border-top: 1px solid var(--grid-line) !important;
    padding-left: 1rem !important;
}

/* Metrics cards */
.metric-card {
    background: linear-gradient(135deg, rgba(0, 255, 65, 0.05), rgba(0, 217, 255, 0.02));
    border: 1px solid var(--grid-line);
    border-left: 3px solid var(--primary-neon);
    padding: 1rem;
    border-radius: 0;
    box-shadow: inset 0 0 20px rgba(0, 255, 65, 0.02);
}

/* Vote bars (progress-like) */
.vote-bar {
    background: var(--darker-bg);
    border: 1px solid var(--grid-line);
    height: 2rem;
    position: relative;
    overflow: hidden;
}

.vote-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary-neon), var(--secondary-neon));
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--dark-bg);
    font-weight: bold;
    font-size: 0.75rem;
    text-shadow: 0 0 5px rgba(0, 255, 65, 0.5);
    box-shadow: 0 0 10px rgba(0, 255, 65, 0.3) inset;
}

.vote-bar-fill.positive {
    background: linear-gradient(90deg, #00FF41, #00FF41);
}

.vote-bar-fill.negative {
    background: linear-gradient(90deg, #FF0055, #FF0055);
}

.vote-bar-fill.neutral {
    background: linear-gradient(90deg, #FFD700, #FFD700);
}

/* Consensus gauge */
.consensus-gauge {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: bold;
    color: var(--secondary-neon);
    text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
    letter-spacing: 0.1em;
}

.consensus-high {
    color: var(--primary-neon);
    text-shadow: 0 0 20px rgba(0, 255, 65, 0.7);
}

.consensus-moderate {
    color: var(--warning-neon);
    text-shadow: 0 0 15px rgba(255, 215, 0, 0.5);
}

.consensus-low {
    color: var(--danger-neon);
    text-shadow: 0 0 10px rgba(255, 0, 85, 0.5);
}

/* Agent badges */
.agent-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    background: linear-gradient(135deg, rgba(0, 255, 65, 0.1), rgba(0, 217, 255, 0.05));
    border: 1px solid var(--primary-neon);
    border-radius: 0;
    color: var(--primary-neon);
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0.25rem;
    box-shadow: 0 0 10px rgba(0, 255, 65, 0.2) inset;
}

.agent-badge.active {
    box-shadow: 0 0 15px rgba(0, 255, 65, 0.5) inset, 0 0 20px rgba(0, 255, 65, 0.3);
    animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 15px rgba(0, 255, 65, 0.5) inset, 0 0 20px rgba(0, 255, 65, 0.3); }
    50% { box-shadow: 0 0 25px rgba(0, 255, 65, 0.8) inset, 0 0 40px rgba(0, 255, 65, 0.5); }
}

/* Trading status indicator */
.status-indicator {
    display: inline-block;
    width: 0.75rem;
    height: 0.75rem;
    border-radius: 50%;
    margin-right: 0.5rem;
    animation: blink 1.5s ease-in-out infinite;
}

.status-bullish {
    background: var(--primary-neon);
    box-shadow: 0 0 10px var(--primary-neon);
}

.status-bearish {
    background: var(--danger-neon);
    box-shadow: 0 0 10px var(--danger-neon);
}

.status-neutral {
    background: var(--warning-neon);
    box-shadow: 0 0 10px var(--warning-neon);
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Sidebar enhancement */
[data-testid="collapsedControl"] > button, [data-testid="stSidebar"] {
    background: var(--darker-bg) !important;
    border-right: 2px solid var(--grid-line) !important;
}

/* Buttons */
button {
    background: linear-gradient(135deg, rgba(0, 255, 65, 0.1), rgba(0, 217, 255, 0.05)) !important;
    border: 1px solid var(--primary-neon) !important;
    color: var(--primary-neon) !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

button:hover {
    background: linear-gradient(135deg, rgba(0, 255, 65, 0.2), rgba(0, 217, 255, 0.1)) !important;
    box-shadow: 0 0 15px rgba(0, 255, 65, 0.4) inset !important;
}

</style>
"""

# ============================================================================
# PAGE SETUP
# ============================================================================

def setup_page():
    """Configure page and inject custom CSS"""
    st.set_page_config(
        page_title="SOLAT Council War Room",
        page_icon="🎖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(WAR_ROOM_CSS, unsafe_allow_html=True)


# ============================================================================
# DATA LAYER
# ============================================================================

@st.cache_resource
def get_db_connection():
    """Get database connection"""
    return sqlite3.connect("data/db/trading_engine.db")


def get_latest_consensus_data() -> Dict:
    """Fetch latest consensus voting data from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get latest market snapshot with voting data
        cursor.execute("""
            SELECT symbol, consensus_score, regime_vote, strategy_vote,
                   sniper_vote, news_vote, seasonality_vote, institutional_vote,
                   updated_at
            FROM market_snapshots
            ORDER BY updated_at DESC
            LIMIT 10
        """)

        snapshots = cursor.fetchall()
        conn.close()

        if not snapshots:
            return {
                'symbol': 'DEMO',
                'consensus_score': 0.0,
                'votes': {
                    'regime': 0.0,
                    'strategy': 0.0,
                    'sniper': 0.0,
                    'news': 0.0,
                    'seasonality': 0.0,
                    'institutional': 0.0
                },
                'timestamp': datetime.utcnow().isoformat()
            }

        latest = snapshots[0]
        return {
            'symbol': latest[0],
            'consensus_score': latest[1] or 0.0,
            'votes': {
                'regime': latest[2] or 0.0,
                'strategy': latest[3] or 0.0,
                'sniper': latest[4] or 0.0,
                'news': latest[5] or 0.0,
                'seasonality': latest[6] or 0.0,
                'institutional': latest[7] or 0.0
            },
            'timestamp': latest[8] or datetime.utcnow().isoformat()
        }

    except Exception as e:
        st.warning(f"Database read error: {e}")
        return {
            'symbol': 'ERROR',
            'consensus_score': 0.0,
            'votes': {k: 0.0 for k in ['regime', 'strategy', 'sniper', 'news', 'seasonality', 'institutional']},
            'timestamp': datetime.utcnow().isoformat()
        }


def get_agent_stats() -> Dict[str, Dict]:
    """Fetch agent trust scores and accuracy"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT agent_name, base_weight, correct_calls, wrong_calls, win_rate
            FROM ai_trust_scores
        """)

        stats = {}
        for row in cursor.fetchall():
            agent_name, weight, correct, wrong, win_rate = row
            stats[agent_name] = {
                'weight': weight or 0.166,
                'correct': correct or 0,
                'wrong': wrong or 0,
                'win_rate': win_rate or 0.50
            }

        conn.close()
        return stats

    except Exception:
        return {
            'regime': {'weight': 0.20, 'correct': 0, 'wrong': 0, 'win_rate': 0.50},
            'strategy': {'weight': 0.22, 'correct': 0, 'wrong': 0, 'win_rate': 0.50},
            'sniper': {'weight': 0.15, 'correct': 0, 'wrong': 0, 'win_rate': 0.50},
            'news': {'weight': 0.12, 'correct': 0, 'wrong': 0, 'win_rate': 0.50},
            'seasonality': {'weight': 0.16, 'correct': 0, 'wrong': 0, 'win_rate': 0.50},
            'institutional': {'weight': 0.15, 'correct': 0, 'wrong': 0, 'win_rate': 0.50},
        }


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_consensus_gauge(score: float) -> str:
    """Render ASCII consensus gauge"""
    abs_score = abs(score)
    filled = int(abs_score * 20)
    direction = "▸" if score > 0 else "◂" if score < 0 else "●"

    gauge = "".join(["■" if i < filled else "□" for i in range(20)])

    return f"{direction} {gauge} {score:+.2f}"


def render_agent_vote_bar(agent_name: str, vote: float, weight: float):
    """Render a single agent vote bar"""
    col1, col2, col3 = st.columns([2, 5, 1])

    with col1:
        st.markdown(f"**{agent_name.upper()}**")

    with col2:
        # Vote bar visualization
        percent = int((vote + 1.0) / 2.0 * 100)  # Convert [-1, 1] to [0, 100]
        color_class = "positive" if vote > 0.3 else "negative" if vote < -0.3 else "neutral"

        st.markdown(f"""
        <div class="vote-bar">
            <div class="vote-bar-fill {color_class}" style="width: {percent}%">
                {vote:+.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"`{weight:.1%}`")


def render_council_voting_matrix(consensus_data: Dict):
    """Render the 6-agent voting matrix"""
    st.markdown("### ⚔️ COUNCIL VOTING MATRIX")

    votes = consensus_data['votes']
    agent_stats = get_agent_stats()

    # Create voting grid
    agents = ['regime', 'strategy', 'sniper', 'news', 'seasonality', 'institutional']

    for agent in agents:
        vote = votes.get(agent, 0.0)
        stats = agent_stats.get(agent, {})
        weight = stats.get('weight', 0.166)

        render_agent_vote_bar(agent, vote, weight)
        st.markdown("---")


def render_consensus_decision(score: float) -> Tuple[str, str, str]:
    """Determine decision and styling from consensus score"""
    abs_score = abs(score)

    if abs_score > 0.80:
        confidence = "EXTREME"
    elif abs_score > 0.60:
        confidence = "HIGH"
    elif abs_score > 0.40:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    if score > 0.60:
        decision = "STRONG BUY"
        css_class = "consensus-high"
    elif score > 0.20:
        decision = "BUY"
        css_class = "consensus-moderate"
    elif score < -0.60:
        decision = "STRONG SELL"
        css_class = "consensus-high"
    elif score < -0.20:
        decision = "SELL"
        css_class = "consensus-moderate"
    else:
        decision = "HOLD"
        css_class = "consensus-low"

    return decision, confidence, css_class


# ============================================================================
# MAIN PAGE
# ============================================================================

def main():
    setup_page()

    # Header
    st.markdown("# 🎖️ COUNCIL WAR ROOM")
    st.markdown("*Real-time Consensus Voting System*")
    st.markdown("---")

    # Fetch data
    consensus_data = get_latest_consensus_data()
    agent_stats = get_agent_stats()
    score = consensus_data['consensus_score']

    # Decision indicator
    decision, confidence, css_class = render_consensus_decision(score)

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #00D9FF; text-transform: uppercase; letter-spacing: 0.1em;">Symbol</div>
            <div style="font-size: 1.5rem; color: #00FF41; font-weight: bold; margin-top: 0.5rem;">{consensus_data['symbol']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #00D9FF; text-transform: uppercase; letter-spacing: 0.1em;">Decision</div>
            <div style="font-size: 1.5rem; color: #00FF41; font-weight: bold; margin-top: 0.5rem; text-transform: uppercase;">{decision}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #00D9FF; text-transform: uppercase; letter-spacing: 0.1em;">Confidence</div>
            <div style="font-size: 1.5rem; color: #FFD700; font-weight: bold; margin-top: 0.5rem; text-transform: uppercase;">{confidence}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        status_color = "#00FF41" if score > 0 else "#FF0055" if score < 0 else "#FFD700"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: #00D9FF; text-transform: uppercase; letter-spacing: 0.1em;">Score</div>
            <div style="font-size: 1.5rem; color: {status_color}; font-weight: bold; margin-top: 0.5rem; text-shadow: 0 0 10px {status_color}66;">{score:+.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Consensus gauge
    st.markdown("### 📊 CONSENSUS GAUGE")
    gauge = render_consensus_gauge(score)
    st.code(gauge, language="text")

    st.markdown("---")

    # Council voting matrix
    render_council_voting_matrix(consensus_data)

    st.markdown("---")

    # Agent performance stats
    st.markdown("### 🎯 AGENT PERFORMANCE")

    agent_df = pd.DataFrame([
        {
            'Agent': agent.upper(),
            'Weight': f"{stats['weight']:.1%}",
            'Win Rate': f"{stats['win_rate']:.1%}",
            'Correct': stats['correct'],
            'Wrong': stats['wrong'],
            'Total': stats['correct'] + stats['wrong']
        }
        for agent, stats in agent_stats.items()
    ])

    st.dataframe(agent_df, use_container_width=True)

    st.markdown("---")

    # Timestamp
    st.caption(f"Last update: {consensus_data['timestamp']}")


if __name__ == "__main__":
    main()
