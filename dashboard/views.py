"""
SOLAT Terminal Mode - Text-First UI Components

Production-grade terminal interface with high-density data display.
Bloomberg Terminal aesthetic: Matrix Green (#00FF41) on Deep Space Black (#0E1117)
No emojis - all indicators use text badges [PASS], [FAIL], [ONLINE], etc.
"""

import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def load_terminal_css() -> None:
    """Load Terminal Mode CSS styling - Matrix Green on Deep Space Black."""
    terminal_css = """
    <style>
    /* Terminal Mode Core Styling */
    :root {
        --terminal-green: #00FF41;
        --terminal-bg: #0E1117;
        --terminal-dark: #1a1f2e;
    }

    /* Force monospace throughout */
    * {
        font-family: 'Courier New', 'JetBrains Mono', monospace !important;
    }

    /* Remove emojis and enforce text badges */
    .metric-label, .metric-value {
        font-family: 'Courier New', monospace !important;
    }

    /* Text badge styling */
    .terminal-badge {
        display: inline-block;
        padding: 2px 6px;
        background: var(--terminal-dark);
        border: 1px solid var(--terminal-green);
        color: var(--terminal-green);
        font-weight: bold;
        margin: 0 2px;
        font-size: 11px;
    }

    .badge-pass {
        border-color: #00FF41;
        color: #00FF41;
    }

    .badge-fail {
        border-color: #FF0055;
        color: #FF0055;
    }

    .badge-warn {
        border-color: #FFD700;
        color: #FFD700;
    }

    .badge-online {
        border-color: #00FF41;
        color: #00FF41;
    }

    .badge-offline {
        border-color: #FF0055;
        color: #FF0055;
    }
    </style>
    """
    st.markdown(terminal_css, unsafe_allow_html=True)


def render_header() -> None:
    """Render Terminal Mode header with system status."""
    # Calculate system status
    now = datetime.now()
    latency_ms = 42  # Simulated

    header_html = f"""
    <div style="
        font-family: 'Courier New', monospace;
        color: #00FF41;
        background: #0E1117;
        padding: 12px;
        border: 1px solid #00FF41;
        border-radius: 0;
        margin-bottom: 16px;
    ">
        <div style="display: flex; justify-content: space-between; font-size: 12px;">
            <span>[SOLAT_TERMINAL_v1.0]</span>
            <span>SYSTEM: [ONLINE] | LATENCY: {latency_ms}ms | AGENTS: 6/6 | MODE: [PAPER]</span>
            <span>{now.strftime('%Y-%m-%d %H:%M:%S')}</span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_council_grid(votes_dict: Dict[str, float], reasons_dict: Dict[str, str] = None) -> None:
    """
    Render 6-column Council voting grid (text-first, no emojis).

    Args:
        votes_dict: Dict of {agent_name: vote_value} where vote is -1.0 to +1.0
        reasons_dict: Dict of {agent_name: reasoning_text} for audit trail
    """
    if reasons_dict is None:
        reasons_dict = {}

    st.markdown("### COUNCIL OF 6 - PARALLEL VOTING", unsafe_allow_html=True)

    # Calculate consensus
    votes = list(votes_dict.values())
    consensus = sum(votes) / len(votes) if votes else 0.0

    # Determine decision
    if consensus > 0.6:
        decision = "STRONG_BUY"
        decision_color = "#00FF41"
    elif consensus > 0.2:
        decision = "BUY"
        decision_color = "#00FF41"
    elif consensus < -0.6:
        decision = "STRONG_SELL"
        decision_color = "#FF0055"
    elif consensus < -0.2:
        decision = "SELL"
        decision_color = "#FF0055"
    else:
        decision = "HOLD"
        decision_color = "#FFD700"

    # Render decision bar
    decision_html = f"""
    <div style="
        font-family: 'Courier New', monospace;
        color: {decision_color};
        background: #1a1f2e;
        padding: 8px;
        border-left: 3px solid {decision_color};
        margin-bottom: 12px;
        font-size: 13px;
    ">
        CONSENSUS: {consensus:+.2f} | DECISION: [{decision}] | CONFIDENCE: {abs(consensus)*100:.0f}%
    </div>
    """
    st.markdown(decision_html, unsafe_allow_html=True)

    # Create 6-column grid
    cols = st.columns(6, gap="small")

    agent_names = ['REGIME', 'STRATEGY', 'SNIPER', 'NEWS', 'SEASONALITY', 'INSTITUTIONAL']

    for col_idx, col in enumerate(cols):
        agent = agent_names[col_idx]
        vote = votes_dict.get(agent.lower(), 0.0)
        reason = reasons_dict.get(agent.lower(), "No reason provided")

        # Determine vote color
        if vote > 0.5:
            vote_color = "#00FF41"
        elif vote > 0:
            vote_color = "#90EE90"
        elif vote < -0.5:
            vote_color = "#FF0055"
        elif vote < 0:
            vote_color = "#FF6B9D"
        else:
            vote_color = "#FFD700"

        agent_html = f"""
        <div style="
            font-family: 'Courier New', monospace;
            color: #00FF41;
            background: #1a1f2e;
            padding: 8px;
            border: 1px solid #00FF41;
            border-radius: 0;
            font-size: 11px;
            line-height: 1.4;
        ">
            <div style="color: #00FF41; font-weight: bold; margin-bottom: 4px;">[{agent}]</div>
            <div style="color: {vote_color}; font-size: 13px; font-weight: bold; margin-bottom: 4px;">{vote:+.2f}</div>
            <div style="color: #8b949e; font-size: 10px;">{reason[:50]}</div>
        </div>
        """
        with col:
            st.markdown(agent_html, unsafe_allow_html=True)


def render_surveillance_table(df: pd.DataFrame) -> None:
    """
    Render high-density surveillance table with Terminal Mode styling.
    Text-first: SYMBOL | PRICE | REGIME | CONSENSUS | ACTION
    """
    if df is None or df.empty:
        st.warning("[ERROR] No market data available")
        return

    st.markdown("### MARKET SURVEILLANCE", unsafe_allow_html=True)

    # Prepare columns for display
    display_cols = ['symbol', 'close_price', 'fitness_score', 'status']
    if 'cloud_status' in df.columns:
        display_cols.append('cloud_status')
    if 'chikou_conf' in df.columns:
        display_cols.append('chikou_conf')

    display_df = df[display_cols].copy()

    # Rename columns
    display_df.columns = ['SYMBOL', 'PRICE', 'FITNESS', 'STATUS', *display_df.columns[4:].tolist()]

    # Format for terminal display
    display_df['PRICE'] = display_df['PRICE'].apply(lambda x: f"${x:,.2f}")
    display_df['FITNESS'] = display_df['FITNESS'].apply(lambda x: f"{x:.4f}")

    # Status badges
    def status_badge(s):
        if s == 'active':
            return '[ACTIVE]'
        elif s == 'normal':
            return '[NORMAL]'
        else:
            return '[DORMANT]'

    display_df['STATUS'] = display_df['STATUS'].apply(status_badge)

    # Signal badges
    if 'chikou_conf' in display_df.columns:
        def signal_badge(s):
            if s == 'BUY':
                return '[BUY]'
            elif s == 'SELL':
                return '[SELL]'
            else:
                return '[HOLD]'
        display_df['chikou_conf'] = display_df['chikou_conf'].apply(signal_badge)

    # Render as table
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        column_config={
            'SYMBOL': st.column_config.TextColumn(width=80),
            'PRICE': st.column_config.TextColumn(width=100),
            'FITNESS': st.column_config.TextColumn(width=90),
            'STATUS': st.column_config.TextColumn(width=80),
            'cloud_status': st.column_config.TextColumn(width=80),
            'chikou_conf': st.column_config.TextColumn(width=80),
        }
    )


def render_system_status() -> None:
    """Render system status panel with text badges."""
    col1, col2, col3, col4, col5 = st.columns(5, gap="small")

    with col1:
        st.metric(
            label="SYSTEM_STATUS",
            value="[ONLINE]",
            delta=None,
        )

    with col2:
        st.metric(
            label="OPEN_TRADES",
            value="2/5",
            delta=None,
        )

    with col3:
        st.metric(
            label="RISK_USAGE",
            value="4%",
            delta=None,
        )

    with col4:
        st.metric(
            label="WIN_RATE",
            value="67%",
            delta=None,
        )

    with col5:
        st.metric(
            label="AGENTS_ONLINE",
            value="6/6",
            delta=None,
        )


def render_agent_stats(agent_stats: Dict[str, Dict[str, float]]) -> None:
    """
    Render agent trust scores and performance metrics.

    Args:
        agent_stats: Dict of {agent_name: {weight, win_rate, correct, total}}
    """
    st.markdown("### AGENT TRUST SCORES", unsafe_allow_html=True)

    stats_data = []
    for agent, stats in agent_stats.items():
        weight = stats.get('weight', 0.0)
        win_rate = stats.get('win_rate', 0.0)
        correct = stats.get('correct', 0)
        total = stats.get('total', 1)

        # Determine status badge
        if win_rate > 0.7:
            status = '[STRONG]'
        elif win_rate > 0.5:
            status = '[GOOD]'
        else:
            status = '[WEAK]'

        stats_data.append({
            'AGENT': agent.upper(),
            'WEIGHT': f"{weight:.3f}",
            'WIN_RATE': f"{win_rate:.1%}",
            'RECORD': f"{correct}/{total}",
            'STATUS': status
        })

    stats_df = pd.DataFrame(stats_data)
    st.dataframe(
        stats_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'AGENT': st.column_config.TextColumn(width=120),
            'WEIGHT': st.column_config.TextColumn(width=80),
            'WIN_RATE': st.column_config.TextColumn(width=100),
            'RECORD': st.column_config.TextColumn(width=100),
            'STATUS': st.column_config.TextColumn(width=80),
        }
    )


def render_sidebar() -> Tuple[bool, bool]:
    """
    Render Terminal Mode sidebar with system controls (text-first).

    Returns:
        Tuple[bool, bool]: (show_only_active, show_only_signals)
    """
    with st.sidebar:
        st.markdown("# COMMAND_CENTER", unsafe_allow_html=True)
        st.markdown("---")

        # System status
        st.markdown("### SYSTEM_STATUS", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("NODE", "[ONLINE]")
        with col2:
            st.metric("UPDATE", "[LIVE]")

        st.markdown("---")

        # Filters
        st.markdown("### FILTERS", unsafe_allow_html=True)
        show_only_active = st.checkbox(
            "Show only [ACTIVE] assets",
            value=False,
            help="Display high-fitness assets only"
        )

        show_only_signals = st.checkbox(
            "Show only [BUY]/[SELL] signals",
            value=False,
            help="Hide [HOLD] signals"
        )

        st.markdown("---")

        # Signal legend
        st.markdown("### SIGNAL_LEGEND", unsafe_allow_html=True)
        st.markdown("""
        **Trading Signals:**
        - [BUY]: Price > Cloud + TK Cross Up
        - [SELL]: Price < Cloud + TK Cross Down
        - [HOLD]: No clear signal

        **Status Badges:**
        - [ACTIVE]: Top 20% fitness
        - [NORMAL]: Middle 60% fitness
        - [DORMANT]: Bottom 20% fitness

        **Agent Badges:**
        - [STRONG]: > 70% win rate
        - [GOOD]: 50-70% win rate
        - [WEAK]: < 50% win rate

        **System Badges:**
        - [ONLINE]: System running
        - [PASS]: Validation passed
        - [FAIL]: Error detected
        """)

        st.markdown("---")

        # Constraints
        st.markdown("### HARD_CONSTRAINTS", unsafe_allow_html=True)
        st.markdown("""
        **Risk Management:**
        - MAX_OPEN_TRADES: 5
        - MAX_POSITION_SIZE: 10%
        - RISK_PER_TRADE: 2%

        **Voting:**
        - CONSENSUS_THRESHOLD: 0.60
        - AGENTS_REQUIRED: 6/6
        - MODE: PARALLEL
        """)

    return show_only_active, show_only_signals


def render_trades_log() -> None:
    """Render recent trades in Terminal Mode format."""
    st.markdown("### TRADE_LOG", unsafe_allow_html=True)

    # Sample trades
    trades_data = {
        'ID': ['001', '002', '003'],
        'SYMBOL': ['BTC/USDT', 'ETH/USDT', 'GBP/USD'],
        'ACTION': ['[BUY]', '[BUY]', '[SELL]'],
        'ENTRY': ['$42350', '$2245', '1.2650'],
        'EXIT': ['$42800', '$2198', '1.2630'],
        'PNL': ['[+$450]', '[-$47]', '[+$20]'],
        'AGENTS': ['5/6', '6/6', '4/6'],
    }

    trades_df = pd.DataFrame(trades_data)
    st.dataframe(
        trades_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'ID': st.column_config.TextColumn(width=50),
            'SYMBOL': st.column_config.TextColumn(width=100),
            'ACTION': st.column_config.TextColumn(width=70),
            'ENTRY': st.column_config.TextColumn(width=80),
            'EXIT': st.column_config.TextColumn(width=80),
            'PNL': st.column_config.TextColumn(width=80),
            'AGENTS': st.column_config.TextColumn(width=80),
        }
    )


def render_footer() -> None:
    """Render Terminal Mode footer."""
    now = datetime.now()
    footer_html = f"""
    <div style="
        font-family: 'Courier New', monospace;
        color: #8b949e;
        font-size: 10px;
        margin-top: 24px;
        padding-top: 12px;
        border-top: 1px solid #1a1f2e;
        text-align: center;
    ">
        SOLAT_TERMINAL | CONSENSUS_VOTING_V1.0 | WAL_MODE_ENABLED | UPDATED: {now.strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
