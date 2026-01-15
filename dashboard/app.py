"""
SOLAT Terminal Dashboard - Council of 6 Mission Control

Terminal Mode UI with Bloomberg-style aesthetics.
Matrix Green (#00FF41) on Deep Space Black (#0E1117).
No emojis - all indicators use text badges [PASS], [FAIL], [ONLINE], etc.
"""

import logging
import json
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Terminal Mode functions (matching views.py)
from dashboard.views import (
    load_terminal_css,
    render_header,
    render_council_grid,
    render_surveillance_table,
    render_system_status,
    render_sidebar,
    render_footer,
)
from src.database.repository import get_db_connection

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SOLAT // TERMINAL",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Terminal CSS
load_terminal_css()

# ============================================================================
# AUTO-REFRESH (30 seconds)
# ============================================================================

count = st_autorefresh(interval=30000)

# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    """Main application entry point."""

    # 1. Sidebar (Controls & Filters)
    show_active, show_signals = render_sidebar()

    # 2. Header (System Status Bar)
    render_header()

    # 3. Live Data Section
    try:
        conn = get_db_connection()

        # A. Fetch Market Data
        query = """
            SELECT
                symbol,
                close_price,
                regime,
                consensus_score,
                signal,
                news_sentiment,
                order_imbalance,
                agent_votes,
                updated_at
            FROM market_snapshots
            ORDER BY updated_at DESC
        """
        df = pd.read_sql(query, conn)

        # Deduplicate to show only latest snapshot per symbol
        if not df.empty:
            df = df.sort_values('updated_at', ascending=False).groupby('symbol').head(1)

            # Apply filters from sidebar
            if show_active:
                # Filter to active assets only (would need join with assets table)
                pass
            if show_signals:
                df = df[df['signal'].isin(['BUY', 'SELL'])]

            # Render the surveillance table
            render_surveillance_table(df)
        else:
            st.info("[WAITING] Sentinel data not yet available...")

        st.divider()

        # B. Fetch Latest Council Votes (For the Grid)
        vote_query = """
            SELECT agent_votes, consensus_score, regime
            FROM market_snapshots
            WHERE agent_votes IS NOT NULL
            ORDER BY updated_at DESC
            LIMIT 1
        """
        cursor = conn.cursor()
        cursor.execute(vote_query)
        row = cursor.fetchone()

        if row and row[0]:
            try:
                votes_data = json.loads(row[0])
                # Convert to expected format {agent_name: vote_value}
                votes_dict = {}
                reasons_dict = {}
                for agent, data in votes_data.items():
                    if isinstance(data, dict):
                        votes_dict[agent] = data.get('vote', 0.0)
                        reasons_dict[agent] = data.get('reason', 'N/A')
                    else:
                        votes_dict[agent] = float(data)
                        reasons_dict[agent] = 'N/A'

                render_council_grid(votes_dict, reasons_dict)
            except (json.JSONDecodeError, TypeError) as e:
                st.warning(f"[PARSE_ERROR] Council votes: {e}")
        else:
            # Placeholder Council Grid with default values
            st.markdown("### COUNCIL OF 6 - AWAITING VOTES")
            default_votes = {
                'regime': 0.0,
                'strategy': 0.0,
                'sniper': 0.0,
                'news': 0.0,
                'seasonality': 0.0,
                'institutional': 0.0,
            }
            render_council_grid(default_votes)

        conn.close()

    except Exception as e:
        st.error(f"[DB_ERROR] {e}")
        logger.error(f"Database error: {e}")

    st.divider()

    # 4. System Status Panel
    render_system_status()

    # 5. Footer
    render_footer()


if __name__ == "__main__":
    main()
