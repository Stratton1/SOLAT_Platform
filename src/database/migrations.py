"""
Database Migrations: Add Consensus Engine Tables
=================================================

This module handles schema updates for the parallel voting system.
Run on first startup to create ai_trust_scores and consensus_votes tables.
"""

import sqlite3
import logging
from typing import Tuple

logger = logging.getLogger("migrations")


def migrate_to_consensus(db_path: str) -> Tuple[bool, str]:
    """
    Add consensus engine tables and columns to existing database.

    Args:
        db_path: Path to SQLite database

    Returns:
        Tuple: (success, message)
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # =========================================================
        # TABLE 1: ai_trust_scores
        # =========================================================
        # Track the evolving trust/accuracy of each AI agent
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_trust_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL UNIQUE,
                base_weight REAL DEFAULT 0.166,
                correct_calls INTEGER DEFAULT 0,
                wrong_calls INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.50,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Initialize 6 agents with equal voting power
        agents = [
            'regime',
            'strategy',
            'sniper',
            'news',
            'seasonality',
            'institutional'
        ]

        for agent in agents:
            try:
                cursor.execute("""
                    INSERT INTO ai_trust_scores (agent_name, base_weight, win_rate)
                    VALUES (?, ?, ?)
                """, (agent, 0.166, 0.50))
            except sqlite3.IntegrityError:
                # Agent already exists
                pass

        # =========================================================
        # TABLE 2: consensus_votes
        # =========================================================
        # Audit trail of all votes for every trade
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consensus_votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                -- Individual agent votes (-1 to +1)
                regime_vote REAL,
                strategy_vote REAL,
                sniper_vote REAL,
                news_vote REAL,
                seasonality_vote REAL,
                institutional_vote REAL,

                -- Weighted votes (vote × weight)
                regime_weighted REAL,
                strategy_weighted REAL,
                sniper_weighted REAL,
                news_weighted REAL,
                seasonality_weighted REAL,
                institutional_weighted REAL,

                -- Final consensus metrics
                consensus_score REAL,
                position_size_ratio REAL,
                executed INTEGER DEFAULT 0,

                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        """)

        # =========================================================
        # EXTEND TABLES: market_snapshots
        # =========================================================
        # Add consensus columns to track market state reasoning
        extend_market_snapshots = [
            ('consensus_score', 'REAL'),
            ('regime_vote', 'REAL'),
            ('strategy_vote', 'REAL'),
            ('sniper_vote', 'REAL'),
            ('news_vote', 'REAL'),
            ('seasonality_vote', 'REAL'),
            ('institutional_vote', 'REAL'),
        ]

        for column_name, column_type in extend_market_snapshots:
            try:
                cursor.execute(
                    f"ALTER TABLE market_snapshots ADD COLUMN {column_name} {column_type}"
                )
                logger.info(f"Added column market_snapshots.{column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    logger.debug(f"Column market_snapshots.{column_name} already exists")
                else:
                    raise

        # =========================================================
        # EXTEND TABLES: trades
        # =========================================================
        # Add voting and consensus tracking to trades
        extend_trades = [
            ('regime_vote', 'REAL'),
            ('strategy_vote', 'REAL'),
            ('sniper_vote', 'REAL'),
            ('news_vote', 'REAL'),
            ('seasonality_vote', 'REAL'),
            ('institutional_vote', 'REAL'),
            ('consensus_score', 'REAL'),
            ('was_correct', 'INTEGER DEFAULT NULL'),  # 1=correct, 0=incorrect, NULL=open
        ]

        for column_name, column_type in extend_trades:
            try:
                cursor.execute(
                    f"ALTER TABLE trades ADD COLUMN {column_name} {column_type}"
                )
                logger.info(f"Added column trades.{column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    logger.debug(f"Column trades.{column_name} already exists")
                else:
                    raise

        # =========================================================
        # CREATE INDEXES for performance
        # =========================================================
        indexes = [
            ("idx_consensus_votes_trade_id", "consensus_votes", "trade_id"),
            ("idx_ai_trust_agent_name", "ai_trust_scores", "agent_name"),
            ("idx_market_snapshots_symbol", "market_snapshots", "symbol"),
            ("idx_trades_symbol", "trades", "symbol"),
        ]

        for index_name, table_name, column_name in indexes:
            try:
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_name})"
                )
                logger.info(f"Created index {index_name}")
            except sqlite3.OperationalError as e:
                logger.warning(f"Could not create index {index_name}: {e}")

        conn.commit()
        conn.close()

        message = (
            "✅ Database migration to consensus engine complete.\n"
            "   - Created ai_trust_scores table (6 agents)\n"
            "   - Created consensus_votes table (audit trail)\n"
            "   - Extended market_snapshots with voting columns\n"
            "   - Extended trades with voting columns\n"
            "   - Created performance indexes"
        )
        logger.info(message)

        return True, message

    except Exception as e:
        message = f"❌ Migration failed: {e}"
        logger.error(message)
        return False, message


if __name__ == "__main__":
    # For manual testing
    import sys

    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "data/db/trading_engine.db"

    success, message = migrate_to_consensus(db_path)
    print(message)
    sys.exit(0 if success else 1)
