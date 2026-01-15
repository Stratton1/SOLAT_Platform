"""
Database repository module for SOLAT.
Handles SQLite connection initialization with WAL mode and schema creation.

Schema includes:
- Core tables: assets, market_snapshots, trades
- Council of 6: All AI agent voting data stored in market_snapshots
"""

import sqlite3
import logging
import os
from typing import Optional

from src.config.settings import DB_PATH

logger = logging.getLogger(__name__)


def init_db() -> None:
    """
    Initialize the SQLite database in WAL mode and create all necessary tables.

    This function:
    1. Ensures the database directory exists
    2. Connects to the database
    3. Enables WAL mode for concurrent read/write access
    4. Creates all required tables with Council schema
    """
    # Ensure the database directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Enable WAL mode for concurrent access
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")

        # ============================================================
        # ASSETS TABLE
        # ============================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                symbol TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                fitness_score REAL DEFAULT 0.0,
                optimal_strategy TEXT DEFAULT 'ichimoku_standard',
                last_scan DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ============================================================
        # MARKET SNAPSHOTS TABLE (Council of 6 Enabled)
        # ============================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                close_price REAL,
                -- Technicals
                cloud_status TEXT,
                chikou_conf TEXT,
                -- THE COUNCIL DATA (Required for Terminal Mode)
                regime TEXT,
                consensus_score REAL,
                agent_votes TEXT,
                order_imbalance REAL,
                news_sentiment REAL,
                signal TEXT,
                -- Metadata
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ============================================================
        # TRADES TABLE
        # ============================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL,
                size REAL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT DEFAULT 'open',
                entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                exit_time DATETIME,
                exit_price REAL,
                pnl REAL,
                exit_reason TEXT,
                strategy_used TEXT DEFAULT 'ichimoku_standard',
                consensus_score REAL,
                agent_votes TEXT
            )
        """)

        # ============================================================
        # EVOLUTION METRICS TABLE
        # ============================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolution_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                win_rate REAL DEFAULT 0.0,
                profit_factor REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,
                total_trades INTEGER DEFAULT 0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ============================================================
        # ACCOUNT BALANCE TABLE (Portfolio Management)
        # ============================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS account_balance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                balance REAL NOT NULL,
                equity REAL NOT NULL,
                daily_pnl REAL DEFAULT 0.0,
                daily_drawdown REAL DEFAULT 0.0,
                peak_equity REAL DEFAULT 0.0,
                is_halted INTEGER DEFAULT 0,
                halt_reason TEXT,
                recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ============================================================
        # TRADING HALTS TABLE (Circuit Breaker)
        # ============================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_halts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reason TEXT NOT NULL,
                halt_start DATETIME NOT NULL,
                halt_end DATETIME,
                daily_drawdown REAL,
                is_active INTEGER DEFAULT 1
            )
        """)

        # ============================================================
        # AGENT WEIGHTS TABLE (Reinforcement Learning)
        # ============================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                weight REAL NOT NULL,
                win_count INTEGER DEFAULT 0,
                loss_count INTEGER DEFAULT 0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(agent_name)
            )
        """)

        # Commit all changes
        conn.commit()
        logger.info("Database initialized with Council Schema")
        print("Database initialized with Council Schema")

    finally:
        # Ensure connection is closed
        conn.close()


def get_connection() -> sqlite3.Connection:
    """
    Get a SQLite connection to the trading engine database.

    Returns:
        sqlite3.Connection: A connection with WAL mode enabled.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def get_db_connection() -> sqlite3.Connection:
    """
    Get a SQLite connection with Row factory for dict-like access.

    Returns:
        sqlite3.Connection: A connection with row_factory set.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


if __name__ == "__main__":
    # Allow direct execution for testing
    init_db()
