"""
Database repository module for SOLAT.
Handles SQLite connection initialization with WAL mode and schema creation.
"""

import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = "data/db/trading_engine.db"


def init_db() -> None:
    """
    Initialize the SQLite database in WAL mode and create all necessary tables.

    This function:
    1. Ensures the database directory exists
    2. Connects to the database
    3. Enables WAL mode for concurrent read/write access
    4. Sets synchronous to NORMAL for performance
    5. Creates all required tables (assets, market_snapshots, trades, evolution_metrics)
    """
    # Ensure the database directory exists
    db_dir = Path(DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Enable WAL mode for concurrent access
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")

        # Create assets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Normal',
                fitness_score REAL DEFAULT 0.0,
                last_scan TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create market_snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                close_price REAL NOT NULL,
                cloud_status TEXT,
                tk_cross TEXT,
                chikou_conf REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (symbol) REFERENCES assets(symbol)
            )
        """)

        # Create trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl REAL,
                exit_reason TEXT,
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exit_time TIMESTAMP,
                FOREIGN KEY (symbol) REFERENCES assets(symbol)
            )
        """)

        # Create evolution_metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolution_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                win_rate REAL DEFAULT 0.0,
                profit_factor REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (symbol) REFERENCES assets(symbol)
            )
        """)

        # Commit all table creations
        conn.commit()

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
    # Ensure WAL mode on every connection
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


if __name__ == "__main__":
    # Allow direct execution for testing
    init_db()
    print(f"Database initialized at {DB_PATH}")
