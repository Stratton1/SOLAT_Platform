"""
Database repository module for SOLAT.
Handles SQLite connection initialization with WAL mode and schema creation.

Schema includes:
- Core tables: assets, market_snapshots, trades, evolution_metrics
- Institutional upgrade: account_balance, strategy_performance, trading_halts, seasonality_patterns
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = "data/db/trading_engine.db"


def init_db() -> None:
    """
    Initialize the SQLite database in WAL mode and create all necessary tables.

    This function:
    1. Ensures the database directory exists
    2. Connects to the database
    3. Enables WAL mode for concurrent read/write access
    4. Sets synchronous to NORMAL for performance
    5. Creates all required tables (core + institutional upgrade)
    6. Runs migrations to add new columns to existing tables
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

        # ============================================================
        # CORE TABLES
        # ============================================================

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
                regime TEXT,
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

        # ============================================================
        # INSTITUTIONAL UPGRADE TABLES
        # ============================================================

        # Account balance tracking for portfolio management
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
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Strategy performance tracking (per asset, per strategy)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                win_rate REAL DEFAULT 0.0,
                sharpe_ratio REAL DEFAULT 0.0,
                max_drawdown REAL DEFAULT 0.0,
                total_trades INTEGER DEFAULT 0,
                profit_factor REAL DEFAULT 0.0,
                avg_win REAL DEFAULT 0.0,
                avg_loss REAL DEFAULT 0.0,
                is_optimal INTEGER DEFAULT 0,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, strategy_name)
            )
        """)

        # Trading halts (circuit breaker)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_halts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reason TEXT NOT NULL,
                halt_start TIMESTAMP NOT NULL,
                halt_end TIMESTAMP,
                daily_drawdown REAL,
                is_active INTEGER DEFAULT 1
            )
        """)

        # Seasonality patterns for time-based analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS seasonality_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                period_value TEXT NOT NULL,
                avg_return REAL,
                win_rate REAL,
                sample_size INTEGER,
                is_significant INTEGER DEFAULT 0,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, pattern_type, period_value)
            )
        """)

        # ============================================================
        # MIGRATIONS: Add new columns to existing tables
        # ============================================================
        _run_migrations(cursor)

        # Commit all changes
        conn.commit()
        logger.info("Database initialized successfully with institutional upgrade tables")

    finally:
        # Ensure connection is closed
        conn.close()


def _run_migrations(cursor: sqlite3.Cursor) -> None:
    """
    Run database migrations to add new columns to existing tables.
    Uses try/except to safely handle columns that already exist.
    """
    migrations = [
        # Trades table extensions for position management
        ("trades", "position_size", "REAL"),
        ("trades", "stop_loss_price", "REAL"),
        ("trades", "take_profit_price", "REAL"),
        ("trades", "strategy_name", "TEXT DEFAULT 'ichimoku_standard'"),
        ("trades", "unit_number", "INTEGER DEFAULT 1"),
        ("trades", "trailing_stop_price", "REAL"),
        ("trades", "is_open", "INTEGER DEFAULT 1"),
        # Market snapshots regime column (if missing)
        ("market_snapshots", "regime", "TEXT"),
        # Market snapshots microstructure columns
        ("market_snapshots", "order_imbalance", "REAL"),
        ("market_snapshots", "spread_bps", "REAL"),
        # Assets table extension for optimal strategy
        ("assets", "optimal_strategy", "TEXT DEFAULT 'ichimoku_standard'"),
    ]

    for table, column, col_type in migrations:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            logger.debug(f"Added column {column} to {table}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                # Column already exists, skip
                pass
            else:
                logger.warning(f"Migration warning for {table}.{column}: {e}")


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
