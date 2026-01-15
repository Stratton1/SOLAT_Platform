"""
Database repository module for SOLAT.
Handles SQLite connection initialization with WAL mode and schema creation.

Complete Institutional Schema:
- assets: Watchlist with fitness scores
- market_snapshots: Council of 6 voting data
- trades: Position management with trailing stops
- seasonality_patterns: Time-based pattern analysis
- ai_trust_scores: Agent weight tracking for RL
- evolution_metrics: Historical performance
- account_balance: Portfolio management
- trading_halts: Circuit breaker state
"""

import sqlite3
import logging
import os
from typing import Optional

from src.config.settings import DB_PATH

logger = logging.getLogger(__name__)


def init_db() -> None:
    """
    Initialize the SQLite database with ALL institutional tables.

    Tables:
    1. assets - Watchlist management
    2. market_snapshots - Council of 6 data
    3. trades - Position tracking with trailing stops
    4. seasonality_patterns - Pattern Hunter data
    5. ai_trust_scores - Consensus Engine weights
    6. evolution_metrics - Fitness calculations
    7. account_balance - Portfolio tracking
    8. trading_halts - Circuit breaker
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Enable WAL Mode for concurrent access
        cursor.execute('PRAGMA journal_mode=WAL;')
        cursor.execute('PRAGMA synchronous=NORMAL;')

        # ============================================================
        # 1. ASSETS TABLE
        # ============================================================
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS assets (
            symbol TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            fitness_score REAL DEFAULT 0.0,
            optimal_strategy TEXT DEFAULT 'ichimoku_standard',
            last_scan DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            meta_data TEXT
        )
        ''')

        # ============================================================
        # 2. MARKET SNAPSHOTS TABLE (Council of 6 Data)
        # ============================================================
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            close_price REAL,
            -- Technicals
            cloud_status TEXT,
            chikou_conf TEXT,
            -- Council of 6 Data
            regime TEXT,
            consensus_score REAL,
            agent_votes TEXT,
            order_imbalance REAL,
            news_sentiment REAL,
            signal TEXT,
            -- Metadata
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # ============================================================
        # 3. TRADES TABLE (with trailing_stop_price)
        # ============================================================
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL,
            size REAL,
            stop_loss REAL,
            take_profit REAL,
            trailing_stop_price REAL,
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
        ''')

        # ============================================================
        # 4. SEASONALITY PATTERNS TABLE (Pattern Hunter)
        # ============================================================
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS seasonality_patterns (
            symbol TEXT NOT NULL,
            hour INTEGER NOT NULL,
            day_of_week INTEGER,
            win_rate REAL,
            avg_return REAL,
            trade_count INTEGER DEFAULT 0,
            is_significant INTEGER DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, hour)
        )
        ''')

        # ============================================================
        # 5. AI TRUST SCORES TABLE (Consensus Engine RL)
        # ============================================================
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_trust_scores (
            agent_name TEXT PRIMARY KEY,
            current_weight REAL NOT NULL,
            correct_calls INTEGER DEFAULT 0,
            wrong_calls INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0.0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # ============================================================
        # 6. EVOLUTION METRICS TABLE
        # ============================================================
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS evolution_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            win_rate REAL DEFAULT 0.0,
            profit_factor REAL DEFAULT 0.0,
            max_drawdown REAL DEFAULT 0.0,
            total_trades INTEGER DEFAULT 0,
            avg_win REAL DEFAULT 0.0,
            avg_loss REAL DEFAULT 0.0,
            sharpe_ratio REAL DEFAULT 0.0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # ============================================================
        # 7. ACCOUNT BALANCE TABLE (Portfolio Management)
        # ============================================================
        cursor.execute('''
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
        ''')

        # ============================================================
        # 8. TRADING HALTS TABLE (Circuit Breaker)
        # ============================================================
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trading_halts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reason TEXT NOT NULL,
            halt_start DATETIME NOT NULL,
            halt_end DATETIME,
            daily_drawdown REAL,
            is_active INTEGER DEFAULT 1
        )
        ''')

        # Commit all changes
        conn.commit()
        logger.info("Database initialized with ALL Institutional Tables")
        print("Database initialized with ALL Institutional Tables")

    finally:
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
    init_db()
