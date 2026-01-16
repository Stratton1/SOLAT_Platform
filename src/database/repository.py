"""
Database repository module for SOLAT.
Handles SQLite connection initialization with WAL mode and schema creation.

FINAL LIVE Schema - All columns required by Council of 6 agents.
"""

import sqlite3
import os
from src.config.settings import DB_PATH


def init_db():
    """Initialize database with FINAL LIVE schema."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Enable WAL Mode for high performance
    cursor.execute('PRAGMA journal_mode=WAL;')
    cursor.execute('PRAGMA synchronous=NORMAL;')

    # 1. ASSETS
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS assets (
        symbol TEXT PRIMARY KEY,
        source TEXT,
        status TEXT DEFAULT 'active',
        fitness_score REAL DEFAULT 0.0,
        optimal_strategy TEXT DEFAULT 'ichimoku_standard',
        last_scan DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        meta_data TEXT
    )
    ''')

    # 2. MARKET SNAPSHOTS (Council of 6 Data)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        close_price REAL,
        cloud_status TEXT,
        chikou_conf TEXT,
        tk_cross TEXT,
        regime TEXT,
        consensus_score REAL,
        agent_votes TEXT,
        order_imbalance REAL,
        news_sentiment REAL,
        signal TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 3. TRADES (with unit_number, is_open, trailing_stop)
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
        unit_number INTEGER DEFAULT 1,
        status TEXT DEFAULT 'open',
        is_open INTEGER DEFAULT 1,
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

    # 4. SEASONALITY PATTERNS (Pattern Hunter)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS seasonality_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        pattern_type TEXT NOT NULL,
        period_value TEXT NOT NULL,
        avg_return REAL,
        win_rate REAL,
        sample_size INTEGER DEFAULT 0,
        is_significant INTEGER DEFAULT 0,
        calculated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(symbol, pattern_type, period_value)
    )
    ''')

    # 5. AI TRUST SCORES (Consensus Engine RL)
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

    # 6. EVOLUTION METRICS
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

    # 7. ACCOUNT BALANCE (Portfolio Management)
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

    # 8. TRADING HALTS (Circuit Breaker)
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

    conn.commit()
    conn.close()
    print("Database initialized with FINAL LIVE Schema")


def get_connection():
    """Get SQLite connection with WAL mode."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def get_db_connection():
    """Get SQLite connection with Row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


if __name__ == "__main__":
    init_db()
