import sqlite3
import os
from src.config.settings import DB_PATH

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Enable WAL Mode for high performance
    cursor.execute('PRAGMA journal_mode=WAL;')

    # 1. ASSETS
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS assets (
        symbol TEXT PRIMARY KEY,
        source TEXT,
        status TEXT,
        fitness_score REAL,
        last_updated DATETIME,
        meta_data TEXT
    )
    ''')

    # 2. MARKET SNAPSHOTS (Fixed: Added tk_cross)
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

    # 3. TRADES (Fixed: Added unit_number, is_open)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        side TEXT,
        entry_price REAL,
        size REAL,
        stop_loss REAL,
        take_profit REAL,
        trailing_stop_price REAL,
        unit_number INTEGER,
        status TEXT,
        is_open BOOLEAN,
        entry_time DATETIME,
        exit_time DATETIME,
        exit_price REAL,
        pnl REAL,
        exit_reason TEXT,
        strategy_used TEXT
    )
    ''')

    # 4. SEASONALITY PATTERNS (Fixed: Added pattern_type, period_value, sample_size, is_significant)
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

    # 5. AI TRUST SCORES
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ai_trust_scores (
        agent_name TEXT PRIMARY KEY,
        current_weight REAL,
        correct_calls INTEGER,
        wrong_calls INTEGER,
        last_updated DATETIME
    )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized with FINAL CORRECTED Schema")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
