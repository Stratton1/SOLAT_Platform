#!/usr/bin/env python3
"""
Test script for the SOLAT core trading logic.

This script:
1. Initializes a fresh database
2. Manually injects BTC/USDT into the assets table
3. Instantiates the Sentinel and runs scan_market() once
4. Prints the market_snapshots and trades tables to verify data flow

This demonstrates the complete data pipeline without running the infinite loop.
"""

import sys
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.database.repository import init_db, get_connection
from src.core.engine import Sentinel


def print_table(cursor: sqlite3.Cursor, table_name: str) -> None:
    """
    Pretty-print a database table.

    Args:
        cursor: SQLite cursor
        table_name: Name of the table to print
    """
    print(f"\n{'=' * 80}")
    print(f"Table: {table_name}")
    print(f"{'=' * 80}")

    try:
        cursor.execute(f"SELECT * FROM {table_name}")
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()

        if not rows:
            print(f"  (empty)")
            return

        # Calculate column widths
        col_widths = {}
        for col in columns:
            col_widths[col] = max(len(col), 15)

        # Print header
        header = " | ".join(col.ljust(col_widths[col]) for col in columns)
        print(f"  {header}")
        print(f"  {'-' * len(header)}")

        # Print rows
        for row in rows:
            row_str = " | ".join(
                str(val).ljust(col_widths[col])
                for col, val in zip(columns, row)
            )
            print(f"  {row_str}")

        print(f"  Total rows: {len(rows)}")

    except Exception as e:
        logger.error(f"Error printing table {table_name}: {e}")


def main() -> int:
    """
    Main test routine.

    Returns:
        int: 0 if successful, 1 if error
    """
    logger.info("=" * 80)
    logger.info("SOLAT CORE TRADING LOGIC TEST")
    logger.info("=" * 80)

    try:
        # Step 1: Initialize fresh database
        logger.info("\n[STEP 1] Initializing database...")
        init_db()
        logger.info("✓ Database initialized with WAL mode")

        # Step 2: Manually inject BTC/USDT into assets table
        logger.info("\n[STEP 2] Injecting BTC/USDT into assets table...")
        conn = get_connection()
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()
        cursor.execute(
            """
            INSERT INTO assets (symbol, source, status, fitness_score, last_scan)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("BTC/USDT", "ccxt", "active", 0.5, now)
        )
        conn.commit()
        logger.info("✓ Injected BTC/USDT (ccxt, active)")

        # Print assets table
        print_table(cursor, "assets")

        # Step 3: Instantiate Sentinel and run scan_market() once
        logger.info("\n[STEP 3] Instantiating Sentinel and running scan_market()...")
        sentinel = Sentinel()

        # Run a single market scan (no loop)
        stats = sentinel.scan_market()
        logger.info(f"✓ Scan completed: {stats}")

        # Step 4: Print market_snapshots and trades tables
        logger.info("\n[STEP 4] Printing database results...")

        # Refresh cursor to see new data
        cursor.execute("SELECT * FROM market_snapshots")
        print_table(cursor, "market_snapshots")

        cursor.execute("SELECT * FROM trades")
        print_table(cursor, "trades")

        # Step 5: Verify data integrity
        logger.info("\n[STEP 5] Verifying data integrity...")

        cursor.execute("SELECT COUNT(*) FROM market_snapshots")
        snapshot_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM trades")
        trade_count = cursor.fetchone()[0]

        if snapshot_count > 0:
            logger.info(f"✓ Found {snapshot_count} market snapshot(s)")
        else:
            logger.error("✗ No market snapshots found!")

        if trade_count > 0:
            logger.info(f"✓ Found {trade_count} trade record(s)")
        else:
            logger.warning("⚠ No trades logged (may be NEUTRAL signals)")

        conn.close()

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Database Status: ✓ Initialized")
        logger.info(f"Assets Injected: ✓ BTC/USDT (1 asset)")
        logger.info(f"Market Scan: ✓ Completed (stats={stats})")
        logger.info(f"Market Snapshots: ✓ {snapshot_count} record(s)")
        logger.info(f"Trades Logged: {'✓' if trade_count > 0 else '⚠'} {trade_count} record(s)")
        logger.info("=" * 80)

        if snapshot_count > 0:
            logger.info("\n✓ ALL TESTS PASSED - Data pipeline is working correctly")
            return 0
        else:
            logger.error("\n✗ TEST FAILED - Market snapshot not recorded")
            return 1

    except Exception as e:
        logger.error(f"\n✗ TEST FAILED WITH ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
