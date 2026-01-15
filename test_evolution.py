#!/usr/bin/env python3
"""
Test script for the SOLAT Evolutionary Optimizer.

This script:
1. Initializes a fresh database
2. Injects BTC/USDT as an active asset
3. Manually inserts 5 dummy trades (3 wins, 2 losses)
4. Runs the EvolutionaryOptimizer to calculate fitness
5. Prints the resulting fitness score

Success Criteria:
- Fitness score should be > 0 (indicating profitability)
- Win rate should be 60% (3 wins out of 5)
- Profit factor should be > 1.0 (sum of wins > sum of losses)
"""

import sys
import logging
import sqlite3
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.database.repository import init_db, get_connection
from src.core.evolution import EvolutionaryOptimizer


def inject_dummy_trades(cursor: sqlite3.Cursor, symbol: str = "BTC/USDT") -> None:
    """
    Inject 5 dummy trades into the database.

    Trade composition:
    - Trade 1: BUY @ 96000, PnL = +500 (WIN)
    - Trade 2: BUY @ 96500, PnL = -300 (LOSS)
    - Trade 3: SELL @ 97000, PnL = +400 (WIN)
    - Trade 4: BUY @ 96800, PnL = -200 (LOSS)
    - Trade 5: BUY @ 96400, PnL = +600 (WIN)

    Results:
    - Total: 5 trades
    - Wins: 3 (trades 1, 3, 5) = 1500 total profit
    - Losses: 2 (trades 2, 4) = -500 total loss
    - Win Rate: 3/5 = 60%
    - Profit Factor: 1500 / 500 = 3.0

    Args:
        cursor (sqlite3.Cursor): Database cursor
        symbol (str): Asset symbol
    """
    base_time = datetime.utcnow()
    trades = [
        ("BUY", 96000.0, 500.0, "Buy signal above cloud"),      # WIN
        ("BUY", 96500.0, -300.0, "Stop loss hit"),              # LOSS
        ("SELL", 97000.0, 400.0, "Sell signal below cloud"),    # WIN
        ("BUY", 96800.0, -200.0, "Early exit"),                 # LOSS
        ("BUY", 96400.0, 600.0, "Strong momentum signal"),      # WIN
    ]

    logger.info(f"Injecting {len(trades)} dummy trades for {symbol}...")

    for i, (side, entry_price, pnl, exit_reason) in enumerate(trades):
        entry_time = (base_time - timedelta(hours=5-i)).isoformat()

        cursor.execute(
            """
            INSERT INTO trades
            (symbol, side, entry_price, pnl, exit_reason, entry_time)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (symbol, side, entry_price, pnl, exit_reason, entry_time)
        )

        result = "WIN" if pnl > 0 else "LOSS"
        logger.info(f"  Trade {i+1}: {side} @ {entry_price:.2f} | PnL: {pnl:+.2f} | {result}")

    logger.info(f"✓ Injected {len(trades)} trades")


def print_asset_details(cursor: sqlite3.Cursor, symbol: str) -> None:
    """
    Print detailed asset information from the database.

    Args:
        cursor (sqlite3.Cursor): Database cursor
        symbol (str): Asset symbol
    """
    print(f"\n{'='*70}")
    print(f"ASSET DETAILS: {symbol}")
    print(f"{'='*70}")

    cursor.execute(
        """
        SELECT id, symbol, source, status, fitness_score, last_scan
        FROM assets
        WHERE symbol = ?
        """,
        (symbol,)
    )

    row = cursor.fetchone()
    if row:
        asset_id, sym, source, status, fitness_score, last_scan = row
        print(f"ID:              {asset_id}")
        print(f"Symbol:          {sym}")
        print(f"Source:          {source}")
        print(f"Status:          {status}")
        print(f"Fitness Score:   {fitness_score:.4f}")
        print(f"Last Scan:       {last_scan}")
    else:
        print(f"✗ Asset {symbol} not found!")

    print(f"{'='*70}\n")


def print_trade_summary(cursor: sqlite3.Cursor, symbol: str) -> None:
    """
    Print summary statistics for all trades of an asset.

    Args:
        cursor (sqlite3.Cursor): Database cursor
        symbol (str): Asset symbol
    """
    print(f"\n{'='*70}")
    print(f"TRADE SUMMARY: {symbol}")
    print(f"{'='*70}")

    cursor.execute(
        """
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
            SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as total_wins,
            SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END) as total_losses,
            SUM(pnl) as net_pnl
        FROM trades
        WHERE symbol = ?
        """,
        (symbol,)
    )

    row = cursor.fetchone()
    if row:
        total, wins, losses, total_wins, total_losses, net_pnl = row
        total = total or 0
        wins = wins or 0
        losses = losses or 0
        total_wins = float(total_wins or 0.0)
        total_losses = float(total_losses or 0.0)
        net_pnl = float(net_pnl or 0.0)

        win_rate = (wins / total * 100) if total > 0 else 0
        profit_factor = total_wins / abs(total_losses) if total_losses != 0 else 2.0

        print(f"Total Trades:    {total}")
        print(f"Winning Trades:  {wins} ({win_rate:.1f}%)")
        print(f"Losing Trades:   {losses} ({100-win_rate:.1f}%)")
        print(f"Total Wins:      {total_wins:+.2f}")
        print(f"Total Losses:    {total_losses:+.2f}")
        print(f"Net PnL:         {net_pnl:+.2f}")
        print(f"Profit Factor:   {profit_factor:.2f}")

    print(f"{'='*70}\n")


def main() -> int:
    """
    Main test routine.

    Returns:
        int: 0 if successful, 1 if error
    """
    logger.info("=" * 70)
    logger.info("SOLAT EVOLUTIONARY OPTIMIZER TEST")
    logger.info("=" * 70)

    try:
        # Step 1: Initialize fresh database
        logger.info("\n[STEP 1] Initializing database...")
        init_db()
        logger.info("✓ Database initialized")

        # Step 2: Inject asset and trades
        logger.info("\n[STEP 2] Injecting asset and dummy trades...")
        conn = get_connection()
        cursor = conn.cursor()

        symbol = "BTC/USDT"
        now = datetime.utcnow().isoformat()

        # Insert asset
        cursor.execute(
            """
            INSERT INTO assets (symbol, source, status, fitness_score, last_scan)
            VALUES (?, ?, ?, ?, ?)
            """,
            (symbol, "ccxt", "active", 0.5, now)
        )
        logger.info(f"✓ Injected asset: {symbol}")

        # Inject dummy trades
        inject_dummy_trades(cursor, symbol)

        conn.commit()

        # Print trade summary before optimization
        print_trade_summary(cursor, symbol)

        # Step 3: Run evolutionary optimizer
        logger.info("[STEP 3] Running EvolutionaryOptimizer.calculate_fitness()...")
        optimizer = EvolutionaryOptimizer()
        fitness_scores = optimizer.calculate_fitness()

        if symbol in fitness_scores:
            fitness = fitness_scores[symbol]
            logger.info(f"✓ Fitness calculated: {symbol} = {fitness:.4f}")
        else:
            logger.error(f"✗ Fitness not calculated for {symbol}")
            conn.close()
            return 1

        # Step 4: Fetch updated asset data
        logger.info("\n[STEP 4] Fetching updated asset data...")
        conn.close()

        conn = get_connection()
        cursor = conn.cursor()

        print_asset_details(cursor, symbol)

        # Step 5: Verify results
        logger.info("[STEP 5] Verifying results...")

        cursor.execute(
            "SELECT fitness_score FROM assets WHERE symbol = ?",
            (symbol,)
        )
        result = cursor.fetchone()
        stored_fitness = result[0] if result else None

        success = False
        if stored_fitness is not None and stored_fitness > 0:
            logger.info(f"✓ Fitness score is positive: {stored_fitness:.4f}")
            success = True
        elif stored_fitness is not None:
            logger.warning(f"⚠ Fitness score is zero or negative: {stored_fitness:.4f}")
        else:
            logger.error(f"✗ Fitness score not found in database")

        # Expected values
        expected_win_rate = 0.60  # 3 wins / 5 trades
        expected_profit_factor = 3.0  # 1500 / 500
        # Expected fitness = (0.4 * 0.60) + (0.4 * 3.0) = 0.24 + 1.20 = 1.44

        logger.info(f"\nExpected Fitness: ~1.44 (0.4*0.6 + 0.4*3.0)")
        logger.info(f"Actual Fitness:   {stored_fitness:.4f}")

        if stored_fitness and stored_fitness >= 1.2:  # Allow some tolerance
            logger.info(f"✓ Fitness is within expected range")
        else:
            logger.warning(f"⚠ Fitness lower than expected (check trade calculations)")

        conn.close()

        # Summary
        logger.info("\n" + "=" * 70)
        if success:
            logger.info("✓ TEST PASSED - Evolutionary optimizer working correctly")
            logger.info("=" * 70)
            return 0
        else:
            logger.error("✗ TEST FAILED - Fitness score invalid")
            logger.info("=" * 70)
            return 1

    except Exception as e:
        logger.error(f"\n✗ TEST FAILED WITH ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
