"""
Evolutionary Optimizer for SOLAT Platform

This module implements the self-optimizing asset selection engine.
Every 4 hours, the system recalculates fitness scores for all assets based on
their trading performance and dynamically reranks them (Active/Normal/Dormant).

This mimics natural selection: high-fitness assets get more scan frequency,
low-fitness assets are demoted to save API credits.
"""

import logging
from datetime import datetime
from typing import Dict, Tuple, List

from src.database.repository import get_connection
from src.config.settings import (
    FITNESS_WEIGHTS,
    ACTIVE_THRESHOLD,
    DORMANT_THRESHOLD,
)

logger = logging.getLogger(__name__)


class EvolutionaryOptimizer:
    """
    Evolutionary Algorithm for Dynamic Asset Selection

    The optimizer:
    1. Calculates fitness scores based on trade performance (win rate, profit factor)
    2. Ranks assets by fitness
    3. Promotes top 20% to "active" (high scan frequency)
    4. Demotes bottom 20% to "dormant" (low scan frequency)
    5. Leaves middle 60% as "normal"

    Fitness Formula:
        F(x) = (0.4 * win_rate) + (0.4 * profit_factor) - (0.2 * max_drawdown)

    Where:
        - win_rate: Fraction of profitable trades (0.0 to 1.0)
        - profit_factor: Sum(wins) / abs(Sum(losses)) (ratio, typically 1.5+)
        - max_drawdown: Largest peak-to-trough decline (0.0 to 1.0, penalized)
    """

    def __init__(self) -> None:
        """Initialize the EvolutionaryOptimizer."""
        self.weights = FITNESS_WEIGHTS
        logger.info(
            f"Initialized EvolutionaryOptimizer with weights: {self.weights}"
        )

    def _calculate_asset_stats(self, symbol: str) -> Dict[str, float]:
        """
        Calculate trade statistics for a single asset.

        Queries the trades table and aggregates:
        - Total trades
        - Winning trades (PnL > 0)
        - Losing trades (PnL < 0)
        - Win rate (wins / total)
        - Total profit from wins
        - Total loss from losses
        - Profit factor (total_wins / abs(total_losses))

        Args:
            symbol (str): Asset symbol (e.g., "BTC/USDT")

        Returns:
            Dict[str, float]: Statistics dictionary with keys:
                - total_trades
                - winning_trades
                - losing_trades
                - win_rate
                - total_wins
                - total_losses
                - profit_factor
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Query trade statistics for the asset
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as total_wins,
                    SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END) as total_losses
                FROM trades
                WHERE symbol = ?
                """,
                (symbol,)
            )

            row = cursor.fetchone()
            conn.close()

            if not row:
                # No trades yet - return neutral stats
                logger.debug(f"No trades found for {symbol}, returning neutral stats")
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.5,  # Neutral
                    "total_wins": 0.0,
                    "total_losses": 0.0,
                    "profit_factor": 1.0,  # Neutral (breakeven)
                }

            total_trades, winning_trades, losing_trades, total_wins, total_losses = row

            # Handle cases with missing data
            total_trades = total_trades or 0
            winning_trades = winning_trades or 0
            losing_trades = losing_trades or 0
            total_wins = float(total_wins or 0.0)
            total_losses = float(total_losses or 0.0)

            # Calculate win rate
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.5

            # Calculate profit factor
            if total_losses != 0:
                profit_factor = total_wins / abs(total_losses)
            elif total_wins > 0:
                profit_factor = 2.0  # Good performance (all wins, no losses)
            else:
                profit_factor = 1.0  # Breakeven or no trades

            stats = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_wins": total_wins,
                "total_losses": total_losses,
                "profit_factor": profit_factor,
            }

            logger.debug(
                f"Stats for {symbol}: trades={total_trades}, "
                f"win_rate={win_rate:.2%}, profit_factor={profit_factor:.2f}"
            )

            return stats

        except Exception as e:
            logger.error(f"Error calculating stats for {symbol}: {e}", exc_info=True)
            # Return neutral stats on error
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.5,
                "total_wins": 0.0,
                "total_losses": 0.0,
                "profit_factor": 1.0,
            }

    def calculate_fitness(self) -> Dict[str, float]:
        """
        Calculate and update fitness scores for all assets.

        Process:
        1. For each asset, fetch trade statistics
        2. Apply fitness formula with configured weights
        3. Update assets.fitness_score in database
        4. Return mapping of symbol -> fitness_score

        Fitness Formula:
            F(x) = (w_wr * win_rate) + (w_pf * profit_factor) - (w_dd * max_drawdown)

        With defaults:
            F(x) = (0.4 * win_rate) + (0.4 * profit_factor) - (0.2 * max_drawdown)

        Returns:
            Dict[str, float]: Mapping of symbol to fitness_score
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Get all assets
            cursor.execute("SELECT DISTINCT symbol FROM assets")
            symbols = [row[0] for row in cursor.fetchall()]

            logger.info(f"Calculating fitness for {len(symbols)} assets")

            fitness_scores = {}

            for symbol in symbols:
                # Get trade statistics
                stats = self._calculate_asset_stats(symbol)

                # Calculate fitness score
                win_rate = stats["win_rate"]
                profit_factor = max(stats["profit_factor"], 0.1)  # Clamp to avoid division errors
                max_drawdown = 0.0  # Simplified: we'll enhance this later with actual drawdown calculation

                # Apply formula with weights
                fitness = (
                    self.weights["win_rate"] * win_rate +
                    self.weights["profit_factor"] * profit_factor -
                    self.weights["drawdown"] * max_drawdown
                )

                # Clamp fitness between 0.0 and 2.0 (practical range)
                fitness = max(0.0, min(fitness, 2.0))

                fitness_scores[symbol] = fitness

                # Update database
                now = datetime.utcnow().isoformat()
                cursor.execute(
                    """
                    UPDATE assets
                    SET fitness_score = ?, updated_at = ?
                    WHERE symbol = ?
                    """,
                    (fitness, now, symbol)
                )

                logger.info(
                    f"Fitness Updated: {symbol} = {fitness:.3f} "
                    f"(wr={win_rate:.2%}, pf={profit_factor:.2f})"
                )

            conn.commit()
            conn.close()

            logger.info(f"Fitness calculation complete: {len(fitness_scores)} assets")
            return fitness_scores

        except Exception as e:
            logger.error(f"Error in calculate_fitness: {e}", exc_info=True)
            return {}

    def optimize_pool(self) -> Tuple[int, int]:
        """
        Optimize the asset pool by ranking and rebalancing statuses.

        Process:
        1. Calculate fitness for all assets
        2. Rank assets by fitness score (descending)
        3. Promote top 20% to "active"
        4. Demote bottom 20% to "dormant"
        5. Set middle 60% to "normal"

        Returns:
            Tuple[int, int]: (promoted_count, demoted_count)
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            logger.info("=" * 60)
            logger.info("EVOLUTIONARY OPTIMIZATION EPOCH")
            logger.info("=" * 60)

            # Step 1: Recalculate fitness scores
            fitness_scores = self.calculate_fitness()

            if not fitness_scores:
                logger.warning("No fitness scores calculated, skipping optimization")
                conn.close()
                return 0, 0

            # Step 2: Rank assets by fitness
            ranked_symbols = sorted(
                fitness_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            total_assets = len(ranked_symbols)
            if total_assets == 0:
                logger.warning("No assets to optimize")
                conn.close()
                return 0, 0

            # Calculate thresholds
            active_count = max(1, int(total_assets * (1 - ACTIVE_THRESHOLD)))
            dormant_count = max(1, int(total_assets * DORMANT_THRESHOLD))

            logger.info(
                f"Asset Pool: Total={total_assets}, "
                f"Active={active_count} (top {1-ACTIVE_THRESHOLD:.0%}), "
                f"Dormant={dormant_count} (bottom {DORMANT_THRESHOLD:.0%})"
            )

            promoted = []
            demoted = []
            now = datetime.utcnow().isoformat()

            # Step 3: Promote top 20% to "active"
            for i, (symbol, fitness) in enumerate(ranked_symbols[:active_count]):
                cursor.execute(
                    """
                    UPDATE assets
                    SET status = ?, updated_at = ?
                    WHERE symbol = ?
                    """,
                    ("active", now, symbol)
                )
                promoted.append(symbol)
                logger.info(f"  ↑ PROMOTED: {symbol} (fitness={fitness:.3f}) → ACTIVE")

            # Step 4: Demote bottom 20% to "dormant"
            for symbol, fitness in ranked_symbols[-dormant_count:]:
                cursor.execute(
                    """
                    UPDATE assets
                    SET status = ?, updated_at = ?
                    WHERE symbol = ?
                    """,
                    ("dormant", now, symbol)
                )
                demoted.append(symbol)
                logger.info(f"  ↓ DEMOTED: {symbol} (fitness={fitness:.3f}) → DORMANT")

            # Step 5: Set middle 60% to "normal"
            normal_range = ranked_symbols[active_count:-dormant_count]
            for symbol, fitness in normal_range:
                cursor.execute(
                    """
                    UPDATE assets
                    SET status = ?, updated_at = ?
                    WHERE symbol = ?
                    """,
                    ("normal", now, symbol)
                )
                logger.debug(f"  → NORMAL: {symbol} (fitness={fitness:.3f})")

            conn.commit()

            # Summary
            logger.info("=" * 60)
            logger.info("EVOLUTION COMPLETE")
            logger.info(f"Promoted: {len(promoted)} assets → ACTIVE")
            logger.info(f"Demoted: {len(demoted)} assets → DORMANT")
            logger.info(f"Normal: {len(normal_range)} assets → NORMAL")
            logger.info("=" * 60)

            conn.close()
            return len(promoted), len(demoted)

        except Exception as e:
            logger.error(f"Critical error in optimize_pool: {e}", exc_info=True)
            return 0, 0
