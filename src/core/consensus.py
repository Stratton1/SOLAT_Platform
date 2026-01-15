"""
Consensus Engine: Multi-Agent Democratic Voting System
=======================================================

Transforms SOLAT from "Sequential Veto" (first negative signal blocks)
to "Parallel Consensus" (all 6 agents vote, trust scores evolve).

The Council of 6:
1. Regime AI (HMM market state detector)
2. Strategy AI (Ichimoku cloud strategy)
3. Sniper AI (Order flow & microstructure)
4. News AI (Sentiment analysis)
5. Seasonality AI (Temporal patterns)
6. Institutional AI (Portfolio constraints)

Vote Aggregation:
  consensus_score = Σ(vote_i × weight_i) / Σ(weights)
  Range: [-1.0 (Strong Sell), 0.0 (Neutral), +1.0 (Strong Buy)]

Execution Threshold:
  Execute if |consensus_score| > 0.6 (60% agreement required)

Position Sizing:
  position_size = base_size × |consensus_score|
  Higher consensus → larger position (confidence scaling)

Reinforcement Learning:
  After trade closes, update agent weights based on accuracy.
  Alpha = 0.02 (conservative, 2-week half-life)
  Bounds = [0.05, 0.35] (prevent domination)
"""

import sqlite3
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json

logger = logging.getLogger("consensus")


@dataclass
class Agent:
    """Represents a voting agent in the council."""
    name: str
    base_weight: float = 0.166  # 1/6 for equal voting power
    correct_calls: int = 0
    wrong_calls: int = 0
    weight: float = 0.166
    win_rate: float = 0.50
    updated_at: datetime = None

    def __post_init__(self):
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class Vote:
    """Represents a single agent's vote on a trade opportunity."""
    agent_name: str
    vote: float  # Range: -1.0 to +1.0
    confidence: float = 1.0  # How certain is the agent? (0.0 to 1.0)
    reason: str = ""  # Why did the agent vote this way?


@dataclass
class ConsensusResult:
    """Result of consensus aggregation."""
    consensus_score: float  # Final aggregated vote [-1.0, +1.0]
    votes: Dict[str, float]  # Individual votes: {'regime': 0.5, 'strategy': 1.0, ...}
    weights: Dict[str, float]  # Applied weights: {'regime': 0.165, ...}
    weighted_votes: Dict[str, float]  # vote × weight for each agent
    position_size_ratio: float  # Scale factor for position size (|consensus_score|)
    decision: str  # "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"
    confidence_level: str  # "HIGH", "MODERATE", "LOW"


class ConsensusEngine:
    """
    The democratic voting system that aggregates signals from 6 AI agents.

    Trust scores evolve through reinforcement learning:
    - Agents that predict correctly get weight increases
    - Agents that predict incorrectly get weight decreases
    - All weights bounded [0.05, 0.35] to prevent domination
    """

    def __init__(self, db_path: str = "data/db/trading_engine.db"):
        self.db_path = db_path
        self.agents: Dict[str, Agent] = {}
        self.learning_rate = 0.02  # Alpha for EMA weight updates
        self.min_weight = 0.05  # Minimum weight (5%)
        self.max_weight = 0.35  # Maximum weight (35%)
        self.consensus_threshold = 0.60  # Execute if |score| > 0.60

        # Initialize 6 agents with equal voting power
        agent_names = [
            'regime',
            'strategy',
            'sniper',
            'news',
            'seasonality',
            'institutional'
        ]

        for name in agent_names:
            self.agents[name] = Agent(name=name, base_weight=0.166, weight=0.166)

        # Load trust scores from database (will initialize on first run)
        self.load_trust_scores()

        logger.info(f"ConsensusEngine initialized with 6 agents")
        logger.debug(f"Initial weights: {self.get_weights()}")

    def get_weights(self) -> Dict[str, float]:
        """Get current weights for all agents."""
        return {agent.name: agent.weight for agent in self.agents.values()}

    def load_trust_scores(self) -> None:
        """Load agent trust scores from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Try to load from DB; if table doesn't exist, use defaults
            try:
                cursor.execute("""
                    SELECT agent_name, base_weight, correct_calls, wrong_calls,
                           win_rate, updated_at
                    FROM ai_trust_scores
                """)
                rows = cursor.fetchall()

                if rows:
                    for row in rows:
                        agent_name, base_w, correct, wrong, win_rate, updated = row
                        if agent_name in self.agents:
                            self.agents[agent_name].base_weight = base_w
                            self.agents[agent_name].correct_calls = correct
                            self.agents[agent_name].wrong_calls = wrong
                            self.agents[agent_name].win_rate = win_rate
                            # Calculate current weight based on win rate
                            self.agents[agent_name].weight = self._calculate_weight(win_rate)

                    logger.info(f"Loaded trust scores for {len(rows)} agents")
                else:
                    self._initialize_trust_scores()

            except sqlite3.OperationalError:
                # Table doesn't exist yet; will be created by migration
                logger.info("ai_trust_scores table not found; using default weights")

            conn.close()

        except Exception as e:
            logger.warning(f"Could not load trust scores: {e}. Using defaults.")

    def _initialize_trust_scores(self) -> None:
        """Initialize trust scores in database (first run)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for agent in self.agents.values():
                cursor.execute("""
                    INSERT OR IGNORE INTO ai_trust_scores
                    (agent_name, base_weight, correct_calls, wrong_calls, win_rate, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    agent.name,
                    agent.base_weight,
                    agent.correct_calls,
                    agent.wrong_calls,
                    agent.win_rate,
                    datetime.utcnow().isoformat()
                ))

            conn.commit()
            conn.close()
            logger.info("Initialized trust scores for 6 agents")

        except sqlite3.OperationalError as e:
            logger.error(f"Could not initialize trust scores: {e}")

    def save_trust_scores(self) -> None:
        """Persist updated trust scores to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for agent in self.agents.values():
                cursor.execute("""
                    UPDATE ai_trust_scores
                    SET base_weight = ?, correct_calls = ?, wrong_calls = ?,
                        win_rate = ?, updated_at = ?
                    WHERE agent_name = ?
                """, (
                    agent.base_weight,
                    agent.correct_calls,
                    agent.wrong_calls,
                    agent.win_rate,
                    datetime.utcnow().isoformat(),
                    agent.name
                ))

            conn.commit()
            conn.close()
            logger.debug("Trust scores persisted to database")

        except Exception as e:
            logger.error(f"Could not save trust scores: {e}")

    def aggregate_consensus(self, votes_dict: Dict[str, float]) -> ConsensusResult:
        """
        Aggregate votes from all 6 agents using weighted voting.

        Args:
            votes_dict: Dictionary of votes like:
                {'regime': 0.5, 'strategy': 1.0, 'sniper': -0.2, ...}

        Returns:
            ConsensusResult with final consensus_score and decision
        """
        # Validate votes
        if not votes_dict:
            logger.warning("No votes provided; returning neutral consensus")
            return ConsensusResult(
                consensus_score=0.0,
                votes={},
                weights=self.get_weights(),
                weighted_votes={},
                position_size_ratio=0.0,
                decision="HOLD",
                confidence_level="LOW"
            )

        # Normalize votes to [-1.0, +1.0] range
        normalized_votes = {}
        for agent_name, vote in votes_dict.items():
            if agent_name in self.agents:
                normalized_votes[agent_name] = max(-1.0, min(1.0, float(vote)))

        # Calculate weighted votes
        weighted_votes = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for agent_name, vote in normalized_votes.items():
            weight = self.agents[agent_name].weight
            weighted_vote = vote * weight
            weighted_votes[agent_name] = weighted_vote
            weighted_sum += weighted_vote
            total_weight += weight

        # Calculate final consensus score (normalized to [-1.0, +1.0])
        if total_weight > 0:
            consensus_score = weighted_sum / total_weight
        else:
            consensus_score = 0.0

        consensus_score = max(-1.0, min(1.0, consensus_score))

        # Determine decision and confidence level
        abs_consensus = abs(consensus_score)

        if abs_consensus > 0.80:
            confidence_level = "HIGH"
        elif abs_consensus > 0.50:
            confidence_level = "MODERATE"
        else:
            confidence_level = "LOW"

        if consensus_score > 0.60:
            decision = "STRONG_BUY"
        elif consensus_score > 0.20:
            decision = "BUY"
        elif consensus_score < -0.60:
            decision = "STRONG_SELL"
        elif consensus_score < -0.20:
            decision = "SELL"
        else:
            decision = "HOLD"

        # Position size scaling (0.0 to 1.0)
        position_size_ratio = abs_consensus  # Execute only if > 0.60

        result = ConsensusResult(
            consensus_score=consensus_score,
            votes=normalized_votes,
            weights=self.get_weights(),
            weighted_votes=weighted_votes,
            position_size_ratio=position_size_ratio,
            decision=decision,
            confidence_level=confidence_level
        )

        logger.debug(
            f"Consensus: {consensus_score:.2f} | Decision: {decision} | "
            f"Votes: {normalized_votes}"
        )

        return result

    def calculate_position_size(
        self,
        base_size: float,
        consensus: ConsensusResult
    ) -> float:
        """
        Scale position size based on consensus confidence.

        Args:
            base_size: Base position size (from portfolio manager)
            consensus: ConsensusResult from aggregation

        Returns:
            Scaled position size
        """
        # Scale by absolute consensus value
        # At 0.60 consensus → 60% of base size
        # At 1.0 consensus → 100% of base size
        scaled_size = base_size * consensus.position_size_ratio
        return scaled_size

    def record_votes(
        self,
        trade_id: int,
        votes_dict: Dict[str, float],
        consensus_score: float,
        position_size_ratio: float,
        executed: bool = False
    ) -> bool:
        """
        Record all votes and consensus for a trade (audit trail + RL training).

        Args:
            trade_id: ID of the trade
            votes_dict: Individual agent votes
            consensus_score: Final consensus score
            position_size_ratio: Position size scaling factor
            executed: Whether trade was actually executed

        Returns:
            True if recorded successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Build weighted votes
            weighted_votes = {}
            for agent_name, vote in votes_dict.items():
                if agent_name in self.agents:
                    weight = self.agents[agent_name].weight
                    weighted_votes[agent_name] = vote * weight

            cursor.execute("""
                INSERT INTO consensus_votes
                (trade_id, timestamp, regime_vote, strategy_vote, sniper_vote,
                 news_vote, seasonality_vote, institutional_vote,
                 regime_weighted, strategy_weighted, sniper_weighted,
                 news_weighted, seasonality_weighted, institutional_weighted,
                 consensus_score, position_size_ratio, executed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                datetime.utcnow().isoformat(),
                votes_dict.get('regime', 0.0),
                votes_dict.get('strategy', 0.0),
                votes_dict.get('sniper', 0.0),
                votes_dict.get('news', 0.0),
                votes_dict.get('seasonality', 0.0),
                votes_dict.get('institutional', 0.0),
                weighted_votes.get('regime', 0.0),
                weighted_votes.get('strategy', 0.0),
                weighted_votes.get('sniper', 0.0),
                weighted_votes.get('news', 0.0),
                weighted_votes.get('seasonality', 0.0),
                weighted_votes.get('institutional', 0.0),
                consensus_score,
                position_size_ratio,
                1 if executed else 0
            ))

            conn.commit()
            conn.close()

            logger.debug(f"Recorded votes for trade {trade_id}")
            return True

        except Exception as e:
            logger.error(f"Could not record votes: {e}")
            return False

    def update_trust_scores(self, trade_id: int, final_pnl: float) -> Dict[str, bool]:
        """
        Update agent trust scores based on trade outcome (Reinforcement Learning).

        Args:
            trade_id: Trade to evaluate
            final_pnl: Realized P&L (positive or negative)

        Returns:
            Dict showing which agents were credited/penalized
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Retrieve recorded votes and trade details
            cursor.execute("""
                SELECT regime_vote, strategy_vote, sniper_vote, news_vote,
                       seasonality_vote, institutional_vote
                FROM consensus_votes WHERE trade_id = ?
            """, (trade_id,))
            vote_row = cursor.fetchone()

            if not vote_row:
                logger.warning(f"No votes found for trade {trade_id}")
                conn.close()
                return {}

            cursor.execute("SELECT side FROM trades WHERE id = ?", (trade_id,))
            trade_row = cursor.fetchone()

            if not trade_row:
                logger.warning(f"Trade {trade_id} not found")
                conn.close()
                return {}

            trade_side = trade_row[0]  # 'BUY' or 'SELL'

            # Determine if trade was profitable
            was_profitable = (trade_side == "BUY" and final_pnl > 0) or \
                            (trade_side == "SELL" and final_pnl > 0)

            # Evaluate each agent
            vote_mapping = {
                'regime': vote_row[0],
                'strategy': vote_row[1],
                'sniper': vote_row[2],
                'news': vote_row[3],
                'seasonality': vote_row[4],
                'institutional': vote_row[5],
            }

            updates = {}

            for agent_name, agent_vote in vote_mapping.items():
                # Determine if agent's vote was correct
                vote_agreed_with_outcome = False

                if trade_side == "BUY":
                    # Agent was correct if it voted positive (>0) for a profitable buy
                    # Or voted negative (<0) for an unprofitable buy
                    vote_agreed_with_outcome = (agent_vote > 0 and was_profitable) or \
                                              (agent_vote < 0 and not was_profitable)
                else:  # SELL
                    # Agent was correct if it voted positive (>0) for a profitable sell
                    # Or voted negative (<0) for an unprofitable sell
                    vote_agreed_with_outcome = (agent_vote > 0 and was_profitable) or \
                                              (agent_vote < 0 and not was_profitable)

                # Update agent's call record
                agent = self.agents[agent_name]

                if agent_vote != 0:  # Only count if agent actually took a position
                    if vote_agreed_with_outcome:
                        agent.correct_calls += 1
                        updates[agent_name] = True  # Credited
                    else:
                        agent.wrong_calls += 1
                        updates[agent_name] = False  # Penalized

                    # Recalculate win rate
                    total_calls = agent.correct_calls + agent.wrong_calls
                    agent.win_rate = agent.correct_calls / total_calls if total_calls > 0 else 0.50

                    # Update weight using EMA
                    agent.weight = self._calculate_weight(agent.win_rate)

                    logger.debug(
                        f"{agent_name}: {agent.correct_calls}W/{agent.wrong_calls}L "
                        f"({agent.win_rate:.1%}) → weight={agent.weight:.3f}"
                    )

            # Persist updated trust scores
            self.save_trust_scores()

            conn.close()

            logger.info(
                f"Updated trust scores for trade {trade_id} (PnL: ${final_pnl:.2f})"
            )

            return updates

        except Exception as e:
            logger.error(f"Could not update trust scores: {e}")
            return {}

    def _calculate_weight(self, win_rate: float) -> float:
        """
        Calculate agent weight based on win rate using EMA.

        Weight = (1 - alpha) × base_weight + alpha × (win_rate / 6)
        This keeps total weight ≈ 1.0 across all agents.

        Args:
            win_rate: Agent's winning percentage [0.0, 1.0]

        Returns:
            New weight bounded [0.05, 0.35]
        """
        alpha = self.learning_rate
        base_weight = 1.0 / 6  # 0.166...

        new_weight = (1 - alpha) * base_weight + alpha * (win_rate * base_weight)

        # Clip to bounds
        return max(self.min_weight, min(self.max_weight, new_weight))

    def get_agent_stats(self) -> Dict[str, Dict]:
        """Get detailed statistics for all agents."""
        stats = {}

        for agent in self.agents.values():
            total_calls = agent.correct_calls + agent.wrong_calls
            stats[agent.name] = {
                'weight': agent.weight,
                'win_rate': agent.win_rate,
                'correct_calls': agent.correct_calls,
                'wrong_calls': agent.wrong_calls,
                'total_calls': total_calls,
                'updated_at': agent.updated_at.isoformat() if agent.updated_at else None,
            }

        return stats

    def reset_weights(self) -> None:
        """Reset all agent weights to equal (1/6 each)."""
        for agent in self.agents.values():
            agent.weight = 0.166
            agent.base_weight = 0.166

        self.save_trust_scores()
        logger.info("Reset all agent weights to equal (1/6)")
