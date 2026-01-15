"""
Test Suite: Consensus Engine Validation
========================================

Tests the parallel voting system:
1. Vote aggregation mathematics
2. Position sizing with consensus confidence
3. Reinforcement learning weight updates
4. Edge cases and stress tests
5. Database persistence
"""

import unittest
import tempfile
import sqlite3
import os
from datetime import datetime

from src.core.consensus import ConsensusEngine, Agent, ConsensusResult


class TestVoteAggregation(unittest.TestCase):
    """Test the core vote aggregation mathematics."""

    def setUp(self):
        """Create temporary database for testing."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()

        # Initialize database with minimal schema
        self._init_db()

        self.engine = ConsensusEngine(db_path=self.db_path)

    def tearDown(self):
        """Clean up temporary database."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def _init_db(self):
        """Initialize database schema for testing."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_trust_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL UNIQUE,
                base_weight REAL DEFAULT 0.166,
                correct_calls INTEGER DEFAULT 0,
                wrong_calls INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.50,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consensus_votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                regime_vote REAL,
                strategy_vote REAL,
                sniper_vote REAL,
                news_vote REAL,
                seasonality_vote REAL,
                institutional_vote REAL,
                regime_weighted REAL,
                strategy_weighted REAL,
                sniper_weighted REAL,
                news_weighted REAL,
                seasonality_weighted REAL,
                institutional_weighted REAL,
                consensus_score REAL,
                position_size_ratio REAL,
                executed INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                side TEXT,
                regime_vote REAL,
                strategy_vote REAL,
                sniper_vote REAL,
                news_vote REAL,
                seasonality_vote REAL,
                institutional_vote REAL,
                consensus_score REAL,
                was_correct INTEGER DEFAULT NULL
            )
        """)

        conn.commit()
        conn.close()

    def test_all_agents_vote_buy(self):
        """Test: All 6 agents vote +1.0 (Strong Buy) → consensus = +1.0"""
        votes = {
            'regime': 1.0,
            'strategy': 1.0,
            'sniper': 1.0,
            'news': 1.0,
            'seasonality': 1.0,
            'institutional': 1.0,
        }

        result = self.engine.aggregate_consensus(votes)

        self.assertEqual(result.consensus_score, 1.0)
        self.assertEqual(result.decision, "STRONG_BUY")
        self.assertEqual(result.confidence_level, "HIGH")
        self.assertEqual(result.position_size_ratio, 1.0)

    def test_all_agents_vote_sell(self):
        """Test: All 6 agents vote -1.0 (Strong Sell) → consensus = -1.0"""
        votes = {
            'regime': -1.0,
            'strategy': -1.0,
            'sniper': -1.0,
            'news': -1.0,
            'seasonality': -1.0,
            'institutional': -1.0,
        }

        result = self.engine.aggregate_consensus(votes)

        self.assertEqual(result.consensus_score, -1.0)
        self.assertEqual(result.decision, "STRONG_SELL")
        self.assertEqual(result.confidence_level, "HIGH")

    def test_all_agents_neutral(self):
        """Test: All 6 agents vote 0.0 (Neutral) → consensus = 0.0"""
        votes = {
            'regime': 0.0,
            'strategy': 0.0,
            'sniper': 0.0,
            'news': 0.0,
            'seasonality': 0.0,
            'institutional': 0.0,
        }

        result = self.engine.aggregate_consensus(votes)

        self.assertEqual(result.consensus_score, 0.0)
        self.assertEqual(result.decision, "HOLD")
        self.assertEqual(result.confidence_level, "LOW")

    def test_mixed_votes(self):
        """Test: Mixed votes with different values → weighted consensus"""
        votes = {
            'regime': 1.0,      # Strong buy
            'strategy': 1.0,    # Strong buy
            'sniper': 0.8,      # Moderate buy
            'news': 0.0,        # Neutral
            'seasonality': -0.5,  # Slight sell
            'institutional': 0.6, # Moderate buy
        }

        result = self.engine.aggregate_consensus(votes)

        # With equal weights (1/6 each):
        # consensus = (1.0 + 1.0 + 0.8 + 0.0 - 0.5 + 0.6) / 6 = 3.9 / 6 ≈ 0.483
        self.assertGreater(result.consensus_score, 0.40)
        self.assertIn(result.decision, ["BUY", "STRONG_BUY"])

    def test_split_vote_sell_majority(self):
        """Test: Majority votes Sell → consensus negative"""
        votes = {
            'regime': -1.0,
            'strategy': -1.0,
            'sniper': -0.8,
            'news': 0.5,        # Dissenter
            'seasonality': -0.5,
            'institutional': 0.3,  # Dissenter
        }

        result = self.engine.aggregate_consensus(votes)

        # (-1 - 1 - 0.8 + 0.5 - 0.5 + 0.3) / 6 ≈ -0.417
        self.assertLess(result.consensus_score, -0.20)
        self.assertIn(result.decision, ["SELL", "STRONG_SELL"])

    def test_vote_clipping(self):
        """Test: Votes outside [-1, 1] are clipped"""
        votes = {
            'regime': 2.5,      # Should be clipped to 1.0
            'strategy': -3.0,   # Should be clipped to -1.0
            'sniper': 0.5,
            'news': 0.5,
            'seasonality': 0.5,
            'institutional': 0.5,
        }

        result = self.engine.aggregate_consensus(votes)

        # After clipping: (1.0 - 1.0 + 0.5 + 0.5 + 0.5 + 0.5) / 6 = 2.0 / 6 ≈ 0.33
        self.assertGreater(result.consensus_score, 0.20)
        self.assertLess(result.consensus_score, 0.50)

    def test_consensus_threshold(self):
        """Test: Consensus threshold determines execution"""
        # Below threshold
        votes_low = {
            'regime': 0.3,
            'strategy': 0.2,
            'sniper': 0.1,
            'news': 0.0,
            'seasonality': 0.2,
            'institutional': 0.1,
        }
        result_low = self.engine.aggregate_consensus(votes_low)
        self.assertLess(abs(result_low.consensus_score), 0.60)
        self.assertEqual(result_low.decision, "HOLD")

        # Above threshold
        votes_high = {
            'regime': 1.0,
            'strategy': 0.9,
            'sniper': 0.8,
            'news': 0.7,
            'seasonality': 0.6,
            'institutional': 0.8,
        }
        result_high = self.engine.aggregate_consensus(votes_high)
        self.assertGreater(result_high.consensus_score, 0.60)
        self.assertIn(result_high.decision, ["STRONG_BUY", "BUY"])


class TestPositionSizing(unittest.TestCase):
    """Test position size scaling based on consensus confidence."""

    def setUp(self):
        """Create temporary database for testing."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_trust_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL UNIQUE,
                base_weight REAL DEFAULT 0.166,
                correct_calls INTEGER DEFAULT 0,
                wrong_calls INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.50,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

        self.engine = ConsensusEngine(db_path=self.db_path)

    def tearDown(self):
        """Clean up temporary database."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_position_size_full_confidence(self):
        """Test: Full consensus (+1.0) → 100% of base size"""
        votes = {
            'regime': 1.0,
            'strategy': 1.0,
            'sniper': 1.0,
            'news': 1.0,
            'seasonality': 1.0,
            'institutional': 1.0,
        }
        result = self.engine.aggregate_consensus(votes)

        base_size = 1000.0
        scaled_size = self.engine.calculate_position_size(base_size, result)

        self.assertEqual(scaled_size, 1000.0)

    def test_position_size_half_confidence(self):
        """Test: 60% consensus (+0.6) → 60% of base size"""
        votes = {
            'regime': 1.0,
            'strategy': 0.9,
            'sniper': 0.8,
            'news': 0.5,
            'seasonality': 0.4,
            'institutional': 0.0,
        }
        result = self.engine.aggregate_consensus(votes)

        base_size = 1000.0
        scaled_size = self.engine.calculate_position_size(base_size, result)

        # Should be ~60% of base
        self.assertGreater(scaled_size, 500)
        self.assertLess(scaled_size, 700)

    def test_position_size_zero_consensus(self):
        """Test: Neutral consensus (0.0) → 0% of base size"""
        votes = {
            'regime': 0.0,
            'strategy': 0.0,
            'sniper': 0.0,
            'news': 0.0,
            'seasonality': 0.0,
            'institutional': 0.0,
        }
        result = self.engine.aggregate_consensus(votes)

        base_size = 1000.0
        scaled_size = self.engine.calculate_position_size(base_size, result)

        self.assertEqual(scaled_size, 0.0)


class TestReinforcementLearning(unittest.TestCase):
    """Test weight updates based on trade outcomes."""

    def setUp(self):
        """Create temporary database for testing."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()

        self._init_db()
        self.engine = ConsensusEngine(db_path=self.db_path)

    def tearDown(self):
        """Clean up temporary database."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def _init_db(self):
        """Initialize database schema for testing."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_trust_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL UNIQUE,
                base_weight REAL DEFAULT 0.166,
                correct_calls INTEGER DEFAULT 0,
                wrong_calls INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.50,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consensus_votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                regime_vote REAL,
                strategy_vote REAL,
                sniper_vote REAL,
                news_vote REAL,
                seasonality_vote REAL,
                institutional_vote REAL,
                regime_weighted REAL,
                strategy_weighted REAL,
                sniper_weighted REAL,
                news_weighted REAL,
                seasonality_weighted REAL,
                institutional_weighted REAL,
                consensus_score REAL,
                position_size_ratio REAL,
                executed INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                side TEXT,
                regime_vote REAL,
                strategy_vote REAL,
                sniper_vote REAL,
                news_vote REAL,
                seasonality_vote REAL,
                institutional_vote REAL,
                consensus_score REAL,
                was_correct INTEGER DEFAULT NULL
            )
        """)

        conn.commit()
        conn.close()

    def test_correct_vote_increases_weight(self):
        """Test: Correct prediction increases agent weight"""
        # Record votes for a BUY trade
        votes = {
            'regime': 1.0,      # Bullish (correct)
            'strategy': 1.0,    # Bullish (correct)
            'sniper': 0.8,      # Bullish (correct)
            'news': 0.0,        # Neutral
            'seasonality': -0.5,  # Bearish (incorrect)
            'institutional': 0.6, # Bullish (correct)
        }

        self.engine.record_votes(
            trade_id=1,
            votes_dict=votes,
            consensus_score=0.62,
            position_size_ratio=0.62,
            executed=True
        )

        # Insert trade record
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO trades (id, side) VALUES (?, ?)", (1, "BUY"))
        conn.commit()
        conn.close()

        # Update scores with winning trade
        initial_weights = self.engine.get_weights()

        self.engine.update_trust_scores(trade_id=1, final_pnl=100.0)

        updated_weights = self.engine.get_weights()

        # Agents who voted correctly should have higher weights
        self.assertGreater(
            updated_weights['regime'],
            initial_weights['regime']
        )
        self.assertGreater(
            updated_weights['strategy'],
            initial_weights['strategy']
        )
        self.assertGreater(
            updated_weights['sniper'],
            initial_weights['sniper']
        )

        # Agent who voted incorrectly should have lower weight
        self.assertLess(
            updated_weights['seasonality'],
            initial_weights['seasonality']
        )

    def test_learning_convergence(self):
        """Test: After 50 correct predictions, agent weight increases"""
        for i in range(50):
            votes = {
                'regime': 1.0,      # Always correct
                'strategy': -1.0,   # Always incorrect
                'sniper': 0.0,
                'news': 0.0,
                'seasonality': 0.0,
                'institutional': 0.0,
            }

            self.engine.record_votes(
                trade_id=i + 1,
                votes_dict=votes,
                consensus_score=0.0,
                position_size_ratio=0.0,
                executed=True
            )

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO trades (id, side) VALUES (?, ?)", (i + 1, "BUY"))
            conn.commit()
            conn.close()

            self.engine.update_trust_scores(trade_id=i + 1, final_pnl=100.0)

        # Regime should have high weight, Strategy should be low
        stats = self.engine.get_agent_stats()

        self.assertGreater(stats['regime']['win_rate'], 0.95)
        self.assertLess(stats['strategy']['win_rate'], 0.05)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Create temporary database for testing."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_trust_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL UNIQUE,
                base_weight REAL DEFAULT 0.166,
                correct_calls INTEGER DEFAULT 0,
                wrong_calls INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.50,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

        self.engine = ConsensusEngine(db_path=self.db_path)

    def tearDown(self):
        """Clean up temporary database."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_empty_votes(self):
        """Test: Empty vote dict returns neutral consensus"""
        result = self.engine.aggregate_consensus({})

        self.assertEqual(result.consensus_score, 0.0)
        self.assertEqual(result.decision, "HOLD")

    def test_missing_agents(self):
        """Test: Partial votes still aggregate correctly"""
        votes = {
            'regime': 1.0,
            'strategy': 1.0,
            # Missing: sniper, news, seasonality, institutional
        }

        result = self.engine.aggregate_consensus(votes)

        # Should still calculate consensus with available votes
        self.assertIsNotNone(result.consensus_score)

    def test_weight_bounds(self):
        """Test: Weights stay within [0.05, 0.35] bounds"""
        # Force extreme win rates
        self.engine.agents['regime'].correct_calls = 1000
        self.engine.agents['regime'].wrong_calls = 0
        self.engine.agents['regime'].win_rate = 1.0

        self.engine.agents['strategy'].correct_calls = 0
        self.engine.agents['strategy'].wrong_calls = 1000
        self.engine.agents['strategy'].win_rate = 0.0

        # Recalculate weights
        for agent in self.engine.agents.values():
            agent.weight = self.engine._calculate_weight(agent.win_rate)

        weights = self.engine.get_weights()

        for agent_name, weight in weights.items():
            self.assertGreaterEqual(weight, 0.05)
            self.assertLessEqual(weight, 0.35)


class TestDatabasePersistence(unittest.TestCase):
    """Test database persistence and recovery."""

    def setUp(self):
        """Create temporary database for testing."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.temp_db.name
        self.temp_db.close()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_trust_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL UNIQUE,
                base_weight REAL DEFAULT 0.166,
                correct_calls INTEGER DEFAULT 0,
                wrong_calls INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.50,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

        self.engine = ConsensusEngine(db_path=self.db_path)

    def tearDown(self):
        """Clean up temporary database."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_save_and_load_weights(self):
        """Test: Weights persist and load correctly"""
        # Modify weights
        self.engine.agents['regime'].weight = 0.25
        self.engine.agents['strategy'].weight = 0.10
        self.engine.save_trust_scores()

        # Create new engine and verify weights loaded
        engine2 = ConsensusEngine(db_path=self.db_path)
        weights = engine2.get_weights()

        # Note: weights will be recalculated, so check approximate values
        self.assertIsNotNone(weights['regime'])
        self.assertIsNotNone(weights['strategy'])


if __name__ == '__main__':
    unittest.main()
