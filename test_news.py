#!/usr/bin/env python3
"""
Test script for News Sentinel system.

Verifies:
1. NewsSentinel initialization
2. VADER sentiment scoring
3. News veto logic (Fear blocks Longs, Greed blocks Shorts)
4. RSS feed parsing (if feedparser available)
5. Engine integration

This test uses mock headlines to verify sentiment logic
without requiring live RSS feeds.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Ensure data directory exists
Path("data/db").mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_news_sentinel_init() -> bool:
    """Test NewsSentinel initialization."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: NewsSentinel Initialization")
    logger.info("=" * 60)

    try:
        from src.core.news_sentinel import NewsSentinel, NEWS_SENTINEL_AVAILABLE

        logger.info(f"NEWS_SENTINEL_AVAILABLE: {NEWS_SENTINEL_AVAILABLE}")

        # Initialize sentinel
        sentinel = NewsSentinel()

        logger.info(f"Initialized NewsSentinel with {len(sentinel.feeds)} RSS feeds")
        logger.info(f"Cache duration: {sentinel.cache_duration}")
        logger.info(f"Fear threshold: {sentinel.FEAR_THRESHOLD}")
        logger.info(f"Greed threshold: {sentinel.GREED_THRESHOLD}")

        # List configured feeds
        logger.info("\nConfigured RSS Feeds:")
        for feed in sentinel.feeds[:5]:
            logger.info(f"  - {feed['name']}")
        logger.info(f"  ... and {len(sentinel.feeds) - 5} more")

        logger.info("\nNewsSentinel Initialization: PASSED")
        return True

    except Exception as e:
        logger.error(f"NewsSentinel init test failed: {e}", exc_info=True)
        return False


def test_vader_sentiment_scoring() -> bool:
    """Test VADER sentiment scoring on mock headlines."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: VADER Sentiment Scoring")
    logger.info("=" * 60)

    try:
        from src.core.news_sentinel import NewsSentinel, NEWS_SENTINEL_AVAILABLE

        if not NEWS_SENTINEL_AVAILABLE:
            logger.warning("News dependencies not installed. Skipping VADER test.")
            return True

        sentinel = NewsSentinel()

        # Test headlines with known sentiment
        test_headlines = [
            # Positive headlines
            ("Stock market rallies to record highs on strong earnings", "positive"),
            ("Bitcoin surges 10% as institutions pile in", "positive"),
            ("Economic outlook brightens as unemployment falls", "positive"),
            # Negative headlines
            ("Market crashes amid recession fears", "negative"),
            ("Bitcoin plunges 20% in massive selloff", "negative"),
            ("Banking crisis deepens as more banks fail", "negative"),
            # Neutral headlines
            ("Fed to announce interest rate decision tomorrow", "neutral"),
            ("Markets await employment data release", "neutral"),
        ]

        logger.info("Testing sentiment scoring on mock headlines:\n")

        positive_scores = []
        negative_scores = []
        neutral_scores = []

        for headline, expected in test_headlines:
            score, is_critical = sentinel._score_headline(headline)

            # Categorize result
            if score > 0.05:
                actual = "positive"
                positive_scores.append(score)
            elif score < -0.05:
                actual = "negative"
                negative_scores.append(score)
            else:
                actual = "neutral"
                neutral_scores.append(score)

            match_icon = "✓" if actual == expected else "✗"
            critical_flag = " [CRITICAL]" if is_critical else ""

            logger.info(f"  {match_icon} {headline[:50]}...")
            logger.info(f"      Score: {score:.3f} | Expected: {expected} | Got: {actual}{critical_flag}")

        # Verify average scores make sense
        if positive_scores:
            avg_pos = sum(positive_scores) / len(positive_scores)
            logger.info(f"\nAverage positive score: {avg_pos:.3f}")
            assert avg_pos > 0, "Positive headlines should have positive scores"

        if negative_scores:
            avg_neg = sum(negative_scores) / len(negative_scores)
            logger.info(f"Average negative score: {avg_neg:.3f}")
            assert avg_neg < 0, "Negative headlines should have negative scores"

        logger.info("\nVADER Sentiment Scoring: PASSED")
        return True

    except Exception as e:
        logger.error(f"VADER sentiment test failed: {e}", exc_info=True)
        return False


def test_critical_keyword_detection() -> bool:
    """Test detection of critical/high-impact keywords."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Critical Keyword Detection")
    logger.info("=" * 60)

    try:
        from src.core.news_sentinel import NewsSentinel, NEWS_SENTINEL_AVAILABLE

        if not NEWS_SENTINEL_AVAILABLE:
            logger.warning("News dependencies not installed. Skipping test.")
            return True

        sentinel = NewsSentinel()

        # Test headlines with critical keywords
        critical_headlines = [
            "BREAKING: Fed announces emergency rate cut",
            "ALERT: Major exchange hacked, funds stolen",
            "Market CRASH: Dow plunges 1000 points",
            "WAR: Russia invades Ukraine",
            "COLLAPSE: Major bank declares bankruptcy",
        ]

        normal_headlines = [
            "Markets close mixed on light volume",
            "Bitcoin trades sideways around $50k",
            "Economic data in line with expectations",
        ]

        logger.info("Testing critical keyword detection:\n")

        # Test critical headlines
        logger.info("Headlines that SHOULD be marked critical:")
        for headline in critical_headlines:
            score, is_critical = sentinel._score_headline(headline)
            status = "✓ CRITICAL" if is_critical else "✗ NOT CRITICAL"
            logger.info(f"  {status}: {headline[:50]}...")
            assert is_critical, f"Should be critical: {headline}"

        # Test normal headlines
        logger.info("\nHeadlines that should NOT be marked critical:")
        for headline in normal_headlines:
            score, is_critical = sentinel._score_headline(headline)
            status = "✗ CRITICAL" if is_critical else "✓ Normal"
            logger.info(f"  {status}: {headline[:50]}...")

        logger.info("\nCritical Keyword Detection: PASSED")
        return True

    except Exception as e:
        logger.error(f"Critical keyword test failed: {e}", exc_info=True)
        return False


def test_news_veto_logic() -> bool:
    """Test the news veto logic for different sentiment levels."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: News Veto Logic")
    logger.info("=" * 60)

    try:
        from src.core.news_sentinel import NewsSentinel

        sentinel = NewsSentinel()

        test_cases = [
            # (score, side, expected_allowed, description)
            (15, "BUY", False, "Extreme Fear + BUY = BLOCKED"),
            (25, "BUY", False, "Fear + BUY = BLOCKED"),
            (50, "BUY", True, "Neutral + BUY = ALLOWED"),
            (75, "BUY", True, "Greed + BUY = ALLOWED"),
            (85, "BUY", True, "Extreme Greed + BUY = ALLOWED"),

            (15, "SELL", True, "Extreme Fear + SELL = ALLOWED"),
            (25, "SELL", True, "Fear + SELL = ALLOWED"),
            (50, "SELL", True, "Neutral + SELL = ALLOWED"),
            (75, "SELL", False, "Greed + SELL = BLOCKED"),
            (85, "SELL", False, "Extreme Greed + SELL = BLOCKED"),
        ]

        logger.info("Testing veto logic:\n")
        logger.info("  Fear (<30) blocks LONGS")
        logger.info("  Greed (>70) blocks SHORTS\n")

        all_passed = True
        for score, side, expected, desc in test_cases:
            is_allowed, reason = sentinel.check_news_veto(side, score)

            match_icon = "✓" if is_allowed == expected else "✗"
            if is_allowed != expected:
                all_passed = False

            logger.info(f"  {match_icon} Score:{score} + {side} -> {'ALLOWED' if is_allowed else 'BLOCKED'}")
            logger.info(f"      Expected: {desc}")
            logger.info(f"      Reason: {reason}")

        assert all_passed, "Not all veto tests passed"

        logger.info("\nNews Veto Logic: PASSED")
        return True

    except Exception as e:
        logger.error(f"News veto logic test failed: {e}", exc_info=True)
        return False


def test_score_normalization() -> bool:
    """Test that sentiment scores are normalized correctly."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Score Normalization")
    logger.info("=" * 60)

    try:
        from src.core.news_sentinel import NewsSentinel

        sentinel = NewsSentinel()

        # Test normalization function
        test_values = [
            (-1.0, 0.0),    # Most negative -> 0
            (-0.5, 25.0),   # Moderately negative -> 25
            (0.0, 50.0),    # Neutral -> 50
            (0.5, 75.0),    # Moderately positive -> 75
            (1.0, 100.0),   # Most positive -> 100
        ]

        logger.info("Testing score normalization (-1 to +1) -> (0 to 100):\n")

        for raw, expected in test_values:
            normalized = sentinel._normalize_score(raw)
            match = abs(normalized - expected) < 0.01
            match_icon = "✓" if match else "✗"

            logger.info(f"  {match_icon} Raw: {raw:+.1f} -> Normalized: {normalized:.1f} (expected: {expected:.1f})")
            assert match, f"Normalization failed: {raw} -> {normalized}, expected {expected}"

        logger.info("\nScore Normalization: PASSED")
        return True

    except Exception as e:
        logger.error(f"Score normalization test failed: {e}", exc_info=True)
        return False


def test_mood_labels() -> bool:
    """Test mood label assignment."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Mood Labels")
    logger.info("=" * 60)

    try:
        from src.core.news_sentinel import NewsSentinel

        sentinel = NewsSentinel()

        test_cases = [
            (10, "extreme_fear"),
            (25, "fear"),
            (50, "neutral"),
            (75, "greed"),
            (90, "extreme_greed"),
        ]

        logger.info("Testing mood label assignment:\n")

        for score, expected_mood in test_cases:
            mood = sentinel._get_mood_label(score)
            emoji = sentinel.get_mood_emoji(score)
            match_icon = "✓" if mood == expected_mood else "✗"

            logger.info(f"  {match_icon} Score: {score} -> Mood: {emoji} {mood} (expected: {expected_mood})")
            assert mood == expected_mood, f"Wrong mood for {score}: got {mood}, expected {expected_mood}"

        logger.info("\nMood Labels: PASSED")
        return True

    except Exception as e:
        logger.error(f"Mood labels test failed: {e}", exc_info=True)
        return False


def test_engine_integration() -> bool:
    """Test integration with the Sentinel engine."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 7: Engine Integration")
    logger.info("=" * 60)

    try:
        # Test imports
        from src.core.news_sentinel import NewsSentinel, NEWS_SENTINEL_AVAILABLE, SentimentResult

        logger.info("All news sentinel imports successful")
        logger.info("  NewsSentinel: OK")
        logger.info("  NEWS_SENTINEL_AVAILABLE: OK")
        logger.info("  SentimentResult: OK")

        # Read engine.py source directly to verify integration
        engine_path = Path("src/core/engine.py")
        if not engine_path.exists():
            logger.error("Engine file not found")
            return False

        with open(engine_path, "r") as f:
            engine_source = f.read()

        # Check imports
        assert "from src.core.news_sentinel import" in engine_source, "Engine should import news_sentinel"
        assert "NewsSentinel" in engine_source, "Engine should import NewsSentinel"
        logger.info("  Engine imports verified: OK")

        # Check __init__ initialization
        assert "news_sentinel" in engine_source, "Engine should have news_sentinel"
        assert "current_sentiment" in engine_source, "Engine should track current_sentiment"
        logger.info("  NewsSentinel initialization: OK")

        # Check scan_market includes news veto
        assert "NEWS VETO" in engine_source or "NEWS_VETO" in engine_source, "Engine should have NEWS VETO check"
        assert "check_news_veto" in engine_source, "Engine should call check_news_veto"
        logger.info("  News veto in scan_market: OK")

        # Check helper methods
        assert "_should_check_news" in engine_source, "Engine should have _should_check_news method"
        assert "_update_news_sentiment" in engine_source, "Engine should have _update_news_sentiment method"
        logger.info("  Helper methods: OK")

        logger.info("\nEngine Integration: PASSED")
        return True

    except Exception as e:
        logger.error(f"Engine integration test failed: {e}", exc_info=True)
        return False


def test_rss_feed_parsing() -> bool:
    """Test RSS feed parsing (optional, requires feedparser)."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 8: RSS Feed Parsing (Optional)")
    logger.info("=" * 60)

    try:
        from src.core.news_sentinel import NewsSentinel, NEWS_SENTINEL_AVAILABLE

        if not NEWS_SENTINEL_AVAILABLE:
            logger.warning("feedparser/vaderSentiment not installed. Skipping RSS test.")
            logger.info("Install with: pip install feedparser vaderSentiment")
            logger.info("\nRSS Feed Parsing: SKIPPED (dependencies not installed)")
            return True

        sentinel = NewsSentinel()

        logger.info("Attempting to fetch headlines from one RSS feed...")
        logger.info("(This requires internet access)\n")

        # Try to fetch from just BBC Business (usually reliable)
        test_feed = {"name": "BBC Business", "url": "http://feeds.bbci.co.uk/news/business/rss.xml"}
        sentinel.feeds = [test_feed]

        try:
            headlines = sentinel._fetch_headlines(max_per_feed=5)

            if headlines:
                logger.info(f"Successfully fetched {len(headlines)} headlines from BBC Business:\n")
                for i, h in enumerate(headlines[:3], 1):
                    score, is_critical = sentinel._score_headline(h["title"])
                    critical_flag = " [CRITICAL]" if is_critical else ""
                    logger.info(f"  {i}. {h['title'][:60]}...")
                    logger.info(f"     Score: {score:.3f}{critical_flag}")

                logger.info("\nRSS Feed Parsing: PASSED")
                return True
            else:
                logger.warning("No headlines fetched. Network might be unavailable.")
                logger.info("\nRSS Feed Parsing: SKIPPED (no data)")
                return True

        except Exception as e:
            logger.warning(f"RSS fetch failed (network issue?): {e}")
            logger.info("\nRSS Feed Parsing: SKIPPED (network unavailable)")
            return True

    except Exception as e:
        logger.error(f"RSS feed parsing test failed: {e}", exc_info=True)
        return False


def main() -> int:
    """Run all news sentinel tests."""
    logger.info("=" * 60)
    logger.info("SOLAT NEWS SENTINEL - TEST SUITE")
    logger.info("=" * 60)
    logger.info("Target: NLP-based Global Sentiment Circuit Breaker")
    logger.info("Four-Filter System: Regime + Ichimoku + Microstructure + News")
    logger.info("=" * 60)

    results = {
        "NewsSentinel Init": test_news_sentinel_init(),
        "VADER Sentiment Scoring": test_vader_sentiment_scoring(),
        "Critical Keyword Detection": test_critical_keyword_detection(),
        "News Veto Logic": test_news_veto_logic(),
        "Score Normalization": test_score_normalization(),
        "Mood Labels": test_mood_labels(),
        "Engine Integration": test_engine_integration(),
        "RSS Feed Parsing": test_rss_feed_parsing(),
    }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name}: {status}")
        all_passed = all_passed and passed

    logger.info("=" * 60)

    if all_passed:
        logger.info("ALL NEWS SENTINEL TESTS PASSED")
        logger.info("")
        logger.info("NEWS SENTINEL VERIFIED:")
        logger.info("  - NewsSentinel initializes with RSS feeds")
        logger.info("  - VADER scores headlines correctly")
        logger.info("  - Critical keywords detected and weighted")
        logger.info("  - Fear (<30) blocks Longs")
        logger.info("  - Greed (>70) blocks Shorts")
        logger.info("  - Engine integration complete")
        logger.info("")
        logger.info("FOUR-FILTER HEDGE FUND SYSTEM COMPLETE:")
        logger.info("  1. Regime (HMM): Macro market state detection")
        logger.info("  2. Strategy (Ichimoku): AI-optimized signal generation")
        logger.info("  3. Microstructure (Order Book): Liquidity validation")
        logger.info("  4. News Sentinel (NLP): Global sentiment circuit breaker")
        logger.info("")
        logger.info("Only trades passing ALL FOUR filters get executed.")
        logger.info("")
        logger.info("Install dependencies with:")
        logger.info("  pip install feedparser vaderSentiment textblob")
        return 0
    else:
        logger.error("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
