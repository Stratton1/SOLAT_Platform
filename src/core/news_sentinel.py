"""
News Sentinel - Global Sentiment Circuit Breaker.

Uses Natural Language Processing (NLP) to analyze financial news RSS feeds
in real-time. Acts as the final filter before trade execution.

Theory:
- Market sentiment drives short-term price action
- Fearful markets (Score < 30) are dangerous for Longs
- Greedy markets (Score > 70) are dangerous for Shorts
- News Sentinel blocks trades that go against market mood

This is the fourth and final filter in the trading pipeline:
1. Regime (HMM): Is the market safe?
2. Strategy (Ichimoku): Is there a trend?
3. Microstructure (Order Book): Is there liquidity?
4. News Sentinel (NLP): Is sentiment aligned?

Only trades that pass ALL FOUR checks get executed.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger.warning("feedparser not installed. News Sentinel disabled.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not installed. News Sentinel disabled.")

NEWS_SENTINEL_AVAILABLE = FEEDPARSER_AVAILABLE and VADER_AVAILABLE


@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""

    score: float  # 0-100 normalized score
    mood: str  # "extreme_fear", "fear", "neutral", "greed", "extreme_greed"
    headlines_analyzed: int
    positive_count: int
    negative_count: int
    neutral_count: int
    critical_headlines: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    is_valid: bool = True
    error_message: Optional[str] = None


class NewsSentinel:
    """
    Global Sentiment Circuit Breaker using NLP analysis of news feeds.

    Monitors financial news RSS feeds and calculates a market sentiment score.
    Blocks trades that go against prevailing market sentiment.

    Score Ranges:
    - 0-20: Extreme Fear (Block ALL Longs)
    - 20-40: Fear (Block Longs)
    - 40-60: Neutral (No restrictions)
    - 60-80: Greed (Block Shorts)
    - 80-100: Extreme Greed (Block ALL Shorts)

    Usage:
        >>> sentinel = NewsSentinel()
        >>> sentiment = sentinel.fetch_sentiment()
        >>> is_allowed = sentinel.check_news_veto("BUY", sentiment.score)
    """

    # RSS Feed Sources - Reliable financial news
    RSS_FEEDS: List[Dict[str, str]] = [
        # General Financial News
        {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rssindex"},
        {"name": "CNBC Top News", "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"},
        {"name": "Reuters Business", "url": "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best"},
        {"name": "MarketWatch", "url": "https://feeds.marketwatch.com/marketwatch/topstories/"},
        {"name": "BBC Business", "url": "http://feeds.bbci.co.uk/news/business/rss.xml"},
        # Crypto-specific
        {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
        {"name": "CoinTelegraph", "url": "https://cointelegraph.com/rss"},
        # Forex-specific
        {"name": "ForexLive", "url": "https://www.forexlive.com/feed/news"},
        {"name": "FXStreet", "url": "https://www.fxstreet.com/rss/news"},
        # Economic Calendar / Central Banks
        {"name": "Fed News", "url": "https://www.federalreserve.gov/feeds/press_all.xml"},
    ]

    # Keywords that indicate critical/high-impact news (double weight)
    CRITICAL_KEYWORDS: List[str] = [
        "breaking", "alert", "crash", "war", "invasion", "collapse",
        "bankruptcy", "default", "emergency", "crisis", "panic",
        "recession", "depression", "inflation", "deflation", "surge",
        "plunge", "soar", "tank", "dump", "moon", "rally", "selloff",
        "fed", "fomc", "interest rate", "cut", "hike", "hawkish", "dovish",
        "trump", "biden", "election", "tariff", "sanction", "hack", "exploit",
    ]

    # Sentiment thresholds
    FEAR_THRESHOLD = 30  # Below this = Fear (block Longs)
    GREED_THRESHOLD = 70  # Above this = Greed (block Shorts)
    EXTREME_FEAR_THRESHOLD = 20
    EXTREME_GREED_THRESHOLD = 80

    # Cache settings
    CACHE_DURATION_MINUTES = 15  # Only fetch news every 15 minutes

    def __init__(
        self,
        feeds: Optional[List[Dict[str, str]]] = None,
        cache_duration_minutes: int = 15,
    ) -> None:
        """
        Initialize the News Sentinel.

        Args:
            feeds: Custom list of RSS feeds. Uses defaults if None.
            cache_duration_minutes: How long to cache sentiment results.
        """
        self.feeds = feeds or self.RSS_FEEDS
        self.cache_duration = timedelta(minutes=cache_duration_minutes)

        # Initialize VADER sentiment analyzer
        if VADER_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None

        # Cache for sentiment results
        self._cached_result: Optional[SentimentResult] = None
        self._last_fetch: Optional[datetime] = None

        logger.info(
            f"Initialized News Sentinel with {len(self.feeds)} RSS feeds. "
            f"Cache duration: {cache_duration_minutes} minutes"
        )

    def _fetch_headlines(self, max_per_feed: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch headlines from all RSS feeds.

        Args:
            max_per_feed: Maximum headlines to fetch per feed.

        Returns:
            List of headline dicts with title, source, and published date.
        """
        if not FEEDPARSER_AVAILABLE:
            return []

        all_headlines = []

        for feed_info in self.feeds:
            try:
                feed = feedparser.parse(feed_info["url"])

                if feed.bozo and not feed.entries:
                    logger.warning(f"Failed to parse {feed_info['name']}: {feed.bozo_exception}")
                    continue

                for entry in feed.entries[:max_per_feed]:
                    headline = {
                        "title": entry.get("title", ""),
                        "source": feed_info["name"],
                        "published": entry.get("published", ""),
                        "link": entry.get("link", ""),
                    }

                    if headline["title"]:
                        all_headlines.append(headline)

                logger.debug(f"Fetched {len(feed.entries[:max_per_feed])} headlines from {feed_info['name']}")

            except Exception as e:
                logger.warning(f"Error fetching {feed_info['name']}: {e}")
                continue

        logger.info(f"Fetched {len(all_headlines)} total headlines from {len(self.feeds)} feeds")
        return all_headlines

    def _score_headline(self, headline: str) -> Tuple[float, bool]:
        """
        Score a single headline using VADER sentiment analysis.

        Args:
            headline: The headline text to analyze.

        Returns:
            Tuple of (score, is_critical):
                - score: -1 (very negative) to +1 (very positive)
                - is_critical: True if headline contains critical keywords
        """
        if not self.analyzer:
            return 0.0, False

        # Get VADER compound score
        sentiment = self.analyzer.polarity_scores(headline)
        score = sentiment["compound"]

        # Check for critical keywords (double weight)
        headline_lower = headline.lower()
        is_critical = any(keyword in headline_lower for keyword in self.CRITICAL_KEYWORDS)

        return score, is_critical

    def _normalize_score(self, raw_score: float) -> float:
        """
        Normalize raw sentiment score (-1 to +1) to 0-100 scale.

        -1.0 -> 0 (Extreme Fear)
         0.0 -> 50 (Neutral)
        +1.0 -> 100 (Extreme Greed)
        """
        return ((raw_score + 1) / 2) * 100

    def _get_mood_label(self, score: float) -> str:
        """Get descriptive mood label from score."""
        if score < self.EXTREME_FEAR_THRESHOLD:
            return "extreme_fear"
        elif score < self.FEAR_THRESHOLD:
            return "fear"
        elif score > self.EXTREME_GREED_THRESHOLD:
            return "extreme_greed"
        elif score > self.GREED_THRESHOLD:
            return "greed"
        else:
            return "neutral"

    def fetch_sentiment(self, force_refresh: bool = False) -> SentimentResult:
        """
        Fetch and analyze sentiment from news feeds.

        Uses caching to avoid excessive API calls. Only fetches new data
        if cache has expired or force_refresh is True.

        Args:
            force_refresh: If True, bypasses cache and fetches fresh data.

        Returns:
            SentimentResult with score (0-100), mood label, and statistics.
        """
        # Check cache
        if not force_refresh and self._cached_result and self._last_fetch:
            time_since_fetch = datetime.utcnow() - self._last_fetch
            if time_since_fetch < self.cache_duration:
                logger.debug(f"Using cached sentiment (age: {time_since_fetch.seconds}s)")
                return self._cached_result

        # Check if dependencies are available
        if not NEWS_SENTINEL_AVAILABLE:
            return SentimentResult(
                score=50.0,  # Neutral default
                mood="neutral",
                headlines_analyzed=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                is_valid=False,
                error_message="News Sentinel dependencies not installed"
            )

        try:
            # Fetch headlines
            headlines = self._fetch_headlines(max_per_feed=10)

            if not headlines:
                logger.warning("No headlines fetched - using neutral sentiment")
                return SentimentResult(
                    score=50.0,
                    mood="neutral",
                    headlines_analyzed=0,
                    positive_count=0,
                    negative_count=0,
                    neutral_count=0,
                    is_valid=False,
                    error_message="No headlines available"
                )

            # Score each headline
            total_score = 0.0
            total_weight = 0.0
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            critical_headlines = []

            for headline_data in headlines:
                title = headline_data["title"]
                score, is_critical = self._score_headline(title)

                # Apply weighting
                weight = 2.0 if is_critical else 1.0
                total_score += score * weight
                total_weight += weight

                # Categorize
                if score > 0.05:
                    positive_count += 1
                elif score < -0.05:
                    negative_count += 1
                else:
                    neutral_count += 1

                # Track critical headlines
                if is_critical:
                    critical_headlines.append(f"{headline_data['source']}: {title}")

            # Calculate weighted average
            if total_weight > 0:
                avg_score = total_score / total_weight
            else:
                avg_score = 0.0

            # Normalize to 0-100 scale
            normalized_score = self._normalize_score(avg_score)
            mood = self._get_mood_label(normalized_score)

            result = SentimentResult(
                score=normalized_score,
                mood=mood,
                headlines_analyzed=len(headlines),
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                critical_headlines=critical_headlines[:5],  # Top 5 critical
                last_updated=datetime.utcnow(),
                is_valid=True,
            )

            # Cache the result
            self._cached_result = result
            self._last_fetch = datetime.utcnow()

            logger.info(
                f"News Sentiment: {normalized_score:.1f}/100 ({mood}) | "
                f"Headlines: {len(headlines)} | "
                f"+{positive_count} -{negative_count} ={neutral_count} | "
                f"Critical: {len(critical_headlines)}"
            )

            return result

        except Exception as e:
            logger.error(f"Error fetching sentiment: {e}", exc_info=True)
            return SentimentResult(
                score=50.0,
                mood="neutral",
                headlines_analyzed=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                is_valid=False,
                error_message=str(e)
            )

    def check_news_veto(
        self,
        side: str,
        sentiment_score: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Check if a trade should be vetoed based on news sentiment.

        Rules:
        - Score < 30 (Fear) AND side == 'BUY' -> VETO
        - Score > 70 (Greed) AND side == 'SELL' -> VETO
        - Otherwise -> ALLOW

        Args:
            side: Trade side ('BUY' or 'SELL')
            sentiment_score: Pre-fetched score (0-100). If None, fetches fresh.

        Returns:
            Tuple[is_allowed, reason]:
                - is_allowed: True if trade passes news filter
                - reason: Explanation of decision
        """
        # Get sentiment score
        if sentiment_score is None:
            result = self.fetch_sentiment()
            sentiment_score = result.score
            mood = result.mood
        else:
            mood = self._get_mood_label(sentiment_score)

        side = side.upper()

        # Check veto conditions
        if side == "BUY":
            if sentiment_score < self.EXTREME_FEAR_THRESHOLD:
                reason = f"EXTREME FEAR ({sentiment_score:.0f}/100) - All Longs blocked"
                logger.warning(f"⛔ NEWS VETO: {side} - {reason}")
                return False, reason
            elif sentiment_score < self.FEAR_THRESHOLD:
                reason = f"Fear sentiment ({sentiment_score:.0f}/100) - Longs restricted"
                logger.warning(f"⛔ NEWS VETO: {side} - {reason}")
                return False, reason
            else:
                reason = f"Sentiment OK ({sentiment_score:.0f}/100 - {mood})"
                logger.info(f"📰 NEWS PASS: {side} - {reason}")
                return True, reason

        elif side == "SELL":
            if sentiment_score > self.EXTREME_GREED_THRESHOLD:
                reason = f"EXTREME GREED ({sentiment_score:.0f}/100) - All Shorts blocked"
                logger.warning(f"⛔ NEWS VETO: {side} - {reason}")
                return False, reason
            elif sentiment_score > self.GREED_THRESHOLD:
                reason = f"Greed sentiment ({sentiment_score:.0f}/100) - Shorts restricted"
                logger.warning(f"⛔ NEWS VETO: {side} - {reason}")
                return False, reason
            else:
                reason = f"Sentiment OK ({sentiment_score:.0f}/100 - {mood})"
                logger.info(f"📰 NEWS PASS: {side} - {reason}")
                return True, reason

        else:
            # NEUTRAL or unknown side - allow
            reason = f"No restriction for {side}"
            return True, reason

    def should_refresh(self) -> bool:
        """Check if sentiment cache should be refreshed."""
        if self._last_fetch is None:
            return True

        time_since_fetch = datetime.utcnow() - self._last_fetch
        return time_since_fetch >= self.cache_duration

    def get_cached_sentiment(self) -> Optional[SentimentResult]:
        """Get cached sentiment without fetching new data."""
        return self._cached_result

    def get_mood_emoji(self, score: Optional[float] = None) -> str:
        """Get emoji representation of market mood."""
        if score is None:
            if self._cached_result:
                score = self._cached_result.score
            else:
                return "❓"

        if score < self.EXTREME_FEAR_THRESHOLD:
            return "😱"  # Extreme Fear
        elif score < self.FEAR_THRESHOLD:
            return "😰"  # Fear
        elif score > self.EXTREME_GREED_THRESHOLD:
            return "🤑"  # Extreme Greed
        elif score > self.GREED_THRESHOLD:
            return "😄"  # Greed
        else:
            return "😐"  # Neutral


# Convenience function for quick sentiment check
def get_market_sentiment() -> Dict[str, Any]:
    """
    Quick function to get current market sentiment.

    Returns:
        Dict with score, mood, and details.
    """
    sentinel = NewsSentinel()
    result = sentinel.fetch_sentiment()

    return {
        "score": result.score,
        "mood": result.mood,
        "emoji": sentinel.get_mood_emoji(result.score),
        "headlines_analyzed": result.headlines_analyzed,
        "is_valid": result.is_valid,
        "last_updated": result.last_updated.isoformat(),
    }
