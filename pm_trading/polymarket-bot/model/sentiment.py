"""
Sentiment analysis on news headlines per market.
Features:
  - sentiment_score: current sentiment in [-1, 1]
  - sentiment_momentum: direction of change (positive = getting more bullish)
Exponential decay weights recent headlines more.
15-minute result cache to avoid API rate limits.
Falls back gracefully when NEWS_API_KEY is missing.
"""
from __future__ import annotations

import logging
import re
import time
from collections import deque
from typing import Optional

import requests

logger = logging.getLogger(__name__)

POSITIVE_WORDS = {
    "win", "won", "victory", "passes", "approve", "approved", "rise", "rises",
    "surge", "surges", "gain", "gains", "beat", "beats", "success", "higher",
    "up", "increase", "breakthrough", "positive", "strong", "confirms", "leads",
    "ahead", "record", "growth", "rally", "bullish", "outperform", "exceed",
}
NEGATIVE_WORDS = {
    "lose", "lost", "defeat", "fail", "fails", "decline", "declines", "drop",
    "drops", "crash", "crashes", "down", "lower", "negative", "weak", "crisis",
    "collapse", "risk", "concern", "fear", "worries", "miss", "misses",
    "below", "bearish", "underperform", "disappoint", "plunge", "slump",
}

CACHE_TTL_SECONDS = 900  # 15 minutes


def _lexicon_sentiment(text: str) -> float:
    words = set(re.findall(r"\b\w+\b", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    return (pos - neg) / total if total else 0.0


def _extract_keywords(question: str, n: int = 3) -> str:
    stop = {"will", "the", "a", "an", "in", "of", "be", "is", "are",
            "by", "to", "or", "and", "for", "at", "on", "with", "has",
            "have", "from", "that", "this", "than"}
    words = re.findall(r"\b[A-Za-z]{3,}\b", question)
    keywords = [w for w in words if w.lower() not in stop][:n]
    return " ".join(keywords) if keywords else question[:50]


class _CacheEntry:
    def __init__(self, score: float, ts: float):
        self.score = score
        self.ts = ts


class SentimentAnalyzer:
    NEWS_API_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self._cache: dict[str, _CacheEntry] = {}
        # Rolling history for momentum: market_id -> deque of (ts, score)
        self._history: dict[str, deque] = {}
        self.session = requests.Session()
        if not api_key:
            logger.warning("NEWS_API_KEY not set — using lexicon-only sentiment (no API calls).")

    def _fetch_headlines(self, query: str) -> list[dict]:
        """Fetch articles from NewsAPI with backoff."""
        if not self.api_key:
            return []
        for attempt in range(3):
            try:
                resp = self.session.get(
                    self.NEWS_API_URL,
                    params={
                        "q": query,
                        "pageSize": 10,
                        "sortBy": "publishedAt",
                        "language": "en",
                        "apiKey": self.api_key,
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    return resp.json().get("articles", [])
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                else:
                    break
            except requests.RequestException as e:
                logger.debug(f"NewsAPI request failed: {e}")
                time.sleep(2 ** attempt)
        return []

    def _score_articles(self, articles: list[dict]) -> float:
        """Score articles with exponential time decay (recent = more weight)."""
        if not articles:
            return 0.0
        now = time.time()
        total_weight = 0.0
        weighted_score = 0.0
        for i, a in enumerate(articles):
            # Recency weight: most recent article gets weight 1, decays by 0.7x per step
            weight = 0.7 ** i
            text = (a.get("title", "") + " " + (a.get("description") or "")).strip()
            score = _lexicon_sentiment(text)
            weighted_score += score * weight
            total_weight += weight
        return weighted_score / total_weight if total_weight else 0.0

    def analyze_market(self, market_id: str, question: str) -> float:
        """Return sentiment_score in [-1, 1]. Uses cache if fresh."""
        now = time.time()
        entry = self._cache.get(market_id)
        if entry and (now - entry.ts) < CACHE_TTL_SECONDS:
            return entry.score

        query = _extract_keywords(question)
        articles = self._fetch_headlines(query)
        if articles:
            score = self._score_articles(articles)
        else:
            score = _lexicon_sentiment(question)

        self._cache[market_id] = _CacheEntry(score, now)

        # Update rolling history for momentum
        hist = self._history.setdefault(market_id, deque(maxlen=10))
        hist.append((now, score))

        return score

    def get_momentum(self, market_id: str) -> float:
        """
        Sentiment momentum: positive = getting more bullish, negative = more bearish.
        Computed as linear trend over the rolling history window.
        """
        hist = self._history.get(market_id)
        if not hist or len(hist) < 2:
            return 0.0
        scores = [s for _, s in hist]
        n = len(scores)
        # Simple first-minus-last normalized by count
        momentum = (scores[-1] - scores[0]) / n
        return float(max(-1.0, min(1.0, momentum * 5)))

    def batch_analyze(self, markets: list) -> dict[str, tuple[float, float]]:
        """Return dict of market_id -> (sentiment_score, sentiment_momentum)."""
        result = {}
        for m in markets:
            score = self.analyze_market(m.market_id, m.question)
            mom = self.get_momentum(m.market_id)
            result[m.market_id] = (score, mom)
        return result

    def clear_cache(self):
        self._cache.clear()
        self._history.clear()
