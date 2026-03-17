"""
ELO-style momentum rating system — the primary edge signal.

Philosophy (from tennis prediction research):
  The single most predictive feature was a custom ELO momentum score,
  not raw win rates. We track separate ELO ratings for:
    • Market CATEGORY (politics, crypto, sports…)
    • Recurring EVENT TYPE (elections, Fed decisions, earnings…)
    • TIME-OF-DAY BUCKET (morning / afternoon / evening)

  ELO-implied probability is compared to current market price.
  The delta IS the primary edge signal.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_ELO = 1500.0
K_BASE = 32.0
ELO_SAVE_PATH = Path("data/elo_state.json")

CATEGORIES = [
    "politics", "crypto", "sports", "economics", "science",
    "entertainment", "geopolitics", "health", "technology", "general",
]

EVENT_TYPES = [
    "election", "fed_decision", "earnings", "referendum",
    "sports_match", "crypto_listing", "legislation", "other",
]

TIME_BUCKETS = ["morning", "afternoon", "evening", "overnight"]


def _elo_to_probability(rating_a: float, rating_b: float = DEFAULT_ELO) -> float:
    """Convert ELO rating to expected win probability."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _expected(rating: float, opponent: float = DEFAULT_ELO) -> float:
    return _elo_to_probability(rating, opponent)


def _new_rating(rating: float, actual: float, expected: float, k: float) -> float:
    return rating + k * (actual - expected)


def _time_bucket(dt: Optional[datetime] = None) -> str:
    """Classify hour of day into a named bucket."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    h = dt.hour
    if 6 <= h < 12:
        return "morning"
    if 12 <= h < 18:
        return "afternoon"
    if 18 <= h < 23:
        return "evening"
    return "overnight"


def _infer_event_type(question: str) -> str:
    """Heuristically classify a market question into an event type."""
    q = question.lower()
    if any(w in q for w in ("election", "elect", "president", "senator", "governor", "vote", "poll")):
        return "election"
    if any(w in q for w in ("fed", "federal reserve", "interest rate", "fomc", "rate hike", "rate cut")):
        return "fed_decision"
    if any(w in q for w in ("earnings", "revenue", "eps", "profit", "quarterly")):
        return "earnings"
    if any(w in q for w in ("referendum", "ballot", "proposition")):
        return "referendum"
    if any(w in q for w in ("win", "lose", "match", "game", "championship", "tournament", "score")):
        return "sports_match"
    if any(w in q for w in ("list", "listing", "etf", "approve", "sec")):
        return "crypto_listing"
    if any(w in q for w in ("bill", "law", "legislation", "congress", "senate", "pass")):
        return "legislation"
    return "other"


@dataclass
class ELOState:
    category: dict[str, float] = field(default_factory=dict)
    event_type: dict[str, float] = field(default_factory=dict)
    time_bucket: dict[str, float] = field(default_factory=dict)
    # history: list of (timestamp, dimension, key, delta, new_rating)
    history: list[dict] = field(default_factory=list)

    def get(self, dimension: str, key: str) -> float:
        store = getattr(self, dimension, {})
        return store.get(key, DEFAULT_ELO)

    def set(self, dimension: str, key: str, value: float):
        getattr(self, dimension)[key] = value


class ELOSystem:
    """
    Maintains ELO ratings across three dimensions.
    Updates ratings when markets resolve.
    Primary output: elo_implied_probability and edge vs. current market price.
    """

    def __init__(self, save_path: Path = ELO_SAVE_PATH):
        self.save_path = save_path
        self.state = ELOState()
        self._init_defaults()
        self.load()

    def _init_defaults(self):
        for c in CATEGORIES:
            if c not in self.state.category:
                self.state.category[c] = DEFAULT_ELO
        for e in EVENT_TYPES:
            if e not in self.state.event_type:
                self.state.event_type[e] = DEFAULT_ELO
        for b in TIME_BUCKETS:
            if b not in self.state.time_bucket:
                self.state.time_bucket[b] = DEFAULT_ELO

    # ── Rating retrieval ───────────────────────────────────────────────────

    def get_category_elo(self, category: str) -> float:
        return self.state.get("category", category.lower())

    def get_event_elo(self, question: str) -> float:
        etype = _infer_event_type(question)
        return self.state.get("event_type", etype)

    def get_time_bucket_elo(self, dt: Optional[datetime] = None) -> float:
        bucket = _time_bucket(dt)
        return self.state.get("time_bucket", bucket)

    def elo_implied_probability(self, category: str, question: str = "", dt: Optional[datetime] = None) -> float:
        """
        Compute a blended ELO-implied YES probability from all three dimensions.
        Weights: category 60%, event_type 30%, time_bucket 10%.
        """
        cat_p = _elo_to_probability(self.get_category_elo(category))
        evt_p = _elo_to_probability(self.get_event_elo(question))
        tod_p = _elo_to_probability(self.get_time_bucket_elo(dt))
        blended = 0.60 * cat_p + 0.30 * evt_p + 0.10 * tod_p
        return float(min(max(blended, 0.01), 0.99))

    def elo_edge(self, current_price: float, category: str, question: str = "") -> float:
        """
        The primary edge signal: ELO-implied probability minus market price.
        Positive = market underprices YES; negative = market overprices YES.
        """
        return self.elo_implied_probability(category, question) - current_price

    # ── Rating updates ─────────────────────────────────────────────────────

    def record_resolution(
        self,
        category: str,
        question: str,
        resolved_yes: bool,
        volume: float = 0.0,
        resolved_at: Optional[datetime] = None,
    ):
        """
        Update ELO after a market resolves.
        Volume-weighted K-factor: larger markets → bigger ELO swing.
        """
        if resolved_at is None:
            resolved_at = datetime.now(timezone.utc)

        # Volume-weight K: base K at $1k volume, up to 3x at $1M
        volume_factor = min(3.0, 1.0 + math.log10(max(volume, 1000) / 1000) * 0.5)
        k = K_BASE * volume_factor
        actual = 1.0 if resolved_yes else 0.0

        updates = [
            ("category", category.lower()),
            ("event_type", _infer_event_type(question)),
            ("time_bucket", _time_bucket(resolved_at)),
        ]
        for dim, key in updates:
            old = self.state.get(dim, key)
            exp = _expected(old)
            new = _new_rating(old, actual, exp, k)
            self.state.set(dim, key, new)
            self.state.history.append({
                "ts": resolved_at.isoformat(),
                "dim": dim,
                "key": key,
                "delta": new - old,
                "new_rating": new,
                "volume": volume,
                "resolved_yes": resolved_yes,
            })
            logger.debug(f"ELO update [{dim}:{key}] {old:.1f} → {new:.1f} (Δ{new-old:+.1f}, k={k:.1f})")

        # Keep history bounded
        if len(self.state.history) > 10_000:
            self.state.history = self.state.history[-5_000:]

        self.save()

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "category": self.state.category,
            "event_type": self.state.event_type,
            "time_bucket": self.state.time_bucket,
            "history": self.state.history[-500:],  # persist last 500 events only
        }
        with open(self.save_path, "w") as f:
            json.dump(payload, f, indent=2)

    def load(self):
        if not self.save_path.exists():
            return
        try:
            with open(self.save_path) as f:
                data = json.load(f)
            self.state.category.update(data.get("category", {}))
            self.state.event_type.update(data.get("event_type", {}))
            self.state.time_bucket.update(data.get("time_bucket", {}))
            self.state.history = data.get("history", [])
            logger.info("ELO state loaded from disk.")
        except Exception as e:
            logger.warning(f"Could not load ELO state: {e}")

    def get_ratings_table(self) -> dict[str, dict[str, float]]:
        return {
            "category": dict(self.state.category),
            "event_type": dict(self.state.event_type),
            "time_bucket": dict(self.state.time_bucket),
        }

    def get_top_categories(self, n: int = 5) -> list[tuple[str, float]]:
        """Return top N categories by ELO rating."""
        return sorted(
            self.state.category.items(), key=lambda x: x[1], reverse=True
        )[:n]
