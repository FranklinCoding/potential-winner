"""
Polymarket CLOB + Gamma API market scanner.
Polls open markets every 5-10 seconds.
Tracks rolling volume windows (1h/6h/24h/7d), resolution history,
liquidity spikes, and time-decay signals.
Persists raw snapshots to SQLite for model training.
"""
from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)


class Market:
    """Represents a Polymarket prediction market with enriched features."""

    def __init__(self, data: dict, volume_history: Optional[dict] = None):
        # Support both camelCase (live Gamma API) and snake_case (tests/legacy)
        self.condition_id: str = data.get("conditionId") or data.get("condition_id", "")
        self.question_id: str = data.get("questionId") or data.get("question_id", "")
        self.question: str = data.get("question", "Unknown")
        self.description: str = data.get("description", "")
        self.category: str = (data.get("groupItemTagged") or
                               data.get("category") or "general").lower()
        self.end_date_iso: str = data.get("endDateIso") or data.get("end_date_iso", "")
        self.active: bool = data.get("active", True)
        self.closed: bool = data.get("closed", False)
        self.volume: float = float(data.get("volume", 0) or 0)
        self.volume_24h: float = float(
            data.get("volume24hr") or data.get("volume_24hr") or data.get("volume_24h") or 0
        )
        self.liquidity: float = float(data.get("liquidity", 0) or 0)
        self.created_at: str = data.get("createdAt") or data.get("created_at", "")
        self.bid_ask_spread: float = 0.0

        # Parse token prices — Gamma API uses outcomePrices+outcomes arrays;
        # CLOB API / test fixtures use a tokens list with outcome/price dicts.
        self.yes_price: float = 0.5
        self.no_price: float = 0.5
        outcome_prices = data.get("outcomePrices")
        outcomes = data.get("outcomes")
        if outcome_prices and outcomes and len(outcome_prices) == len(outcomes):
            for outcome, raw_price in zip(outcomes, outcome_prices):
                try:
                    price = float(raw_price)
                except (TypeError, ValueError):
                    continue
                if outcome.upper() in ("YES", "Y"):
                    self.yes_price = price
                elif outcome.upper() in ("NO", "N"):
                    self.no_price = price
        else:
            for token in data.get("tokens", []):
                outcome = token.get("outcome", "").upper()
                price = float(token.get("price", 0) or 0)
                if outcome == "YES":
                    self.yes_price = price
                elif outcome == "NO":
                    self.no_price = price

        # Spread from token prices (approximation)
        if self.yes_price > 0 and self.no_price > 0:
            total = self.yes_price + self.no_price
            self.bid_ask_spread = abs(total - 1.0)

        # Rolling volume windows (filled in by scanner after snapshot lookup)
        self.volume_1h: float = 0.0
        self.volume_6h: float = 0.0
        self.volume_7d: float = float(
            data.get("volumeNum") or data.get("volume_num") or self.volume
        )
        self.volume_acceleration: float = 0.0   # d(volume)/dt
        self.price_velocity: float = 0.0        # d(price)/dt
        self.whale_activity: bool = False        # large order spike detected
        self.prev_snapshot: Optional[dict] = None

        self._raw = data
        if volume_history:
            self._apply_volume_history(volume_history)

    def _apply_volume_history(self, history: dict):
        """Populate rolling volume windows from historical snapshots."""
        snaps = history.get(self.market_id, [])
        if not snaps:
            return
        now_ts = time.time()
        v1h = v6h = 0.0
        for snap in reversed(snaps):
            age = now_ts - snap.get("ts", now_ts)
            vol = snap.get("volume_24h", 0)
            if age <= 3600:
                v1h += vol / max(len(snaps), 1)
            if age <= 21600:
                v6h += vol / max(len(snaps), 1)
        self.volume_1h = v1h
        self.volume_6h = v6h
        if len(snaps) >= 2:
            dt = snaps[-1].get("ts", now_ts) - snaps[-2].get("ts", now_ts - 60)
            if dt > 0:
                dv = snaps[-1].get("volume_24h", 0) - snaps[-2].get("volume_24h", 0)
                self.volume_acceleration = dv / dt
                dp = snaps[-1].get("yes_price", 0.5) - snaps[-2].get("yes_price", 0.5)
                self.price_velocity = dp / dt
        # Whale: last-hour volume > 3× typical 1h volume
        typical_1h = self.volume_24h / 24 if self.volume_24h else 0
        self.whale_activity = self.volume_1h > typical_1h * 3 and self.volume_1h > 500

    @property
    def market_id(self) -> str:
        return self.condition_id or self.question_id

    @property
    def days_to_resolution(self) -> float:
        return self.hours_to_resolution / 24.0

    @property
    def hours_to_resolution(self) -> float:
        if not self.end_date_iso:
            return 720.0
        try:
            end = datetime.fromisoformat(self.end_date_iso.replace("Z", "+00:00"))
            now = datetime.now(end.tzinfo)
            return max(0.0, (end - now).total_seconds() / 3600)
        except Exception:
            return 720.0

    @property
    def market_age_hours(self) -> float:
        if not self.created_at:
            return 168.0
        try:
            created = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
            now = datetime.now(created.tzinfo)
            return max(0.0, (now - created).total_seconds() / 3600)
        except Exception:
            return 168.0

    @property
    def market_age_days(self) -> float:
        return self.market_age_hours / 24.0

    @property
    def is_approaching_deadline(self) -> bool:
        """True if resolving within 24 hours — time-decay volatility zone."""
        return 0 < self.hours_to_resolution <= 24

    @property
    def age_ratio(self) -> float:
        """market_age / (market_age + time_to_resolution) — normalized age."""
        total = self.market_age_hours + self.hours_to_resolution
        return self.market_age_hours / total if total > 0 else 0.5

    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "question": self.question,
            "category": self.category,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "volume": self.volume,
            "volume_24h": self.volume_24h,
            "volume_1h": self.volume_1h,
            "volume_6h": self.volume_6h,
            "volume_7d": self.volume_7d,
            "volume_acceleration": self.volume_acceleration,
            "price_velocity": self.price_velocity,
            "liquidity": self.liquidity,
            "bid_ask_spread": self.bid_ask_spread,
            "hours_to_resolution": self.hours_to_resolution,
            "market_age_hours": self.market_age_hours,
            "age_ratio": self.age_ratio,
            "end_date_iso": self.end_date_iso,
            "is_approaching_deadline": self.is_approaching_deadline,
            "whale_activity": self.whale_activity,
        }


def _with_backoff(fn, retries: int = 3, base_delay: float = 1.0):
    """Call fn() with exponential backoff on failure."""
    for attempt in range(retries):
        try:
            return fn()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(f"Request failed (attempt {attempt+1}/{retries}): {e}. Retrying in {delay:.1f}s")
            time.sleep(delay)
    return None


class MarketScanner:
    """Polls Polymarket for open markets. Persists snapshots for model training."""

    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    def __init__(
        self,
        api_key: str = "",
        poll_interval: int = 7,
        snapshot_store: Optional["SnapshotStore"] = None,
    ):
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.snapshot_store = snapshot_store
        self._markets: dict[str, Market] = {}
        self._callbacks: list[Callable[[list[Market]], None]] = []
        self._running = False
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "polymarket-bot/2.0",
            "Accept": "application/json",
        })

    def on_update(self, callback: Callable[[list[Market]], None]):
        self._callbacks.append(callback)

    def fetch_markets(self, limit: int = 100) -> list[Market]:
        """Fetch active markets from Gamma API (public, no auth required)."""
        markets = []
        try:
            def _do_fetch():
                params = {
                    "active": "true",
                    "closed": "false",
                    "limit": limit,
                    "order": "volume24hr",
                    "ascending": "false",
                }
                resp = self.session.get(
                    f"{self.GAMMA_API}/markets", params=params, timeout=15
                )
                resp.raise_for_status()
                return resp.json()

            data = _with_backoff(_do_fetch)
            raw_markets = data if isinstance(data, list) else data.get("markets", [])

            # Load volume history for rolling windows
            vol_history = {}
            if self.snapshot_store:
                vol_history = self.snapshot_store.get_volume_history()

            for item in raw_markets:
                try:
                    m = Market(item, volume_history=vol_history)
                    if m.active and not m.closed and m.liquidity > 50:
                        markets.append(m)
                except Exception as e:
                    logger.debug(f"Skipping malformed market: {e}")
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch markets after retries: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in fetch_markets: {e}")
        return markets

    def scan_once(self) -> list[Market]:
        markets = self.fetch_markets()
        self._markets = {m.market_id: m for m in markets}

        # Persist snapshots
        if self.snapshot_store and markets:
            self.snapshot_store.save_snapshots(markets)

        logger.info(
            f"Scanned {len(markets)} active markets "
            f"(approaching deadline: {sum(1 for m in markets if m.is_approaching_deadline)})"
        )
        for cb in self._callbacks:
            try:
                cb(markets)
            except Exception as e:
                logger.error(f"Scanner callback error: {e}")
        return markets

    def start(self, run_once: bool = False):
        self._running = True
        logger.info("Market scanner started.")
        while self._running:
            try:
                self.scan_once()
            except Exception as e:
                logger.error(f"Scanner loop error: {e}")
            if run_once:
                break
            jitter = random.uniform(-1.5, 1.5)
            time.sleep(self.poll_interval + jitter)

    def stop(self):
        self._running = False

    def get_cached_markets(self) -> list[Market]:
        return list(self._markets.values())


class SnapshotStore:
    """
    Persists market snapshots to SQLite so the XGBoost model has
    historical training data that grows over time as the bot runs.
    """

    def __init__(self, db_path: str = "data/snapshots.db"):
        import sqlite3
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                market_id TEXT NOT NULL,
                question TEXT,
                category TEXT,
                yes_price REAL,
                no_price REAL,
                volume_24h REAL,
                liquidity REAL,
                hours_to_resolution REAL,
                market_age_hours REAL,
                bid_ask_spread REAL,
                whale_activity INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_snap_market_ts ON market_snapshots(market_id, ts);
            CREATE INDEX IF NOT EXISTS idx_snap_ts ON market_snapshots(ts);

            CREATE TABLE IF NOT EXISTS resolved_markets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resolved_at REAL NOT NULL,
                market_id TEXT NOT NULL UNIQUE,
                question TEXT,
                category TEXT,
                resolved_yes INTEGER,
                volume REAL,
                final_price REAL
            );
            CREATE INDEX IF NOT EXISTS idx_resolved_market ON resolved_markets(market_id);
        """)
        self._conn.commit()

    def save_snapshots(self, markets: list[Market]):
        ts = time.time()
        rows = [
            (
                ts, m.market_id, m.question, m.category,
                m.yes_price, m.no_price, m.volume_24h, m.liquidity,
                m.hours_to_resolution, m.market_age_hours,
                m.bid_ask_spread, int(m.whale_activity),
            )
            for m in markets
        ]
        try:
            self._conn.executemany(
                """INSERT INTO market_snapshots
                   (ts, market_id, question, category, yes_price, no_price,
                    volume_24h, liquidity, hours_to_resolution, market_age_hours,
                    bid_ask_spread, whale_activity)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"Snapshot save failed: {e}")

    def record_resolution(self, market_id: str, question: str, category: str,
                          resolved_yes: bool, volume: float, final_price: float):
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO resolved_markets
                   (resolved_at, market_id, question, category, resolved_yes, volume, final_price)
                   VALUES (?,?,?,?,?,?,?)""",
                (time.time(), market_id, question, category,
                 int(resolved_yes), volume, final_price),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"Resolution record failed: {e}")

    def get_volume_history(self) -> dict[str, list[dict]]:
        """Return recent snapshots grouped by market_id for rolling window calc."""
        cutoff = time.time() - 7 * 86400  # 7 days
        try:
            cur = self._conn.execute(
                "SELECT market_id, ts, volume_24h, yes_price FROM market_snapshots "
                "WHERE ts > ? ORDER BY ts ASC",
                (cutoff,),
            )
            result: dict[str, list] = {}
            for row in cur.fetchall():
                mid, ts, v24, price = row
                result.setdefault(mid, []).append({"ts": ts, "volume_24h": v24, "yes_price": price})
            return result
        except Exception:
            return {}

    def get_resolved_for_training(self) -> list[dict]:
        """Return resolved markets for model retraining."""
        try:
            cur = self._conn.execute(
                "SELECT market_id, question, category, resolved_yes, volume, final_price, resolved_at "
                "FROM resolved_markets ORDER BY resolved_at DESC"
            )
            cols = ["market_id", "question", "category", "resolved_yes",
                    "volume", "final_price", "resolved_at"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception:
            return []

    def get_snapshot_count(self) -> int:
        try:
            return self._conn.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0]
        except Exception:
            return 0
