"""Tests for market scanner and snapshot store."""
import time
import pytest
from scanner.market_scanner import Market, MarketScanner, SnapshotStore


def test_market_from_data(sample_market_data):
    m = Market(sample_market_data)
    assert m.market_id == "test_market_001"
    assert m.question == "Will the Fed raise rates in December?"
    assert m.yes_price == 0.35
    assert m.no_price == 0.65
    assert m.volume == 50000
    assert m.liquidity == 10000


def test_market_hours_to_resolution(sample_market_data):
    m = Market(sample_market_data)
    assert isinstance(m.hours_to_resolution, float)
    assert m.hours_to_resolution >= 0
    assert m.days_to_resolution == m.hours_to_resolution / 24.0


def test_market_age_hours(sample_market_data):
    m = Market(sample_market_data)
    assert m.market_age_hours > 0  # created in 2024


def test_market_age_ratio(sample_market_data):
    m = Market(sample_market_data)
    assert 0 <= m.age_ratio <= 1


def test_market_to_dict(sample_market_data):
    m = Market(sample_market_data)
    d = m.to_dict()
    required = ["market_id", "question", "yes_price", "no_price",
                "volume", "hours_to_resolution", "age_ratio", "whale_activity"]
    for k in required:
        assert k in d


def test_market_bid_ask_spread(sample_market_data):
    m = Market(sample_market_data)
    # YES=0.35, NO=0.65 → total=1.0 → spread≈0
    assert m.bid_ask_spread < 0.05


def test_market_approaching_deadline():
    data = {
        "condition_id": "x",
        "question": "Q?",
        "active": True,
        "closed": False,
        "volume": 100,
        "volume_24hr": 10,
        "liquidity": 200,
        "tokens": [{"outcome": "YES", "price": 0.5}, {"outcome": "NO", "price": 0.5}],
        # end_date = 1 hour from now
        "end_date_iso": "",  # blank = 720h, so NOT approaching
    }
    m = Market(data)
    assert not m.is_approaching_deadline


def test_market_with_missing_tokens():
    data = {
        "condition_id": "abc",
        "question": "Q?",
        "active": True,
        "closed": False,
        "volume": 0,
        "volume_24hr": 0,
        "liquidity": 500,
        "tokens": [],
    }
    m = Market(data)
    assert m.yes_price == 0.5
    assert m.no_price == 0.5


def test_scanner_instantiation():
    s = MarketScanner(poll_interval=5)
    assert s.poll_interval == 5
    assert s.get_cached_markets() == []


def test_snapshot_store_basic(tmp_path):
    store = SnapshotStore(db_path=str(tmp_path / "snaps.db"))
    data = {
        "condition_id": "mkt_snap_001",
        "question": "Test?",
        "active": True,
        "closed": False,
        "volume": 1000,
        "volume_24hr": 100,
        "liquidity": 500,
        "tokens": [{"outcome": "YES", "price": 0.4}, {"outcome": "NO", "price": 0.6}],
    }
    m = Market(data)
    store.save_snapshots([m])
    count = store.get_snapshot_count()
    assert count == 1


def test_snapshot_store_volume_history(tmp_path):
    store = SnapshotStore(db_path=str(tmp_path / "snaps2.db"))
    data = {
        "condition_id": "mkt_snap_002",
        "question": "History test?",
        "active": True,
        "closed": False,
        "volume": 5000,
        "volume_24hr": 500,
        "liquidity": 1000,
        "tokens": [{"outcome": "YES", "price": 0.5}, {"outcome": "NO", "price": 0.5}],
    }
    m = Market(data)
    store.save_snapshots([m])
    hist = store.get_volume_history()
    assert "mkt_snap_002" in hist
    assert len(hist["mkt_snap_002"]) == 1


def test_snapshot_store_record_resolution(tmp_path):
    store = SnapshotStore(db_path=str(tmp_path / "snaps3.db"))
    store.record_resolution(
        market_id="mkt_res_001",
        question="Did it resolve?",
        category="politics",
        resolved_yes=True,
        volume=50000,
        final_price=0.98,
    )
    resolved = store.get_resolved_for_training()
    assert len(resolved) == 1
    assert resolved[0]["resolved_yes"] == 1
    assert resolved[0]["category"] == "politics"
