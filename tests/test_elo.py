"""Tests for the ELO rating system — the primary edge signal."""
import math
import pytest
from model.elo import (
    ELOSystem, DEFAULT_ELO, _elo_to_probability,
    _expected, _new_rating, _time_bucket, _infer_event_type,
)


def test_default_ratings_initialized(elo):
    assert elo.get_category_elo("politics") == DEFAULT_ELO
    assert elo.get_category_elo("crypto") == DEFAULT_ELO
    assert elo.get_event_elo("some election") > 0


def test_elo_to_probability_baseline():
    p = _elo_to_probability(DEFAULT_ELO, DEFAULT_ELO)
    assert abs(p - 0.5) < 1e-6


def test_elo_to_probability_higher_rating():
    p = _elo_to_probability(1700, DEFAULT_ELO)
    assert p > 0.5


def test_elo_to_probability_lower_rating():
    p = _elo_to_probability(1300, DEFAULT_ELO)
    assert p < 0.5


def test_resolution_yes_increases_rating(elo):
    elo.record_resolution("politics", "Will X happen?", resolved_yes=True, volume=10000)
    assert elo.get_category_elo("politics") > DEFAULT_ELO


def test_resolution_no_decreases_rating(elo):
    elo.record_resolution("crypto", "Will BTC hit 100k?", resolved_yes=False, volume=10000)
    assert elo.get_category_elo("crypto") < DEFAULT_ELO


def test_volume_weighting(elo):
    """High-volume resolution should produce larger ELO swing."""
    elo2 = ELOSystem.__new__(ELOSystem)
    from model.elo import ELOState, DEFAULT_ELO, CATEGORIES, EVENT_TYPES, TIME_BUCKETS
    elo2.state = ELOState()
    elo2.save_path = elo.save_path
    elo2._init_defaults()

    elo.record_resolution("sports", "q", resolved_yes=True, volume=1000)
    elo2.record_resolution("sports", "q", resolved_yes=True, volume=1_000_000)

    delta_low = elo.get_category_elo("sports") - DEFAULT_ELO
    delta_high = elo2.get_category_elo("sports") - DEFAULT_ELO
    assert delta_high > delta_low


def test_elo_implied_probability_is_blended(elo):
    p = elo.elo_implied_probability("politics", "Will the president win?")
    assert 0.01 <= p <= 0.99


def test_elo_edge_zero_at_baseline(elo):
    # When ELO is at default, implied prob ≈ 0.5
    # Edge vs 0.5 should be ~0
    edge = elo.elo_edge(0.5, "general")
    assert abs(edge) < 0.05


def test_elo_edge_positive_when_underpriced(elo):
    # Boost category ELO to make YES more likely
    for _ in range(20):
        elo.record_resolution("politics", "q", resolved_yes=True, volume=50000)
    edge = elo.elo_edge(0.3, "politics")
    assert edge > 0  # ELO says YES > 30%


def test_infer_event_type_election():
    assert _infer_event_type("Will Biden win the 2024 presidential election?") == "election"


def test_infer_event_type_fed():
    assert _infer_event_type("Will the Fed raise interest rates?") == "fed_decision"


def test_infer_event_type_sports():
    assert _infer_event_type("Will the Lakers win the championship?") == "sports_match"


def test_infer_event_type_fallback():
    assert _infer_event_type("Will aliens visit Earth?") == "other"


def test_time_bucket_morning():
    from datetime import datetime, timezone
    dt = datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc)
    assert _time_bucket(dt) == "morning"


def test_time_bucket_overnight():
    from datetime import datetime, timezone
    dt = datetime(2025, 1, 1, 2, 0, tzinfo=timezone.utc)
    assert _time_bucket(dt) == "overnight"


def test_save_and_load(elo, tmp_path):
    elo.record_resolution("politics", "q", resolved_yes=True, volume=5000)
    old_rating = elo.get_category_elo("politics")

    elo2 = ELOSystem(save_path=elo.save_path)
    assert abs(elo2.get_category_elo("politics") - old_rating) < 0.1


def test_history_bounded(elo):
    for i in range(15000):
        elo.state.history.append({"ts": "x", "dim": "category", "key": "politics",
                                   "delta": 1, "new_rating": 1500, "volume": 100,
                                   "resolved_yes": True})
    elo.save()
    # After reload, history should be bounded
    elo2 = ELOSystem(save_path=elo.save_path)
    assert len(elo2.state.history) <= 500


def test_get_top_categories(elo):
    elo.record_resolution("sports", "q", resolved_yes=True, volume=100000)
    top = elo.get_top_categories(3)
    assert len(top) == 3
    assert top[0][0] == "sports"
    cats = [t[0] for t in top]
    ratings = [t[1] for t in top]
    assert ratings == sorted(ratings, reverse=True)
