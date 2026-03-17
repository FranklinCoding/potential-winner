"""Tests for fair value model, feature engineering, edge detector."""
import pytest
import numpy as np
from model.fair_value_model import (
    FairValueModel, _generate_synthetic_data, FEATURE_COLS,
)
from model.edge_detector import EdgeDetector, kelly_fraction
from model.sentiment import SentimentAnalyzer, _lexicon_sentiment
from scanner.market_scanner import Market


@pytest.fixture
def sample_market():
    data = {
        "condition_id": "mkt_model_001",
        "question": "Will Bitcoin hit $100k in 2025?",
        "category": "crypto",
        "end_date_iso": "2025-12-31T00:00:00Z",
        "created_at": "2024-01-01T00:00:00Z",
        "active": True,
        "closed": False,
        "volume": 200000,
        "volume_24hr": 20000,
        "liquidity": 50000,
        "tokens": [
            {"outcome": "YES", "price": 0.32},
            {"outcome": "NO", "price": 0.68},
        ],
    }
    return Market(data)


class TestFeatureEngineering:
    def test_synthetic_data_has_all_features(self):
        df = _generate_synthetic_data(n_samples=100)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"

    def test_synthetic_data_resolution_bounded(self):
        df = _generate_synthetic_data(n_samples=500)
        assert (df["resolution_probability"] >= 0.01).all()
        assert (df["resolution_probability"] <= 0.99).all()

    def test_elo_edge_is_in_features(self):
        df = _generate_synthetic_data(n_samples=100)
        assert "elo_edge" in df.columns
        assert "elo_implied_probability" in df.columns

    def test_20_features(self):
        assert len(FEATURE_COLS) == 20


class TestFairValueModel:
    def test_train_and_predict(self, fresh_model, sample_market, elo):
        from model.edge_detector import EdgeDetector
        detector = EdgeDetector(model=fresh_model, elo_system=elo)
        features = detector.build_features(sample_market)
        pred = fresh_model.predict(features)
        assert 0.01 <= pred <= 0.99

    def test_predict_unknown_category(self, fresh_model, elo):
        from model.edge_detector import EdgeDetector
        data = {
            "condition_id": "x",
            "question": "Q?",
            "category": "unknown_xyz",
            "active": True,
            "closed": False,
            "volume": 1000,
            "volume_24hr": 100,
            "liquidity": 2000,
            "end_date_iso": "2025-12-31T00:00:00Z",
            "created_at": "2024-01-01T00:00:00Z",
            "tokens": [{"outcome": "YES", "price": 0.5}, {"outcome": "NO", "price": 0.5}],
        }
        market = Market(data)
        detector = EdgeDetector(model=fresh_model, elo_system=elo)
        features = detector.build_features(market)
        pred = fresh_model.predict(features)
        assert 0.01 <= pred <= 0.99

    def test_batch_predict(self, fresh_model, elo):
        import pandas as pd
        from model.fair_value_model import _generate_synthetic_data
        df = _generate_synthetic_data(n_samples=20)
        preds = fresh_model.batch_predict(df)
        assert len(preds) == 20
        assert (preds >= 0.01).all()
        assert (preds <= 0.99).all()

    def test_should_retrain_false_on_fresh(self, fresh_model):
        # Just trained — should not immediately retrain
        fresh_model.retrain_interval_hours = 24
        assert not fresh_model.should_retrain()

    def test_should_retrain_true_when_old(self, fresh_model):
        import time
        fresh_model._last_train_time = time.time() - 999999  # trained a long time ago
        fresh_model.retrain_interval_hours = 0.001
        assert fresh_model.should_retrain()


class TestEdgeDetector:
    def test_build_features_has_all_cols(self, fresh_model, elo, sample_market):
        detector = EdgeDetector(model=fresh_model, elo_system=elo)
        features = detector.build_features(sample_market)
        for col in FEATURE_COLS:
            assert col in features, f"Missing: {col}"

    def test_elo_edge_is_primary_signal(self, fresh_model, elo, sample_market):
        detector = EdgeDetector(model=fresh_model, elo_system=elo)
        features = detector.build_features(sample_market)
        assert "elo_edge" in features
        assert isinstance(features["elo_edge"], float)

    def test_scan_markets_returns_list(self, fresh_model, elo, sample_market):
        detector = EdgeDetector(model=fresh_model, elo_system=elo, min_edge=0.0)
        results = detector.scan_markets([sample_market])
        assert isinstance(results, list)

    def test_signals_ranked_by_confidence(self, fresh_model, elo):
        detector = EdgeDetector(model=fresh_model, elo_system=elo, min_edge=0.0)
        markets = []
        for i, price in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
            data = {
                "condition_id": f"mkt_{i}",
                "question": f"Will event {i} happen?",
                "category": "politics",
                "active": True, "closed": False,
                "volume": 100000, "volume_24hr": 10000,
                "liquidity": 20000,
                "end_date_iso": "2025-12-31T00:00:00Z",
                "created_at": "2024-01-01T00:00:00Z",
                "tokens": [
                    {"outcome": "YES", "price": price},
                    {"outcome": "NO", "price": 1 - price},
                ],
            }
            markets.append(Market(data))
        signals = detector.scan_markets(markets)
        if len(signals) >= 2:
            confs = [s.confidence for s in signals]
            assert confs == sorted(confs, reverse=True)

    def test_low_liquidity_filtered(self, fresh_model, elo):
        detector = EdgeDetector(model=fresh_model, elo_system=elo, min_edge=0.01, min_liquidity=1000)
        data = {
            "condition_id": "thin_mkt",
            "question": "Q?",
            "category": "general",
            "active": True, "closed": False,
            "volume": 100, "volume_24hr": 10,
            "liquidity": 50,  # below min
            "end_date_iso": "2025-12-31T00:00:00Z",
            "created_at": "2024-01-01T00:00:00Z",
            "tokens": [{"outcome": "YES", "price": 0.1}, {"outcome": "NO", "price": 0.9}],
        }
        signal = detector.analyze(Market(data))
        assert signal is None


class TestKellyCriterion:
    def test_zero_edge_returns_zero(self):
        assert kelly_fraction(0.0, 1.0) == 0.0

    def test_negative_edge_returns_zero(self):
        assert kelly_fraction(-0.1, 1.0) == 0.0

    def test_positive_edge_returns_positive(self):
        f = kelly_fraction(0.15, 2.0)
        assert f > 0

    def test_kelly_bounded_at_quarter(self):
        # Even extreme edge should not exceed 0.25 (quarter-Kelly cap)
        f = kelly_fraction(0.9, 10.0)
        assert f <= 0.25

    def test_larger_edge_larger_kelly(self):
        f_small = kelly_fraction(0.08, 1.0)
        f_large = kelly_fraction(0.25, 1.0)
        assert f_large >= f_small


class TestSentiment:
    def test_lexicon_positive(self):
        assert _lexicon_sentiment("win victory success strong") > 0

    def test_lexicon_negative(self):
        assert _lexicon_sentiment("crash fail collapse defeat") < 0

    def test_lexicon_neutral(self):
        score = _lexicon_sentiment("the market will resolve in December 2025")
        assert -1 <= score <= 1

    def test_analyze_no_api(self):
        sa = SentimentAnalyzer(api_key="")
        score = sa.analyze_market("mkt1", "Will the economy improve?")
        assert -1 <= score <= 1

    def test_cache_hit(self):
        sa = SentimentAnalyzer(api_key="")
        s1 = sa.analyze_market("mkt_cache", "test question")
        s2 = sa.analyze_market("mkt_cache", "test question")
        assert s1 == s2

    def test_sentiment_momentum_insufficient_history(self):
        sa = SentimentAnalyzer(api_key="")
        mom = sa.get_momentum("new_market")
        assert mom == 0.0

    def test_sentiment_momentum_builds_over_calls(self):
        sa = SentimentAnalyzer(api_key="")
        # Make multiple calls to build history
        for q in ["win strong rally", "crash fail decline", "win gain surge"]:
            sa._cache.clear()  # force re-analysis
            sa.analyze_market("mkt_mom", q)
        mom = sa.get_momentum("mkt_mom")
        assert -1 <= mom <= 1

    def test_batch_returns_tuple(self):
        sa = SentimentAnalyzer(api_key="")
        data = {
            "condition_id": "m1", "question": "Q?", "active": True, "closed": False,
            "volume": 1000, "volume_24hr": 100, "liquidity": 500,
            "tokens": [{"outcome": "YES", "price": 0.5}, {"outcome": "NO", "price": 0.5}],
        }
        markets = [Market(data)]
        result = sa.batch_analyze(markets)
        assert "m1" in result
        score, mom = result["m1"]
        assert -1 <= score <= 1
        assert -1 <= mom <= 1

    def test_clear_cache(self):
        sa = SentimentAnalyzer(api_key="")
        sa.analyze_market("mkt1", "Q")
        sa.clear_cache()
        assert len(sa._cache) == 0
