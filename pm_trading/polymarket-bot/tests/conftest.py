"""Shared pytest fixtures."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from logger.trade_logger import TradeLogger
from model.elo import ELOSystem


@pytest.fixture
def tmp_db(tmp_path):
    return TradeLogger(db_path=str(tmp_path / "test_trades.db"))


@pytest.fixture
def elo(tmp_path):
    return ELOSystem(save_path=tmp_path / "elo.json")


@pytest.fixture
def sample_market_data():
    return {
        "condition_id": "test_market_001",
        "question": "Will the Fed raise rates in December?",
        "category": "economics",
        "end_date_iso": "2025-12-31T00:00:00Z",
        "created_at": "2024-01-01T00:00:00Z",
        "active": True,
        "closed": False,
        "volume": 50000,
        "volume_24hr": 5000,
        "liquidity": 10000,
        "tokens": [
            {"outcome": "YES", "price": 0.35},
            {"outcome": "NO", "price": 0.65},
        ],
    }


@pytest.fixture
def trained_model(tmp_path, elo):
    import model.fair_value_model as fvm
    from pathlib import Path
    import monkeypatch as mp  # not used directly — use monkeypatch fixture below
    return elo  # placeholder — actual model fixture built per test


@pytest.fixture
def fresh_model(tmp_path, monkeypatch, elo):
    import model.fair_value_model as fvm
    monkeypatch.setattr(fvm, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(fvm, "META_PATH", tmp_path / "meta.json")
    from model.fair_value_model import FairValueModel
    m = FairValueModel(elo_system=elo)
    m.train()
    return m
