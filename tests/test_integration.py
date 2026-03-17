"""
Integration test: full paper trade cycle end-to-end.
Verifies: scan → ELO → model → edge → execute → manage position → close.
"""
import time
import pytest
from unittest.mock import patch, MagicMock


def test_full_paper_trade_cycle(tmp_path, monkeypatch):
    """
    Full pipeline:
    1. Create a market with known mispricing
    2. Build ELO features
    3. Run edge detection
    4. Execute paper trade
    5. Simulate price reaching profit target
    6. Run position manager cycle → trade should close
    """
    import model.fair_value_model as fvm
    monkeypatch.setattr(fvm, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(fvm, "META_PATH", tmp_path / "meta.json")

    from logger.trade_logger import TradeLogger
    from model.elo import ELOSystem
    from model.fair_value_model import FairValueModel
    from model.edge_detector import EdgeDetector
    from model.sentiment import SentimentAnalyzer
    from executor.trade_executor import TradeExecutor
    from position_manager.position_manager import PositionManager
    from scanner.market_scanner import Market

    # Init all components
    db = TradeLogger(db_path=str(tmp_path / "trades.db"))
    elo = ELOSystem(save_path=tmp_path / "elo.json")
    model = FairValueModel(elo_system=elo)
    model.train()

    sentiment = SentimentAnalyzer(api_key="")

    # Aggressively boost politics ELO to create a large, detectable edge
    for _ in range(50):
        elo.record_resolution("politics", "Will X win?", resolved_yes=True, volume=500000)

    detector = EdgeDetector(model=model, elo_system=elo, min_edge=0.01, min_liquidity=100)
    executor = TradeExecutor(db, paper_trading=True, bankroll=1000, max_position_size=100)
    pos_mgr = PositionManager(db, profit_target=0.10, stop_loss=0.50, paper_trading=True)

    # Create a market priced well below ELO-implied probability
    market_data = {
        "condition_id": "integration_mkt_001",
        "question": "Will the election result be announced?",
        "category": "politics",
        "end_date_iso": "2025-12-31T00:00:00Z",
        "created_at": "2024-01-01T00:00:00Z",
        "active": True,
        "closed": False,
        "volume": 500000,
        "volume_24hr": 50000,
        "liquidity": 100000,
        "tokens": [
            {"outcome": "YES", "price": 0.20},  # very cheap vs. ELO-boosted category
            {"outcome": "NO", "price": 0.80},
        ],
    }
    market = Market(market_data)

    # Step 1: sentiment analysis
    sent_score = sentiment.analyze_market(market.market_id, market.question)
    sent_mom = sentiment.get_momentum(market.market_id)

    # Step 2: analyze for edge
    signal = detector.analyze(market, sent_score, sent_mom)
    assert signal is not None, "Expected edge signal with boosted ELO"
    assert signal.elo_edge > 0
    assert signal.trade_side == "YES"

    # Step 3: execute paper trade
    trade = executor.execute(signal)
    assert trade is not None
    assert trade.is_paper is True
    assert trade.status == "open"
    assert trade.elo_diff == signal.elo_edge

    # Verify trade is in open trades
    open_trades = db.get_open_trades(paper=True)
    assert any(t.id == trade.id for t in open_trades)

    # Step 4: simulate price reaching profit target
    profit_price = trade.entry_price * (1 + 0.15)
    with patch.object(pos_mgr, "_fetch_current_price", return_value=profit_price):
        closed_trades = pos_mgr.run_cycle()

    assert len(closed_trades) == 1
    closed = closed_trades[0]
    assert closed.status == "closed"
    assert closed.pnl > 0
    assert closed.exit_reason == "profit_target"

    # Step 5: verify stats
    stats = db.get_stats(paper=True)
    assert stats["total_trades"] == 1
    assert stats["wins"] == 1
    assert stats["win_rate"] == 100.0
    assert stats["open_positions"] == 0


def test_elo_updates_after_trade_close(tmp_path, monkeypatch):
    """Verify ELO rating changes after a resolved position."""
    import model.fair_value_model as fvm
    monkeypatch.setattr(fvm, "MODEL_PATH", tmp_path / "model2.pkl")
    monkeypatch.setattr(fvm, "META_PATH", tmp_path / "meta2.json")

    from logger.trade_logger import TradeLogger
    from model.elo import ELOSystem, DEFAULT_ELO
    from model.fair_value_model import FairValueModel
    from executor.trade_executor import TradeExecutor
    from position_manager.position_manager import PositionManager
    from scanner.market_scanner import Market
    from model.edge_detector import EdgeDetector

    db = TradeLogger(db_path=str(tmp_path / "trades2.db"))
    elo = ELOSystem(save_path=tmp_path / "elo2.json")
    model = FairValueModel(elo_system=elo)
    model.train()
    detector = EdgeDetector(model=model, elo_system=elo, min_edge=0.01, min_liquidity=100)
    executor = TradeExecutor(db, paper_trading=True, bankroll=500)
    pos_mgr = PositionManager(db, profit_target=0.10, stop_loss=0.50,
                               paper_trading=True, elo_system=elo)

    initial_elo = elo.get_category_elo("sports")

    # Log a trade manually
    trade = db.log_trade(
        market_id="elo_update_mkt",
        market_question="Will the team win?",
        category="sports",
        side="YES",
        entry_price=0.40,
        size=50,
        is_paper=True,
    )
    # Close as winner
    from unittest.mock import patch
    with patch.object(pos_mgr, "_fetch_current_price", return_value=0.70):
        closed_trades = pos_mgr.run_cycle()

    # ELO should have changed
    new_elo = elo.get_category_elo("sports")
    assert new_elo != initial_elo
