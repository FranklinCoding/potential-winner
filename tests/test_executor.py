"""Tests for the trade executor (paper mode, Kelly sizing, slippage)."""
import pytest
from unittest.mock import patch
from executor.trade_executor import TradeExecutor, OrderResult, _simulate_slippage
from model.edge_detector import EdgeSignal
from scanner.market_scanner import Market


@pytest.fixture
def mock_signal():
    data = {
        "condition_id": "mkt_exec_001",
        "question": "Will test pass?",
        "category": "technology",
        "end_date_iso": "2025-12-31T00:00:00Z",
        "created_at": "2024-01-01T00:00:00Z",
        "active": True,
        "closed": False,
        "volume": 100000,
        "volume_24hr": 10000,
        "liquidity": 30000,
        "tokens": [{"outcome": "YES", "price": 0.35}, {"outcome": "NO", "price": 0.65}],
    }
    market = Market(data)
    return EdgeSignal(
        market=market,
        fair_value=0.55,
        elo_implied_prob=0.58,
        current_price=0.35,
        raw_edge=0.23,
        elo_edge=0.23,
        final_edge=0.20,
        trade_side="YES",
        trade_price=0.35,
        confidence=0.75,
        position_size_factor=0.65,
        sentiment_agrees=True,
        sentiment_score=0.3,
        sentiment_momentum=0.1,
        kelly_fraction=0.06,
    )


def test_paper_trade_executes(tmp_db, mock_signal):
    executor = TradeExecutor(trade_logger=tmp_db, paper_trading=True, bankroll=1000)
    trade = executor.execute(mock_signal)
    assert trade is not None
    assert trade.is_paper is True
    assert trade.side == "YES"
    assert trade.entry_price > 0
    assert trade.status == "open"
    assert trade.elo_diff == mock_signal.elo_edge


def test_paper_trade_records_slippage(tmp_db, mock_signal):
    executor = TradeExecutor(trade_logger=tmp_db, paper_trading=True, bankroll=1000)
    trade = executor.execute(mock_signal)
    # Fill price may differ from signal.trade_price due to slippage
    assert trade is not None
    # Slippage should be non-negative
    assert (trade.slippage or 0) >= 0


def test_kelly_position_sizing(tmp_db, mock_signal):
    executor = TradeExecutor(
        trade_logger=tmp_db, paper_trading=True,
        bankroll=1000, max_position_size=200, max_position_pct=0.05,
    )
    size = executor.kelly_position_size(mock_signal)
    # kelly=6% of $1000 = $60, cap at max_pos_pct=5%*1000=$50
    assert size <= 50.0
    assert size >= 1.0


def test_position_size_respects_max_usdc(tmp_db, mock_signal):
    executor = TradeExecutor(
        trade_logger=tmp_db, paper_trading=True,
        bankroll=10000, max_position_size=30, max_position_pct=0.20,
    )
    size = executor.calculate_position_size(mock_signal)
    assert size <= 30.0


def test_no_private_key_forces_paper(tmp_db, mock_signal):
    # No private key → must force paper regardless of flag
    executor = TradeExecutor(
        trade_logger=tmp_db,
        paper_trading=False,  # tries to go live
        wallet_private_key="",  # but no key → should switch to paper
        bankroll=1000,
    )
    assert executor.paper_trading is True


def test_slippage_simulation_bounded():
    fill, slip = _simulate_slippage(0.50, 100, 10000)
    assert 0.01 <= fill <= 0.99
    assert slip >= 0


def test_slippage_larger_for_bigger_orders():
    _, slip_small = _simulate_slippage(0.50, 10, 10000)
    _, slip_large = _simulate_slippage(0.50, 5000, 10000)
    assert slip_large >= slip_small


def test_order_result_fields():
    r = OrderResult(success=True, order_id="abc", fill_price=0.35)
    assert r.success
    assert r.order_id == "abc"
    r_fail = OrderResult(success=False, error="network error")
    assert not r_fail.success
    assert r_fail.error == "network error"
