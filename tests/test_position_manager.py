"""Tests for position manager: trailing stop, profit ladder, daily drawdown."""
import pytest
from unittest.mock import patch
from position_manager.position_manager import PositionManager


def _trade(db, entry_price=0.40, **kw):
    defaults = dict(
        market_id="mkt_pm", market_question="PM test?",
        side="YES", size=50.0, is_paper=True,
    )
    defaults.update(kw)
    return db.log_trade(entry_price=entry_price, **defaults)


def test_profit_target_exit(tmp_db):
    trade = _trade(tmp_db, entry_price=0.40)
    pm = PositionManager(tmp_db, profit_target=0.15, stop_loss=0.10, paper_trading=True)
    with patch.object(pm, "_fetch_current_price", return_value=0.50):
        reason = pm.check_position(trade)
    assert reason == "profit_target"


def test_stop_loss_exit(tmp_db):
    trade = _trade(tmp_db, entry_price=0.60)
    pm = PositionManager(tmp_db, profit_target=0.15, stop_loss=0.10, paper_trading=True)
    with patch.object(pm, "_fetch_current_price", return_value=0.50):
        reason = pm.check_position(trade)
    assert reason == "stop_loss"


def test_trailing_stop(tmp_db):
    trade = _trade(tmp_db, entry_price=0.40)
    pm = PositionManager(tmp_db, profit_target=0.30, stop_loss=0.10, paper_trading=True)
    # Price moves up to 0.60 (peak), then falls back to 0.52 (>10% from peak)
    with patch.object(pm, "_fetch_current_price", return_value=0.60):
        pm.check_position(trade)  # establishes peak
    pm._peak_price[trade.id] = 0.60  # ensure peak recorded
    with patch.object(pm, "_fetch_current_price", return_value=0.52):
        reason = pm.check_position(trade)
    # 0.52/0.60 - 1 = -13.3% from peak → trailing stop
    assert reason == "stop_loss"


def test_hold_position(tmp_db):
    trade = _trade(tmp_db, entry_price=0.50)
    pm = PositionManager(tmp_db, profit_target=0.15, stop_loss=0.10, paper_trading=True)
    with patch.object(pm, "_fetch_current_price", return_value=0.53):
        reason = pm.check_position(trade)
    assert reason is None


def test_resolved_exit(tmp_db):
    trade = _trade(tmp_db, entry_price=0.50)
    pm = PositionManager(tmp_db, profit_target=0.15, stop_loss=0.10, paper_trading=True)
    with patch.object(pm, "_fetch_current_price", return_value=0.99):
        reason = pm.check_position(trade)
    assert reason == "resolved"


def test_resolved_low_price(tmp_db):
    trade = _trade(tmp_db, entry_price=0.50)
    pm = PositionManager(tmp_db, profit_target=0.15, stop_loss=0.10, paper_trading=True)
    with patch.object(pm, "_fetch_current_price", return_value=0.01):
        reason = pm.check_position(trade)
    assert reason == "resolved"


def test_close_position(tmp_db):
    trade = _trade(tmp_db, entry_price=0.30)
    pm = PositionManager(tmp_db, profit_target=0.15, stop_loss=0.10, paper_trading=True)
    pm._price_cache["mkt_pm"] = (0.50, 0)
    closed = pm.close_position(trade, reason="profit_target")
    assert closed.status == "closed"
    assert closed.pnl > 0
    assert closed.exit_reason == "profit_target"


def test_daily_drawdown_halts_trading(tmp_db):
    # Start with $1000 bankroll, max drawdown 5% = $50
    pm = PositionManager(tmp_db, profit_target=0.15, stop_loss=0.10,
                          max_daily_drawdown_pct=0.05, paper_trading=True, bankroll=1000)
    # Create losing trades summing to > $50 loss today
    for i in range(3):
        t = tmp_db.log_trade(
            market_id=f"m{i}", market_question="Q?",
            side="YES", entry_price=0.80, size=100,
            is_paper=True,
        )
        tmp_db.close_trade(t.id, 0.60)  # ~$20 loss each
    pm._check_daily_drawdown()
    assert tmp_db.is_trading_halted_today(paper=True)


def test_run_cycle_closes_profit_target(tmp_db):
    trade = _trade(tmp_db, entry_price=0.40, market_id="run_cycle_mkt")
    pm = PositionManager(tmp_db, profit_target=0.15, stop_loss=0.10, paper_trading=True)
    with patch.object(pm, "_fetch_current_price", return_value=0.55):
        closed = pm.run_cycle()
    assert len(closed) == 1
    assert closed[0].status == "closed"
