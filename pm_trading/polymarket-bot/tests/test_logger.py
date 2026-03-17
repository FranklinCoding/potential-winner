"""Tests for the trade logger (SQLAlchemy + SQLite)."""
import math
import pytest
from logger.trade_logger import TradeLogger, Trade


def _open(db, **kw):
    defaults = dict(
        market_id="mkt1", market_question="Q?", side="YES",
        entry_price=0.40, size=50.0, is_paper=True,
    )
    defaults.update(kw)
    return db.log_trade(**defaults)


def test_log_and_retrieve(tmp_db):
    t = _open(tmp_db)
    assert t.id is not None
    assert t.status == "open"
    assert t.entry_price == 0.40


def test_close_trade_pnl(tmp_db):
    t = _open(tmp_db, entry_price=0.40, size=100.0)
    closed = tmp_db.close_trade(t.id, exit_price=0.55)
    assert closed.status == "closed"
    # cost=40, proceeds=55, pnl=15
    assert abs(closed.pnl - 15.0) < 0.01
    assert closed.pnl_pct > 0


def test_close_trade_exit_reason(tmp_db):
    t = _open(tmp_db)
    closed = tmp_db.close_trade(t.id, 0.6, exit_reason="profit_target")
    assert closed.exit_reason == "profit_target"


def test_hold_time_recorded(tmp_db):
    t = _open(tmp_db)
    closed = tmp_db.close_trade(t.id, 0.5)
    assert closed.hold_time_minutes is not None
    assert closed.hold_time_minutes >= 0


def test_get_open_trades_paper_filter(tmp_db):
    _open(tmp_db, market_id="m1", is_paper=True)
    _open(tmp_db, market_id="m2", is_paper=False)
    paper_open = tmp_db.get_open_trades(paper=True)
    live_open = tmp_db.get_open_trades(paper=False)
    assert len(paper_open) == 1
    assert len(live_open) == 1


def test_stats_empty(tmp_db):
    s = tmp_db.get_stats()
    assert s["total_trades"] == 0
    assert s["win_rate"] == 0.0


def test_stats_after_trades(tmp_db):
    t1 = _open(tmp_db, market_id="m1", entry_price=0.3, size=100)
    t2 = _open(tmp_db, market_id="m2", entry_price=0.6, size=100)
    tmp_db.close_trade(t1.id, 0.5)   # win
    tmp_db.close_trade(t2.id, 0.4)   # loss
    s = tmp_db.get_stats()
    assert s["total_trades"] == 2
    assert s["wins"] == 1
    assert s["losses"] == 1
    assert abs(s["win_rate"] - 50.0) < 0.01


def test_sharpe_ratio_computed(tmp_db):
    trades = [0.3, 0.5, -0.1, 0.4, -0.2, 0.6, 0.1, 0.3, -0.15, 0.2]
    for i, entry in enumerate(trades):
        t = _open(tmp_db, market_id=f"m{i}", entry_price=0.4, size=100)
        exit_p = 0.4 + entry / 100  # encode pnl signal
        tmp_db.close_trade(t.id, max(0.01, min(0.99, exit_p)))
    s = tmp_db.get_stats()
    assert "sharpe" in s
    assert isinstance(s["sharpe"], float)


def test_is_winner(tmp_db):
    t = _open(tmp_db, entry_price=0.3, size=10)
    closed = tmp_db.close_trade(t.id, 0.8)
    assert closed.is_winner is True


def test_daily_drawdown_halt(tmp_db):
    assert not tmp_db.is_trading_halted_today(paper=True)
    tmp_db.halt_trading_today(paper=True)
    assert tmp_db.is_trading_halted_today(paper=True)
    # Paper and live are separate
    assert not tmp_db.is_trading_halted_today(paper=False)


def test_log_retrain(tmp_db):
    import json
    tmp_db.log_retrain(mae=0.05, n_samples=500, feature_importance=json.dumps({"a": 0.5}))
    row = tmp_db.get_last_retrain()
    assert row is not None
    assert abs(row.mae - 0.05) < 1e-6
    assert row.n_samples == 500


def test_daily_paper_report_empty(tmp_db):
    report = tmp_db.daily_paper_report()
    assert "date" in report


def test_paper_live_stats_separate(tmp_db):
    _open(tmp_db, market_id="mp", is_paper=True)
    _open(tmp_db, market_id="ml", is_paper=False)
    t_p = tmp_db.get_open_trades(paper=True)
    t_l = tmp_db.get_open_trades(paper=False)
    assert all(t.is_paper for t in t_p)
    assert all(not t.is_paper for t in t_l)
