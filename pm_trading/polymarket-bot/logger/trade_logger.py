"""
SQLAlchemy trade logger.
Separate tables for paper trades and live trades — never mixed.
Includes: entry/exit price, hold time, edge at entry, sentiment, ELO diff,
exit reason, running Sharpe/Sortino, daily drawdown tracking.
"""
from __future__ import annotations

import math
import os
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Index, Integer, String,
    create_engine, func, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_id = Column(String, nullable=False)
    market_question = Column(String)
    category = Column(String)
    side = Column(String, nullable=False)           # YES | NO
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    size = Column(Float, nullable=False)            # USDC amount
    fair_value = Column(Float)
    edge_at_entry = Column(Float)
    elo_diff = Column(Float)                        # ELO-implied prob - market price
    sentiment_score = Column(Float)
    momentum_score = Column(Float)
    is_paper = Column(Boolean, nullable=False, default=True)
    status = Column(String, default="open")         # open | closed | cancelled
    opened_at = Column(DateTime, default=_now)
    closed_at = Column(DateTime)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    hold_time_minutes = Column(Float)
    exit_reason = Column(String)                    # profit_target|stop_loss|resolved|time_exit
    tx_hash = Column(String)
    slippage = Column(Float, default=0.0)
    notes = Column(String)

    __table_args__ = (
        Index("idx_trade_market", "market_id"),
        Index("idx_trade_status", "status"),
        Index("idx_trade_paper", "is_paper"),
        Index("idx_trade_opened", "opened_at"),
    )

    @property
    def is_winner(self) -> Optional[bool]:
        if self.pnl is None:
            return None
        return self.pnl > 0


class DailyStats(Base):
    """Track per-day P&L for drawdown monitoring."""
    __tablename__ = "daily_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_date = Column(String, nullable=False, unique=True)  # YYYY-MM-DD
    is_paper = Column(Boolean, nullable=False, default=True)
    starting_pnl = Column(Float, default=0.0)
    ending_pnl = Column(Float, default=0.0)
    trade_count = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    trading_halted = Column(Boolean, default=False)


class ModelRetrainLog(Base):
    __tablename__ = "model_retrain_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    retrained_at = Column(DateTime, default=_now)
    mae = Column(Float)
    n_samples = Column(Integer)
    feature_importance = Column(String)  # JSON


class TradeLogger:
    def __init__(self, db_path: str = "data/trades.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        Base.metadata.create_all(self.engine)

    def _session(self) -> Session:
        return Session(self.engine)

    # ── Trade CRUD ─────────────────────────────────────────────────────────

    def log_trade(self, **kwargs) -> Trade:
        with self._session() as s:
            trade = Trade(**kwargs)
            s.add(trade)
            s.commit()
            trade_id = trade.id
        return self._get_trade(trade_id)

    def _get_trade(self, trade_id: int) -> Trade:
        with self._session() as s:
            trade = s.get(Trade, trade_id)
            self._touch(trade)
            s.expunge(trade)
            return trade

    def update_trade(self, trade_id: int, **kwargs) -> Optional[Trade]:
        with self._session() as s:
            trade = s.get(Trade, trade_id)
            if trade is None:
                return None
            for k, v in kwargs.items():
                setattr(trade, k, v)
            s.commit()
            self._touch(trade)
            s.expunge(trade)
            return trade

    def close_trade(self, trade_id: int, exit_price: float,
                    exit_reason: str = "manual") -> Optional[Trade]:
        with self._session() as s:
            trade = s.get(Trade, trade_id)
            if trade is None:
                return None
            trade.exit_price = exit_price
            trade.closed_at = _now()
            trade.status = "closed"
            trade.exit_reason = exit_reason
            cost = trade.entry_price * trade.size
            proceeds = exit_price * trade.size
            trade.pnl = proceeds - cost - (trade.slippage or 0.0)
            trade.pnl_pct = (trade.pnl / cost) * 100 if cost else 0.0
            if trade.opened_at:
                delta = (trade.closed_at - trade.opened_at).total_seconds()
                trade.hold_time_minutes = delta / 60.0
            s.commit()
            self._touch(trade)
            s.expunge(trade)
            return trade

    @staticmethod
    def _touch(trade: Trade):
        """Force-load all columns to avoid DetachedInstanceError."""
        _ = (
            trade.id, trade.market_id, trade.market_question, trade.category,
            trade.side, trade.entry_price, trade.exit_price, trade.size,
            trade.fair_value, trade.edge_at_entry, trade.elo_diff,
            trade.sentiment_score, trade.is_paper, trade.status,
            trade.opened_at, trade.closed_at, trade.pnl, trade.pnl_pct,
            trade.hold_time_minutes, trade.exit_reason, trade.tx_hash,
            trade.slippage, trade.notes,
        )

    def get_open_trades(self, paper: Optional[bool] = None) -> list[Trade]:
        with self._session() as s:
            q = s.query(Trade).filter(Trade.status == "open")
            if paper is not None:
                q = q.filter(Trade.is_paper == paper)
            trades = q.all()
            for t in trades:
                self._touch(t)
            s.expunge_all()
            return trades

    def get_recent_trades(self, limit: int = 20, paper: Optional[bool] = None) -> list[Trade]:
        with self._session() as s:
            q = s.query(Trade).order_by(Trade.opened_at.desc()).limit(limit)
            if paper is not None:
                q = s.query(Trade).filter(Trade.is_paper == paper).order_by(Trade.opened_at.desc()).limit(limit)
            trades = q.all()
            for t in trades:
                self._touch(t)
            s.expunge_all()
            return trades

    # ── Statistics ─────────────────────────────────────────────────────────

    def get_stats(self, paper: Optional[bool] = None, last_n: int = 0) -> dict:
        with self._session() as s:
            q = s.query(Trade).filter(Trade.status == "closed")
            if paper is not None:
                q = q.filter(Trade.is_paper == paper)
            if last_n > 0:
                q = q.order_by(Trade.closed_at.desc()).limit(last_n)
            closed = q.all()

            open_q = s.query(Trade).filter(Trade.status == "open")
            if paper is not None:
                open_q = open_q.filter(Trade.is_paper == paper)
            open_count = open_q.count()

        if not closed:
            return {
                "total_trades": 0, "wins": 0, "losses": 0,
                "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
                "open_positions": open_count,
                "sharpe": 0.0, "sortino": 0.0,
            }

        wins = [t for t in closed if t.pnl and t.pnl > 0]
        losses = [t for t in closed if t.pnl and t.pnl <= 0]
        pnls = [t.pnl for t in closed if t.pnl is not None]
        total_pnl = sum(pnls)

        sharpe, sortino = self._risk_ratios(pnls)

        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) * 100,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(closed),
            "open_positions": open_count,
            "sharpe": sharpe,
            "sortino": sortino,
        }

    @staticmethod
    def _risk_ratios(pnls: list[float]) -> tuple[float, float]:
        if len(pnls) < 2:
            return 0.0, 0.0
        n = len(pnls)
        mean = sum(pnls) / n
        variance = sum((x - mean) ** 2 for x in pnls) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 1e-9
        sharpe = mean / std

        downside = [min(x, 0) for x in pnls]
        d_var = sum(x ** 2 for x in downside) / max(len(downside), 1)
        d_std = math.sqrt(d_var) if d_var > 0 else 1e-9
        sortino = mean / d_std
        return round(sharpe, 3), round(sortino, 3)

    def get_today_pnl(self, paper: bool = True) -> float:
        # Use UTC date to match _now() which stores UTC datetimes
        from datetime import datetime as _dt
        today = _dt.utcnow().date().isoformat()  # "YYYY-MM-DD" in UTC
        with self._session() as s:
            rows = s.query(Trade).filter(
                Trade.status == "closed",
                Trade.is_paper == paper,
                func.strftime("%Y-%m-%d", Trade.closed_at) == today,
            ).all()
            return sum(t.pnl or 0 for t in rows)

    def is_trading_halted_today(self, paper: bool = True) -> bool:
        from datetime import datetime as _dt
        today = _dt.utcnow().date().isoformat()
        with self._session() as s:
            row = s.query(DailyStats).filter(
                DailyStats.trade_date == today,
                DailyStats.is_paper == bool(paper),
            ).first()
            return bool(row and row.trading_halted)

    def halt_trading_today(self, paper: bool = True, reason: str = ""):
        from datetime import datetime as _dt
        today = _dt.utcnow().date().isoformat()
        with self._session() as s:
            row = s.query(DailyStats).filter(
                DailyStats.trade_date == today,
                DailyStats.is_paper == bool(paper),
            ).first()
            if row is None:
                row = DailyStats(trade_date=today, is_paper=bool(paper))
                s.add(row)
            row.trading_halted = True
            s.commit()

    # ── Model retrain log ──────────────────────────────────────────────────

    def log_retrain(self, mae: float, n_samples: int, feature_importance: str):
        with self._session() as s:
            s.add(ModelRetrainLog(mae=mae, n_samples=n_samples,
                                  feature_importance=feature_importance))
            s.commit()

    def get_last_retrain(self) -> Optional[ModelRetrainLog]:
        with self._session() as s:
            row = s.query(ModelRetrainLog).order_by(ModelRetrainLog.retrained_at.desc()).first()
            if row:
                _ = (row.id, row.retrained_at, row.mae, row.n_samples, row.feature_importance)
                s.expunge(row)
            return row

    # ── Daily paper report ─────────────────────────────────────────────────

    def daily_paper_report(self) -> dict:
        from datetime import datetime as _dt
        today = _dt.utcnow().date().isoformat()
        with self._session() as s:
            trades = s.query(Trade).filter(
                Trade.is_paper == True,
                Trade.status == "closed",
                func.strftime("%Y-%m-%d", Trade.opened_at) == today,
            ).all()
            for t in trades:
                self._touch(t)
            s.expunge_all()

        if not trades:
            return {"date": today, "trades": 0, "message": "No paper trades today."}

        pnls = [t.pnl for t in trades if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        best = max(pnls) if pnls else 0
        worst = min(pnls) if pnls else 0
        return {
            "date": today,
            "trades": len(trades),
            "win_rate": len(wins) / len(pnls) * 100 if pnls else 0,
            "total_pnl": sum(pnls),
            "best_trade": best,
            "worst_trade": worst,
        }
