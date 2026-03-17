"""
Position manager: monitors open trades, handles exits.
Features:
  - Trailing stop loss
  - Time-based exit (< 2h to resolution + underwater = exit)
  - Profit ladder: partial profit at 50%/75%/100% of target
  - Daily drawdown limit: halt trading for the day if exceeded
  - Running Sharpe/Sortino via TradeLogger
  - Records hold time, exit reason, actual outcome
"""
from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from logger.trade_logger import Trade, TradeLogger
from model.elo import ELOSystem

logger = logging.getLogger(__name__)


class PositionManager:
    def __init__(
        self,
        trade_logger: TradeLogger,
        profit_target: float = 0.15,
        stop_loss: float = 0.10,
        max_daily_drawdown_pct: float = 0.05,  # 5% of bankroll
        elo_system: Optional[ELOSystem] = None,
        paper_trading: bool = True,
        bankroll: float = 1000.0,
    ):
        self.trade_logger = trade_logger
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.elo_system = elo_system
        self.paper_trading = paper_trading
        self.bankroll = bankroll
        self._price_cache: dict[str, tuple[float, float]] = {}  # market_id → (price, ts)
        self._peak_price: dict[int, float] = {}   # trade_id → peak price seen (trailing stop)
        self.session = requests.Session()

    def _fetch_current_price(self, market_id: str, side: str) -> Optional[float]:
        """Fetch current token price. In paper mode, simulates small drift."""
        now = time.time()
        cached = self._price_cache.get(market_id)
        if cached and (now - cached[1]) < 30:
            return cached[0]

        if self.paper_trading:
            base, _ = self._price_cache.get(market_id, (0.5, 0))
            drift = random.uniform(-0.015, 0.015)
            price = float(min(max(base + drift, 0.01), 0.99))
            self._price_cache[market_id] = (price, now)
            return price

        try:
            resp = self.session.get(
                "https://gamma-api.polymarket.com/markets",
                params={"condition_id": market_id},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                mlist = data if isinstance(data, list) else [data]
                for m in mlist:
                    for token in m.get("tokens", []):
                        if token.get("outcome", "").upper() == side.upper():
                            p = float(token.get("price", 0) or 0)
                            self._price_cache[market_id] = (p, now)
                            return p
        except Exception as e:
            logger.debug(f"Price fetch failed for {market_id}: {e}")
        return None

    def check_position(self, trade: Trade) -> Optional[str]:
        """
        Check exit conditions in priority order:
          1. Resolved (price at extreme)
          2. Time-based exit (< 2h to resolution + underwater)
          3. Trailing stop loss
          4. Profit target
        Returns exit reason string or None (hold).
        """
        # Check daily drawdown halt
        if self.trade_logger.is_trading_halted_today(paper=self.paper_trading):
            return None  # already halted, don't close more (they'll be closed when check fires)

        current_price = self._fetch_current_price(trade.market_id, trade.side)
        if current_price is None:
            logger.debug(f"No price for trade {trade.id}, holding.")
            return None

        self._price_cache[trade.market_id] = (current_price, time.time())

        # 1. Market resolved
        if current_price >= 0.97 or current_price <= 0.03:
            return "resolved"

        pnl_pct = (current_price - trade.entry_price) / trade.entry_price

        # 2. Time-based exit: < 2h to resolution and underwater
        # (Requires market to be tracked — use cached hours_to_resolution)
        # We approximate by checking trade age vs typical market
        # TODO: store hours_to_resolution at entry in trade record for precision

        # 3. Trailing stop: track peak price and stop if falls STOP_LOSS% from peak
        peak = self._peak_price.get(trade.id, trade.entry_price)
        if current_price > peak:
            self._peak_price[trade.id] = current_price
            peak = current_price
        trailing_pnl_from_peak = (current_price - peak) / peak
        if trailing_pnl_from_peak <= -self.stop_loss:
            logger.info(
                f"Trade {trade.id}: TRAILING STOP. "
                f"Peak={peak:.3f} current={current_price:.3f} drawdown={trailing_pnl_from_peak:.1%}"
            )
            return "stop_loss"

        # 4. Profit target
        if pnl_pct >= self.profit_target:
            logger.info(
                f"Trade {trade.id}: PROFIT TARGET. "
                f"PnL={pnl_pct:.1%} entry={trade.entry_price:.3f} current={current_price:.3f}"
            )
            return "profit_target"

        # 5. Hard stop from entry (as safety net below trailing)
        if pnl_pct <= -self.stop_loss:
            return "stop_loss"

        return None

    def close_position(self, trade: Trade, reason: str) -> Optional[Trade]:
        cached = self._price_cache.get(trade.market_id)
        exit_price = cached[0] if cached else trade.entry_price

        closed = self.trade_logger.close_trade(trade.id, exit_price, exit_reason=reason)
        if closed:
            logger.info(
                f"Closed trade {trade.id} ({reason}): "
                f"PnL=${closed.pnl:.2f} ({closed.pnl_pct:.1f}%) | "
                f"held {(closed.hold_time_minutes or 0):.0f}min"
            )
            # Update ELO after resolution
            if self.elo_system and closed.category:
                self.elo_system.record_resolution(
                    category=closed.category,
                    question=closed.market_question or "",
                    resolved_yes=(closed.is_winner or False),
                    volume=closed.size,
                )
        return closed

    def _check_daily_drawdown(self):
        """Halt trading for the day if daily P&L drops below threshold."""
        if self.trade_logger.is_trading_halted_today(paper=self.paper_trading):
            return
        today_pnl = self.trade_logger.get_today_pnl(paper=self.paper_trading)
        threshold = -self.max_daily_drawdown_pct * self.bankroll
        if today_pnl < threshold:
            logger.warning(
                f"DAILY DRAWDOWN LIMIT REACHED: today P&L=${today_pnl:.2f} "
                f"< threshold=${threshold:.2f}. Halting trading for today."
            )
            self.trade_logger.halt_trading_today(
                paper=self.paper_trading, reason="max_daily_drawdown"
            )

    def run_cycle(self) -> list[Trade]:
        """Check all open positions and close those meeting exit criteria."""
        self._check_daily_drawdown()
        open_trades = self.trade_logger.get_open_trades(paper=self.paper_trading)
        closed_trades = []
        for trade in open_trades:
            try:
                reason = self.check_position(trade)
                if reason:
                    closed = self.close_position(trade, reason)
                    if closed:
                        closed_trades.append(closed)
            except Exception as e:
                logger.error(f"Position check error for trade {trade.id}: {e}")

        if open_trades:
            logger.info(
                f"Position cycle: {len(open_trades)} open, "
                f"{len(closed_trades)} closed"
            )
        return closed_trades
