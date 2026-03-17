"""
Trade executor: places orders (real or paper).
Paper mode simulates realistic slippage (0.1–0.3%) and order book impact.
Paper and real P&L are tracked in SEPARATE tables forever.
PAPER_TRADING=true is the hardcoded fallback — real orders require
explicit PAPER_TRADING=false + CONFIRM prompt at startup.
"""
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

from model.edge_detector import EdgeSignal
from logger.trade_logger import Trade, TradeLogger

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    success: bool
    order_id: str = ""
    tx_hash: str = ""
    fill_price: float = 0.0
    slippage: float = 0.0
    error: str = ""


def _simulate_slippage(price: float, size: float, liquidity: float) -> tuple[float, float]:
    """
    Simulate realistic slippage:
    - Base: 0.1–0.3% random
    - Extra market impact for large orders vs. available liquidity
    Returns (fill_price, slippage_usdc).
    """
    base_slip_pct = random.uniform(0.001, 0.003)
    # Market impact: for every 10% of liquidity consumed, add 0.1% slip
    if liquidity > 0:
        impact = (size / liquidity) * 0.10
    else:
        impact = 0.002
    total_slip_pct = base_slip_pct + impact
    fill_price = float(min(max(price * (1 + total_slip_pct), 0.01), 0.99))
    slippage_usdc = abs(fill_price - price) * size
    return fill_price, slippage_usdc


class TradeExecutor:
    """
    Executes trades. Paper by default — real wallet requires explicit opt-in.
    """

    def __init__(
        self,
        trade_logger: TradeLogger,
        paper_trading: bool = True,           # hardcoded safe default
        max_position_size: float = 100.0,
        max_position_pct: float = 0.05,       # 5% of bankroll (Kelly cap)
        bankroll: float = 1000.0,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        wallet_private_key: str = "",
    ):
        # Safety: if no private key, force paper mode regardless of flag
        if not wallet_private_key and not paper_trading:
            logger.warning(
                "WALLET_PRIVATE_KEY missing — forcing PAPER_TRADING=true for safety."
            )
            paper_trading = True

        self.paper_trading = paper_trading
        self.trade_logger = trade_logger
        self.max_position_size = max_position_size
        self.max_position_pct = max_position_pct
        self.bankroll = bankroll
        self._clob_client = None

        if not paper_trading:
            self._init_clob_client(api_key, api_secret, api_passphrase, wallet_private_key)

    def _init_clob_client(self, key: str, secret: str, passphrase: str, pk: str):
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.constants import POLYGON
            self._clob_client = ClobClient(
                host="https://clob.polymarket.com",
                key=pk,
                chain_id=POLYGON,
                creds={"api_key": key, "api_secret": secret, "api_passphrase": passphrase},
            )
            logger.info("CLOB client initialized for LIVE trading.")
        except ImportError:
            logger.error("py-clob-client not installed.")
        except Exception as e:
            logger.error(f"CLOB init failed: {e}")

    def kelly_position_size(self, signal: EdgeSignal) -> float:
        """
        Kelly Criterion: size = kelly_fraction * bankroll, capped at:
        - max_position_size (absolute USDC cap)
        - max_position_pct * bankroll (relative cap)
        """
        kelly_size = signal.kelly_fraction * self.bankroll
        pct_cap = self.max_position_pct * self.bankroll
        size = min(kelly_size, pct_cap, self.max_position_size)
        return round(max(1.0, size), 2)

    def calculate_position_size(self, signal: EdgeSignal) -> float:
        """Primary sizing: Kelly-based, scaled by edge strength."""
        return self.kelly_position_size(signal)

    def execute(self, signal: EdgeSignal, webhook_url: str = "") -> Optional[Trade]:
        """Execute a trade based on an EdgeSignal. Returns logged Trade or None."""
        # PAPER_TRADING guard — required before every real order
        if not self.paper_trading:
            logger.warning("LIVE ORDER ABOUT TO BE PLACED — verify PAPER_TRADING=false is intentional")

        size = self.calculate_position_size(signal)
        market = signal.market

        logger.info(
            f"{'[PAPER]' if self.paper_trading else '[LIVE!!]'} "
            f"{signal.trade_side} '{market.question[:55]}' "
            f"@ {signal.trade_price:.3f} | elo_edge={signal.elo_edge:.1%} | "
            f"conf={signal.confidence:.0%} | kelly=${size:.1f}"
        )

        if self.paper_trading:
            result = self._paper_execute(signal, size)
        else:
            result = self._live_execute(signal, size)

        if not result.success:
            logger.warning(f"Order failed: {result.error}")
            return None

        trade = self.trade_logger.log_trade(
            market_id=market.market_id,
            market_question=market.question,
            category=market.category,
            side=signal.trade_side,
            entry_price=result.fill_price or signal.trade_price,
            size=size,
            fair_value=signal.fair_value,
            edge_at_entry=signal.final_edge,
            elo_diff=signal.elo_edge,
            sentiment_score=signal.sentiment_score,
            is_paper=self.paper_trading,
            status="open",
            slippage=result.slippage,
            tx_hash=result.tx_hash,
        )

        if webhook_url and signal.final_edge >= 0.15:
            self._send_webhook(signal, size, webhook_url)

        return trade

    def _paper_execute(self, signal: EdgeSignal, size: float) -> OrderResult:
        fill_price, slippage = _simulate_slippage(
            signal.trade_price, size, signal.market.liquidity
        )
        order_id = f"PAPER_{signal.market.market_id[:8]}_{int(time.time())}"
        return OrderResult(
            success=True,
            order_id=order_id,
            fill_price=fill_price,
            slippage=slippage,
        )

    def _live_execute(self, signal: EdgeSignal, size: float) -> OrderResult:
        """
        PAPER_TRADING guard check before every real order.
        """
        # Final safety: double-check paper flag (belt + suspenders)
        assert not self.paper_trading, "Safety check: paper_trading should be False here"

        if self._clob_client is None:
            return OrderResult(success=False, error="CLOB client not initialized")

        try:
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import BUY

            # Select correct token_id for the side
            token_id = signal.market.market_id
            for token in signal.market._raw.get("tokens", []):
                if token.get("outcome", "").upper() == signal.trade_side.upper():
                    token_id = token.get("token_id") or token_id
                    break

            order_args = OrderArgs(
                token_id=token_id,
                price=round(signal.trade_price, 4),
                size=size,
                side=BUY,
            )
            resp = self._clob_client.create_and_post_order(order_args)
            if not isinstance(resp, dict):
                return OrderResult(success=False, error=f"Unexpected API response: {resp}")
            return OrderResult(
                success=True,
                order_id=resp.get("order_id", ""),
                tx_hash=resp.get("transactionHash", ""),
                fill_price=signal.trade_price,
            )
        except Exception as e:
            return OrderResult(success=False, error=str(e))

    def _send_webhook(self, signal: EdgeSignal, size: float, webhook_url: str):
        try:
            import requests as req
            payload = {
                "text": (
                    f"*Large Edge Alert* | {signal.trade_side} "
                    f"'{signal.market.question[:60]}'\n"
                    f"ELO Edge: {signal.elo_edge:.1%} | Confidence: {signal.confidence:.0%} | "
                    f"Price: {signal.trade_price:.3f} | ELO Prob: {signal.elo_implied_prob:.3f} | "
                    f"Size: ${size:.0f} | Kelly: {signal.kelly_fraction:.1%}"
                )
            }
            req.post(webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.debug(f"Webhook failed: {e}")
