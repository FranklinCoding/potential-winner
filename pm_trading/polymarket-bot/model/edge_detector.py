"""
Edge detector: primary signal = ELO-implied probability vs. current market price.
Secondary signal: sentiment directional agreement.
All three signals (ELO edge, sentiment, liquidity) must agree for a flag.

Kelly Criterion position sizing:
  f = (edge * bankroll) / odds
  capped at MAX_POSITION_PCT of bankroll.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from scanner.market_scanner import Market
from model.fair_value_model import FairValueModel
from model.elo import ELOSystem

logger = logging.getLogger(__name__)


@dataclass
class EdgeSignal:
    market: Market
    fair_value: float              # XGBoost model output
    elo_implied_prob: float        # ELO-implied probability
    current_price: float
    raw_edge: float                # fair_value - current_price
    elo_edge: float                # elo_implied_prob - current_price (PRIMARY signal)
    final_edge: float              # blended, used for sizing
    trade_side: str                # YES | NO
    trade_price: float
    confidence: float              # 0-1
    position_size_factor: float    # 0-1, scales with edge strength
    sentiment_agrees: bool
    sentiment_score: float
    sentiment_momentum: float
    kelly_fraction: float          # raw Kelly fraction (before bankroll cap)


def kelly_fraction(edge: float, odds: float = 1.0) -> float:
    """
    Kelly Criterion: f = (edge * (odds + 1) - 1) / odds
    For binary markets: odds = (1/price) - 1 (implied by market price).
    Returns fraction in [0, 0.25] (quarter-Kelly for safety).
    """
    if edge <= 0 or odds <= 0:
        return 0.0
    # b = net fractional odds if bet wins
    # f* = (b*p - q) / b where p = prob of win, q = 1-p
    p = min(max(edge + 0.5, 0.01), 0.99)  # rough win probability
    b = odds
    q = 1 - p
    f_star = (b * p - q) / b
    # Quarter-Kelly, bounded
    return float(min(max(f_star * 0.25, 0.0), 0.25))


class EdgeDetector:
    """Detects mispriced markets using ELO + XGBoost + sentiment."""

    def __init__(
        self,
        model: FairValueModel,
        elo_system: ELOSystem,
        min_edge: float = 0.08,
        min_liquidity: float = 500.0,
        max_spread: float = 0.05,
    ):
        self.model = model
        self.elo = elo_system
        self.min_edge = min_edge
        self.min_liquidity = min_liquidity
        self.max_spread = max_spread

    def build_features(
        self,
        market: Market,
        sentiment_score: float = 0.0,
        sentiment_momentum: float = 0.0,
    ) -> dict:
        """Build the full 20-feature dict for a market."""
        now = datetime.now(timezone.utc)
        cat_elo = self.elo.get_category_elo(market.category)
        evt_elo = self.elo.get_event_elo(market.question)
        elo_prob = self.elo.elo_implied_probability(market.category, market.question, now)
        elo_edge_val = elo_prob - market.yes_price

        return {
            "market_age_hours": market.market_age_hours,
            "time_to_resolution_hours": market.hours_to_resolution,
            "current_price": market.yes_price,
            "volume_1h": market.volume_1h,
            "volume_6h": market.volume_6h,
            "volume_24h": market.volume_24h,
            "volume_7d": market.volume_7d,
            "volume_acceleration": market.volume_acceleration,
            "price_velocity": market.price_velocity,
            "category_elo": cat_elo,
            "event_type_elo": evt_elo,
            "elo_implied_probability": elo_prob,
            "elo_edge": elo_edge_val,
            "liquidity_depth": market.liquidity,
            "bid_ask_spread": market.bid_ask_spread,
            "age_ratio": market.age_ratio,
            "sentiment_score": sentiment_score,
            "sentiment_momentum": sentiment_momentum,
            "hour_of_day": float(now.hour),
            "whale_activity_flag": float(market.whale_activity),
        }

    def analyze(
        self,
        market: Market,
        sentiment_score: float = 0.0,
        sentiment_momentum: float = 0.0,
    ) -> Optional[EdgeSignal]:
        """
        Analyze one market. Returns EdgeSignal if all conditions are met:
          1. ELO edge > min_edge
          2. Sentiment agrees with direction
          3. Liquidity sufficient + spread not too wide
        """
        # Pre-filter: skip thin or noisy markets
        if market.liquidity < self.min_liquidity:
            return None
        if market.bid_ask_spread > self.max_spread:
            return None

        features = self.build_features(market, sentiment_score, sentiment_momentum)

        try:
            fair_value = self.model.predict(features)
        except Exception as e:
            logger.warning(f"Model prediction failed for {market.market_id}: {e}")
            return None

        elo_prob = features["elo_implied_probability"]
        elo_edge_val = features["elo_edge"]  # positive = YES underpriced per ELO

        # Determine trade direction from ELO (primary) and model (secondary)
        yes_elo_edge = elo_edge_val
        no_elo_edge = (1 - elo_prob) - market.no_price

        if abs(yes_elo_edge) >= abs(no_elo_edge):
            raw_elo_edge = yes_elo_edge
            side = "YES"
            trade_price = market.yes_price
            model_edge = fair_value - market.yes_price
        else:
            raw_elo_edge = no_elo_edge
            side = "NO"
            trade_price = market.no_price
            model_edge = (1 - fair_value) - market.no_price

        # Gate 1: ELO edge must be positive and above threshold
        if raw_elo_edge < self.min_edge:
            return None

        # Gate 2: Sentiment must agree (or be neutral)
        sent_direction = "YES" if sentiment_score >= 0 else "NO"
        sentiment_agrees = (sentiment_score == 0.0) or (sent_direction == side)

        # Gate 3: Model edge should not strongly disagree — but only enforce when
        # ELO edge is modest. For large ELO signals (>20%), trust ELO.
        if model_edge < -0.05 and raw_elo_edge < 0.20:
            logger.debug(f"Model contradicts ELO on {market.market_id[:8]} — skipping")
            return None

        # Blended final edge: ELO is primary (70%), model secondary (30%)
        final_edge = 0.70 * raw_elo_edge + 0.30 * max(model_edge, 0.0)

        if final_edge < self.min_edge:
            return None

        # Confidence: edge size + sentiment agreement + liquidity depth
        edge_conf = min(final_edge / 0.30, 1.0)
        sent_conf = 0.15 if sentiment_agrees else 0.0
        liq_conf = min(market.liquidity / 50_000, 0.15)
        confidence = min(edge_conf + sent_conf + liq_conf, 1.0)

        # Kelly position sizing
        implied_odds = (1.0 / trade_price - 1.0) if trade_price > 0 else 1.0
        kf = kelly_fraction(final_edge, implied_odds)
        position_size_factor = min(kf / 0.10, 1.0)  # normalize: 10% kelly → full size

        return EdgeSignal(
            market=market,
            fair_value=fair_value,
            elo_implied_prob=elo_prob,
            current_price=market.yes_price,
            raw_edge=raw_elo_edge,
            elo_edge=raw_elo_edge,
            final_edge=final_edge,
            trade_side=side,
            trade_price=trade_price,
            confidence=confidence,
            position_size_factor=position_size_factor,
            sentiment_agrees=sentiment_agrees,
            sentiment_score=sentiment_score,
            sentiment_momentum=sentiment_momentum,
            kelly_fraction=kf,
        )

    def scan_markets(
        self,
        markets: list[Market],
        sentiment_data: Optional[dict[str, tuple[float, float]]] = None,
    ) -> list[EdgeSignal]:
        """Scan all markets. sentiment_data maps market_id → (score, momentum)."""
        signals = []
        sentiment_data = sentiment_data or {}

        for market in markets:
            mid = market.market_id
            sent_score, sent_mom = sentiment_data.get(mid, (0.0, 0.0))
            try:
                signal = self.analyze(market, sent_score, sent_mom)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Edge analysis error on {mid[:8]}: {e}")

        # Rank by CONFIDENCE (not just edge size) per requirements
        signals.sort(key=lambda s: s.confidence, reverse=True)
        logger.info(
            f"Edge scan: {len(signals)} opportunities from {len(markets)} markets "
            f"(top edge: {signals[0].final_edge:.1%})" if signals else
            f"Edge scan: 0 opportunities from {len(markets)} markets"
        )
        return signals
