"""
Legacy momentum module — thin wrapper kept for backward compatibility.
The full ELO system now lives in model/elo.py.
"""
from __future__ import annotations

from model.elo import ELOSystem


class MomentumScorer:
    """Backward-compatible wrapper around ELOSystem."""

    def __init__(self, elo_system: ELOSystem = None):
        self.elo = elo_system or ELOSystem()

    def record_trade_outcome(self, category: str, won: bool, volume: float = 0.0):
        self.elo.record_resolution(
            category=category,
            question="",
            resolved_yes=won,
            volume=volume,
        )

    def get_category_score(self, category: str) -> float:
        from model.elo import DEFAULT_ELO
        rating = self.elo.get_category_elo(category)
        return max(-1.0, min(1.0, (rating - DEFAULT_ELO) / 400.0))

    def score_market(self, market) -> float:
        return self.get_category_score(market.category)

    def batch_score(self, markets: list) -> dict[str, float]:
        return {m.market_id: self.score_market(m) for m in markets}
