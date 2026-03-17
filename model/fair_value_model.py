"""
XGBoost fair value model — 20-feature engineering philosophy.
Inspired by tennis prediction research: feature quality > model complexity.
The ELO-implied probability vs. current price IS the primary edge signal.

Features (20 total):
  1  market_age_hours
  2  time_to_resolution_hours
  3  current_price
  4  volume_1h
  5  volume_6h
  6  volume_24h
  7  volume_7d
  8  volume_acceleration
  9  price_velocity
  10 category_elo
  11 event_type_elo
  12 elo_implied_probability
  13 elo_edge  (THE primary signal: elo_prob - current_price)
  14 liquidity_depth
  15 bid_ask_spread
  16 age_ratio  (age / total_market_life)
  17 sentiment_score
  18 sentiment_momentum
  19 hour_of_day
  20 whale_activity_flag

Model auto-retrains every 24h using resolved markets from SQLite.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "market_age_hours",
    "time_to_resolution_hours",
    "current_price",
    "volume_1h",
    "volume_6h",
    "volume_24h",
    "volume_7d",
    "volume_acceleration",
    "price_velocity",
    "category_elo",
    "event_type_elo",
    "elo_implied_probability",
    "elo_edge",
    "liquidity_depth",
    "bid_ask_spread",
    "age_ratio",
    "sentiment_score",
    "sentiment_momentum",
    "hour_of_day",
    "whale_activity_flag",
]

_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
MODEL_PATH = _DATA_DIR / "fair_value_model.pkl"
META_PATH = _DATA_DIR / "model_meta.json"


def _generate_synthetic_data(n_samples: int = 8000, elo_system=None) -> pd.DataFrame:
    """
    Generate synthetic training data approximating Polymarket resolution patterns.
    If elo_system is provided, ELO features are real; otherwise sampled.
    """
    from model.elo import DEFAULT_ELO, _elo_to_probability

    rng = np.random.default_rng(42)
    cats = ["politics", "crypto", "sports", "economics", "science",
            "entertainment", "geopolitics", "health", "technology", "general"]
    rows = []
    for _ in range(n_samples):
        cat = rng.choice(cats)
        age_h = rng.uniform(1, 4320)        # 1 hour to 6 months
        ttres_h = rng.uniform(0.1, 8760)    # up to 1 year
        current_price = rng.uniform(0.02, 0.98)
        volume_24h = rng.exponential(50_000)
        volume_1h = volume_24h / 24 * rng.uniform(0.3, 3.0)
        volume_6h = volume_24h / 4 * rng.uniform(0.5, 2.0)
        volume_7d = volume_24h * rng.uniform(5, 14)
        volume_acc = rng.normal(0, 100)
        price_vel = rng.normal(0, 0.002)
        liq = rng.exponential(20_000)
        spread = rng.uniform(0.001, 0.05)
        age_ratio = age_h / (age_h + ttres_h + 1e-9)
        sent = rng.uniform(-1, 1)
        sent_mom = rng.uniform(-0.5, 0.5)
        hour = rng.integers(0, 24)
        whale = int(rng.random() < 0.08)

        if elo_system:
            cat_elo = elo_system.get_category_elo(cat)
            evt_elo = elo_system.get_event_elo("")
        else:
            cat_elo = rng.normal(DEFAULT_ELO, 80)
            evt_elo = rng.normal(DEFAULT_ELO, 60)

        elo_prob = _elo_to_probability(cat_elo * 0.6 + evt_elo * 0.4)
        elo_edge = elo_prob - current_price

        # Resolution probability: mostly follows price but ELO edge adds signal
        noise = rng.normal(0, 0.06 + 0.12 * (ttres_h / 8760))
        res_prob = np.clip(current_price + elo_edge * 0.3 + noise, 0.01, 0.99)

        rows.append({
            "market_age_hours": age_h,
            "time_to_resolution_hours": ttres_h,
            "current_price": current_price,
            "volume_1h": volume_1h,
            "volume_6h": volume_6h,
            "volume_24h": volume_24h,
            "volume_7d": volume_7d,
            "volume_acceleration": volume_acc,
            "price_velocity": price_vel,
            "category_elo": cat_elo,
            "event_type_elo": evt_elo,
            "elo_implied_probability": elo_prob,
            "elo_edge": elo_edge,
            "liquidity_depth": liq,
            "bid_ask_spread": spread,
            "age_ratio": age_ratio,
            "sentiment_score": sent,
            "sentiment_momentum": sent_mom,
            "hour_of_day": hour,
            "whale_activity_flag": whale,
            "resolution_probability": res_prob,
        })
    return pd.DataFrame(rows)


class FairValueModel:
    """XGBoost regressor estimating fair YES probability."""

    def __init__(self, elo_system=None):
        self.model: Optional[xgb.XGBRegressor] = None
        self.elo_system = elo_system
        self._loaded = False
        self._last_train_time: float = 0.0
        self._last_mae: float = 0.0
        self._last_n_samples: int = 0
        self._feature_importance: dict = {}
        self.retrain_interval_hours: float = 24.0

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        return df[FEATURE_COLS].fillna(0.0).values

    def train(self, data: Optional[pd.DataFrame] = None,
              trade_logger=None) -> float:
        """Train on provided data or synthetic fallback. Returns MAE."""
        if data is None:
            logger.info("No real training data — using synthetic dataset.")
            data = _generate_synthetic_data(n_samples=8000, elo_system=self.elo_system)

        X = self._prepare_X(data)
        y = data["resolution_probability"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_preds = self.model.predict(X_val)
        mae = float(mean_absolute_error(y_val, val_preds))
        self._last_mae = mae
        self._last_n_samples = len(data)
        self._last_train_time = time.time()
        self._loaded = True

        # Feature importance
        scores = self.model.feature_importances_
        self._feature_importance = dict(zip(FEATURE_COLS, scores.tolist()))
        sorted_fi = sorted(self._feature_importance.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Model trained. Val MAE={mae:.4f} | n={len(data)}")
        logger.info("Top 5 features: " + ", ".join(f"{k}:{v:.3f}" for k, v in sorted_fi[:5]))

        if trade_logger:
            trade_logger.log_retrain(
                mae=mae,
                n_samples=len(data),
                feature_importance=json.dumps(self._feature_importance),
            )

        self._save()
        return mae

    def _save(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        meta = {
            "last_train_time": self._last_train_time,
            "last_mae": self._last_mae,
            "last_n_samples": self._last_n_samples,
            "feature_importance": self._feature_importance,
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Model saved to {MODEL_PATH}")

    def load(self) -> bool:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            self._loaded = True
            if META_PATH.exists():
                with open(META_PATH) as f:
                    meta = json.load(f)
                self._last_train_time = meta.get("last_train_time", 0)
                self._last_mae = meta.get("last_mae", 0)
                self._last_n_samples = meta.get("last_n_samples", 0)
                self._feature_importance = meta.get("feature_importance", {})
            logger.info(f"Model loaded (MAE={self._last_mae:.4f})")
            return True
        return False

    def ensure_ready(self, trade_logger=None, snapshot_store=None):
        if not self._loaded:
            if not self.load():
                logger.info("No saved model. Training on synthetic data...")
                self.train(trade_logger=trade_logger)

    def should_retrain(self) -> bool:
        """Return True if enough time has elapsed since last training."""
        if not self._loaded:
            return False
        elapsed_h = (time.time() - self._last_train_time) / 3600
        return elapsed_h >= self.retrain_interval_hours

    def retrain_from_resolved(self, snapshot_store, trade_logger=None) -> Optional[float]:
        """Retrain using resolved markets from SQLite + synthetic fallback."""
        resolved = snapshot_store.get_resolved_for_training()
        if not resolved:
            logger.info("No resolved markets yet — retraining with synthetic data.")
            return self.train(trade_logger=trade_logger)

        from model.elo import _elo_to_probability, DEFAULT_ELO

        rows = []
        for r in resolved:
            cat_elo = self.elo_system.get_category_elo(r["category"]) if self.elo_system else DEFAULT_ELO
            elo_p = _elo_to_probability(cat_elo)
            rows.append({
                "market_age_hours": 720,
                "time_to_resolution_hours": 1,
                "current_price": r["final_price"],
                "volume_1h": 0, "volume_6h": 0,
                "volume_24h": r["volume"] / 30,
                "volume_7d": r["volume"] / 4,
                "volume_acceleration": 0, "price_velocity": 0,
                "category_elo": cat_elo,
                "event_type_elo": DEFAULT_ELO,
                "elo_implied_probability": elo_p,
                "elo_edge": elo_p - r["final_price"],
                "liquidity_depth": r["volume"],
                "bid_ask_spread": 0.02,
                "age_ratio": 0.9,
                "sentiment_score": 0, "sentiment_momentum": 0,
                "hour_of_day": 12, "whale_activity_flag": 0,
                "resolution_probability": float(r["resolved_yes"]),
            })
        real_df = pd.DataFrame(rows)
        synth_df = _generate_synthetic_data(n_samples=max(200, len(rows) * 2),
                                             elo_system=self.elo_system)
        combined = pd.concat([real_df, synth_df], ignore_index=True)
        logger.info(f"Retraining on {len(real_df)} real + {len(synth_df)} synthetic samples")
        return self.train(data=combined, trade_logger=trade_logger)

    def predict(self, features: dict) -> float:
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call ensure_ready() first.")
        df = pd.DataFrame([features])
        X = self._prepare_X(df)
        pred = self.model.predict(X)[0]
        return float(np.clip(pred, 0.01, 0.99))

    def batch_predict(self, features_df: pd.DataFrame) -> np.ndarray:
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded.")
        X = self._prepare_X(features_df)
        return np.clip(self.model.predict(X), 0.01, 0.99)
