"""Closed-loop feedback system — learns from every trade.

When a trade closes:
  1. Attribute P&L to entry factors
  2. Update rolling factor performance metrics
  3. Update online weights via gradient descent
  4. Persist factor_performance to Supabase
  5. Check for strategy decay
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config import DB_PATH, ONLINE_LEARNING_ENABLED
from learning.attribution import (
    attribute_trade,
    compute_factor_ic,
    compute_hit_rate,
    rolling_factor_stats,
)
from learning.weight_optimizer import OnlineWeightOptimizer, FactorWeights
from learning.decay import DecayDetector, DecayState

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """Closed-loop: trades → attribution → weight updates → decay checks."""

    def __init__(
        self,
        weights: Optional[FactorWeights] = None,
        lr: float = 0.05,
    ):
        self.optimizer = OnlineWeightOptimizer(weights=weights, lr=lr)
        self.decay_detector = DecayDetector()
        self._trade_history: list[dict] = []
        self._attribution_history: list[dict] = []
        self._current_weights = weights or FactorWeights()

    def process_closed_trades(
        self,
        closed_trades: list[dict],
        forward_returns: Optional[dict[str, float]] = None,
    ) -> dict:
        """Process newly closed trades through the feedback loop.

        Args:
            closed_trades: List of trades with status='CLOSED' and pnl filled
            forward_returns: Optional dict of ticker → realized forward return

        Returns:
            Summary dict with updated weights, decay state, and attribution
        """
        if not closed_trades:
            return {"trades_processed": 0}

        forward_returns = forward_returns or {}
        attributions = []

        for trade in closed_trades:
            attr = attribute_trade(trade, forward_returns)
            attributions.append(attr)
            self._attribution_history.append(attr)
            self._trade_history.append(trade)

        # Compute per-factor performance
        factor_ic = compute_factor_ic(closed_trades, pd.DataFrame())
        hit_rate = compute_hit_rate(closed_trades)

        # Update online weights using IC as signal (disabled per Phase 4 campaign findings)
        if ONLINE_LEARNING_ENABLED:
            self.optimizer.update(factor_ic)
            self._current_weights = self.optimizer.weights
        else:
            logger.debug("Online learning disabled — keeping equal weights")

        # Check for decay
        signal_scores = [t.get("composite_score", 0.5) or 0.5 for t in closed_trades]
        decay_state = self.decay_detector.check(
            closed_trades,
            signal_scores=signal_scores if signal_scores else None,
        )

        # Persist factor performance
        self._persist_factor_performance(factor_ic, hit_rate)

        # Persist decay alerts if any
        if decay_state.level.value != "GREEN":
            self._persist_decay_alert(decay_state)

        summary = {
            "trades_processed": len(closed_trades),
            "factor_ic": factor_ic,
            "hit_rate": hit_rate,
            "updated_weights": self._current_weights.as_dict(),
            "decay_level": decay_state.level.value,
            "decay_alerts": decay_state.details,
            "total_trade_count": len(self._trade_history),
        }

        logger.info(
            f"Feedback loop: {len(closed_trades)} trades processed | "
            f"Decay: {decay_state.level.value} ({decay_state.alert_count}/3) | "
            f"Weights: {self._current_weights.as_dict()}"
        )

        return summary

    def get_current_state(self) -> dict:
        """Current system state snapshot."""
        return {
            "weights": self._current_weights.as_dict(),
            "total_trades": len(self._trade_history),
            "total_attributions": len(self._attribution_history),
            "optimizer_updates": self.optimizer._update_count,
        }

    def _persist_factor_performance(self, ic: dict[str, float], hit: dict[str, float]):
        """Save factor performance to Supabase."""
        try:
            from db.client import rest_upsert, table_exists

            if not table_exists("factor_performance"):
                return

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            rows = []
            for factor in ["quality", "momentum", "value", "low_vol"]:
                rows.append({
                    "factor_name": factor,
                    "date": today,
                    "ic_60d": ic.get(factor, 0),
                    "hit_rate_60d": hit.get(factor, 0.5),
                    "weight": self._current_weights.as_dict().get(factor, 0.25),
                    "weight_target": self.optimizer._velocity[0] if factor == "quality" else None,
                })

            rest_upsert("factor_performance", rows, on_conflict="factor_name,date")
            logger.info(f"Persisted factor performance for {today}")
        except Exception as e:
            logger.error(f"Failed to persist factor performance: {e}")

    def _persist_decay_alert(self, state: DecayState):
        """Save decay alert to Supabase."""
        try:
            from db.client import rest_insert, table_exists

            if not table_exists("decay_alerts"):
                return

            import uuid
            alert_id = str(uuid.uuid4())[:8]

            rest_insert("decay_alerts", [{
                "alert_id": alert_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "detector": "CUSUM|SHARPE|PSI",
                "severity": state.level.value,
                "value": state.alert_count,
                "threshold": 2.0,
                "details": "; ".join(state.details),
                "acknowledged": False,
            }])
        except Exception as e:
            logger.error(f"Failed to persist decay alert: {e}")
