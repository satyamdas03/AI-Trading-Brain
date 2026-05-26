"""Online weight optimizer — adjusts factor weights based on rolling performance.

Gradient-based online learning:
  w_new = w + lr * tanh(perf)
  where perf = rolling IC or hit rate deviation from baseline

Bayesian weight optimization (Phase 4 weekend retraining):
  Optuna 500 trials, maximize Sharpe on walk-forward backtest
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import RETRAIN_MAX_TRIALS

logger = logging.getLogger(__name__)


@dataclass
class FactorWeights:
    quality: float = 0.20
    momentum: float = 0.20
    value: float = 0.20
    low_vol: float = 0.20
    vol_rank: float = 0.20

    def as_dict(self) -> dict[str, float]:
        return {
            "quality": self.quality,
            "momentum": self.momentum,
            "value": self.value,
            "low_vol": self.low_vol,
            "vol_rank": self.vol_rank,
        }

    def as_array(self) -> np.ndarray:
        return np.array([self.quality, self.momentum, self.value, self.low_vol, self.vol_rank])

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "FactorWeights":
        return cls(
            quality=d.get("quality", 0.20),
            momentum=d.get("momentum", 0.20),
            value=d.get("value", 0.20),
            low_vol=d.get("low_vol", 0.20),
            vol_rank=d.get("vol_rank", 0.20),
        )

    def normalize(self):
        """Ensure weights sum to 1.0."""
        total = self.quality + self.momentum + self.value + self.low_vol + self.vol_rank
        if total > 0:
            self.quality /= total
            self.momentum /= total
            self.value /= total
            self.low_vol /= total
            self.vol_rank /= total

    def clamp(self, min_w: float = 0.05, max_w: float = 0.50):
        """Clamp each weight to [min_w, max_w] range."""
        self.quality = max(min_w, min(max_w, self.quality))
        self.momentum = max(min_w, min(max_w, self.momentum))
        self.value = max(min_w, min(max_w, self.value))
        self.low_vol = max(min_w, min(max_w, self.low_vol))
        self.vol_rank = max(min_w, min(max_w, self.vol_rank))
        self.normalize()


class OnlineWeightOptimizer:
    """Online gradient-based weight adjustment.

    Formula: w_new = w + lr * tanh(perf), then normalize + clamp.
    """

    def __init__(
        self,
        weights: Optional[FactorWeights] = None,
        lr: float = 0.05,
        momentum_decay: float = 0.5,
    ):
        self.weights = weights or FactorWeights()
        self.lr = lr
        self.momentum_decay = momentum_decay
        self._velocity = np.zeros(5)
        self._update_count = 0

    def update(self, factor_performance: dict[str, float]):
        """Update weights based on per-factor performance metrics.

        Args:
            factor_performance: dict of factor_name → performance metric (IC, hit rate, etc.)
                Range should be approximately -1 to 1 (for IC) or 0 to 1 (for hit rate).
                Values are centered: perf_centered = value - baseline
        """
        factors = ["quality", "momentum", "value", "low_vol", "vol_rank"]
        current = self.weights.as_array()

        # Compute gradient: tanh of performance deviation from baseline
        perf_array = np.array([factor_performance.get(f, 0.0) for f in factors])

        # Baseline: 0 for IC, 0.5 for hit rate. Determine from data range.
        # If most values < 1, treat as IC-like (baseline 0)
        # If most values > 0.3, treat as hit-rate-like (baseline 0.5)
        baseline = 0.0 if np.abs(perf_array).max() < 0.5 else 0.5
        deviation = perf_array - baseline
        gradient = np.tanh(deviation * 2)  # Scale to make signal stronger

        # Momentum-accelerated update
        self._velocity = self.momentum_decay * self._velocity + self.lr * gradient
        new_weights = current + self._velocity

        # Update weights object
        self.weights = FactorWeights(
            quality=float(new_weights[0]),
            momentum=float(new_weights[1]),
            value=float(new_weights[2]),
            low_vol=float(new_weights[3]),
            vol_rank=float(new_weights[4]),
        )
        self.weights.clamp()
        self._update_count += 1

        logger.info(
            f"Weight update #{self._update_count}: "
            f"Q={self.weights.quality:.3f} M={self.weights.momentum:.3f} "
            f"V={self.weights.value:.3f} LV={self.weights.low_vol:.3f} "
            f"VR={self.weights.vol_rank:.3f}"
        )

    @property
    def weights_dict(self) -> dict[str, float]:
        return self.weights.as_dict()


class BayesianWeightOptimizer:
    """Bayesian weight optimization using Optuna for weekend retraining."""

    def __init__(self, n_trials: int = RETRAIN_MAX_TRIALS):
        self.n_trials = n_trials

    def optimize(
        self,
        trade_history: list[dict],
        signal_history: list[dict],
        benchmark_returns: Optional[list[float]] = None,
    ) -> tuple[FactorWeights, dict]:
        """Run Bayesian optimization to find optimal factor weights.

        Args:
            trade_history: List of trades with factor scores and P&L
            signal_history: Daily signal DataFrames history
            benchmark_returns: SPX benchmark returns for Sharpe comparison

        Returns:
            (optimal_weights, study_summary_dict)
        """
        try:
            import optuna
        except ImportError:
            logger.warning("optuna not installed — using fallback equal weights")
            return FactorWeights(), {"error": "optuna not installed"}

        # Prepare historical data for objective
        trades_df = self._prepare_data(trade_history, signal_history)

        def objective(trial: optuna.Trial) -> float:
            w = FactorWeights(
                quality=trial.suggest_float("quality", 0.05, 0.50),
                momentum=trial.suggest_float("momentum", 0.05, 0.50),
                value=trial.suggest_float("value", 0.05, 0.50),
                low_vol=trial.suggest_float("low_vol", 0.05, 0.50),
                vol_rank=trial.suggest_float("vol_rank", 0.05, 0.50),
            )
            w.normalize()

            # Simulate portfolio with these weights
            sharpe = self._simulate_sharpe(trades_df, w)
            return sharpe if not np.isnan(sharpe) else -1.0

        study = optuna.create_study(
            direction="maximize",
            study_name="factor_weight_optimization",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best = FactorWeights(
            quality=study.best_params["quality"],
            momentum=study.best_params["momentum"],
            value=study.best_params["value"],
            low_vol=study.best_params["low_vol"],
            vol_rank=study.best_params["vol_rank"],
        )
        best.normalize()

        summary = {
            "best_sharpe": study.best_value,
            "best_weights": best.as_dict(),
            "n_trials": self.n_trials,
            "baseline_equal_weight": self._simulate_sharpe(trades_df, FactorWeights()),
        }

        logger.info(f"Bayesian optimization: best Sharpe={study.best_value:.3f}, weights={best.as_dict()}")
        return best, summary

    def _prepare_data(self, trades: list[dict], signals: list[dict]) -> list[dict]:
        """Merge trades with their entry signals."""
        signal_map = {}
        for s in signals:
            key = f"{s.get('ticker')}:{s.get('date')}"
            signal_map[key] = s

        merged = []
        for t in trades:
            key = f"{t.get('ticker')}:{t.get('entry_date')}"
            signal = signal_map.get(key, {})
            merged.append({**t, **signal})
        return merged

    def _simulate_sharpe(self, trades: list[dict], weights: "FactorWeights") -> float:
        """Simulate Sharpe ratio using weighted composite scores."""
        if not trades:
            return 0.0

        w = weights.as_array()
        returns = []
        factor_cols = ["quality_pct", "momentum_pct", "value_pct", "low_vol_pct", "vol_rank_pct"]

        for t in trades:
            scores = np.array([t.get(c, 0.5) or 0.5 for c in factor_cols])
            weighted_score = np.dot(w, scores)
            # Map composite to expected return (simple linear mapping)
            expected_return = (weighted_score - 0.5) * 0.10  # 10% max annualized

            pnl = t.get("pnl_pct", 0) or 0
            returns.append(pnl)

        if len(returns) < 5:
            return 0.0

        returns_arr = np.array(returns)
        mean_ret = returns_arr.mean()
        std_ret = returns_arr.std()

        if std_ret == 0:
            return 0.0
        return float(mean_ret / std_ret * math.sqrt(252))  # Annualized Sharpe
