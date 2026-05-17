"""Strategy decay detection — three independent detectors.

Three detectors:
  1. CUSUM — detects regime shifts (sudden change in win rate)
  2. Rolling Sharpe — detects performance deterioration
  3. PSI (Population Stability Index) — detects distribution shift in signal scores

2/3 vote triggers alert (YELLOW). 3/3 triggers RED.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass
class DecayState:
    cusum_triggered: bool = False
    cusum_value: float = 0.0
    sharpe_triggered: bool = False
    sharpe_value: float = 0.0
    psi_triggered: bool = False
    psi_value: float = 0.0
    alert_count: int = 0
    level: AlertLevel = AlertLevel.GREEN
    details: list[str] = field(default_factory=list)


class DecayDetector:
    """Three-detector strategy decay monitoring."""

    def __init__(
        self,
        cusum_threshold: float = 5.0,
        cusum_drift: float = 0.05,
        sharpe_threshold: float = 0.0,
        psi_threshold: float = 0.25,
        baseline_win_rate: float = 0.50,
        baseline_sharpe: float = 1.0,
    ):
        self.cusum_threshold = cusum_threshold
        self.cusum_drift = cusum_drift
        self.sharpe_threshold = sharpe_threshold
        self.psi_threshold = psi_threshold
        self.baseline_win_rate = baseline_win_rate
        self.baseline_sharpe = baseline_sharpe

        # CUSUM state
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0

        # Reference distribution for PSI
        self._reference_scores: Optional[np.ndarray] = None

    def check(
        self,
        recent_trades: list[dict],
        signal_scores: Optional[list[float]] = None,
    ) -> DecayState:
        """Run all three detectors.

        Args:
            recent_trades: Last N trades (ideally last 20-30)
            signal_scores: Current signal composite scores for PSI check

        Returns:
            DecayState with alert level and details
        """
        state = DecayState()

        # 1. CUSUM: detect shift in win rate
        if len(recent_trades) >= 10:
            state.cusum_triggered, state.cusum_value = self._check_cusum(recent_trades)
        if state.cusum_triggered:
            state.details.append(
                f"CUSUM triggered: {state.cusum_value:.2f} > threshold {self.cusum_threshold}"
            )

        # 2. Rolling Sharpe: detect performance decay
        if len(recent_trades) >= 15:
            state.sharpe_triggered, state.sharpe_value = self._check_sharpe(recent_trades)
        if state.sharpe_triggered:
            state.details.append(
                f"Rolling Sharpe decay: {state.sharpe_value:.3f} < threshold {self.sharpe_threshold}"
            )

        # 3. PSI: detect distribution shift
        if signal_scores and len(signal_scores) >= 10:
            state.psi_triggered, state.psi_value = self._check_psi(signal_scores)
        if state.psi_triggered:
            state.details.append(
                f"PSI distribution shift: {state.psi_value:.3f} > threshold {self.psi_threshold}"
            )

        # Determine level (2/3 vote)
        state.alert_count = sum([state.cusum_triggered, state.sharpe_triggered, state.psi_triggered])

        if state.alert_count >= 3:
            state.level = AlertLevel.RED
        elif state.alert_count >= 2:
            state.level = AlertLevel.YELLOW
        else:
            state.level = AlertLevel.GREEN

        if state.level != AlertLevel.GREEN:
            logger.warning(
                f"Decay alert ({state.level.value}): {state.alert_count}/3 detectors triggered — {state.details}"
            )

        return state

    def _check_cusum(self, trades: list[dict]) -> tuple[bool, float]:
        """CUSUM detection on win rate shifts.

        CUSUM tracks cumulative deviation from expected win rate.
        When it exceeds threshold, a regime shift is detected.
        """
        for t in reversed(trades):
            pnl = t.get("pnl_pct", 0) or 0
            is_win = 1 if pnl > 0 else 0
            deviation = is_win - self.baseline_win_rate - self.cusum_drift

            self._cusum_pos = max(0, self._cusum_pos + deviation)
            self._cusum_neg = max(0, self._cusum_neg - deviation)

        max_cusum = max(self._cusum_pos, self._cusum_neg)
        triggered = max_cusum > self.cusum_threshold
        return triggered, max_cusum

    def _check_sharpe(self, trades: list[dict]) -> tuple[bool, float]:
        """Check rolling Sharpe ratio from trade P&L stream."""
        returns = [t.get("pnl_pct", 0) or 0 for t in trades[-20:]]
        returns_arr = np.array(returns)

        mean_ret = returns_arr.mean()
        std_ret = returns_arr.std()

        if std_ret == 0:
            rolling_sharpe = 0.0
        else:
            rolling_sharpe = float(mean_ret / std_ret * np.sqrt(252))

        triggered = rolling_sharpe < self.sharpe_threshold
        return triggered, rolling_sharpe

    def _check_psi(self, scores: list[float]) -> tuple[bool, float]:
        """Population Stability Index — detect distribution shift in signal scores.

        PSI = sum((actual_i - expected_i) * ln(actual_i / expected_i))
        PSI < 0.1: no shift
        PSI 0.1-0.25: moderate shift
        PSI > 0.25: significant shift (triggered)
        """
        scores_arr = np.array(scores)
        scores_arr = scores_arr[np.isfinite(scores_arr)]

        if len(scores_arr) < 10:
            return False, 0.0

        # Initialize or update reference distribution
        if self._reference_scores is None:
            self._reference_scores = scores_arr
            return False, 0.0

        # Compute PSI with 10 equal-width bins
        try:
            bins = np.linspace(0, 1, 11)
            ref_hist, _ = np.histogram(self._reference_scores, bins=bins, density=True)
            cur_hist, _ = np.histogram(scores_arr, bins=bins, density=True)

            # Add small epsilon to avoid division by zero
            eps = 1e-6
            ref_hist = ref_hist + eps
            cur_hist = cur_hist + eps

            ref_hist = ref_hist / ref_hist.sum()
            cur_hist = cur_hist / cur_hist.sum()

            psi = float(np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist)))
            triggered = psi > self.psi_threshold

            return triggered, psi
        except Exception as e:
            logger.error(f"PSI computation failed: {e}")
            return False, 0.0

    def reset_cusum(self):
        """Reset CUSUM accumulators after a regime change is acknowledged."""
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0

    def update_reference(self, scores: list[float]):
        """Update reference distribution for PSI."""
        scores_arr = np.array(scores)
        self._reference_scores = scores_arr[np.isfinite(scores_arr)]
