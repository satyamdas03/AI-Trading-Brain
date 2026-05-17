"""Migration gates — three checks for paper→live transition readiness.

Gates:
  1. Consistency: rolling 60-day Sharpe > 0.5
  2. Drawdown: max drawdown < 10% over 60 trading days
  3. Stability: 30-day rolling Sharpe > 0.3

All three GREEN for 5 consecutive days → migration proposal.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class GateResult:
    consistency: str = "UNKNOWN"  # GREEN / YELLOW / RED
    drawdown: str = "UNKNOWN"
    stability: str = "UNKNOWN"
    sharpe_60d: float = 0.0
    max_drawdown_60d: float = 0.0
    sharpe_30d: float = 0.0
    all_green: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MigrationProposal:
    ready: bool = False
    days_green: int = 0
    gates: GateResult = field(default_factory=GateResult)
    proposal_timestamp: str = ""


class LiveGate:
    """Three-gate migration checklist evaluator."""

    def __init__(
        self,
        sharpe_threshold: float = 0.5,
        max_dd_threshold: float = 0.10,
        stability_threshold: float = 0.3,
        consistency_lookback: int = 60,
        drawdown_lookback: int = 60,
        stability_lookback: int = 30,
    ):
        self.sharpe_threshold = sharpe_threshold
        self.max_dd_threshold = max_dd_threshold
        self.stability_threshold = stability_threshold
        self.consistency_lookback = consistency_lookback
        self.drawdown_lookback = drawdown_lookback
        self.stability_lookback = stability_lookback

    def evaluate(
        self,
        trades: list[dict],
        equity_curve: list[dict],
    ) -> GateResult:
        """Run all three gate checks.

        Args:
            trades: list of closed trades with pnl_pct field
            equity_curve: list of daily snapshots with equity field
        """
        result = GateResult()

        # Gate 1: Consistency — rolling 60-day Sharpe
        if equity_curve and len(equity_curve) >= self.consistency_lookback:
            recent = equity_curve[-self.consistency_lookback:]
            daily_pnl = [s.get("daily_pnl_pct", 0) or 0 for s in recent]
            result.sharpe_60d = self._compute_sharpe(daily_pnl)
            result.consistency = self._level(result.sharpe_60d, self.sharpe_threshold, higher_better=True)
        elif equity_curve and len(equity_curve) >= 5:
            recent = equity_curve[-min(len(equity_curve), self.consistency_lookback):]
            daily_pnl = [s.get("daily_pnl_pct", 0) or 0 for s in recent]
            result.sharpe_60d = self._compute_sharpe(daily_pnl)
            result.consistency = "UNKNOWN"
        else:
            result.consistency = "UNKNOWN"

        # Gate 2: Drawdown — max DD over lookback window
        if equity_curve and len(equity_curve) >= max(5, self.drawdown_lookback):
            recent = equity_curve[-self.drawdown_lookback:]
            equities = [s.get("equity", 0) for s in recent]
            result.max_drawdown_60d = self._compute_max_drawdown(equities)
            result.drawdown = self._level(result.max_drawdown_60d, self.max_dd_threshold, higher_better=False)
        elif equity_curve and len(equity_curve) >= 5:
            equities = [s.get("equity", 0) for s in equity_curve]
            result.max_drawdown_60d = self._compute_max_drawdown(equities)
            result.drawdown = "UNKNOWN"
        else:
            result.drawdown = "UNKNOWN"

        # Gate 3: Stability — 30-day rolling Sharpe
        if equity_curve and len(equity_curve) >= self.stability_lookback:
            recent = equity_curve[-self.stability_lookback:]
            daily_pnl = [s.get("daily_pnl_pct", 0) or 0 for s in recent]
            result.sharpe_30d = self._compute_sharpe(daily_pnl)
            result.stability = self._level(result.sharpe_30d, self.stability_threshold, higher_better=True)
        elif equity_curve and len(equity_curve) >= 5:
            daily_pnl = [s.get("daily_pnl_pct", 0) or 0 for s in equity_curve]
            result.sharpe_30d = self._compute_sharpe(daily_pnl)
            result.stability = "UNKNOWN"
        else:
            result.stability = "UNKNOWN"

        result.all_green = (
            result.consistency == "GREEN"
            and result.drawdown == "GREEN"
            and result.stability == "GREEN"
        )

        return result

    def check_migration_readiness(
        self,
        gate_history: list[dict],
    ) -> MigrationProposal:
        """Check if we've had 5 consecutive all-green days.

        Args:
            gate_history: list of prior GateResult dicts with 'all_green' field
        """
        if not gate_history:
            return MigrationProposal()

        consecutive = 0
        for h in reversed(gate_history):
            if h.get("all_green"):
                consecutive += 1
            else:
                break

        ready = consecutive >= 5
        last = gate_history[-1]
        gates = GateResult(
            consistency=last.get("consistency", "UNKNOWN"),
            drawdown=last.get("drawdown", "UNKNOWN"),
            stability=last.get("stability", "UNKNOWN"),
            sharpe_60d=last.get("sharpe_60d", 0),
            max_drawdown_60d=last.get("max_drawdown_60d", 0),
            sharpe_30d=last.get("sharpe_30d", 0),
            all_green=last.get("all_green", False),
        )

        return MigrationProposal(
            ready=ready,
            days_green=consecutive,
            gates=gates,
            proposal_timestamp=datetime.now(timezone.utc).isoformat() if ready else "",
        )

    # ── internal ─────────────────────────────────────────────────

    @staticmethod
    def _compute_sharpe(daily_returns: list[float]) -> float:
        arr = np.array(daily_returns, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 5:
            return 0.0
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return 0.0
        return float(mean / std * math.sqrt(TRADING_DAYS_PER_YEAR))

    @staticmethod
    def _compute_max_drawdown(equities: list[float]) -> float:
        arr = np.array(equities, dtype=np.float64)
        if len(arr) < 2:
            return 0.0
        peak = np.maximum.accumulate(arr)
        drawdowns = (peak - arr) / peak
        return float(drawdowns.max())

    @staticmethod
    def _level(value: float, threshold: float, higher_better: bool) -> str:
        if higher_better:
            if value >= threshold:
                return "GREEN"
            elif value >= threshold * 0.5:
                return "YELLOW"
            return "RED"
        else:
            if value <= threshold:
                return "GREEN"
            elif value <= threshold * 1.5:
                return "YELLOW"
            return "RED"
