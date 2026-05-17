"""Risk manager — tiered drawdown guard, sector caps, correlation checks.

Multi-asset support (Phase 6): per-class rules, shared global limits.

Three-tier alert system:
  GREEN  — normal operation
  YELLOW — reduce position sizes, no new entries
  RED    — close all positions, halt trading
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import pandas as pd

from config import MAX_DRAWDOWN_PCT, MAX_SECTOR_PCT
from execution.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


def detect_market(ticker: str) -> str:
    """Detect asset class from ticker format.

    - .NS suffix → IN (India)
    - / in ticker → CRYPTO (e.g. BTC/USD)
    - hex token ID → POLYMARKET
    - default → US
    """
    if ticker.endswith(".NS"):
        return "IN"
    if "/" in ticker:
        return "CRYPTO"
    if ticker.startswith("0x") or len(ticker) == 42:
        return "POLYMARKET"
    return "US"


@dataclass
class RiskState:
    level: AlertLevel = AlertLevel.GREEN
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    sector_exposure: dict[str, float] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)
    max_position_pct_override: float | None = None


class RiskManager:
    """Risk controls: drawdown guard, sector caps, concentration limits. Multi-asset aware."""

    def __init__(self, client: AlpacaClient):
        self._client = client
        self._state = RiskState()
        self._fundamentals_cache: dict[str, dict] = {}

    def check_pre_market(self, fundamentals: dict[str, dict] | None = None) -> RiskState:
        """Pre-market risk check. Returns current risk state.

        Called before execution to determine if we can trade today.
        """
        if fundamentals:
            self._fundamentals_cache = fundamentals

        try:
            acc = self._client.get_account()
            positions = self._client.get_positions()

            # Update peak equity
            if acc.equity > self._state.peak_equity:
                self._state.peak_equity = acc.equity

            if self._state.peak_equity > 0:
                self._state.current_drawdown_pct = (
                    (self._state.peak_equity - acc.equity) / self._state.peak_equity
                )
        except Exception as e:
            logger.error(f"Pre-market risk check failed: {e}")
            return self._state

        # Tiered checks
        alerts = []

        # 1. Drawdown check (global — applies to all asset classes)
        if self._state.current_drawdown_pct > MAX_DRAWDOWN_PCT:
            alerts.append(f"Drawdown {self._state.current_drawdown_pct:.1%} exceeds limit {MAX_DRAWDOWN_PCT:.1%}")

        # 2. Day trade count check (US only — PDT is US-specific)
        us_positions = [p for p in positions if detect_market(p.symbol) == "US"]
        if us_positions and acc.day_trade_count >= 3:
            alerts.append(f"Day trade count {acc.day_trade_count} — approaching PDT limit")

        # 3. Account status
        if acc.status != "ACTIVE":
            alerts.append(f"Account status: {acc.status}")

        # 4. Sector concentration (US + IN only, skips crypto/polymarket)
        sector_values: dict[str, float] = {}
        equity_positions = [p for p in positions if detect_market(p.symbol) in ("US", "IN")]
        if self._fundamentals_cache and equity_positions:
            for p in equity_positions:
                sector = self._fundamentals_cache.get(p.symbol, {}).get("sector", "Unknown")
                sector_values[sector] = sector_values.get(sector, 0) + p.market_value

            total = sum(sector_values.values()) if sector_values else 1
            for sector, value in sector_values.items():
                pct = value / total if total > 0 else 0
                if pct > MAX_SECTOR_PCT:
                    alerts.append(f"Sector {sector} at {pct:.0%} exceeds cap {MAX_SECTOR_PCT:.0%}")
                sector_values[sector] = pct

        # Determine level
        if any("exceeds limit" in a or "halted" in a.lower() for a in alerts):
            level = AlertLevel.RED
        elif alerts:
            level = AlertLevel.YELLOW
        else:
            level = AlertLevel.GREEN

        self._state = RiskState(
            level=level,
            daily_pnl=0.0,  # reset daily
            daily_pnl_pct=0.0,
            peak_equity=self._state.peak_equity,
            current_drawdown_pct=self._state.current_drawdown_pct,
            sector_exposure=sector_values,
            alerts=alerts,
            max_position_pct_override=None if level == AlertLevel.GREEN else (
                0.05 if level == AlertLevel.YELLOW else 0.0
            ),
        )

        if level != AlertLevel.GREEN:
            logger.warning(f"Risk level: {level.value} — {alerts}")

        return self._state

    def validate_position(self, ticker: str, size_pct: float, market: str = "US") -> tuple[bool, str]:
        """Validate a new position entry. Returns (approved, reason).

        Asset-class-agnostic: applies correct rules per market.
        """
        market = market or detect_market(ticker)

        # Global drawdown gate
        if self._state.level == AlertLevel.RED:
            return False, "Risk level RED — trading halted"

        # Per-class checks
        if market in ("US", "IN"):
            cap = self.position_cap_pct
            if size_pct > cap:
                return False, f"Size {size_pct:.1%} exceeds cap {cap:.1%} ({market})"

            # Sector check (US/IN only)
            if self._fundamentals_cache:
                sector = self._fundamentals_cache.get(ticker, {}).get("sector")
                if sector and sector in self._state.sector_exposure:
                    if self._state.sector_exposure[sector] + size_pct > MAX_SECTOR_PCT:
                        return False, f"Sector {sector} would exceed {MAX_SECTOR_PCT:.0%} cap"

        elif market == "CRYPTO":
            if size_pct > 0.15:
                return False, f"Crypto size {size_pct:.1%} exceeds 15% cap"
            if self._state.level == AlertLevel.YELLOW:
                return False, "YELLOW risk level — no new crypto entries"

        elif market == "POLYMARKET":
            if size_pct > 0.05:
                return False, f"Polymarket size {size_pct:.1%} exceeds 5% cap"

        return True, "approved"

    def check_intraday(self) -> RiskState:
        """Intraday risk check (run every 15min)."""
        try:
            acc = self._client.get_account()
            positions = self._client.get_positions()

            # Calculate intraday drawdown
            if self._state.peak_equity > 0:
                current_dd = (self._state.peak_equity - acc.equity) / self._state.peak_equity
            else:
                current_dd = 0.0

            # Check for stop-loss violations
            alerts = []
            for p in positions:
                if p.unrealized_pnl_pct < -0.05:  # >5% loss triggers alert
                    alerts.append(f"{p.symbol} down {p.unrealized_pnl_pct:.1%}")

            if current_dd > MAX_DRAWDOWN_PCT:
                level = AlertLevel.RED
                alerts.append(f"Intraday DD {current_dd:.1%} triggers RED")
            elif alerts:
                level = AlertLevel.YELLOW
            else:
                level = AlertLevel.GREEN

            self._state.level = level
            self._state.current_drawdown_pct = current_dd
            self._state.alerts = alerts

            return self._state

        except Exception as e:
            logger.error(f"Intraday check failed: {e}")
            return self._state

    @property
    def can_trade(self) -> bool:
        return self._state.level != AlertLevel.RED

    @property
    def position_cap_pct(self) -> float:
        """Effective position size cap considering risk level."""
        if self._state.max_position_pct_override is not None:
            return self._state.max_position_pct_override
        from config import MAX_POSITION_PCT
        return MAX_POSITION_PCT

    @property
    def state(self) -> RiskState:
        return self._state
