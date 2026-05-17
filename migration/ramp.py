"""Graduated live allocation ramp — 10%→25%→50%→100% weekly.

Each week advance requires no RED alerts in prior week.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

RAMP_SCHEDULE = [
    (1, 0.10, "Week 1 — 10% allocation"),
    (2, 0.25, "Week 2 — 25% allocation"),
    (3, 0.50, "Week 3 — 50% allocation"),
    (4, 1.00, "Week 4 — 100% allocation — FULLY LIVE"),
]


@dataclass
class RampState:
    current_week: int = 0  # 0 = not started
    allocation_pct: float = 0.0
    label: str = "Pre-ramp"
    is_fully_live: bool = False
    weeks_completed: int = 0
    started_at: str = ""
    last_advance: str = ""


class RampController:
    """Controls graduated migration from paper to live allocation."""

    def __init__(self, state: Optional[dict] = None):
        self._state = RampState()
        if state:
            self._state = RampState(
                current_week=state.get("current_week", 0),
                allocation_pct=state.get("allocation_pct", 0.0),
                label=state.get("label", "Pre-ramp"),
                is_fully_live=state.get("is_fully_live", False),
                weeks_completed=state.get("weeks_completed", 0),
                started_at=state.get("started_at", ""),
                last_advance=state.get("last_advance", ""),
            )

    def current_allocation_pct(self) -> float:
        return self._state.allocation_pct

    def is_fully_live(self) -> bool:
        return self._state.is_fully_live

    def advance_week(self, gates_all_green: bool) -> bool:
        """Advance to next ramp week if gates pass. Returns True if advanced."""
        if self._state.is_fully_live:
            logger.info("Already fully live — no further advance")
            return False

        if not gates_all_green:
            logger.warning(f"Ramp week {self._state.current_week + 1}: gates not all green — holding")
            return False

        next_week = self._state.current_week + 1
        step = next((s for s in RAMP_SCHEDULE if s[0] == next_week), None)

        if step is None:
            logger.warning(f"No ramp step defined for week {next_week}")
            return False

        week_num, allocation, label = step
        now = datetime.now(timezone.utc).isoformat()

        if self._state.current_week == 0:
            self._state.started_at = now

        self._state.current_week = week_num
        self._state.allocation_pct = allocation
        self._state.label = label
        self._state.is_fully_live = allocation >= 1.0
        self._state.weeks_completed += 1
        self._state.last_advance = now

        logger.info(f"Ramp advanced: {label}")
        return True

    def to_dict(self) -> dict:
        return {
            "current_week": self._state.current_week,
            "allocation_pct": self._state.allocation_pct,
            "label": self._state.label,
            "is_fully_live": self._state.is_fully_live,
            "weeks_completed": self._state.weeks_completed,
            "started_at": self._state.started_at,
            "last_advance": self._state.last_advance,
        }
