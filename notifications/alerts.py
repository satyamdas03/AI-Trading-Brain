"""Push alert stub — logs alerts instead of pushing to devices."""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


def push_alert(message: str, severity: AlertSeverity = AlertSeverity.INFO) -> bool:
    """Stub: log the alert instead of pushing to devices.

    Returns True to indicate success.
    """
    log_fn = {
        AlertSeverity.INFO: logger.info,
        AlertSeverity.WARNING: logger.warning,
        AlertSeverity.CRITICAL: logger.critical,
    }.get(severity, logger.info)

    log_fn(f"[PUSH {severity.value}] {message}")
    return True


def push_trade_executed(ticker: str, side: str, quantity: float, price: float) -> bool:
    """Stub: log trade execution notification."""
    logger.info(f"[PUSH] Trade: {side} {quantity} {ticker} @ ${price:.2f}")
    return True


def push_risk_alert(message: str, level: str) -> bool:
    """Stub: log risk alert."""
    severity = AlertSeverity.CRITICAL if level == "RED" else AlertSeverity.WARNING
    return push_alert(f"RISK {level}: {message}", severity)
