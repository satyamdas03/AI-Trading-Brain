"""Email notification stub — logs summaries instead of sending."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def send_summary(recipient: str, subject: str, body: str, html: str = "") -> bool:
    """Stub: log the email instead of sending.

    Returns True to indicate success (for API compatibility).
    """
    logger.info(f"[EMAIL STUB] To: {recipient} | Subject: {subject}")
    logger.info(f"[EMAIL STUB] Body preview: {body[:200]}")
    return True


def send_trade_alert(recipient: str, trade: dict) -> bool:
    """Stub: log trade alert."""
    logger.info(
        f"[EMAIL STUB] Trade alert to {recipient}: "
        f"{trade.get('ticker')} {trade.get('side')} "
        f"${trade.get('pnl_usd', 0):+,.2f}"
    )
    return True
