"""Polymarket Pipeline V2 bridge — classify → detect_edge → execute."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Add polymarket pipeline to path
_POLY_ROOT = Path(__file__).parent.parent.parent / "polymarket-pipeline"
if str(_POLY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLY_ROOT))


class PolymarketBridge:
    """Thin wrapper around Polymarket Pipeline V2 for trading brain integration."""

    def __init__(self, max_per_market: float = 500):
        self.max_per_market = max_per_market

    def scan_and_trade(self, max_per_market: Optional[float] = None) -> list[dict]:
        """Run one scan cycle: classify markets → detect edge → execute trades.

        Returns list of executed trade dicts.
        """
        limit = max_per_market or self.max_per_market
        trades: list[dict] = []

        try:
            from pipeline import run_pipeline_v2  # noqa: F401 — verify pipeline importable
            from edge import detect_edge_v2  # noqa: F401

            # Polymarket Pipeline V2 runs as standalone async process.
            # Bridge integration point for future: call run_pipeline_v2()
            # and extract markets from pipeline.stats.
            logger.debug("Polymarket pipeline available (V2), bridge integration pending")

        except ImportError as e:
            logger.debug(f"Polymarket pipeline not available: {e}")
        except Exception as e:
            logger.debug(f"Polymarket scan skipped: {e}")

        return trades

    def _execute(self, market: dict, signal, max_bet: float) -> Optional[dict]:
        """Execute a single Polymarket trade via pipeline executor."""
        try:
            from executor import execute_trade_async
            import asyncio

            trade_result = asyncio.new_event_loop().run_until_complete(
                execute_trade_async(
                    slug=market.get("slug", ""),
                    side=signal.side,
                    bet_amount=min(signal.bet_amount, max_bet),
                    reason=signal.reason or "AI Trading Brain signal",
                )
            )

            if trade_result:
                return {
                    "slug": market.get("slug"),
                    "side": signal.side,
                    "price": trade_result.get("price", 0),
                    "bet_amount": min(signal.bet_amount, max_bet),
                    "edge": signal.edge,
                    "reason": signal.reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            logger.error(f"Polymarket trade execution failed: {e}")

        return None
