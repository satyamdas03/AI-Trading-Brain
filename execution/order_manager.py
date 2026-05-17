"""Order lifecycle manager — stop-loss, trailing stop, take-profit, cancel/replace."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from alpaca_trade_api.entity import Order
from alpaca_trade_api.rest import APIError

from execution.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order lifecycle: placement, stop-loss updates, cancellations."""

    def __init__(self, client: AlpacaClient):
        self._client = client

    def place_stop_loss(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
        side: str = "sell",
        time_in_force: str = "gtc",
    ) -> Optional[Order]:
        """Place a GTC stop-loss order."""
        try:
            order = self._client._rest.submit_order(
                symbol=symbol, qty=qty, side=side,
                type="stop", stop_price=round(stop_price, 2),
                time_in_force=time_in_force,
            )
            logger.info(f"Stop-loss placed: {side.upper()} {qty} {symbol} @ stop ${stop_price:.2f}")
            return order
        except APIError as e:
            logger.error(f"Stop-loss failed for {symbol}: {e}")
            return None

    def place_trailing_stop(
        self,
        symbol: str,
        qty: float,
        trail_percent: float,
        side: str = "sell",
    ) -> Optional[Order]:
        """Place a trailing stop order (trail_percent as decimal, e.g. 0.05 = 5%)."""
        try:
            order = self._client._rest.submit_order(
                symbol=symbol, qty=qty, side=side,
                type="trailing_stop", trail_percent=str(trail_percent * 100),
                time_in_force="gtc",
            )
            logger.info(f"Trailing stop: {side.upper()} {qty} {symbol} @ {trail_percent:.1%}")
            return order
        except APIError as e:
            logger.error(f"Trailing stop failed for {symbol}: {e}")
            return None

    def replace_stop_loss(
        self, order_id: str, new_stop_price: float, qty: Optional[float] = None
    ) -> Optional[Order]:
        """Cancel existing stop and replace with updated stop price."""
        try:
            old = self._client.get_order(order_id)
            self._client.cancel_order(order_id)
            new_order = self._client._rest.submit_order(
                symbol=old.symbol, qty=qty or float(old.qty),
                side=old.side, type="stop",
                stop_price=round(new_stop_price, 2),
                time_in_force="gtc",
            )
            logger.info(f"Stop updated: {old.symbol} ${new_stop_price:.2f}")
            return new_order
        except APIError as e:
            logger.error(f"Stop replacement failed: {e}")
            return None

    def cancel_stops_for(self, symbol: str) -> int:
        """Cancel all stop orders for a symbol. Returns count cancelled."""
        cancelled = 0
        try:
            orders = self._client.get_orders(status="open")
            for o in orders:
                if o.symbol == symbol and o.type in ("stop", "trailing_stop"):
                    self._client.cancel_order(o.id)
                    cancelled += 1
        except Exception as e:
            logger.error(f"Cancel stops failed for {symbol}: {e}")
        return cancelled

    def cancel_all_stops(self) -> int:
        """Cancel all open stop orders. Returns count cancelled."""
        cancelled = 0
        try:
            orders = self._client.get_orders(status="open")
            for o in orders:
                if o.type in ("stop", "trailing_stop"):
                    self._client.cancel_order(o.id)
                    cancelled += 1
        except Exception as e:
            logger.error(f"Cancel all stops failed: {e}")
        return cancelled

    def has_open_order(self, symbol: str) -> bool:
        """Check if there's an open order for a symbol."""
        try:
            orders = self._client.get_orders(status="open")
            return any(o.symbol == symbol for o in orders)
        except Exception:
            return False
