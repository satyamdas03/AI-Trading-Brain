"""Alpaca REST API wrapper — paper trading client.

Uses alpaca-trade-api SDK. Auto-detects paper vs live from APCA_API_BASE_URL env var.
Set APCA_API_BASE_URL=https://paper-api.alpaca.markets for paper trading.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
from alpaca_trade_api import REST
from alpaca_trade_api.entity import Order, Position, Clock
from alpaca_trade_api.rest import APIError
from tenacity import retry, wait_exponential, stop_after_attempt

from config import ALPACA_PAPER, ALPACA_API_KEY, ALPACA_SECRET_KEY

logger = logging.getLogger(__name__)

# Ensure SDK env vars are set for REST() auto-detection
if ALPACA_PAPER and not os.environ.get("APCA_API_BASE_URL"):
    os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
if not os.environ.get("APCA_API_KEY_ID"):
    os.environ["APCA_API_KEY_ID"] = ALPACA_API_KEY
    os.environ["APCA_API_SECRET_KEY"] = ALPACA_SECRET_KEY

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"


@dataclass
class AlpacaAccount:
    account_number: str
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    day_trade_count: int
    pattern_day_trader: bool
    status: str


@dataclass
class AlpacaPosition:
    symbol: str
    qty: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    current_price: float
    avg_entry_price: float
    side: str  # "long" | "short"


class AlpacaClient:
    """Thin wrapper around alpaca-trade-api REST client."""

    def __init__(self):
        if not ALPACA_API_KEY:
            raise RuntimeError("APCA_API_KEY_ID or ALPACA_API_KEY required")
        base_url = PAPER_BASE_URL if ALPACA_PAPER else LIVE_BASE_URL
        self._rest = REST(base_url=base_url)
        self.paper = ALPACA_PAPER

    def get_account(self) -> AlpacaAccount:
        acc = self._rest.get_account()
        return AlpacaAccount(
            account_number=acc.account_number,
            equity=float(acc.equity),
            cash=float(acc.cash),
            buying_power=float(acc.buying_power),
            portfolio_value=float(acc.portfolio_value),
            day_trade_count=int(acc.daytrade_count),
            pattern_day_trader=acc.pattern_day_trader,
            status=acc.status,
        )

    def get_positions(self) -> list[AlpacaPosition]:
        positions = self._rest.list_positions()
        return [
            AlpacaPosition(
                symbol=p.symbol,
                qty=float(p.qty),
                market_value=float(p.market_value),
                cost_basis=float(p.cost_basis),
                unrealized_pnl=float(p.unrealized_pl),
                unrealized_pnl_pct=float(p.unrealized_plpc),
                current_price=float(p.current_price),
                avg_entry_price=float(p.avg_entry_price),
                side=p.side,
            )
            for p in positions
        ]

    def get_position(self, symbol: str) -> Optional[AlpacaPosition]:
        try:
            p = self._rest.get_position(symbol)
            return AlpacaPosition(
                symbol=p.symbol, qty=float(p.qty),
                market_value=float(p.market_value),
                cost_basis=float(p.cost_basis),
                unrealized_pnl=float(p.unrealized_pl),
                unrealized_pnl_pct=float(p.unrealized_plpc),
                current_price=float(p.current_price),
                avg_entry_price=float(p.avg_entry_price),
                side=p.side,
            )
        except APIError as e:
            if "position does not exist" in str(e).lower():
                return None
            raise

    def get_clock(self) -> Clock:
        return self._rest.get_clock()

    def is_market_open(self) -> bool:
        clock = self.get_clock()
        return clock.is_open

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,  # "buy" | "sell"
        order_type: str = "market",  # "market" | "limit"
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        order_class: str = "simple",  # "simple" | "bracket" | "oco"
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> Order:
        if order_class == "bracket":
            order = self._rest.submit_order(
                symbol=symbol, qty=qty, side=side,
                type=order_type,
                limit_price=limit_price,
                time_in_force=time_in_force,
                order_class="bracket",
                take_profit={"limit_price": take_profit},
                stop_loss={"stop_price": stop_loss},
            )
        elif order_type == "market":
            order = self._rest.submit_order(
                symbol=symbol, qty=qty, side=side,
                type="market", time_in_force=time_in_force,
            )
        else:
            order = self._rest.submit_order(
                symbol=symbol, qty=qty, side=side,
                type=order_type, limit_price=limit_price,
                time_in_force=time_in_force,
            )
        logger.info(f"Order placed: {side.upper()} {qty} {symbol} [{order_type}]")
        return order

    def cancel_order(self, order_id: str) -> None:
        self._rest.cancel_order(order_id)
        logger.info(f"Order cancelled: {order_id}")

    def cancel_all_orders(self) -> None:
        self._rest.cancel_all_orders()
        logger.info("All orders cancelled")

    def get_order(self, order_id: str) -> Order:
        return self._rest.get_order(order_id)

    def get_orders(self, status: str = "open", limit: int = 50) -> list[Order]:
        return self._rest.list_orders(status=status, limit=limit)

    def get_bars(
        self, symbol: str, days: int = 252, timeframe: str = "1Day"
    ) -> pd.DataFrame:
        bars = self._rest.get_bars(symbol, timeframe, limit=days)
        return bars.df.reset_index()

    def get_last_price(self, symbol: str) -> Optional[float]:
        try:
            bars = self._rest.get_bars(symbol, "1Min", limit=1)
            if bars.df.empty:
                return None
            return float(bars.df["close"].iloc[-1])
        except Exception:
            return None
