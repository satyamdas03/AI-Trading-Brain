"""Execution engine — converts signals into trades via Alpaca REST API.

Phase 2: Market-open top-N buy with bracket orders (take-profit + stop-loss).
Phase 3+: Adds 7-agent debate before execution.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config import DRY_RUN, MAX_POSITION_PCT, DAILY_TRADE_CAP, SIGNAL_TOP_N, TAKE_PROFIT_PCT, STOP_LOSS_PCT
from execution.alpaca_client import AlpacaClient, AlpacaPosition
from execution.order_manager import OrderManager

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Converts daily signal rankings into paper trades."""

    def __init__(self):
        self._client = AlpacaClient()
        self._orders = OrderManager(self._client)
        self._dry_run = DRY_RUN

    def execute_market_open(
        self,
        signals: pd.DataFrame,
        account_equity: float,
        max_new_positions: int = DAILY_TRADE_CAP,
    ) -> list[dict]:
        """Execute buys for top-N signals at market open.

        Args:
            signals: DataFrame from signals/aggregator, sorted by composite_score desc
            account_equity: current account equity for position sizing
            max_new_positions: cap on new positions per session

        Returns:
            List of trade result dicts for logging
        """
        if signals.empty:
            logger.info("No signals — skipping execution")
            return []

        # Get existing positions
        existing = {p.symbol: p for p in self._client.get_positions()}
        logger.info(f"Existing positions: {list(existing.keys())}")

        trades = []
        placed = 0

        for _, row in signals.iterrows():
            if placed >= max_new_positions:
                break

            ticker = row["ticker"]

            # Skip if already in position
            if ticker in existing:
                continue

            # Position sizing: max_position_pct of equity
            position_pct = MAX_POSITION_PCT
            position_value = account_equity * position_pct

            try:
                price = self._client.get_last_price(ticker)
            except Exception:
                logger.warning(f"No price for {ticker}, skipping")
                continue

            if not price or price <= 0:
                continue

            qty = max(1, int(position_value / price))

            trade_result = {
                "trade_id": str(uuid.uuid4())[:8],
                "ticker": ticker,
                "market": "US",
                "side": "BUY",
                "quantity": qty,
                "entry_price": price,
                "entry_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "status": "OPEN",
                "composite_score": float(row.get("composite_score", 0.5)),
                "quality_pct": float(row.get("quality_score", 0.5)),
                "momentum_pct": float(row.get("momentum_score", 0.5)),
                "value_pct": float(row.get("value_score", 0.5)),
                "low_vol_pct": float(row.get("low_vol_score", 0.5)),
                "regime_id": int(row.get("regime_id", 2)),
                "position_size_pct": position_pct,
                "stop_loss_price": price * (1 - STOP_LOSS_PCT),
                "max_favorable_excursion": 0.0,
                "max_adverse_excursion": 0.0,
            }

            if self._dry_run:
                logger.info(f"DRY RUN BUY: {qty} {ticker} @ ${price:.2f} ({position_pct:.0%} of equity)")
            else:
                try:
                    order = self._client.place_order(
                        symbol=ticker, qty=qty, side="buy",
                        order_type="market", time_in_force="day",
                        order_class="bracket",
                        take_profit=round(price * (1 + TAKE_PROFIT_PCT), 2),
                        stop_loss=round(price * (1 - STOP_LOSS_PCT), 2)
                    )
                    trade_result["alpaca_order_id"] = order.id
                    logger.info(f"EXECUTED: BUY {qty} {ticker} @ ${price:.2f}")
                except Exception as e:
                    logger.error(f"Order failed {ticker}: {e}")
                    trade_result["status"] = "FAILED"
                    trade_result["exit_reason"] = str(e)[:200]

            trades.append(trade_result)
            placed += 1

        logger.info(f"Execution complete: {placed} new positions, {len(existing)} existing")
        return trades

    def execute_sell(self, ticker: str, reason: str = "manual") -> Optional[dict]:
        """Sell an existing position."""
        pos = self._client.get_position(ticker)
        if not pos:
            logger.warning(f"No position in {ticker}")
            return None

        price = self._client.get_last_price(ticker) or pos.current_price

        trade_result = {
            "trade_id": str(uuid.uuid4())[:8],
            "ticker": ticker,
            "side": "SELL",
            "quantity": pos.qty,
            "exit_price": price,
            "exit_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "exit_reason": reason,
            "status": "CLOSED",
            "pnl_usd": (price - pos.avg_entry_price) * pos.qty,
            "pnl_pct": (price - pos.avg_entry_price) / pos.avg_entry_price if pos.avg_entry_price > 0 else 0,
        }

        if self._dry_run:
            logger.info(f"DRY RUN SELL: {pos.qty} {ticker} @ ${price:.2f} ({reason})")
        else:
            # Cancel existing stops first
            self._orders.cancel_stops_for(ticker)
            try:
                order = self._client.place_order(
                    symbol=ticker, qty=pos.qty, side="sell",
                    order_type="market", time_in_force="day",
                )
                trade_result["alpaca_order_id"] = order.id
            except Exception as e:
                logger.error(f"Sell failed {ticker}: {e}")
                trade_result["status"] = "FAILED"

        return trade_result

    def get_portfolio_summary(self) -> dict:
        """Snapshot of current account + positions."""
        acc = self._client.get_account()
        positions = self._client.get_positions()

        return {
            "equity": acc.equity,
            "cash": acc.cash,
            "buying_power": acc.buying_power,
            "num_positions": len(positions),
            "positions_value": sum(p.market_value for p in positions),
            "unrealized_pnl": sum(p.unrealized_pnl for p in positions),
            "account_status": acc.status,
        }
