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

from config import (
    DRY_RUN, MAX_POSITION_PCT, DAILY_TRADE_CAP, SIGNAL_TOP_N,
    TAKE_PROFIT_PCT, STOP_LOSS_PCT, MAX_POSITIONS_PER_SECTOR,
    ATR_ENABLED, ATR_TP_MULTIPLIER, ATR_SL_MULTIPLIER,
)
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
        fundamentals: dict[str, dict] | None = None,
        max_per_sector: int = MAX_POSITIONS_PER_SECTOR,
        atr_values: dict[str, float | None] | None = None,
    ) -> list[dict]:
        """Execute buys for top-N signals at market open.

        Args:
            signals: DataFrame from signals/aggregator, sorted by composite_score desc
            account_equity: current account equity for position sizing
            max_new_positions: cap on new positions per session
            fundamentals: ticker -> {sector, ...} for sector diversification
            max_per_sector: max positions per GICS sector (default 1)
            atr_values: ticker -> latest ATR value for volatility-adjusted exits

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
        sector_counts: dict[str, int] = {}

        for _, row in signals.iterrows():
            if placed >= max_new_positions:
                break

            ticker = row["ticker"]

            # --- Sector diversification enforcement ---
            if fundamentals and max_per_sector > 0:
                sector = (fundamentals.get(ticker, {}) or {}).get("sector", "Unknown")
                if sector != "Unknown" and sector_counts.get(sector, 0) >= max_per_sector:
                    logger.info(
                        f"Skipping {ticker}: sector '{sector}' at cap "
                        f"({sector_counts[sector]}/{max_per_sector})"
                    )
                    continue

            # Skip if already in position
            if ticker in existing:
                continue

            # Position sizing: max_position_pct of equity
            position_pct = MAX_POSITION_PCT
            position_value = account_equity * position_pct

            try:
                price = self._client.get_last_price(ticker)
            except Exception as exc:
                logger.warning(f"No price for {ticker}, skipping: {exc}")
                continue

            if not price or price <= 0:
                logger.warning(f"Invalid price for {ticker}: {price!r}, skipping")
                continue

            qty = max(1, int(position_value / price))

            # Determine TP/SL — use ATR-based if available, else fixed %
            ticker_atr = (atr_values or {}).get(ticker)
            if ATR_ENABLED and ticker_atr and ticker_atr > 0:
                take_profit_price = round(price + ATR_TP_MULTIPLIER * ticker_atr, 2)
                stop_loss_price = round(price - ATR_SL_MULTIPLIER * ticker_atr, 2)
                exit_mode = f"ATR(${ticker_atr:.2f})"
            else:
                take_profit_price = round(price * (1 + TAKE_PROFIT_PCT), 2)
                stop_loss_price = round(price * (1 - STOP_LOSS_PCT), 2)
                exit_mode = "fixed%"

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
                "vol_rank_pct": float(row.get("vol_rank_score", 0.5)),
                "regime_id": int(row.get("regime_id", 2)),
                "position_size_pct": position_pct,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "exit_mode": exit_mode,
                "atr_value": ticker_atr if ATR_ENABLED else None,
                "max_favorable_excursion": 0.0,
                "max_adverse_excursion": 0.0,
            }

            if self._dry_run:
                logger.info(
                    f"DRY RUN BUY: {qty} {ticker} @ ${price:.2f} "
                    f"({position_pct:.0%} of equity) | TP=${take_profit_price:.2f} SL=${stop_loss_price:.2f} [{exit_mode}]"
                )
            else:
                try:
                    order = self._client.place_order(
                        symbol=ticker, qty=qty, side="buy",
                        order_type="market", time_in_force="day",
                        order_class="bracket",
                        take_profit=take_profit_price,
                        stop_loss=stop_loss_price,
                    )
                    trade_result["alpaca_order_id"] = order.id
                    logger.info(f"EXECUTED: BUY {qty} {ticker} @ ${price:.2f}")
                except Exception as e:
                    logger.error(f"Order failed {ticker}: {e}")
                    trade_result["status"] = "FAILED"
                    trade_result["exit_reason"] = str(e)[:200]

            trades.append(trade_result)
            placed += 1

            # Track sector for diversification cap
            if fundamentals and max_per_sector > 0:
                sector = (fundamentals.get(ticker, {}) or {}).get("sector", "Unknown")
                if sector != "Unknown":
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1

        if placed == 0 and len(signals) > 0:
            skipped_tickers = signals["ticker"].tolist()
            logger.warning(
                f"All {len(signals)} signals skipped ({skipped_tickers}). "
                f"Existing: {list(existing.keys())}. "
                f"Check get_last_price() for data subscription issues."
            )

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

    def reconcile_positions(self) -> list[dict]:
        """Sync Alpaca positions with local DB open trades.

        Detects positions closed by Alpaca bracket orders (TP/SL fills)
        and returns closure records. Does NOT write to DB — caller handles that.

        Returns:
            List of close-result dicts for any trades that were auto-closed by Alpaca.
        """
        from db.client import rest_get

        alpaca_positions = {p.symbol: p for p in self._client.get_positions()}
        try:
            db_open = rest_get("trades", select="*", filters={"status": "eq.OPEN"})
        except Exception as e:
            logger.warning(f"reconcile_positions: DB query failed: {e}")
            return []

        closed_trades = []
        for trade in (db_open or []):
            ticker = trade["ticker"]
            if ticker in alpaca_positions:
                continue  # still open at Alpaca

            # Position gone from Alpaca — bracket leg filled
            entry_price = trade.get("entry_price", 0)
            qty = trade.get("quantity", 0)

            # Get exit price from last trade or current price
            try:
                exit_price = self._client.get_last_price(ticker)
            except Exception:
                exit_price = entry_price  # fallback, won't be exact

            pnl_usd = (exit_price - entry_price) * qty if entry_price and exit_price else 0
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0

            # Determine exit reason: use stored TP/SL if available, else compute from config
            tp_price = trade.get("take_profit_price") or entry_price * (1 + TAKE_PROFIT_PCT)
            sl_price = trade.get("stop_loss_price") or entry_price * (1 - STOP_LOSS_PCT)
            if exit_price and exit_price >= tp_price * 0.995:
                reason = "take_profit"
            elif exit_price and exit_price <= sl_price * 1.005:
                reason = "stop_loss"
            else:
                reason = "alpaca_bracket"

            close_record = {
                "trade_id": trade.get("trade_id", ""),
                "ticker": ticker,
                "status": "CLOSED",
                "exit_price": exit_price,
                "exit_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "exit_reason": reason,
                "pnl_usd": round(pnl_usd, 2),
                "pnl_pct": round(pnl_pct, 4),
                "quantity": qty,
                "entry_price": entry_price,
            }
            closed_trades.append(close_record)
            logger.info(
                f"Reconciled CLOSE: {ticker} @ ${exit_price:.2f} "
                f"({reason}), P&L=${pnl_usd:.2f} ({pnl_pct:.1%})"
            )

        return closed_trades
