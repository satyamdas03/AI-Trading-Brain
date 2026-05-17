"""P&L tracker — equity curve, trade P&L, benchmark comparison.

Persists to portfolio_snapshots table, logs to trades table.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config import DB_PATH
from execution.alpaca_client import AlpacaClient, AlpacaPosition

logger = logging.getLogger(__name__)


class PnLTracker:
    """Tracks P&L and maintains equity curve."""

    def __init__(self, client: AlpacaClient):
        self._client = client
        self._trade_history: list[dict] = []
        self._equity_curve: list[dict] = []

    def snapshot(self) -> dict:
        """Take current portfolio snapshot for equity curve."""
        try:
            acc = self._client.get_account()
            positions = self._client.get_positions()
            spx_price = self._get_spx_price()
        except Exception as e:
            logger.error(f"Snapshot failed: {e}")
            return {}

        total_positions_value = sum(p.market_value for p in positions)

        snap = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "equity": acc.equity,
            "cash": acc.cash,
            "positions_value": total_positions_value,
            "daily_pnl": 0.0,  # computed externally by comparing snapshots
            "daily_pnl_pct": 0.0,
            "cumulative_return_pct": 0.0,
            "num_positions": len(positions),
            "spx_close": spx_price,
            "spx_return_pct": 0.0,
        }

        self._equity_curve.append(snap)
        return snap

    def record_trade(self, trade: dict) -> None:
        """Record a trade to history."""
        self._trade_history.append(trade)

    def compute_daily_pnl(
        self, prev_snapshot: Optional[dict], current_snapshot: dict
    ) -> dict:
        """Compute daily P&L from two consecutive snapshots."""
        if prev_snapshot and prev_snapshot.get("equity", 0) > 0:
            prev_equity = prev_snapshot["equity"]
            current_snapshot["daily_pnl"] = current_snapshot["equity"] - prev_equity
            current_snapshot["daily_pnl_pct"] = (
                (current_snapshot["equity"] - prev_equity) / prev_equity
            )

            # Cumulative return
            if self._equity_curve and len(self._equity_curve) > 1:
                starting = self._equity_curve[0].get("equity", 100000)
                if starting > 0:
                    current_snapshot["cumulative_return_pct"] = (
                        (current_snapshot["equity"] - starting) / starting
                    )

        if current_snapshot.get("spx_close") and prev_snapshot and prev_snapshot.get("spx_close"):
            prev_spx = prev_snapshot["spx_close"]
            if prev_spx > 0:
                current_snapshot["spx_return_pct"] = (
                    (current_snapshot["spx_close"] - prev_spx) / prev_spx
                )

        return current_snapshot

    def persist_position_pnl(self, trades: list[dict]) -> None:
        """Log executed trades to DB."""
        try:
            from db.client import rest_upsert, table_exists

            if not table_exists("trades"):
                logger.warning("Trades table not yet created — run migration 001")
                return

            rows = []
            for t in trades:
                rows.append({
                    "trade_id": t.get("trade_id", ""),
                    "ticker": t["ticker"],
                    "market": t.get("market", "US"),
                    "side": t["side"],
                    "quantity": t["quantity"],
                    "entry_price": t.get("entry_price"),
                    "exit_price": t.get("exit_price"),
                    "entry_date": t.get("entry_date"),
                    "exit_date": t.get("exit_date"),
                    "status": t.get("status", "OPEN"),
                    "exit_reason": t.get("exit_reason"),
                    "pnl_usd": t.get("pnl_usd"),
                    "pnl_pct": t.get("pnl_pct"),
                    "composite_score": t.get("composite_score"),
                    "quality_pct": t.get("quality_pct"),
                    "momentum_pct": t.get("momentum_pct"),
                    "value_pct": t.get("value_pct"),
                    "low_vol_pct": t.get("low_vol_pct"),
                    "regime_id": t.get("regime_id"),
                    "max_favorable_excursion": t.get("max_favorable_excursion"),
                    "max_adverse_excursion": t.get("max_adverse_excursion"),
                    "stop_loss_price": t.get("stop_loss_price"),
                    "position_size_pct": t.get("position_size_pct"),
                })

            if rows:
                rest_upsert("trades", rows, on_conflict="trade_id")
                logger.info(f"Persisted {len(rows)} trades to DB")
        except Exception as e:
            logger.error(f"Failed to persist trades: {e}")

    def persist_snapshot(self, snapshot: dict) -> None:
        """Save portfolio snapshot to DB."""
        try:
            from db.client import rest_upsert, table_exists

            if not table_exists("portfolio_snapshots"):
                logger.warning("portfolio_snapshots table not yet created")
                return

            rest_upsert("portfolio_snapshots", [snapshot], on_conflict="date")
            logger.info(f"Snapshot {snapshot['date']}: equity=${snapshot['equity']:,.2f}")
        except Exception as e:
            logger.error(f"Failed to persist snapshot: {e}")

    def get_trade_summary(self) -> dict:
        """Summary stats for all trades."""
        if not self._trade_history:
            return {"total_trades": 0}

        closed = [t for t in self._trade_history if t.get("status") == "CLOSED" and t.get("pnl_usd") is not None]
        wins = [t for t in closed if t.get("pnl_usd", 0) > 0]
        losses = [t for t in closed if t.get("pnl_usd", 0) < 0]

        return {
            "total_trades": len(self._trade_history),
            "closed_trades": len(closed),
            "open_trades": len([t for t in self._trade_history if t.get("status") == "OPEN"]),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "total_pnl": sum(t.get("pnl_usd", 0) for t in closed),
            "avg_win": sum(t["pnl_usd"] for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t["pnl_usd"] for t in losses) / len(losses) if losses else 0,
        }

    def _get_spx_price(self) -> Optional[float]:
        try:
            import yfinance as yf
            spx = yf.Ticker("^GSPC")
            hist = spx.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        return None
