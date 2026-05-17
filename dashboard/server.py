"""Dashboard FastAPI server — REST API + WebSocket for live trading state."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse

from config import (
    DASHBOARD_PORT, DB_DIR, ONLINE_LEARNING_ENABLED,
    IN_MARKET_ENABLED, CRYPTO_ENABLED, POLYMARKET_ENABLED,
    TREND_FILTER_ENABLED,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Trading Brain Dashboard", version="0.1.0")

STATE_FILE = DB_DIR / "brain_state.json"
_start_time = time.time()

# WebSocket clients
_ws_clients: list[WebSocket] = []


# ── helpers ──────────────────────────────────────────────────────────

def _load_brain_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _query_supabase(table: str, **kwargs) -> list[dict]:
    """Safe Supabase query — returns empty list on failure.

    Converts string filters (\"field.eq.value\") to dict format rest_get expects.
    """
    try:
        from db.client import rest_get, table_exists
        if not table_exists(table):
            return []
        select = kwargs.pop("select", "*")

        # Convert string filter to dict: \"field.eq.value\" → {\"field\": \"value\"}
        filters_str = kwargs.pop("filters", None)
        if isinstance(filters_str, str):
            parts = filters_str.split(".eq.", 1)
            if len(parts) == 2:
                kwargs["filters"] = {parts[0]: parts[1]}

        return rest_get(table, select=select, **kwargs) or []
    except Exception as e:
        logger.warning(f"Supabase query failed ({table}): {e}")
        return []


def _collect_dashboard_state() -> dict:
    """Collect live dashboard state from brain_state.json + Supabase."""
    brain = _load_brain_state()

    # Latest snapshot from brain state
    last_snap = brain.get("last_snapshot", {})

    # Open positions from Supabase trades table
    open_trades = _query_supabase("trades", select="ticker,entry_price,entry_date,pnl_pct,composite_score,position_size_pct", filters="status.eq.OPEN", limit=20)

    # Recent closed trades
    closed_trades = _query_supabase("trades", select="ticker,pnl_usd,pnl_pct,exit_date,exit_reason", filters="status.eq.CLOSED", order="exit_date.desc", limit=10)

    # Latest signals
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    latest_signals = _query_supabase("daily_signals", select="ticker,composite_score,score_1_10,momentum_percentile,low_vol_percentile,regime_id", filters=f"date.eq.{today}", order="composite_score.desc", limit=10)

    # Recent decay alerts
    alerts = _query_supabase("decay_alerts", select="*", filters="acknowledged.eq.false", order="timestamp.desc", limit=5)

    # Factor performance
    factor_perf = _query_supabase("factor_performance", select="factor_name,ic_60d,hit_rate_60d,weight", filters=f"date.eq.{today}", limit=10)

    # Regime
    regime = _query_supabase("regime_history", select="*", order="date.desc", limit=1)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": time.time() - _start_time,
        "online_learning_enabled": ONLINE_LEARNING_ENABLED,
        "trend_filter_enabled": TREND_FILTER_ENABLED,
        "markets_active": _active_markets(),
        "equity": last_snap.get("equity", 0),
        "cash": last_snap.get("cash", 0),
        "daily_pnl": last_snap.get("daily_pnl", 0),
        "daily_pnl_pct": last_snap.get("daily_pnl_pct", 0),
        "cumulative_return_pct": last_snap.get("cumulative_return_pct", 0),
        "num_positions": last_snap.get("num_positions", 0),
        "spx_return_pct": last_snap.get("spx_return_pct", 0),
        "open_positions": open_trades,
        "recent_trades": closed_trades,
        "signals": latest_signals,
        "alerts": alerts,
        "factor_performance": factor_perf,
        "regime": regime[0] if regime else {},
        "weights": brain.get("last_retrain_report", {}).get("best_sharpe", {}),
        "last_scoring": brain.get("last_scoring_time"),
        "last_trades_executed": brain.get("last_trades"),
    }


def _active_markets() -> list[str]:
    markets = ["US"]
    if IN_MARKET_ENABLED:
        markets.append("IN")
    if CRYPTO_ENABLED:
        markets.append("CRYPTO")
    if POLYMARKET_ENABLED:
        markets.append("POLYMARKET")
    return markets


# ── REST endpoints ───────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "AI Trading Brain",
        "version": "0.4.0",
        "endpoints": {
            "state": "/api/state",
            "equity": "/api/equity",
            "trades": "/api/trades",
            "signals": "/api/signals",
            "regime": "/api/regime",
            "alerts": "/api/alerts",
            "migration": "/api/migration",
            "ws": "/ws",
        },
        "uptime_seconds": time.time() - _start_time,
    }

@app.get("/health")
async def health():
    return {"status": "ok", "uptime_seconds": time.time() - _start_time}


@app.get("/api/state")
async def api_state():
    return _collect_dashboard_state()


@app.get("/api/equity")
async def api_equity(days: int = Query(default=90, ge=1, le=3650)):
    rows = _query_supabase("portfolio_snapshots", select="date,equity,cash,daily_pnl,daily_pnl_pct,cumulative_return_pct,spx_return_pct", order="date.desc", limit=days)
    return list(reversed(rows))  # chronological order


@app.get("/api/trades")
async def api_trades(limit: int = Query(default=50, ge=1, le=500), status: Optional[str] = Query(default=None)):
    kwargs = {"order": "entry_date.desc", "limit": limit}
    if status:
        kwargs["filters"] = f"status.eq.{status}"
    return _query_supabase("trades", select="*", **kwargs)


@app.get("/api/signals")
async def api_signals(date: Optional[str] = Query(default=None), limit: int = Query(default=50, ge=1, le=500)):
    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _query_supabase("daily_signals", select="*", filters=f"date.eq.{date}", order="composite_score.desc", limit=limit)


@app.get("/api/regime")
async def api_regime(days: int = Query(default=90, ge=1, le=3650)):
    rows = _query_supabase("regime_history", select="*", order="date.desc", limit=days)
    return list(reversed(rows))


@app.get("/api/alerts")
async def api_alerts():
    return _query_supabase("decay_alerts", select="*", filters="acknowledged.eq.false", order="timestamp.desc", limit=20)


@app.get("/api/migration")
async def api_migration():
    """Return migration gate status from brain state."""
    brain = _load_brain_state()
    gate_history = brain.get("migration_gate_history", [])
    ramp_state = brain.get("migration_ramp", {})

    # Compute current gates if we have trade data
    gates = {"consistency": "UNKNOWN", "drawdown": "UNKNOWN", "stability": "UNKNOWN", "ready": False, "days_green": 0}
    try:
        trades = _query_supabase("trades", select="pnl_pct,entry_date,exit_date", filters="status.eq.CLOSED", order="exit_date.desc", limit=500)
        snapshots = _query_supabase("portfolio_snapshots", select="date,equity", order="date.desc", limit=120)

        if trades and snapshots:
            from migration.live_gate import LiveGate
            gate = LiveGate()
            result = gate.evaluate(trades, snapshots)
            gates = {
                "consistency": result.consistency,
                "drawdown": result.drawdown,
                "stability": result.stability,
                "ready": result.all_green,
                "days_green": 0,
            }

            # Count consecutive green days from history
            if result.all_green and gate_history:
                consecutive = 0
                for h in reversed(gate_history):
                    if h.get("all_green"):
                        consecutive += 1
                    else:
                        break
                gates["days_green"] = consecutive
    except Exception as e:
        logger.warning(f"Migration gate computation failed: {e}")

    return {
        "gates": gates,
        "ramp": ramp_state,
    }


# ── WebSocket ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    logger.info(f"WebSocket client connected ({len(_ws_clients)} total)")

    try:
        while True:
            try:
                state = _collect_dashboard_state()
                await ws.send_json(state)
            except Exception as e:
                logger.error(f"WebSocket send failed: {e}")

            # Check if client sent anything (ping/pong)
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
                if data == "ping":
                    await ws.send_text("pong")
            except asyncio.TimeoutError:
                pass  # normal — no client message
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("WebSocket error")
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ── startup / shutdown ───────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info(f"Dashboard server starting on port {DASHBOARD_PORT}")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Dashboard server shutting down")
    for ws in _ws_clients:
        try:
            await ws.close()
        except Exception:
            pass
    _ws_clients.clear()
