"""AI Trading Brain — main daemon entry point.

Run: python brain.py

Daily routines:
  1. Pre-Market (6:07AM ET): Data Ingestor -> Signal Aggregator -> Regime Detection
  2. Market Open (9:37AM ET): Risk check -> Top-30 -> Execution (Phase 2)
  3. Intraday (every 15min): Position monitor, stop-loss checks (Phase 2)
  4. Market Close (4:03PM ET): P&L calc, factor attribution, journal
  5. Friday 4:07PM ET: Weekend retraining (Phase 4+)
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import (
    DRY_RUN, TRADE_ENABLED, DB_DIR, SIGNAL_TOP_N, MAX_POSITION_PCT, DASHBOARD_PORT,
    ONLINE_LEARNING_ENABLED, TREND_FILTER_ENABLED,
    IN_MARKET_ENABLED, IN_UNIVERSE_SIZE, CRYPTO_ENABLED, CRYPTO_UNIVERSE_SIZE,
    CRYPTO_TOP_N, POLYMARKET_ENABLED, POLYMARKET_MAX_PER_MARKET,
    SENTIMENT_ENABLED, SENTIMENT_SCAN_INTERVAL_MINUTES,
    POLYMARKET_SCAN_INTERVAL_MINUTES,
    OPTION_SENTIMENT_ENABLED, OPTION_SENTIMENT_MAX_TICKERS,
)
from scheduler import BrainScheduler, get_ny_time

# Setup logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
STATE_FILE = DB_DIR / "brain_state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "brain.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("brain")


# --- State helpers ---

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, default=str, indent=2))


# --- Routine 1: Pre-Market Signal Computation (6:07 AM ET) ---

def run_daily_scoring():
    """Pre-market signal computation. Stores top signals to state file for market-open execution."""
    start = time.perf_counter()
    logger.info("=== Daily Signal Computation Starting ===")

    try:
        from ingestion.universe import us_tickers
        from ingestion.macro_data import fetch_macro_snapshot
        from ingestion.market_data import build_snapshot, compute_all_atrs
        from signals.aggregator import compute_signals
        from config import ATR_PERIOD
        from db.client import rest_upsert, table_exists

        macro = fetch_macro_snapshot()
        logger.info(f"Macro: VIX={macro.get('vix')}, spx_vs_200ma={macro.get('spx_vs_200ma', 0):.1%}")

        us_ticker_list = us_tickers()
        logger.info(f"US universe: {len(us_ticker_list)} tickers")

        snapshot = build_snapshot(us_ticker_list[:200], "US")
        us_scores = compute_signals(snapshot.prices, snapshot.fundamentals, macro, "US")

        # Compute ATR for volatility-adjusted exits
        atr_values = compute_all_atrs(snapshot.prices, ATR_PERIOD)
        atr_count = sum(1 for v in atr_values.values() if v is not None and v > 0)
        logger.info(f"ATR computed: {atr_count}/{len(atr_values)} tickers have valid ATR")

        top_us = us_scores.head(SIGNAL_TOP_N)
        logger.info(
            f"US top-5: {', '.join(f'{r.ticker}={r.composite_score:.3f}' for _, r in top_us.head(5).iterrows())}"
        )

        # Persist to Supabase
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = []
        for _, row in us_scores.iterrows():
            rows.append({
                "ticker": row["ticker"],
                "market": "US",
                "date": today,
                "composite_score": float(row["composite_score"]),
                "score_1_10": int(max(1, min(10, round(row["composite_score"] * 9 + 1)))),
                "quality_percentile": float(row.get("quality_score", 0.5)),
                "momentum_percentile": float(row.get("momentum_score", 0.5)),
                "value_percentile": float(row.get("value_score", 0.5)),
                "low_vol_percentile": float(row.get("low_vol_score", 0.5)),
                "vol_rank_percentile": float(row.get("vol_rank_score", 0.5)),
                "short_interest_percentile": 0.5,
                "regime_id": int(row.get("regime_id", 2)),
                "regime_confidence": float(row.get("regime_confidence", 0.5)),
            })

        # Save top signals to state file first (survives DB failures)
        top_list = top_us[["ticker", "composite_score", "quality_score", "momentum_score",
                            "value_score", "low_vol_score", "vol_rank_score", "regime_id"]].to_dict(orient="records")
        state = _load_state()
        state["latest_signals"] = top_list
        state["last_scoring_time"] = datetime.now(timezone.utc).isoformat()
        state["last_atr_values"] = {k: v for k, v in atr_values.items() if v is not None}
        _save_state(state)
        logger.info(f"State saved: {len(top_list)} signals, {len(state['last_atr_values'])} ATRs")

        # Persist to Supabase (non-fatal if fails)
        if table_exists("daily_signals"):
            try:
                rest_upsert("daily_signals", rows)
                logger.info(f"Saved {len(rows)} US signals to daily_signals")
            except Exception as e:
                logger.warning(f"Supabase upsert failed (non-fatal): {e}")

        elapsed = time.perf_counter() - start
        logger.info(f"Daily scoring complete in {elapsed:.1f}s ({len(rows)} tickers)")

    except Exception:
        logger.exception("Daily scoring failed")


# --- Routine 2: Market Open Execution (9:37 AM ET) ---

def run_market_open_execution():
    """Execute trades at market open: risk check -> top signals -> 7-agent debate -> buy with bracket orders."""
    logger.info("=== Market Open Execution (Phase 3: PARA-DEBATE Enhanced) ===")

    try:
        from execution.engine import ExecutionEngine
        from execution.alpaca_client import AlpacaClient
        from risk.manager import RiskManager
        from paper.pnl import PnLTracker
        from strategy.decider import StrategyDecider
        from ingestion.macro_data import fetch_macro_snapshot
        from ingestion.market_data import fetch_fundamentals
        import pandas as pd

        client = AlpacaClient()
        risk = RiskManager(client)
        engine = ExecutionEngine()
        pnl = PnLTracker(client)
        decider = StrategyDecider()

        # 1. Check market hours
        if not client.is_market_open():
            logger.info("Market closed — skipping execution")
            return

        # 1.5. Trend filter: block entries when SPX below 200MA
        if TREND_FILTER_ENABLED:
            macro = fetch_macro_snapshot()
            spx_vs_ma = macro.get("spx_vs_200ma", 0.02)
            if spx_vs_ma < 0:
                logger.warning(f"SPX below 200MA ({spx_vs_ma:.2%}) — skipping all entries")
                return

        # 2. Risk check
        state = risk.check_pre_market()
        if not risk.can_trade:
            logger.warning(f"Trading halted: {state.level.value} — {state.alerts}")
            return

        logger.info(f"Risk check: {state.level.value} | Drawdown: {state.current_drawdown_pct:.2%}")

        # 2.5. Trade enabled gate
        if not TRADE_ENABLED:
            logger.info("TRADE_ENABLED=false — skipping execution")
            return

        # 3. Load signals from state file
        stored = _load_state()
        signals_list = stored.get("latest_signals", [])

        if not signals_list:
            logger.warning("No signals found — run daily_scoring first")
            return

        signals_df = pd.DataFrame(signals_list)
        top_n = min(SIGNAL_TOP_N, len(signals_df))
        top_tickers = signals_df.head(top_n)
        logger.info(f"Top {top_n} signals: {', '.join(top_tickers['ticker'].values)}")

        # 4. Enrich each ticker with full context
        macro = fetch_macro_snapshot()
        contexts = {}
        fundamentals_map: dict[str, dict] = {}
        for _, row in top_tickers.iterrows():
            ticker = row["ticker"]
            try:
                fund = fetch_fundamentals(ticker)
            except Exception:
                fund = {}
            fundamentals_map[ticker] = fund

            context = {
                "ticker": ticker,
                "composite_score": float(row.get("composite_score", 0.5)),
                "quality_percentile": float(row.get("quality_score", 0.5)),
                "momentum_percentile": float(row.get("momentum_score", 0.5)),
                "value_percentile": float(row.get("value_score", 0.5)),
                "low_vol_percentile": float(row.get("low_vol_score", 0.5)),
                "vol_rank_percentile": float(row.get("vol_rank_score", 0.5)),
                "regime_id": int(row.get("regime_id", 2)),
                "regime_label": {1: "Risk-On", 2: "Late-Cycle", 3: "Bear", 4: "Recovery"}.get(int(row.get("regime_id", 2)), "Late-Cycle"),
                **macro,
                **fund,
            }
            contexts[ticker] = context

        # 4.5. Enrich contexts with X sentiment data (Phase 6 extension)
        sentiment_data = stored.get("last_sentiment_scan", {})
        if sentiment_data:
            sentiment_signals = {
                s["ticker"]: s
                for s in sentiment_data.get("ticker_signals", [])
            }
            for ticker, ctx in contexts.items():
                ss = sentiment_signals.get(ticker)
                if ss:
                    ctx["x_sentiment"] = (
                        f"{ss['sentiment']} ({ss['bullish_count']}B/{ss['bearish_count']}S, "
                        f"net={ss['net_score']:.3f}, conf={ss['avg_confidence']:.2f}, "
                        f"high_mat={ss['high_mat_count']})"
                    )
                    ctx["x_sentiment_label"] = ss["sentiment"]
                    ctx["x_sentiment_score"] = ss["net_score"]

        # 4.6. Enrich with put/call ratio options sentiment (live-only)
        if OPTION_SENTIMENT_ENABLED:
            try:
                from sentiment.option_scanner import OptionSentimentScanner
                pcr_scanner = OptionSentimentScanner(max_tickers=OPTION_SENTIMENT_MAX_TICKERS)
                pcr_data = pcr_scanner.fetch_put_call_ratios(list(top_tickers["ticker"].values))
                pcr_context = pcr_scanner.format_sentiment_context(pcr_data)
                logger.info(f"Option PCR: {pcr_context}")
                for ticker, ctx in contexts.items():
                    pcr = pcr_data.get(ticker, {})
                    if pcr.get("pcr") is not None:
                        ctx["pcr_ratio"] = pcr["pcr"]
                        ctx["pcr_sentiment"] = pcr["sentiment"]
                        ctx["pcr_context"] = (
                            f"PCR={pcr['pcr']:.3f} ({pcr['sentiment']}), "
                            f"put_vol={pcr['put_vol']:.0f}, call_vol={pcr['call_vol']:.0f}"
                        )
            except Exception as e:
                logger.warning(f"Option sentiment enrichment failed: {e}")

        # 5. Run 7-agent debate on each ticker
        logger.info(f"Running PARA-DEBATE on {len(top_tickers)} tickers...")
        debate_results = decider.debate_batch(
            list(top_tickers["ticker"].values), "US", contexts
        )

        # Log debate results
        for dr in debate_results:
            logger.info(f"  {dr.ticker}: {dr.verdict} (consensus={dr.consensus_score:.2f}, {dr.latency_ms:.0f}ms)")
            if dr.error:
                logger.error(f"    Error: {dr.error}")

        # 6. Filter to BUY/STRONG BUY only
        buy_signals = []
        skipped = []
        for dr in debate_results:
            if dr.verdict in ("STRONG BUY", "BUY"):
                buy_signals.append(dr.ticker)
            else:
                skipped.append(f"{dr.ticker}={dr.verdict}")

        if skipped:
            logger.info(f"Skipped (non-BUY verdicts): {', '.join(skipped)}")

        if not buy_signals:
            logger.info("No BUY verdicts — no trades executed")
            # Still persist debate records
            stored["last_debates"] = [
                {
                    "ticker": dr.ticker, "verdict": dr.verdict,
                    "consensus_score": dr.consensus_score,
                    "investment_thesis": dr.investment_thesis[:300],
                    "debate_json": dr.debate_json,
                }
                for dr in debate_results
            ]
            _save_state(stored)
            return

        # 7. Execute buys for BUY-verdict signals
        buy_df = signals_df[signals_df["ticker"].isin(buy_signals)]
        logger.info(f"Executing buys for: {buy_signals}")

        atr_values = stored.get("last_atr_values", {})

        acc = client.get_account()
        trades = engine.execute_market_open(
            buy_df, acc.equity,
            fundamentals=fundamentals_map,
            atr_values=atr_values,
        )

        # 8. Attach debate data to trades
        debate_map = {dr.ticker: dr for dr in debate_results}
        for t in trades:
            dr = debate_map.get(t["ticker"])
            if dr:
                t["debate_json"] = dr.debate_json
                t["agent_verdict"] = dr.verdict

        # 9. Record & persist
        if trades:
            for t in trades:
                pnl.record_trade(t)
            pnl.persist_position_pnl(trades)

        # 10. Update state
        stored["last_trades"] = trades
        stored["last_debates"] = [
            {
                "ticker": dr.ticker, "verdict": dr.verdict,
                "consensus_score": dr.consensus_score,
                "investment_thesis": dr.investment_thesis[:300],
                "debate_json": dr.debate_json,
            }
            for dr in debate_results
        ]
        _save_state(stored)

        logger.info(f"Executed {len(trades)} trades from {len(buy_signals)} BUY signals")

    except Exception:
        logger.exception("Market open execution failed")


# --- Routine 3: Intraday Monitor (every 15 min, 9:45AM-4PM ET) ---

_decay_detector = None  # Singleton DecayDetector, initialized on first use


def _get_decay_detector():
    global _decay_detector
    if _decay_detector is None:
        from learning.decay import DecayDetector
        _decay_detector = DecayDetector()
    return _decay_detector


def run_intraday_monitor():
    """Intraday check: position P&L, stop-loss triggers, drawdown, decay detection."""
    try:
        from execution.alpaca_client import AlpacaClient
        from execution.engine import ExecutionEngine
        from risk.manager import RiskManager
        from db.client import rest_upsert, table_exists

        client = AlpacaClient()
        risk = RiskManager(client)
        engine = ExecutionEngine()

        if not client.is_market_open():
            return

        # Reconcile: detect positions closed by Alpaca bracket orders
        try:
            if table_exists("trades"):
                closed = engine.reconcile_positions()
                if closed:
                    rest_upsert("trades", closed)
                    logger.info(f"Reconciled {len(closed)} closed positions from Alpaca")
        except Exception:
            logger.exception("Position reconciliation failed")

        state = risk.check_intraday()

        if state.level.value == "RED":
            logger.critical(f"INTRADAY RED ALERT: {state.alerts}")
            # Emergency: close all positions
            positions = client.get_positions()
            for p in positions:
                try:
                    result = engine.execute_sell(p.symbol, reason=f"RISK_RED: {state.alerts}")
                    if result:
                        logger.info(f"Emergency sell: {p.symbol}")
                except Exception:
                    logger.exception(f"Emergency sell failed: {p.symbol}")

        elif state.alerts:
            logger.warning(f"Intraday alerts ({state.level.value}): {state.alerts}")

        # Decay detection — check last 20 trades for strategy deterioration
        try:
            from db.client import rest_get, table_exists
            if table_exists("trades"):
                recent = rest_get("trades", select="ticker,pnl_pct,composite_score,exit_date", order="exit_date.desc", limit=20)
                if recent and len(recent) >= 10:
                    detector = _get_decay_detector()
                    signal_scores = [t.get("composite_score", 0.5) or 0.5 for t in recent]
                    decay_state = detector.check(recent, signal_scores=signal_scores)
                    if decay_state.level.value != "GREEN":
                        logger.warning(
                            f"Decay alert ({decay_state.level.value}): "
                            f"{decay_state.alert_count}/3 detectors triggered — {decay_state.details}"
                        )
        except Exception:
            pass  # Non-critical, don't block intraday monitor

        # Log current positions
        positions = client.get_positions()
        if positions:
            for p in positions:
                if abs(p.unrealized_pnl_pct) > 0.01:
                    logger.info(f"  {p.symbol}: {p.unrealized_pnl_pct:+.2%} (${p.unrealized_pnl:+.2f})")

    except Exception:
        logger.exception("Intraday monitor failed")


# --- Routine 4: Market Close Journal (4:03 PM ET) ---

def run_market_close_journal():
    """Market close: snapshot portfolio, compute daily P&L, save journal."""
    logger.info("=== Market Close Journal ===")
    try:
        from ingestion.macro_data import fetch_macro_snapshot
        from execution.alpaca_client import AlpacaClient
        from execution.engine import ExecutionEngine
        from paper.pnl import PnLTracker
        from db.client import rest_upsert, table_exists

        client = AlpacaClient()
        pnl = PnLTracker(client)
        macro = fetch_macro_snapshot()

        # Reconcile: detect positions closed by Alpaca during the day
        try:
            if table_exists("trades"):
                engine = ExecutionEngine()
                closed = engine.reconcile_positions()
                if closed:
                    rest_upsert("trades", closed)
                    logger.info(f"EOD reconcile: {len(closed)} positions closed by Alpaca")
        except Exception:
            logger.exception("EOD reconciliation failed")

        # Take current snapshot
        snap = pnl.snapshot()
        if not snap:
            return

        # Load previous snapshot for P&L computation
        stored = _load_state()
        prev = stored.get("last_snapshot")
        snap = pnl.compute_daily_pnl(prev, snap)
        pnl.persist_snapshot(snap)

        # Save to state
        stored["last_snapshot"] = snap
        _save_state(stored)

        # Trade summary
        summary = pnl.get_trade_summary()
        logger.info(
            f"Portfolio: ${snap['equity']:,.2f} | "
            f"Positions: {snap['num_positions']} | "
            f"Daily P&L: ${snap['daily_pnl']:+,.2f} ({snap['daily_pnl_pct']:+.2%}) | "
            f"SPX: {snap.get('spx_return_pct', 0):+.2%}"
        )
        if summary.get("closed_trades", 0) > 0:
            logger.info(f"Trades: {summary['win_count']}W/{summary['loss_count']}L "
                        f"(WR: {summary['win_rate']:.0%}) | Total P&L: ${summary['total_pnl']:+,.2f}")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Phase 4: Feedback loop — learn from closed trades
        try:
            from db.client import rest_get, table_exists as te2
            from learning.feedback import FeedbackLoop

            if te2("trades"):
                today_closed = rest_get("trades",
                    select="*",
                    filters={"exit_date": today},
                    limit=50)
                if today_closed:
                    feedback = FeedbackLoop()
                    fb_result = feedback.process_closed_trades(today_closed)
                    logger.info(
                        f"Feedback: {fb_result['trades_processed']} trades processed | "
                        f"Decay: {fb_result['decay_level']} | "
                        f"Weights: {fb_result['updated_weights']}"
                    )
        except Exception:
            logger.exception("Feedback loop failed")

        # Save macro to regime_history
        if table_exists("regime_history"):
            rest_upsert("regime_history", [{
                "date": today,
                "vix": macro.get("vix", 18.0),
                "vix_20d_change": macro.get("vix_20d_change", 0.0),
                "spx_vs_200ma": macro.get("spx_vs_200ma", 0.02),
                "hy_spread_oas": macro.get("hy_spread_oas", 350.0),
                "ism_pmi": macro.get("ism_pmi", 51.0),
                "yield_spread_2y10y": macro.get("yield_spread_2y10y", 0.10),
                "cpi_yoy": macro.get("cpi_yoy", 3.0),
                "fed_funds_rate": macro.get("fed_funds_rate", 5.25),
            }], on_conflict="date")

    except Exception:
        logger.exception("Market close journal failed")


# --- Routine 5: Historical Replay (market-closed learning, Phase 4) ---

def run_historical_replay():
    """When market is closed, trade on historical data to learn and improve."""
    try:
        from execution.alpaca_client import AlpacaClient

        client = AlpacaClient()
        if client.is_market_open():
            return  # Live trading has priority

        from learning.historical_replay import HistoricalReplay

        replay = HistoricalReplay()
        stats = replay.run(max_days=30)  # Process 30 days per batch

        if stats.days_processed > 0:
            wr = stats.wins / max(1, stats.trades_taken)
            logger.info(
                f"Replay: {stats.days_processed} days, {stats.trades_taken} trades, "
                f"WR={wr:.0%}, Equity=${stats.current_equity:,.0f} "
                f"({(stats.current_equity / 100000 - 1) * 100:+.1f}%), "
                f"DD={stats.current_drawdown:.1%}"
            )

    except Exception:
        logger.exception("Historical replay failed")


# --- Routine 6: Weekend Retraining (Friday 4:07 PM ET, Phase 4) ---

def run_weekend_retraining():
    """Weekend retraining — 7-stage pipeline (Phase 4)."""
    logger.info("=== Weekend Retraining Pipeline (Phase 4) ===")
    try:
        from learning.retrainer import WeekendRetrainer

        retrainer = WeekendRetrainer()
        report = retrainer.run()

        logger.info(f"Retraining complete: {len(report.stages_completed)}/7 stages in {report.total_duration_seconds:.0f}s")
        if report.stages_failed:
            logger.warning(f"Failed stages: {report.stages_failed}")
        if report.errors:
            for e in report.errors:
                logger.error(f"  Retrain error: {e}")

        # Persist report summary to state file
        stored = _load_state()
        stored["last_retrain_report"] = {
            "timestamp": report.timestamp,
            "stages_completed": report.stages_completed,
            "stages_failed": report.stages_failed,
            "hmm_ic": report.hmm_ic,
            "lgbm_icir": report.lgbm_icir,
            "best_sharpe": report.best_sharpe,
            "champion_sharpe": report.champion_sharpe,
            "challenger_sharpe": report.challenger_sharpe,
            "total_duration_seconds": report.total_duration_seconds,
        }
        _save_state(stored)

    except Exception:
        logger.exception("Weekend retraining failed")


# --- Routine 7: India Daily Scoring (Phase 6) ---

def run_daily_scoring_india():
    """Pre-market signal computation for Indian stocks (Nifty 200)."""
    if not IN_MARKET_ENABLED:
        return
    start = time.perf_counter()
    logger.info("=== India Daily Signal Computation ===")
    try:
        from ingestion.universe import in_tickers
        from ingestion.macro_data import fetch_macro_snapshot
        from ingestion.market_data import build_snapshot
        from signals.aggregator import compute_signals
        from db.client import rest_upsert, table_exists

        macro = fetch_macro_snapshot()
        in_list = in_tickers()
        snapshot = build_snapshot(in_list[:IN_UNIVERSE_SIZE], "IN")
        scores = compute_signals(snapshot.prices, snapshot.fundamentals, macro, "IN")

        top_in = scores.head(SIGNAL_TOP_N)
        logger.info(f"IN top-5: {', '.join(f'{r.ticker}={r.composite_score:.3f}' for _, r in top_in.head(5).iterrows())}")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = []
        for _, row in scores.iterrows():
            rows.append({
                "ticker": row["ticker"],
                "market": "IN",
                "date": today,
                "composite_score": float(row["composite_score"]),
                "score_1_10": int(max(1, min(10, round(row["composite_score"] * 9 + 1)))),
                "quality_percentile": float(row.get("quality_score", 0.5)),
                "momentum_percentile": float(row.get("momentum_score", 0.5)),
                "value_percentile": float(row.get("value_score", 0.5)),
                "low_vol_percentile": float(row.get("low_vol_score", 0.5)),
                "short_interest_percentile": 0.5,
                "regime_id": int(row.get("regime_id", 2)),
                "regime_confidence": float(row.get("regime_confidence", 0.5)),
            })

        if table_exists("daily_signals"):
            rest_upsert("daily_signals", rows)
            logger.info(f"Saved {len(rows)} IN signals to daily_signals")

        elapsed = time.perf_counter() - start
        logger.info(f"India scoring complete in {elapsed:.1f}s")
    except Exception:
        logger.exception("India daily scoring failed")


# --- Routine 8: India Market Open Execution (Phase 6) ---

def run_market_open_execution_india():
    """Execute India trades — dry-run only (no India broker connected yet)."""
    if not IN_MARKET_ENABLED:
        return
    logger.info("=== India Market Open Execution (Dry-Run Only) ===")
    try:
        stored = _load_state()
        signals_list = stored.get("latest_signals_in", [])
        if not signals_list:
            logger.info("No India signals — skipping execution")
            return

        import pandas as pd
        signals_df = pd.DataFrame(signals_list)
        top_n = min(SIGNAL_TOP_N, len(signals_df))
        logger.info(f"India top {top_n} signals (dry-run): {', '.join(signals_df.head(top_n)['ticker'].values)}")
        logger.info("India execution: DRY-RUN ONLY — no broker connected")
    except Exception:
        logger.exception("India market open execution failed")


# --- Routine 9: India Market Close Journal (Phase 6) ---

def run_market_close_journal_india():
    """India market close — placeholder for future broker integration."""
    if not IN_MARKET_ENABLED:
        return
    logger.info("=== India Market Close Journal (Placeholder) ===")


# --- Routine 10: Crypto Daily Scoring (Phase 6) ---

def run_crypto_scoring():
    """Compute crypto signals and persist."""
    if not CRYPTO_ENABLED:
        return
    start = time.perf_counter()
    logger.info("=== Crypto Signal Computation ===")
    try:
        from ingestion.universe import crypto_tickers
        from ingestion.macro_data import fetch_macro_snapshot
        from ingestion.market_data import fetch_prices
        from signals.aggregator import compute_crypto_signals
        from db.client import rest_upsert, table_exists

        macro = fetch_macro_snapshot()
        tickers = crypto_tickers()[:CRYPTO_UNIVERSE_SIZE]
        prices = fetch_prices(tickers, period="6mo")

        scores = compute_crypto_signals(prices, macro)
        top_crypto = scores.head(CRYPTO_TOP_N)
        logger.info(f"Crypto top-{CRYPTO_TOP_N}: {', '.join(f'{r.ticker}={r.composite_score:.3f}' for _, r in top_crypto.iterrows())}")

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = []
        for _, row in scores.iterrows():
            rows.append({
                "ticker": row["ticker"],
                "market": "CRYPTO",
                "date": today,
                "composite_score": float(row["composite_score"]),
                "score_1_10": int(max(1, min(10, round(row["composite_score"] * 9 + 1)))),
                "momentum_percentile": float(row.get("momentum_score", 0.5)),
                "low_vol_percentile": float(row.get("low_vol_score", 0.5)),
                "quality_percentile": 0.5,
                "value_percentile": 0.5,
                "short_interest_percentile": 0.5,
                "regime_id": int(row.get("regime_id", 2)),
                "regime_confidence": 0.5,
            })

        if table_exists("daily_signals"):
            rest_upsert("daily_signals", rows)
            logger.info(f"Saved {len(rows)} crypto signals")

        elapsed = time.perf_counter() - start
        logger.info(f"Crypto scoring complete in {elapsed:.1f}s")
    except Exception:
        logger.exception("Crypto scoring failed")


# --- Routine 11: Crypto Monitor (Phase 6) ---

def run_crypto_monitor():
    """Periodic crypto position check. Placeholder for Alpaca crypto execution."""
    if not CRYPTO_ENABLED:
        return
    logger.debug("Crypto monitor tick — no active positions")


# --- Routine 12: Polymarket Scan (Phase 6) ---

def run_polymarket_scan():
    """Scan Polymarket for edges using Pipeline V2 classifier + edge detector."""
    if not POLYMARKET_ENABLED:
        return
    logger.debug("=== Polymarket Edge Scan ===")
    try:
        from execution.polymarket_bridge import PolymarketBridge
        bridge = PolymarketBridge()
        trades = bridge.scan_and_trade(POLYMARKET_MAX_PER_MARKET)
        if trades:
            logger.info(f"Polymarket scan: {len(trades)} trades executed")
            for t in trades:
                logger.info(f"  {t.get('slug', '?')}: {t.get('side', '?')} @ ${t.get('price', 0):.4f} — {t.get('reason', '?')}")
    except Exception:
        logger.exception("Polymarket scan failed")


# --- Routine 13: Multi-Source Sentiment Scanner (free: Reddit + Finnhub + RSS) ---

def run_sentiment_scan():
    """Scan Reddit, Finnhub, RSS for financial sentiment using Grox-pattern classifier + Claude Haiku."""
    if not SENTIMENT_ENABLED:
        return
    logger.debug("=== Multi-Source Sentiment Scan ===")
    try:
        from sentiment.multi_scanner import run_scan

        scan = run_scan()
        if scan.errors:
            for e in scan.errors:
                logger.warning(f"Sentiment scan error: {e}")

        if scan.tweets_scanned == 0:
            logger.debug("Sentiment scan: 0 tweets fetched")
            return

        logger.info(
            f"Sentiment scan: {scan.tweets_scanned} tweets -> "
            f"{scan.bullish_count}B/{scan.bearish_count}S "
            f"({scan.overall_sentiment}), {len(scan.ticker_signals)} ticker signals, "
            f"{scan.latency_ms:.0f}ms"
        )

        # Cross-reference with latest factor signals
        stored = _load_state()
        latest_signals = stored.get("latest_signals", [])
        signal_tickers = {s["ticker"] for s in latest_signals}

        for ts in scan.ticker_signals[:10]:
            ticker = ts["ticker"]
            overlap = " *OVERLAP" if ticker in signal_tickers else ""
            logger.info(
                f"  {ticker}: {ts['sentiment']} (net={ts['net_score']:.3f}, "
                f"{ts['bullish_count']}B/{ts['bearish_count']}S, "
                f"mat={ts['high_mat_count']}, conf={ts['avg_confidence']:.2f}){overlap}"
            )

        # Persist to state file for debate enrichment
        stored["last_sentiment_scan"] = {
            "timestamp": scan.timestamp,
            "tweets_scanned": scan.tweets_scanned,
            "overall_sentiment": scan.overall_sentiment,
            "bullish_count": scan.bullish_count,
            "bearish_count": scan.bearish_count,
            "ticker_signals": scan.ticker_signals,
            "ticker_sentiment": {
                t: {k: v for k, v in d.items() if k != "tweets"}
                for t, d in scan.ticker_sentiment.items()
            },
            "latency_ms": scan.latency_ms,
        }
        _save_state(stored)

    except Exception:
        logger.exception("Sentiment scan failed")


# --- Daemon setup and main ---

def setup_signal_handlers(loop: asyncio.AbstractEventLoop, scheduler: BrainScheduler):
    def shutdown(sig_name: str):
        logger.info(f"Received {sig_name}, shutting down...")
        scheduler.stop()
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: shutdown(s.name))
        except NotImplementedError:
            pass


def _check_port_available(port: int):
    """Raise RuntimeError if port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            logger.error(f"Port {port} already in use — is another daemon running?")
            raise RuntimeError(
                f"Port {port} is already occupied. Stop the existing daemon first."
            )


def main():
    logger.info("=" * 60)
    logger.info("AI Trading Brain — Starting Daemon (Phase 5)")
    logger.info(f"  Dry Run: {DRY_RUN}")
    logger.info(f"  Trade Enabled: {TRADE_ENABLED}")
    logger.info(f"  Online Learning: {ONLINE_LEARNING_ENABLED}")
    logger.info(f"  Dashboard: http://0.0.0.0:{DASHBOARD_PORT}")
    logger.info(f"  NY Time: {get_ny_time().strftime('%Y-%m-%d %H:%M %Z')}")
    logger.info("=" * 60)

    # Guard: check if dashboard port is already in use
    _check_port_available(DASHBOARD_PORT)

    # Setup scheduler FIRST (before any blocking calls)
    sched = BrainScheduler()

    # Phase 1 routines
    sched.add_daily("daily_scoring", run_daily_scoring, hour=6, minute=7)

    # Phase 2 routines
    sched.add_daily("market_open", run_market_open_execution, hour=9, minute=37)
    sched.add_interval("intraday_monitor", run_intraday_monitor, minutes=15)
    sched.add_daily("market_close", run_market_close_journal, hour=16, minute=3)

    # Phase 4
    sched.add_interval("historical_replay", run_historical_replay, minutes=20)
    sched.add_weekly("weekend_retraining", run_weekend_retraining, day_of_week="fri", hour=16, minute=7)

    # Phase 6: Multi-Asset
    if IN_MARKET_ENABLED:
        sched.add_daily("daily_scoring_in", run_daily_scoring_india, hour=1, minute=37)
        sched.add_daily("market_open_in", run_market_open_execution_india, hour=4, minute=7)
        sched.add_daily("market_close_in", run_market_close_journal_india, hour=10, minute=33)

    if CRYPTO_ENABLED:
        sched.add_daily("crypto_scoring", run_crypto_scoring, hour=6, minute=22)
        sched.add_interval("crypto_monitor", run_crypto_monitor, minutes=240)

    if POLYMARKET_ENABLED:
        sched.add_interval("polymarket_scan", run_polymarket_scan, minutes=POLYMARKET_SCAN_INTERVAL_MINUTES)

    if SENTIMENT_ENABLED:
        sched.add_interval("sentiment_scan", run_sentiment_scan, minutes=SENTIMENT_SCAN_INTERVAL_MINUTES)

    sched.start()
    logger.info("Scheduler started:")
    logger.info("  6:07 AM ET  — daily signal computation")
    logger.info("  9:37 AM ET  — market open execution (risk check + debate + buys)")
    logger.info("  every 15min — intraday monitor + stop-loss + decay detection")
    logger.info("  4:03 PM ET  — market close journal + P&L + feedback loop")
    logger.info("  every 20min — historical replay (market-closed learning)")
    logger.info("  Fri 4:07PM  — weekend retraining (7-stage pipeline)")
    if IN_MARKET_ENABLED:
        logger.info("  India: 6:07a/9:37a/4:03p IST  — IN daily scoring + execution")
    if CRYPTO_ENABLED:
        logger.info("  Crypto: 6:22a daily + every 4hr — crypto scoring + monitor")
    if POLYMARKET_ENABLED:
        logger.info(f"  Polymarket: every {POLYMARKET_SCAN_INTERVAL_MINUTES}min — edge scan + trade")
    if SENTIMENT_ENABLED:
        logger.info(f"  X Sentiment: every {SENTIMENT_SCAN_INTERVAL_MINUTES}min — financial tweet scanner")

    # Start FastAPI dashboard in background thread (health check available immediately)
    import threading
    import uvicorn
    from dashboard.server import app

    def _start_dashboard():
        config = uvicorn.Config(app, host="0.0.0.0", port=DASHBOARD_PORT, log_level="info")
        server = uvicorn.Server(config)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    dashboard_thread = threading.Thread(target=_start_dashboard, daemon=True, name="dashboard")
    dashboard_thread.start()
    logger.info(f"Dashboard starting on port {DASHBOARD_PORT} (background thread)")

    # Run initial scans in scheduler threads (non-blocking, after health check is live)
    ny_now = get_ny_time()
    premarket_start = ny_now.replace(hour=5, minute=0, second=0, microsecond=0)
    premarket_end = ny_now.replace(hour=9, minute=37, second=0, microsecond=0)
    if premarket_start <= ny_now < premarket_end:
        logger.info("Scheduling immediate daily scoring...")
        sched._scheduler.add_job(
            run_daily_scoring,
            trigger="date",
            run_date=datetime.now(timezone.utc),
            id="initial_scoring",
            misfire_grace_time=300,
        )

    if SENTIMENT_ENABLED:
        logger.info("Scheduling immediate sentiment scan...")
        sched._scheduler.add_job(
            run_sentiment_scan,
            trigger="date",
            run_date=datetime.now(timezone.utc),
            id="initial_sentiment",
            misfire_grace_time=300,
        )

    try:
        # Register signal handlers in main thread (no event loop needed)
        def _shutdown(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            sched.stop()
            import os as _os
            _os._exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        try:
            signal.signal(signal.SIGTERM, _shutdown)
        except (AttributeError, ValueError):
            pass  # SIGTERM not available on Windows

        logger.info(f"Brain daemon running. http://localhost:{DASHBOARD_PORT} | Ctrl+C to stop.")
        # Keep main thread alive while dashboard runs in background
        while dashboard_thread.is_alive():
            dashboard_thread.join(timeout=1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    finally:
        sched.stop()
        logger.info("AI Trading Brain stopped.")


if __name__ == "__main__":
    main()
