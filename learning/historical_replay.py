"""Historical replay engine — brain trades on past data to learn.

When market is closed, this engine:
  1. Steps through historical data day by day
  2. Runs signal computation (same as live)
  3. Simulates trade execution with bracket order logic
  4. Feeds simulated P&L through FeedbackLoop
  5. Checkpoints progress, resumes on restart

Goal: brain gains experience, weights evolve, strategy improves 24/7.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import DB_DIR, SIGNAL_TOP_N, MAX_POSITION_PCT, DAILY_TRADE_CAP, TAKE_PROFIT_PCT, STOP_LOSS_PCT, US_UNIVERSE_SIZE, TREND_FILTER_ENABLED
from learning.weight_optimizer import FactorWeights

logger = logging.getLogger(__name__)

REPLAY_DIR = DB_DIR / "replay"
REPLAY_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = REPLAY_DIR / "checkpoint.json"
SIMULATED_TRADES_FILE = REPLAY_DIR / "simulated_trades.json"
SIMULATED_STATE_FILE = REPLAY_DIR / "simulated_state.json"
DEBATE_RESULTS_FILE = REPLAY_DIR / "debate_results.json"

# How many days to process per invocation (avoid blocking daemon too long)
MAX_DAYS_PER_RUN = 60
# How often to print equity update lines
EQUITY_PRINT_INTERVAL = 30
# Position holding period in trading days
HOLD_DAYS = 21
# Entry threshold: composite_score must exceed this
ENTRY_THRESHOLD = 0.55
# Max concurrent positions in simulation
MAX_SIM_POSITIONS = 5


@dataclass
class ReplayStats:
    days_processed: int = 0
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_pct: float = 0.0
    current_equity: float = 100_000.0
    peak_equity: float = 100_000.0
    current_drawdown: float = 0.0
    last_processed_date: str = ""
    start_date: str = ""
    end_date: str = ""


class HistoricalReplay:
    """Replay trading engine for closed-market learning."""

    def __init__(self):
        self.stats = ReplayStats()
        self._open_positions: list[dict] = []
        self._feedback_loop = None
        self._prices_df: Optional[pd.DataFrame] = None
        self._spx_trend_cache: dict[str, bool] = {}
        self._last_weights: dict[str, float] = {}
        self._weight_updates: int = 0

    @property
    def feedback(self):
        if self._feedback_loop is None:
            from learning.feedback import FeedbackLoop
            self._feedback_loop = FeedbackLoop(lr=0.01)
        return self._feedback_loop

    def run(self, max_days: int = MAX_DAYS_PER_RUN, verbose: bool = False) -> ReplayStats:
        """Run historical replay for up to max_days. Resumes from checkpoint."""
        self._verbose = verbose
        start_time = time.perf_counter()
        logger.info("=" * 50)
        logger.info("HISTORICAL REPLAY: trading on past data")

        # Load or build price data
        prices_df = self._load_or_fetch_prices()
        if prices_df.empty:
            logger.warning("No price data available for replay")
            return self.stats

        # Load checkpoint BEFORE computing scores (need saved weights)
        checkpoint = self._load_checkpoint()
        processed_dates = set(checkpoint.get("processed_dates", []))
        self._open_positions = checkpoint.get("open_positions", [])
        self.stats = ReplayStats(**checkpoint.get("stats", {}))

        # Restore learned weights from checkpoint
        saved_weights = checkpoint.get("weights")
        if saved_weights:
            self.feedback._current_weights = FactorWeights.from_dict(saved_weights)
            self.feedback.optimizer.weights = FactorWeights.from_dict(saved_weights)
            self._last_weights = dict(saved_weights)
            logger.info(f"  Restored weights from checkpoint: {saved_weights}")

        # Pre-compute factor scores on full price history (uses restored weights)
        prices_df = self._compute_factor_scores(prices_df)
        self._prices_df = prices_df

        # Build SPX trend filter cache (date → above_200ma)
        self._spx_trend_cache = self._build_spx_trend_cache()

        if not self.stats.last_processed_date:
            # Fresh start — skip first 252 days (need data for factor computation)
            dates = sorted(prices_df["date"].unique())
            if len(dates) > 252:
                self.stats.start_date = str(dates[252])
                self.stats.last_processed_date = str(dates[252])
            else:
                self.stats.start_date = str(dates[0])
                self.stats.last_processed_date = str(dates[0])

        # Get unique dates, skip already processed AND pre-warmup
        all_dates = sorted(prices_df["date"].unique())
        start_str = self.stats.start_date
        pending = [
            d for d in all_dates
            if str(d) not in processed_dates and str(d) >= start_str
        ]

        if not pending:
            logger.info("All historical data processed. Resetting...")
            processed_dates = set()
            pending = [d for d in all_dates if str(d) >= start_str]
            self._open_positions = []
            self.stats = ReplayStats()
            self.stats.start_date = start_str

        self.stats.end_date = str(all_dates[-1])

        days_to_run = min(max_days, len(pending))
        logger.info(f"  Date range: {self.stats.start_date} → {self.stats.end_date}")
        logger.info(f"  Pending: {len(pending)} days, running {days_to_run}")
        logger.info(f"  Starting equity: ${self.stats.current_equity:,.0f}")

        for i, date_str in enumerate(pending[:days_to_run]):
            day_data = prices_df[prices_df["date"] == date_str]
            if day_data.empty:
                processed_dates.add(str(date_str))
                continue

            self._process_day(day_data, date_str)
            processed_dates.add(str(date_str))
            self.stats.days_processed += 1
            self.stats.last_processed_date = str(date_str)

            # Periodic equity print
            if self._verbose and (i + 1) % EQUITY_PRINT_INTERVAL == 0:
                trades_taken = self.stats.trades_taken
                wr = self.stats.wins / max(1, trades_taken)
                w = self.feedback._current_weights.as_dict()
                print(f"  === [{date_str}] day {i+1}/{days_to_run} | equity=${self.stats.current_equity:,.0f} | trades={trades_taken} | WR={wr:.0%} | DD={self.stats.current_drawdown:.1%} | w:M={w['momentum']:.2f}/LV={w['low_vol']:.2f} ===")

            # Save checkpoint periodically
            if (i + 1) % 10 == 0:
                self._save_checkpoint(list(processed_dates))
                self._save_stats()

        # Final save
        self._save_checkpoint(list(processed_dates))
        self._save_stats()

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Replay complete: {self.stats.days_processed} days, "
            f"{self.stats.trades_taken} trades, "
            f"Win rate: {self.stats.wins / max(1, self.stats.trades_taken):.0%}, "
            f"Equity: ${self.stats.current_equity:,.0f} "
            f"({(self.stats.current_equity / 100000 - 1) * 100:+.1f}%) "
            f"in {elapsed:.0f}s"
        )

        return self.stats

    def _process_day(self, day_data: pd.DataFrame, date_str: str):
        """Process a single historical day: check exits → compute signals → enter new positions."""
        # 1. Check exits: positions held >= HOLD_DAYS get closed
        self._check_exits(day_data, date_str)

        # 2. Compute signals for today's tickers
        signals = self._compute_historical_signals(day_data)
        if signals.empty:
            return

        # 2.5. Trend filter: skip entries when SPX below 200MA
        if TREND_FILTER_ENABLED and self._spx_trend_cache:
            date_str_clean = str(date_str)[:10]  # "YYYY-MM-DD" from Timestamp or string
            above_200ma = self._spx_trend_cache.get(date_str_clean)
            if above_200ma is False:
                if self._verbose:
                    print(f"  TREND BLOCKED {date_str_clean} — SPX below 200MA, no entries")
                return

        # 3. Enter new positions if under max
        slots = MAX_SIM_POSITIONS - len(self._open_positions)
        if slots <= 0:
            return

        top = signals.head(slots)
        for _, row in top.iterrows():
            ticker = row["ticker"]
            if any(p["ticker"] == ticker for p in self._open_positions):
                continue

            composite = float(row.get("composite_score", 0.5))
            if composite < ENTRY_THRESHOLD:
                continue

            entry_price = float(row["close"])
            position_pct = MAX_POSITION_PCT
            position_value = self.stats.current_equity * position_pct
            qty = max(1, int(position_value / entry_price))

            trade = {
                "trade_id": f"sim_{date_str}_{ticker}",
                "ticker": ticker,
                "side": "BUY",
                "quantity": qty,
                "entry_price": entry_price,
                "entry_date": date_str,
                "composite_score": composite,
                "quality_pct": float(row.get("quality_score", 0.5)),
                "momentum_pct": float(row.get("momentum_score", 0.5)),
                "value_pct": float(row.get("value_score", 0.5)),
                "low_vol_pct": float(row.get("low_vol_score", 0.5)),
                "regime_id": 2,
                "position_size_pct": position_pct,
                "stop_loss_price": entry_price * (1 - STOP_LOSS_PCT),
                "take_profit_price": entry_price * (1 + TAKE_PROFIT_PCT),
            }
            self._open_positions.append(trade)
            self.stats.trades_taken += 1

            if self._verbose:
                print(f"  BUY  {date_str} | {ticker:5s} @ ${entry_price:>8.2f} | qty={qty} | score={composite:.3f}")

    def _check_exits(self, day_data: pd.DataFrame, date_str: str):
        """Check open positions for exit conditions: time, take-profit, stop-loss."""
        ticker_prices = {}
        for _, row in day_data.iterrows():
            ticker_prices[row["ticker"]] = float(row["close"])

        closed = []
        for pos in self._open_positions:
            ticker = pos["ticker"]
            current_price = ticker_prices.get(ticker)
            if current_price is None:
                continue

            entry_date = pd.Timestamp(pos["entry_date"])
            current_date = pd.Timestamp(date_str)
            days_held = (current_date - entry_date).days

            exit_reason = None
            exit_price = current_price

            # Check take-profit
            if current_price >= pos.get("take_profit_price", float("inf")):
                exit_reason = "take_profit"
            # Check stop-loss
            elif current_price <= pos.get("stop_loss_price", 0):
                exit_reason = "stop_loss"
            # Check time exit
            elif days_held >= HOLD_DAYS:
                exit_reason = "time_exit"

            if exit_reason:
                pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
                pnl_usd = (exit_price - pos["entry_price"]) * pos["quantity"]

                pos["exit_price"] = exit_price
                pos["exit_date"] = date_str
                pos["exit_reason"] = exit_reason
                pos["pnl_pct"] = pnl_pct
                pos["pnl_usd"] = pnl_usd
                pos["status"] = "CLOSED"

                self.stats.current_equity += pnl_usd
                self.stats.total_pnl_pct += pnl_pct
                if pnl_pct > 0:
                    self.stats.wins += 1
                else:
                    self.stats.losses += 1

                if self.stats.current_equity > self.stats.peak_equity:
                    self.stats.peak_equity = self.stats.current_equity

                self.stats.current_drawdown = (
                    (self.stats.peak_equity - self.stats.current_equity)
                    / self.stats.peak_equity
                )

                closed.append(pos)

                if self._verbose:
                    tag = "WIN" if pnl_pct > 0 else "LOSS"
                    print(f"  SELL {date_str} | {ticker:5s} @ ${exit_price:>8.2f} | {tag:4s} {pnl_pct:+.1%} | {exit_reason}")

        # Feed closed trades through feedback loop
        if closed:
            try:
                result = self.feedback.process_closed_trades(closed)
                logger.debug(
                    f"  Feedback: {len(closed)} trades, "
                    f"decay={result['decay_level']}, "
                    f"weights={result['updated_weights']}"
                )

                # Recompute composite scores for all remaining dates with updated weights
                new_w = result["updated_weights"]
                if new_w != self._last_weights and self._prices_df is not None:
                    self._recompute_composites(date_str, new_w)
                    self._last_weights = dict(new_w)
                    self._weight_updates += 1
                    if self._verbose:
                        w = new_w
                        print(f"  WT   {date_str} | M={w['momentum']:.3f} LV={w['low_vol']:.3f} Q={w['quality']:.3f} V={w['value']:.3f} | update #{self._weight_updates}")
            except Exception as e:
                logger.error(f"Feedback loop failed in replay: {e}")

        # Remove closed from open
        for c in closed:
            self._open_positions.remove(c)

        # Save closed trades to history
        if closed:
            self._append_simulated_trades(closed)

    def _recompute_composites(self, from_date_str: str, weights: dict[str, float]):
        """Recompute composite_score for all dates >= from_date_str using new weights.

        Called after feedback loop updates weights so future signal rankings
        immediately reflect the learned weight adjustments.
        """
        if self._prices_df is None:
            return
        future = self._prices_df["date"] >= pd.Timestamp(from_date_str)
        if not future.any():
            return
        scores_df = self._prices_df.loc[future, :].copy()
        factor_cols = ["quality_score", "momentum_score", "value_score", "low_vol_score"]
        for col in factor_cols:
            if col not in scores_df.columns:
                scores_df[col] = 0.5
        self._prices_df.loc[future, "composite_score"] = (
            weights.get("quality", 0.25) * scores_df["quality_score"]
            + weights.get("momentum", 0.25) * scores_df["momentum_score"]
            + weights.get("value", 0.25) * scores_df["value_score"]
            + weights.get("low_vol", 0.25) * scores_df["low_vol_score"]
        )

    def _compute_factor_scores(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute factor scores across full price history.

        Computes: momentum_score (12-1 cross-sectional), low_vol_score (inverse 60d vol),
        quality_score, value_score, and composite_score.
        Fundamentals fetched from yfinance (current) and applied as constant proxy.
        """
        logger.info("  Computing factor scores on historical data...")
        tickers = prices_df["ticker"].unique()
        fundamentals = self._fetch_replay_fundamentals(tickers)
        scores_list = []

        for t in tickers:
            t_data = prices_df[prices_df["ticker"] == t].sort_values("date")
            if len(t_data) < 252:
                continue

            closes = t_data["close"].values
            dates = t_data["date"].values

            for i in range(252, len(closes)):
                # Momentum: 12-1 return (252d - 21d)
                mom_ret = (closes[i - 21] - closes[i - 252]) / closes[i - 252] if closes[i - 252] > 0 else 0
                # Low-vol: inverse of 60-day realized volatility
                if i >= 60:
                    daily_rets = np.diff(closes[i - 60:i + 1]) / closes[i - 60:i]
                    vol = np.std(daily_rets) if len(daily_rets) > 0 else 0.01
                else:
                    vol = 0.02
                low_vol = 1.0 / (1.0 + vol * 100)  # Normalize to ~0-1

                # Fetch fundamental scores for this ticker (constant over time from current data)
                f = fundamentals.get(t, {})
                quality_raw = _compute_quality_raw(f)
                value_raw = _compute_value_raw(f)

                scores_list.append({
                    "ticker": t,
                    "date": dates[i],
                    "close": closes[i],
                    "momentum_raw": mom_ret,
                    "vol_raw": vol,
                    "quality_raw": quality_raw,
                    "value_raw": value_raw,
                })

        if not scores_list:
            return prices_df

        scores_df = pd.DataFrame(scores_list)

        # Cross-sectional percentile ranking within each date
        scores_df["momentum_score"] = scores_df.groupby("date")["momentum_raw"].rank(pct=True)
        scores_df["low_vol_score"] = scores_df.groupby("date")["vol_raw"].rank(pct=True, ascending=False)

        # Quality and value: cross-sectional rank per date from (constant) raw scores
        # Each ticker gets same raw score every date → rank stays constant, but
        # differing universe composition means ranks shift slightly as tickers enter/exit
        scores_df["quality_score"] = scores_df.groupby("date")["quality_raw"].rank(pct=True)
        scores_df["value_score"] = scores_df.groupby("date")["value_raw"].rank(pct=True)

        # Composite score with current weights
        try:
            w = self.feedback._current_weights.as_dict()
        except Exception:
            w = {"quality": 0.25, "momentum": 0.25, "value": 0.25, "low_vol": 0.25}

        scores_df["composite_score"] = (
            w["quality"] * scores_df["quality_score"]
            + w["momentum"] * scores_df["momentum_score"]
            + w["value"] * scores_df["value_score"]
            + w["low_vol"] * scores_df["low_vol_score"]
        )

        # Merge back with prices
        prices_df = prices_df.merge(
            scores_df[["ticker", "date", "momentum_score", "low_vol_score",
                        "quality_score", "value_score", "composite_score"]],
            on=["ticker", "date"], how="left"
        )

        # Fill missing factor scores with neutral values
        for col in ["momentum_score", "low_vol_score", "quality_score", "value_score"]:
            if col in prices_df.columns:
                prices_df[col] = prices_df[col].fillna(0.5)
        if "composite_score" in prices_df.columns:
            prices_df["composite_score"] = prices_df["composite_score"].fillna(0.5)

        logger.info(f"  Factor scores computed: {len(scores_df)} rows, {len(scores_df['ticker'].unique())} tickers")
        return prices_df

    def _compute_historical_signals(self, day_data: pd.DataFrame) -> pd.DataFrame:
        """Generate signal rankings from pre-computed factor scores."""
        if day_data.empty:
            return pd.DataFrame()

        # Factor scores already pre-computed in columns
        required_cols = {"ticker", "close", "composite_score"}
        if not required_cols.issubset(day_data.columns):
            return pd.DataFrame()

        # Ensure all factor columns exist
        for col in ["momentum_score", "quality_score", "value_score", "low_vol_score"]:
            if col not in day_data.columns:
                day_data[col] = 0.5

        df = day_data[["ticker", "close", "composite_score",
                        "momentum_score", "quality_score", "value_score", "low_vol_score"]].copy()
        if df.empty:
            return df
        return df.sort_values("composite_score", ascending=False)

    def _build_spx_trend_cache(self) -> dict[str, bool]:
        """Fetch ^GSPC data and compute 200MA status for each date. Returns dict[date_str] → above_200ma."""
        cache = {}
        try:
            import yfinance as yf
            spx = yf.download("^GSPC", period="20y", progress=False)
            if spx.empty:
                logger.warning("Could not fetch ^GSPC data for trend filter")
                return cache
            # Flatten MultiIndex columns if present (yfinance quirk for single ticker)
            if isinstance(spx.columns, pd.MultiIndex):
                spx.columns = spx.columns.get_level_values(0)
            spx["ma200"] = spx["Close"].rolling(200).mean()
            for idx, row in spx.iterrows():
                ma = row.get("ma200")
                close = row.get("Close")
                try:
                    ma_val = float(ma.iloc[0]) if hasattr(ma, 'iloc') else float(ma)
                    close_val = float(close.iloc[0]) if hasattr(close, 'iloc') else float(close)
                except (TypeError, ValueError, IndexError):
                    continue
                if pd.notna(ma_val) and pd.notna(close_val) and ma_val > 0:
                    date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                    cache[date_str] = close_val >= ma_val
        except Exception as e:
            logger.warning(f"SPX trend cache build failed: {e}")
        logger.info(f"  SPX trend cache: {len(cache)} dates with 200MA status")
        return cache

    def _load_or_fetch_prices(self) -> pd.DataFrame:
        """Load cached prices or fetch fresh."""
        cache_path = REPLAY_DIR / "replay_prices.parquet"
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                # Age check: if older than 7 days, refresh
                mtime = cache_path.stat().st_mtime
                if time.time() - mtime < 30 * 86400:
                    logger.info(f"Using cached prices: {len(df)} rows")
                    return df
            except Exception:
                pass

        logger.info("Fetching fresh price data for replay...")
        try:
            from ingestion.universe import us_tickers
            from ingestion.market_data import fetch_prices

            tickers = us_tickers()[:US_UNIVERSE_SIZE]
            df = fetch_prices(tickers, period="20y")
            if not df.empty:
                df.to_parquet(cache_path)
                logger.info(f"Cached {len(df)} rows to {cache_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch prices: {e}")
            return pd.DataFrame()

    def _load_checkpoint(self) -> dict:
        if CHECKPOINT_FILE.exists():
            try:
                return json.loads(CHECKPOINT_FILE.read_text())
            except Exception:
                pass
        return {}

    def _save_checkpoint(self, processed_dates: list[str]):
        def _serialize(obj):
            if isinstance(obj, (pd.Timestamp, np.datetime64)):
                return str(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        open_pos = []
        for p in self._open_positions:
            open_pos.append({k: _serialize(v) for k, v in p.items()})

        CHECKPOINT_FILE.write_text(json.dumps({
            "processed_dates": processed_dates[-5000:],
            "open_positions": open_pos,
            "stats": {
                "days_processed": self.stats.days_processed,
                "trades_taken": self.stats.trades_taken,
                "wins": self.stats.wins,
                "losses": self.stats.losses,
                "total_pnl_pct": self.stats.total_pnl_pct,
                "current_equity": self.stats.current_equity,
                "peak_equity": self.stats.peak_equity,
                "current_drawdown": self.stats.current_drawdown,
                "last_processed_date": self.stats.last_processed_date,
                "start_date": self.stats.start_date,
                "end_date": self.stats.end_date,
            },
            "weights": self.feedback._current_weights.as_dict(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

    def _save_stats(self):
        """Persist stats to disk so dashboard/verify can read them."""
        SIMULATED_STATE_FILE.write_text(json.dumps({
            "stats": {
                "days_processed": self.stats.days_processed,
                "trades_taken": self.stats.trades_taken,
                "wins": self.stats.wins,
                "losses": self.stats.losses,
                "win_rate": self.stats.wins / max(1, self.stats.trades_taken),
                "total_pnl_pct": self.stats.total_pnl_pct,
                "current_equity": self.stats.current_equity,
                "peak_equity": self.stats.peak_equity,
                "current_drawdown": self.stats.current_drawdown,
            },
            "last_processed_date": self.stats.last_processed_date,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

    def _append_simulated_trades(self, trades: list[dict]):
        """Append closed simulated trades to history file."""
        existing = []
        if SIMULATED_TRADES_FILE.exists():
            try:
                existing = json.loads(SIMULATED_TRADES_FILE.read_text())
            except Exception:
                pass

        for t in trades:
            entry_date = t.get("entry_date", "")
            exit_date = t.get("exit_date", "")
            if isinstance(entry_date, (pd.Timestamp, np.datetime64)):
                entry_date = str(entry_date)
            if isinstance(exit_date, (pd.Timestamp, np.datetime64)):
                exit_date = str(exit_date)

            existing.append({
                "trade_id": str(t.get("trade_id", "")),
                "ticker": str(t["ticker"]),
                "entry_date": str(entry_date),
                "exit_date": str(exit_date),
                "entry_price": float(t.get("entry_price", 0)),
                "exit_price": float(t.get("exit_price", 0)),
                "pnl_pct": float(t.get("pnl_pct", 0)),
                "pnl_usd": float(t.get("pnl_usd", 0)),
                "exit_reason": str(t.get("exit_reason", "")),
                "composite_score": float(t.get("composite_score", 0.5)),
            })

        # Keep last 10000 trades
        SIMULATED_TRADES_FILE.write_text(json.dumps(existing[-10000:], indent=2))

    def _fetch_replay_fundamentals(self, tickers: list) -> dict[str, dict]:
        """Fetch current fundamentals for replay scoring. Cached to disk for speed."""
        import hashlib
        cache_path = REPLAY_DIR / "replay_fundamentals.json"
        tickers_key = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:8]

        # Check cache — valid for 30 days
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
                cached_key = cache.get("_tickers_key", "")
                cached_at = cache.get("_cached_at", 0)
                if cached_key == tickers_key and time.time() - cached_at < 30 * 86400:
                    logger.info(f"  Using cached fundamentals ({len(cache) - 2} tickers)")
                    return {k: v for k, v in cache.items() if not k.startswith("_")}
            except Exception:
                pass

        logger.info(f"  Fetching fundamentals for {len(tickers)} tickers...")
        try:
            from ingestion.market_data import fetch_fundamentals
            result = {}
            batch_size = 20
            for i in range(0, len(tickers), batch_size):
                chunk = tickers[i:i + batch_size]
                for t in chunk:
                    result[t] = fetch_fundamentals(t)
                if (i + batch_size) % 60 == 0:
                    logger.info(f"    Fundamentals: {min(i + batch_size, len(tickers))}/{len(tickers)}")
        except Exception as e:
            logger.warning(f"Fundamentals fetch failed: {e}")
            return {}

        # Cache
        result["_tickers_key"] = tickers_key
        result["_cached_at"] = time.time()
        try:
            cache_path.write_text(json.dumps(result, default=str))
        except Exception:
            pass

        return {k: v for k, v in result.items() if not k.startswith("_")}

    def run_debate_replay(
        self,
        sample_interval: int = 30,
        lookback_years: int = 3,
        verbose: bool = True,
    ) -> list[dict]:
        """Run PARA-DEBATE on sampled historical dates to validate AI decision-making.

        Every `sample_interval` trading days, runs full 7-agent debate on the top-scored
        ticker. Records debate verdict vs factor-score decision for comparison.

        Args:
            sample_interval: How many trading days between debate samples (default 30).
            lookback_years: How many years back from today to sample (default 3).

        Returns:
            List of debate result dicts with ticker, date, verdict, consensus, factor_score.
        """
        logger.info("=" * 60)
        logger.info("PARA-DEBATE HISTORICAL REPLAY: validating AI decisions on past data")
        logger.info(f"  Sample every {sample_interval} days, last {lookback_years} years")

        prices_df = self._load_or_fetch_prices()
        if prices_df.empty:
            logger.warning("No price data available for debate replay")
            return []

        prices_df = self._compute_factor_scores(prices_df)
        self._prices_df = prices_df

        from strategy.decider import StrategyDecider
        decider = StrategyDecider()

        all_dates = sorted(prices_df["date"].unique())
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        recent_dates = [d for d in all_dates if pd.Timestamp(d) >= cutoff]
        sample_dates = recent_dates[::sample_interval]

        logger.info(f"  {len(sample_dates)} sample dates from {sample_dates[0]} to {sample_dates[-1]}")
        results = []

        for i, date_str in enumerate(sample_dates):
            day_data = prices_df[prices_df["date"] == date_str]
            if day_data.empty:
                continue

            signals = self._compute_historical_signals(day_data)
            if signals.empty:
                continue

            top = signals.head(1).iloc[0]
            ticker = top["ticker"]

            # Build debate context from available data
            context = {
                "ticker": ticker,
                "composite_score": float(top.get("composite_score", 0.5)),
                "momentum_score": float(top.get("momentum_score", 0.5)),
                "quality_score": float(top.get("quality_score", 0.5)),
                "value_score": float(top.get("value_score", 0.5)),
                "low_vol_score": float(top.get("low_vol_score", 0.5)),
                "entry_price": float(top["close"]),
                "date": str(date_str)[:10],
                "regime_id": int(top.get("regime_id", 2)),
            }

            # Add fundamentals if available
            fundamentals = self._fetch_replay_fundamentals([ticker])
            if ticker in fundamentals:
                f = fundamentals[ticker]
                for k, v in f.items():
                    if v is not None:
                        context[f"fund_{k}"] = v

            if verbose:
                print(f"\n  [{i+1}/{len(sample_dates)}] {date_str} | {ticker} | debating...")

            try:
                debate = decider.debate(ticker, "US", context)
                result = {
                    "sample": i + 1,
                    "date": str(date_str)[:10],
                    "ticker": ticker,
                    "factor_score": float(top.get("composite_score", 0.5)),
                    "debate_verdict": debate.verdict,
                    "consensus_score": round(debate.consensus_score, 3),
                    "investment_thesis": debate.investment_thesis[:500],
                    "bull_case": debate.bull_case[:300],
                    "bear_case": debate.bear_case[:300],
                    "risk_factors": debate.risk_factors,
                    "agent_stances": [
                        {"agent": o.agent, "stance": o.stance, "conviction": o.conviction}
                        for o in debate.agent_outputs
                    ],
                    "latency_ms": round(debate.latency_ms, 0),
                    "error": debate.error,
                }
                results.append(result)

                if verbose:
                    action = decider.verdict_to_action(debate.verdict)
                    print(f"    → {debate.verdict} (consensus={debate.consensus_score:.2f}) | {action} | {debate.latency_ms:.0f}ms")

            except Exception as e:
                logger.error(f"Debate failed for {ticker} on {date_str}: {e}")
                results.append({
                    "sample": i + 1, "date": str(date_str)[:10], "ticker": ticker,
                    "factor_score": float(top.get("composite_score", 0.5)),
                    "debate_verdict": "ERROR", "consensus_score": 0.0,
                    "error": str(e),
                })

            # Save intermediate results
            DEBATE_RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

        # Summary
        buy_signals = [r for r in results if r["debate_verdict"] in ("BUY", "STRONG BUY")]
        hold_signals = [r for r in results if r["debate_verdict"] == "HOLD"]
        sell_signals = [r for r in results if r["debate_verdict"] in ("SELL", "STRONG SELL")]
        errors = [r for r in results if r["debate_verdict"] == "ERROR"]

        logger.info(f"\nDebate Replay Complete: {len(results)} debates")
        logger.info(f"  BUY: {len(buy_signals)} | HOLD: {len(hold_signals)} | SELL: {len(sell_signals)} | ERRORS: {len(errors)}")
        if results:
            avg_consensus = sum(r.get("consensus_score", 0) for r in results) / max(1, len(results) - len(errors))
            avg_latency = sum(r.get("latency_ms", 0) for r in results if r.get("latency_ms")) / max(1, len(results) - len(errors))
            logger.info(f"  Avg Consensus: {avg_consensus:.2f} | Avg Latency: {avg_latency:.0f}ms")

        return results

    def get_stats(self) -> dict:
        """Current replay stats for dashboard/status."""
        return {
            "days_processed": self.stats.days_processed,
            "trades_taken": self.stats.trades_taken,
            "wins": self.stats.wins,
            "losses": self.stats.losses,
            "win_rate": self.stats.wins / max(1, self.stats.trades_taken),
            "total_pnl_pct": self.stats.total_pnl_pct,
            "current_equity": self.stats.current_equity,
            "peak_equity": self.stats.peak_equity,
            "current_drawdown": self.stats.current_drawdown,
            "last_processed_date": self.stats.last_processed_date,
            "date_range": f"{self.stats.start_date} → {self.stats.end_date}",
            "open_positions": len(self._open_positions),
        }


def _compute_quality_raw(fundamentals: dict) -> float:
    """Compute raw quality score from fundamentals (0-1 scale, before cross-sectional ranking)."""
    if not fundamentals:
        return 0.5
    roe = fundamentals.get("roe")
    margin = fundamentals.get("profit_margin")
    score = 0.5
    if roe is not None and isinstance(roe, (int, float)):
        score += 0.25 * min(max(roe, 0) / 0.30, 1.0)
    if margin is not None and isinstance(margin, (int, float)):
        score += 0.25 * min(max(margin, 0) / 0.25, 1.0)
    return min(1.0, max(0.0, score))


def _compute_value_raw(fundamentals: dict) -> float:
    """Compute raw value score from fundamentals (0-1 scale, lower P/E & P/B = higher score)."""
    if not fundamentals:
        return 0.5
    pe = fundamentals.get("pe_ratio")
    pb = fundamentals.get("pb_ratio")
    score = 0.5
    if pe is not None and isinstance(pe, (int, float)) and pe > 0:
        score -= 0.25 * min(pe / 30, 1.0)
    if pb is not None and isinstance(pb, (int, float)) and pb > 0:
        score -= 0.25 * min(pb / 5, 1.0)
    return max(0.0, min(1.0, score))
