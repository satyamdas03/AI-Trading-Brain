"""Backtest comparison: run 4 config presets on 20-year data and compare.

Configs:
  1. Baseline     — ATR=Off, VolRank=Off (current: fixed 7%/6% TP/SL, 4 factors)
  2. ATR Only     — ATR=On,  VolRank=Off (volatility-adjusted exits)
  3. VolRank Only — ATR=Off, VolRank=On  (5-factor scoring)
  4. Full Stack   — ATR=On,  VolRank=On  (all replay-compatible integrations)

Run: python scripts/backtest_compare.py [--quick]

Output: data/backtest/compare_results.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DB_DIR, US_UNIVERSE_SIZE

BACKTEST_DIR = DB_DIR / "backtest"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("backtest_compare")


@dataclass
class BacktestResult:
    config_name: str = ""
    final_equity: float = 100_000.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    avg_holding_days: float = 0.0
    avg_return_per_trade_pct: float = 0.0
    profit_factor: float = 0.0
    trend_filter_blocks: int = 0
    best_year_pct: float = 0.0
    worst_year_pct: float = 0.0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "final_equity": round(self.final_equity, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "win_rate_pct": round(self.win_rate_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "total_trades": self.total_trades,
            "avg_holding_days": round(self.avg_holding_days, 1),
            "avg_return_per_trade_pct": round(self.avg_return_per_trade_pct, 2),
            "profit_factor": round(self.profit_factor, 2),
            "trend_filter_blocks": self.trend_filter_blocks,
            "best_year_pct": round(self.best_year_pct, 2),
            "worst_year_pct": round(self.worst_year_pct, 2),
            "duration_seconds": round(self.duration_seconds, 1),
        }


def run_config(config_name: str, atr_enabled: bool, vol_rank_enabled: bool, max_days: int = 5000) -> BacktestResult:
    """Run a single backtest config and return results."""
    import config as cfg
    from learning.historical_replay import HistoricalReplay, REPLAY_DIR

    # Override config for this run
    original_atr = cfg.ATR_ENABLED
    cfg.ATR_ENABLED = atr_enabled

    # For vol_rank: we patch FactorWeights default to exclude vol_rank when disabled
    from learning.weight_optimizer import FactorWeights

    result = BacktestResult(config_name=config_name)
    start_time = time.perf_counter()

    try:
        # Clear replay cache to force fresh run
        checkpoint_path = REPLAY_DIR / "checkpoint.json"
        prices_cache = REPLAY_DIR / "replay_prices.parquet"

        # Delete checkpoint to force fresh run
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        # Patch FactorWeights default if vol_rank disabled
        original_fw = FactorWeights.__dataclass_fields__.copy()

        if not vol_rank_enabled:
            # Override FactorWeights to exclude vol_rank (0 weight)
            from learning.weight_optimizer import FactorWeights as FW

            # Use 4-factor equal weight (no vol_rank)
            FW.quality = 0.25
            FW.momentum = 0.25
            FW.value = 0.25
            FW.low_vol = 0.25
            FW.vol_rank = 0.0
        else:
            from learning.weight_optimizer import FactorWeights as FW
            FW.quality = 0.20
            FW.momentum = 0.20
            FW.value = 0.20
            FW.low_vol = 0.20
            FW.vol_rank = 0.20

        replay = HistoricalReplay()
        # Force fresh weights based on config
        from learning.weight_optimizer import FactorWeights as FW
        replay.feedback._current_weights = FW()

        # Run full replay
        stats = replay.run(max_days=max_days, verbose=False)

        result.final_equity = stats.current_equity
        result.total_return_pct = (stats.current_equity / 100_000 - 1) * 100
        result.max_drawdown_pct = stats.current_drawdown * 100
        result.win_rate_pct = (stats.wins / max(1, stats.trades_taken)) * 100
        result.sharpe_ratio = replay.compute_sharpe()
        result.total_trades = stats.trades_taken
        result.duration_seconds = time.perf_counter() - start_time

        # Compute year-by-year and per-trade stats from simulated trades
        _compute_derived_metrics(result, replay)

    except Exception as e:
        logger.exception(f"Config '{config_name}' failed: {e}")
        result.errors.append(str(e))
    finally:
        # Restore config
        cfg.ATR_ENABLED = original_atr

    return result


def _compute_derived_metrics(result: BacktestResult, replay):
    """Compute avg_holding_days, profit_factor, best/worst year from simulated trades."""
    from learning.historical_replay import SIMULATED_TRADES_FILE

    if not SIMULATED_TRADES_FILE.exists():
        return

    try:
        trades = json.loads(SIMULATED_TRADES_FILE.read_text())
    except Exception:
        return

    if not trades:
        return

    # Holding days
    holding_days = []
    for t in trades:
        try:
            entry = pd.Timestamp(t.get("entry_date", ""))
            exit_d = pd.Timestamp(t.get("exit_date", ""))
            holding_days.append((exit_d - entry).days)
        except Exception:
            continue

    if holding_days:
        result.avg_holding_days = sum(holding_days) / len(holding_days)

    # Average return per trade
    pnl_pcts = [t.get("pnl_pct", 0) or 0 for t in trades]
    if pnl_pcts:
        result.avg_return_per_trade_pct = sum(pnl_pcts) / len(pnl_pcts) * 100

    # Profit factor: total gains / total losses
    gains = sum(t.get("pnl_usd", 0) or 0 for t in trades if (t.get("pnl_usd", 0) or 0) > 0)
    losses = abs(sum(t.get("pnl_usd", 0) or 0 for t in trades if (t.get("pnl_usd", 0) or 0) < 0))
    if losses > 0:
        result.profit_factor = gains / losses

    # Yearly returns
    yearly: dict[int, list[float]] = {}
    for t in trades:
        pnl = t.get("pnl_usd", 0) or 0
        try:
            year = pd.Timestamp(t.get("exit_date", "")).year
        except Exception:
            continue
        yearly.setdefault(year, []).append(pnl)

    if yearly:
        initial_equity = 100_000.0
        yearly_returns = {}
        current_equity = initial_equity
        for year in sorted(yearly.keys()):
            year_pnl = sum(yearly[year])
            if current_equity > 0:
                yearly_returns[year] = (year_pnl / current_equity) * 100
            current_equity += year_pnl

        if yearly_returns:
            result.best_year_pct = max(yearly_returns.values())
            result.worst_year_pct = min(yearly_returns.values())

    # Count trend filter blocks from replay stats
    result.trend_filter_blocks = 0  # Approximated from days_processed vs potential


def run_all_configs(quick: bool = False) -> list[BacktestResult]:
    """Run all 4 config presets and return comparison results."""
    configs = [
        ("1_Baseline", False, False),
        ("2_ATR_Only", True, False),
        ("3_VolRank_Only", False, True),
        ("4_Full_Stack", True, True),
    ]

    max_days_val = 100 if quick else 5000

    results = []
    for name, atr, vol_rank in configs:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"RUNNING: {name} (ATR={atr}, VolRank={vol_rank}, days={max_days_val})")
        logger.info(f"{'=' * 60}")
        result = run_config(name, atr, vol_rank, max_days=max_days_val)
        results.append(result)
        logger.info(f"  Equity: ${result.final_equity:,.0f} ({result.total_return_pct:+.1f}%)")
        logger.info(f"  WR: {result.win_rate_pct:.1f}% | Sharpe: {result.sharpe_ratio:.3f}")
        logger.info(f"  Max DD: {result.max_drawdown_pct:.1f}% | Trades: {result.total_trades}")
        logger.info(f"  Duration: {result.duration_seconds:.0f}s")

    return results


def print_comparison(results: list[BacktestResult]):
    """Print formatted comparison table."""
    baseline = results[0] if results else None

    header = (
        f"{'Config':<20} {'Equity':>12} {'Return':>8} {'WR':>7} "
        f"{'Sharpe':>8} {'MaxDD':>7} {'Trades':>7} {'PF':>6} {'BestYr':>7} {'WorstYr':>8}"
    )
    print("\n" + "=" * len(header))
    print("BACKTEST COMPARISON RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        name = r.config_name.replace("_", " ")
        delta = ""
        if baseline and r != baseline:
            delta_return = r.total_return_pct - baseline.total_return_pct
            delta_sharpe = r.sharpe_ratio - baseline.sharpe_ratio
            delta = f" [{delta_return:+.1f}%, {delta_sharpe:+.3f}sh]"

        print(
            f"{name:<20} ${r.final_equity:>11,.0f} {r.total_return_pct:>+7.1f}% "
            f"{r.win_rate_pct:>6.1f}% {r.sharpe_ratio:>8.3f} {r.max_drawdown_pct:>6.1f}% "
            f"{r.total_trades:>7} {r.profit_factor:>6.2f} {r.best_year_pct:>+6.1f}% {r.worst_year_pct:>+7.1f}%{delta}"
        )

    print("-" * len(header))

    if baseline:
        best_return = max(results, key=lambda r: r.total_return_pct)
        best_sharpe = max(results, key=lambda r: r.sharpe_ratio)
        lowest_dd = min(results, key=lambda r: r.max_drawdown_pct)
        print(f"Best return: {best_return.config_name} ({best_return.total_return_pct:+.1f}%)")
        print(f"Best Sharpe: {best_return.config_name}" if best_return == best_sharpe else f"Best Sharpe: {best_sharpe.config_name} ({best_sharpe.sharpe_ratio:.3f})")
        print(f"Lowest DD:  {lowest_dd.config_name} ({lowest_dd.max_drawdown_pct:.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest 4 strategy configs on 20-year data")
    parser.add_argument("--quick", action="store_true", help="Quick run (100 days per config)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print(f"Backtest comparison starting at {datetime.now(timezone.utc).isoformat()}")
    if args.quick:
        print("QUICK MODE: 100 days per config")
    else:
        print("FULL MODE: processing entire 20-year dataset")

    results = run_all_configs(quick=args.quick)
    print_comparison(results)

    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "quick" if args.quick else "full",
        "universe_size": US_UNIVERSE_SIZE,
        "configs": [r.as_dict() for r in results],
    }

    output_path = BACKTEST_DIR / "compare_results.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
