"""Run 20-year historical paper trading with live terminal output."""
import sys
from pathlib import Path

# Use cached 20y data if available; fetch fresh if missing
cache = Path("data/replay/replay_prices.parquet")
if not cache.exists():
    print("No cache found. Fetching 20 years of data...\n")

from learning.historical_replay import HistoricalReplay

replay = HistoricalReplay()
print("=" * 60)
print("AI TRADING BRAIN — PAPER TRADING ON 20 YEARS OF DATA")
print("=" * 60)

stats = replay.run(max_days=10000, verbose=True)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Days processed:   {stats.days_processed}")
print(f"Trades taken:     {stats.trades_taken}")
if stats.trades_taken > 0:
    print(f"Win rate:         {stats.wins / stats.trades_taken:.0%} ({stats.wins}W / {stats.losses}L)")
print(f"Starting equity:  $100,000")
print(f"Final equity:     ${stats.current_equity:,.0f}")
print(f"Total return:     {(stats.current_equity / 100000 - 1) * 100:+.1f}%")
print(f"Max drawdown:     {stats.current_drawdown:.1%}")
print(f"Date range:       {stats.start_date} -> {stats.last_processed_date}")
