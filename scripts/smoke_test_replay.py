"""Quick smoke test of historical replay engine."""
from learning.historical_replay import HistoricalReplay

replay = HistoricalReplay()
stats = replay.run(max_days=60)

print(f"Processed: {stats.days_processed} days")
print(f"Trades: {stats.trades_taken}")
print(f"Wins: {stats.wins}, Losses: {stats.losses}")
if stats.trades_taken > 0:
    print(f"Win rate: {stats.wins / stats.trades_taken:.0%}")
print(f"Equity: ${stats.current_equity:,.0f}")
print(f"P&L: {(stats.current_equity / 100000 - 1) * 100:+.1f}%")
print(f"DD: {stats.current_drawdown:.1%}")
print(f"Date range: {stats.start_date} -> {stats.last_processed_date}")
print("\nReplay engine: WORKING")
