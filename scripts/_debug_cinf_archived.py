import pandas as pd
from execution.engine import ExecutionEngine

engine = ExecutionEngine()

signals = pd.DataFrame([{
    'ticker': 'CINF',
    'composite_score': 0.981,
    'quality_score': 0.748,
    'momentum_score': 0.624,
    'value_score': 0.880,
    'low_vol_score': 0.952,
    'regime_id': 1
}])

acc = engine._client.get_account()
print(f"Equity: {acc.equity}, Cash: {acc.cash}")
print(f"Dry run: {engine._dry_run}")
print(f"Existing: {[p.symbol for p in engine._client.get_positions()]}")

trades = engine.execute_market_open(signals, float(acc.equity))
print(f"Trades: {len(trades)}")
for t in trades:
    print(f"  {t['ticker']}: {t['quantity']} shares @ ${t['entry_price']:.2f}, status={t['status']}")
