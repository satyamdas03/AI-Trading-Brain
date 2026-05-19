"""Dry-run market open execution — full pipeline test without placing orders."""
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from dotenv import load_dotenv; load_dotenv(PROJECT / ".env")
from datetime import datetime, timezone
from strategy.decider import StrategyDecider
from ingestion.macro_data import fetch_macro_snapshot
from ingestion.market_data import fetch_fundamentals
from risk.manager import RiskManager
from execution.alpaca_client import AlpacaClient
import logging

logging.basicConfig(level=logging.WARNING)  # suppress httpx noise

print("=" * 60)
print("MARKET OPEN DRY RUN")
print(f"Time: {datetime.now(timezone.utc).isoformat()}")
print("=" * 60)

# 1. Account check
print("\n--- 1. ACCOUNT ---")
try:
    client = AlpacaClient()
    acc = client.get_account()
    print(f"  Equity: ${acc.equity:,.2f}")
    print(f"  Cash: ${acc.cash:,.2f}")
    print(f"  Buying power: ${acc.buying_power:,.2f}")
    print(f"  Day trades: {acc.day_trade_count}/3")
    print(f"  Positions: {len(client.get_positions())}")
except Exception as e:
    print(f"  FAILED: {e}")
    acc = None

# 2. Market check
print("\n--- 2. MARKET ---")
try:
    clock = client.get_clock()
    print(f"  Is open: {clock.is_open}")
    print(f"  Next open: {clock.next_open}")
    if clock.is_open:
        print("  WARNING: Market open — orders WOULD be placed!")
except Exception as e:
    print(f"  FAILED: {e}")

# 3. Risk check
print("\n--- 3. RISK ---")
try:
    risk = RiskManager(client)
    state = risk.check_pre_market()
    print(f"  Level: {state.level}")
    print(f"  Drawdown: {state.current_drawdown_pct:.2%}")
    print(f"  Can trade: {state.level.value != 'RED'}")
except Exception as e:
    print(f"  FAILED: {e}")
    state = None

# 4. Load signals
print("\n--- 4. SIGNALS ---")
state_path = PROJECT / "data" / "brain_state.json"
if state_path.exists():
    brain = json.loads(state_path.read_text())
    signals = brain.get("latest_signals", [])[:5]
    print(f"  Loaded {len(signals)} signals:")
    for s in signals:
        print(f"    {s['ticker']}: composite={s.get('composite_score',0):.3f} "
              f"Q={s.get('quality_score',0):.2f} M={s.get('momentum_score',0):.2f} "
              f"V={s.get('value_score',0):.2f} LV={s.get('low_vol_score',0):.2f} "
              f"regime={s.get('regime_id')}")
else:
    print("  No brain state found!")
    signals = []

# 5. Macro + fundamentals
print("\n--- 5. MACRO ---")
try:
    macro = fetch_macro_snapshot()
    print(f"  VIX: {macro.get('vix')}")
    print(f"  SPX vs 200MA: {macro.get('spx_vs_200ma', 0):.1%}")
    print(f"  Trend block: {'YES' if macro.get('spx_vs_200ma', 0) < 0 else 'NO'}")
except Exception as e:
    print(f"  FAILED: {e}")
    macro = {}

# 6. Build contexts with fundamentals
print("\n--- 6. ENRICHMENT ---")
contexts = {}
for s in signals:
    ticker = s["ticker"]
    try:
        fund = fetch_fundamentals(ticker)
    except Exception:
        fund = {}
    ctx = {
        "ticker": ticker,
        "composite_score": float(s.get("composite_score", 0.5)),
        "quality_percentile": float(s.get("quality_score", 0.5)),
        "momentum_percentile": float(s.get("momentum_score", 0.5)),
        "value_percentile": float(s.get("value_score", 0.5)),
        "low_vol_percentile": float(s.get("low_vol_score", 0.5)),
        "regime_id": int(s.get("regime_id", 2)),
        "regime_label": {1: "Risk-On", 2: "Late-Cycle", 3: "Bear", 4: "Recovery"}.get(
            int(s.get("regime_id", 2)), "Late-Cycle"
        ),
        **macro,
        **fund,
    }
    contexts[ticker] = ctx
    has_fund = bool(fund)
    print(f"  {ticker}: fundamentals={'yes' if has_fund else 'no'} "
          f"({len(fund)} fields)")

# 7. DEBATE — with per-agent tracking
print("\n--- 7. DEBATE ---")
decider = StrategyDecider()
tickers = [s["ticker"] for s in signals]

# Run debate
t0 = time.perf_counter()
results = decider.debate_batch(tickers, "US", contexts)
elapsed = time.perf_counter() - t0

print(f"\n  Total time: {elapsed:.1f}s ({elapsed/len(tickers):.1f}s per ticker)")
print()

buy_count = 0
for r in results:
    verdict_icon = {"STRONG BUY": "[BUY+]", "BUY": "[BUY]", "HOLD": "[HOLD]", "SELL": "[SELL]", "STRONG SELL": "[SELL+]"}.get(r.verdict, "[???]")
    print(f"  {verdict_icon} {r.ticker}: {r.verdict} "
          f"(consensus={r.consensus_score:+.2f}, {r.latency_ms/1000:.1f}s)")
    if r.error:
        print(f"      ERROR: {r.error}")
    if r.investment_thesis:
        thesis = r.investment_thesis[:250].replace('\n', ' ')
        print(f"      Thesis: {thesis}")
    if r.verdict in ("STRONG BUY", "BUY"):
        buy_count += 1

print(f"\n  BUY signals: {buy_count}/{len(results)}")
print(f"  HOLD signals: {len(results) - buy_count}/{len(results)}")

# 8. What would happen
print("\n--- 8. EXECUTION SIMULATION ---")
if buy_count == 0:
    print("  RESULT: No trades today. All signals HOLD/SELL.")
    print("  This is SAFE — no capital at risk.")
    print("  But means debate may be too conservative.")
elif state and state.level.value == "RED":
    print(f"  RESULT: Blocked by risk manager: {state.block_reason}")
else:
    print(f"  RESULT: Would place {buy_count} bracket orders:")
    position_size = (acc.equity * 0.05) if acc else 5000  # 5% per position
    for r in results:
        if r.verdict in ("STRONG BUY", "BUY"):
            print(f"    {r.ticker}: ~${position_size:,.0f} with 7%TP/6%SL bracket")

print()
print("=" * 60)
print(f"DRY RUN COMPLETE — no orders placed")
print(f"Market opens: ~{((clock.next_open - clock.timestamp).total_seconds()/3600):.1f}h from now")
print("=" * 60)
