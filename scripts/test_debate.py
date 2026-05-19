"""Test PARA-DEBATE flow end-to-end with real Claude calls."""
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

from strategy.decider import StrategyDecider
from ingestion.macro_data import fetch_macro_snapshot
from ingestion.market_data import fetch_fundamentals

print("=== PARA-DEBATE Flow Test ===\n")

# Load current signals
state_path = PROJECT / "data" / "brain_state.json"
if state_path.exists():
    state = json.loads(state_path.read_text())
    signals = state.get("latest_signals", [])[:3]  # Test top 3
    print(f"Loaded {len(signals)} signals from brain state")
else:
    print("WARNING: No brain state, using hardcoded test tickers")
    signals = [
        {"ticker": "AAPL", "composite_score": 0.85, "quality_score": 0.7, "momentum_score": 0.8, "value_score": 0.6, "low_vol_score": 0.5, "regime_id": 1},
        {"ticker": "MSFT", "composite_score": 0.82, "quality_score": 0.9, "momentum_score": 0.7, "value_score": 0.5, "low_vol_score": 0.6, "regime_id": 1},
    ]

# Get macro + fundamentals
macro = fetch_macro_snapshot()
print(f"Macro: VIX={macro.get('vix')}, SPX vs 200MA={macro.get('spx_vs_200ma', 0):.1%}")

# Build contexts
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
        "regime_label": {1: "Risk-On", 2: "Late-Cycle", 3: "Bear", 4: "Recovery"}.get(int(s.get("regime_id", 2)), "Late-Cycle"),
        **macro,
        **fund,
    }
    contexts[ticker] = ctx
    print(f"Context [{ticker}]: score={ctx['composite_score']}, regime={ctx['regime_label']}")

# Run debate
decider = StrategyDecider()
tickers = [s["ticker"] for s in signals]
print(f"\nRunning 7-agent debate on {tickers}...")
t0 = time.perf_counter()

results = decider.debate_batch(tickers, "US", contexts)

elapsed = time.perf_counter() - t0
print(f"\nDebate complete in {elapsed:.1f}s\n")

# Show results
all_ok = True
for r in results:
    status = "OK" if not r.error else f"ERROR: {r.error}"
    print(f"  {r.ticker}: {r.verdict} (consensus={r.consensus_score:+.2f}, {r.latency_ms:.0f}ms) [{status}]")
    if r.error:
        all_ok = False
    if r.investment_thesis:
        print(f"    Thesis: {r.investment_thesis[:200]}")

buy_count = sum(1 for r in results if r.verdict in ("STRONG BUY", "BUY"))
print(f"\nBUY signals: {buy_count}/{len(results)}")

if all_ok:
    print("\n*** Debate flow PASSED ***")
else:
    print("\n*** Debate flow has errors ***")
    sys.exit(1)
