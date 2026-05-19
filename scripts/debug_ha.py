"""Debug single debate to check Head Analyst output."""
import logging, json, sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
from dotenv import load_dotenv; load_dotenv(PROJECT / ".env")

import strategy.decider as decider_mod
from strategy.decider import StrategyDecider, _run_head_analyst, _compute_consensus, _parse_head_output
from ingestion.macro_data import fetch_macro_snapshot
from ingestion.market_data import fetch_fundamentals

# Monkey-patch _call_claude to capture raw HA response
_original_call_claude = decider_mod._call_claude
_captured_raw = []

def _capturing_call_claude(system, user_message, model, max_tokens=2048):
    raw = _original_call_claude(system, user_message, model, max_tokens)
    _captured_raw.append((model, system[:100], raw))
    return raw

decider_mod._call_claude = _capturing_call_claude

# Load signals
state = json.loads((PROJECT / "data" / "brain_state.json").read_text())
signals = state.get("latest_signals", [])[:1]
ticker = signals[0]["ticker"]
print(f"Testing: {ticker}")

# Build context
macro = fetch_macro_snapshot()
fund = fetch_fundamentals(ticker)

ctx = {
    "ticker": ticker,
    "composite_score": float(signals[0].get("composite_score", 0.5)),
    "quality_percentile": float(signals[0].get("quality_score", 0.5)),
    "momentum_percentile": float(signals[0].get("momentum_score", 0.5)),
    "value_percentile": float(signals[0].get("value_score", 0.5)),
    "low_vol_percentile": float(signals[0].get("low_vol_score", 0.5)),
    "regime_id": int(signals[0].get("regime_id", 2)),
    **macro,
    **fund,
}

decider = StrategyDecider()
result = decider.debate(ticker, "US", ctx)

print(f"\nVerdict: {result.verdict}")
print(f"Consensus: {result.consensus_score:.3f}")
print(f"Thesis present: {bool(result.investment_thesis)}")
print(f"Thesis length: {len(result.investment_thesis) if result.investment_thesis else 0}")
print(f"Bull case present: {bool(result.bull_case)}")
print(f"Error: {result.error}")

# Check agent outputs
for o in result.agent_outputs:
    print(f"  {o.agent}: {o.stance}/{o.conviction}")

if result.investment_thesis:
    print(f"\nThesis preview: {result.investment_thesis[:300]}")
else:
    print("\nNO THESIS - checking debate_json for raw HA output...")
    debate_data = json.loads(result.debate_json)
    ha = debate_data.get("head_analyst", {})
    print(f"HA verdict: {ha.get('verdict')}")
    print(f"HA thesis: {ha.get('investment_thesis', '')[:200]}")
    print(f"HA raw keys: {list(ha.keys())}")

# Print captured raw responses
print("\n" + "="*80)
print("RAW CLAUDE RESPONSES:")
print("="*80)
for i, (model, sys_prefix, raw) in enumerate(_captured_raw):
    print(f"\n--- Call {i+1}: model={model}, system_prefix={sys_prefix}... ---")
    # Write raw to file to avoid cp1252 encoding crashes
    outfile = PROJECT / "data" / f"debug_raw_call_{i+1}.txt"
    outfile.write_text(raw, encoding="utf-8")
    # Print safely
    safe = raw.encode('cp1252', errors='replace').decode('cp1252')
    print(safe[:2000])
    print(f"\n--- END (total len={len(raw)}, saved to {outfile.name}) ---")
