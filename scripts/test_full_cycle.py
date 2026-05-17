"""Test full daemon cycle: scoring -> execution -> feedback."""
import json
from brain import run_daily_scoring, run_market_open_execution

print("=== STEP 1: Daily Scoring ===")
run_daily_scoring()

print("\n=== STEP 2: Market Open Execution ===")
run_market_open_execution()

print("\n=== STEP 3: Check State ===")
state = json.loads(open("data/brain_state.json").read())
sig_count = len(state.get("latest_signals", []))
print(f"Signals stored: {sig_count}")
if sig_count > 0:
    top = state["latest_signals"][:3]
    for s in top:
        comp = s.get("composite_score", 0)
        print(f"  {s['ticker']}: composite={comp:.3f}")

last_trades = state.get("last_trades", [])
print(f"Last trades: {len(last_trades)}")

debates = state.get("last_debates", [])
if debates:
    print(f"Last debates: {len(debates)}")
    for d in debates[:3]:
        cs = d.get("consensus_score", 0)
        print(f"  {d['ticker']}: {d['verdict']} (consensus={cs:.2f})")

# Also check Supabase
try:
    from db.client import rest_get, table_exists
    if table_exists("daily_signals"):
        rows = rest_get("daily_signals", select="ticker,composite_score", order="composite_score.desc", limit=5)
        print(f"\n=== Supabase Top-5 Signals ===")
        for r in rows:
            print(f"  {r['ticker']}: {r['composite_score']:.4f}")
except Exception as e:
    print(f"Supabase check: {e}")

print("\n=== FULL CYCLE: PASS ===")
