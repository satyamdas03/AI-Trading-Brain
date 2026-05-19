"""Pre-flight check: verify all systems ready for market open paper trading."""
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

RESULTS = []

def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")

print("=" * 60)
print("AI Trading Brain -- Pre-Flight Check")
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
print("=" * 60)

# 1. Config
print("\n--- Config ---")
from config import (
    DRY_RUN, TRADE_ENABLED, ALPACA_PAPER, ALPACA_API_KEY,
    TAKE_PROFIT_PCT, STOP_LOSS_PCT, SIGNAL_TOP_N,
    MAX_POSITION_PCT, MAX_DRAWDOWN_PCT, DASHBOARD_PORT,
    TREND_FILTER_ENABLED, ONLINE_LEARNING_ENABLED,
    SUPABASE_URL,
)

check("ALPACA_API_KEY set", bool(ALPACA_API_KEY))
check("ALPACA_PAPER=true", ALPACA_PAPER is True)
check("DRY_RUN=false", DRY_RUN is False, "Real paper orders will be placed")
check("TRADE_ENABLED=true", TRADE_ENABLED is True, "Execution gate open")
check("TREND_FILTER_ENABLED=true", TREND_FILTER_ENABLED is True)
check("ONLINE_LEARNING_ENABLED=false", ONLINE_LEARNING_ENABLED is False, "Production-safe")
check(f"TAKE_PROFIT={TAKE_PROFIT_PCT:.0%} / STOP_LOSS={STOP_LOSS_PCT:.0%}",
     TAKE_PROFIT_PCT == 0.07 and STOP_LOSS_PCT == 0.06)
check(f"SIGNAL_TOP_N={SIGNAL_TOP_N}", SIGNAL_TOP_N == 5,
      "Max 5 new positions per day")
check(f"MAX_DRAWDOWN={MAX_DRAWDOWN_PCT:.0%}", MAX_DRAWDOWN_PCT == 0.10)
check("SUPABASE_URL set", bool(SUPABASE_URL))

# 2. Alpaca Account
print("\n--- Alpaca Account ---")
try:
    from execution.alpaca_client import AlpacaClient
    c = AlpacaClient()
    acc = c.get_account()
    check(f"Account active: {acc.account_number}", acc.status == "ACTIVE")
    check(f"Equity: ${acc.equity:,.2f}", acc.equity > 0)
    check(f"Cash: ${acc.cash:,.2f}", acc.cash > 0)
    check(f"Day trades: {acc.day_trade_count}/3", acc.day_trade_count < 3,
          "PDT limit OK" if acc.day_trade_count < 3 else "WARNING: approaching PDT limit")

    pos = c.get_positions()
    check(f"Open positions: {len(pos)}", True,
          ", ".join(f"{p.symbol}({p.qty}@{p.avg_entry_price:.1f},PnL={p.unrealized_pnl:+.0f})" for p in pos))

    clock = c.get_clock()
    ny_time = time.strftime('%Y-%m-%d %H:%M ET', time.localtime(time.time() - 4*3600))
    check(f"Market: {'OPEN' if clock.is_open else 'CLOSED'}", True,
          f"NY time: {ny_time} | Next open: {clock.next_open}")

    # Verify we can fetch live prices for common tickers
    test_tickers = ["AAPL", "SPY", "MSFT"]
    prices_ok = 0
    for t in test_tickers:
        try:
            p = c.get_last_price(t)
            if p and p > 0:
                prices_ok += 1
        except Exception:
            pass
    check(f"Price feed: {prices_ok}/{len(test_tickers)} tickers", prices_ok >= 2)

except Exception as e:
    check("Alpaca connection", False, str(e)[:200])

# 3. Supabase
print("\n--- Supabase ---")
try:
    from db.client import table_exists, rest_get
    tables = ["daily_signals", "trades", "portfolio_snapshots", "regime_history",
              "debate_records", "factor_performance", "decay_alerts"]
    ok_count = sum(1 for t in tables if table_exists(t))
    check(f"Tables: {ok_count}/{len(tables)}", ok_count == len(tables),
          "All 7 present" if ok_count == 7 else f"Missing: {[t for t in tables if not table_exists(t)]}")

    # Check recent signals
    signals = rest_get("daily_signals", select="date,market", order="date.desc", limit=5)
    if signals:
        check(f"Recent signals: {len(signals)} rows", True,
              f"Latest dates: {', '.join(set(s['date'] for s in signals))}")

except Exception as e:
    check("Supabase", False, str(e)[:200])

# 4. Signals Freshness
print("\n--- Signals ---")
try:
    state_path = PROJECT / "data" / "brain_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        last_time = state.get("last_scoring_time", "NEVER")
        check(f"Last scoring: {last_time[:19]}", bool(last_time),
              f"Top signals: {', '.join(s.get('ticker','?') + '=' + str(round(s.get('composite_score',0),2)) for s in state.get('latest_signals',[]))}")
    else:
        check("brain_state.json", False, "File missing -- daemon may not have completed initial scoring")
except Exception as e:
    check("Signal state file", False, str(e)[:200])

# 5. Claude API
print("\n--- Claude API ---")
try:
    from config import ANTHROPIC_API_KEY
    if ANTHROPIC_API_KEY:
        import httpx
        # POST a minimal request to validate auth
        r = httpx.post("https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                                 "content-type": "application/json"},
                        json={"model": "claude-haiku-4-5-20251001", "max_tokens": 1,
                              "messages": [{"role": "user", "content": "hi"}]},
                        timeout=15)
        check("Claude API reachable", r.status_code == 200,
              f"Haiku responded in {r.json().get('usage',{}).get('output_tokens',0)} tokens")
    else:
        check("ANTHROPIC_API_KEY", False, "Not set")
except Exception as e:
    check("Claude API", False, str(e)[:200])

# 6. Execution Engine Import Test
print("\n--- Execution ---")
try:
    from execution.engine import ExecutionEngine
    from risk.manager import RiskManager
    from strategy.decider import StrategyDecider
    check("Engine imports", True, "Engine + Risk + Decider all importable")
except Exception as e:
    check("Engine imports", False, str(e)[:200])

# 7. Dashboard Health
print("\n--- Dashboard ---")
try:
    import httpx
    r = httpx.get(f"http://localhost:{DASHBOARD_PORT}/api/state", timeout=10)
    check(f"Dashboard :{DASHBOARD_PORT}", r.status_code == 200,
          f"Uptime: {r.json().get('uptime_seconds', 0):.0f}s")
except Exception as e:
    check(f"Dashboard :{DASHBOARD_PORT}", False, "Not responding -- daemon may still be starting")

# Summary
print("\n" + "=" * 60)
passed = sum(1 for _, ok, _ in RESULTS if ok)
failed = len(RESULTS) - passed
print(f"RESULTS: {passed}/{len(RESULTS)} passed, {failed} failed")
if failed:
    print("\nFAILED CHECKS:")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAIL: {name} -- {detail}")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
