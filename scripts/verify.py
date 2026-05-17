"""Verification script — confirms all APIs reachable and config valid.

Run: python scripts/verify.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def main():
    errors = []

    def check(name: str, condition: bool, detail: str = ""):
        if condition:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}  {detail}")
            errors.append(name)

    print("=" * 50)
    print("AI Trading Brain — Verification")
    print("=" * 50)

    # 1. Config
    print("\n[Config]")
    check(".env loaded", config._ENV_PATH is not None)
    check("ANTHROPIC_BASE_URL killed", "ANTHROPIC_BASE_URL" not in __import__("os").environ)
    check("ANTHROPIC_API_KEY set", bool(config.ANTHROPIC_API_KEY), "Set ANTHROPIC_API_KEY in .env")

    # 2. Supabase
    print("\n[Supabase]")
    check("SUPABASE_URL set", bool(config.SUPABASE_URL), "Set SUPABASE_URL in .env")
    check("SUPABASE_SERVICE_ROLE_KEY set", bool(config.SUPABASE_SERVICE_ROLE_KEY))

    if config.SUPABASE_URL and config.SUPABASE_SERVICE_ROLE_KEY:
        try:
            from db.client import rest_get, table_exists
            # Check any accessible table — score_cache exists from NeuralQuant
            rows = rest_get("score_cache", limit=1)
            check("Supabase reachable (score_cache)", True, f"({len(rows)} rows)")
        except Exception as e:
            check("Supabase reachable", False, str(e)[:80])
        try:
            from db.client import rest_get
            rows = rest_get("daily_signals", limit=1)
            check("daily_signals table exists", True, f"({len(rows)} rows)")
        except Exception:
            check("daily_signals table exists", False, "Run db/migrations/001_init_trading.sql in Supabase SQL Editor")

    # 3. Alpaca
    print("\n[Alpaca]")
    check("APCA_API_KEY_ID set", bool(config.ALPACA_API_KEY), "Set APCA_API_KEY_ID in .env")
    check("APCA_API_SECRET_KEY set", bool(config.ALPACA_SECRET_KEY))
    check("Paper trading mode", config.ALPACA_PAPER)

    if config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY:
        try:
            import os
            os.environ.setdefault("APCA_API_KEY_ID", config.ALPACA_API_KEY)
            os.environ.setdefault("APCA_API_SECRET_KEY", config.ALPACA_SECRET_KEY)
            from execution.alpaca_client import AlpacaClient
            client = AlpacaClient()
            acc = client.get_account()
            check("Alpaca API reachable", True,
                  f"Account: {acc.account_number}, Equity=${acc.equity:,.2f}")
        except Exception as e:
            check("Alpaca API reachable", False, str(e)[:80])

    # 4. Anthropic
    print("\n[Anthropic]")
    if config.ANTHROPIC_API_KEY:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
            resp = client.messages.create(
                model=config.CLASSIFICATION_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'OK'"}],
            )
            text = resp.content[0].text if hasattr(resp.content[0], "text") else str(resp.content[0])
            check("Anthropic API reachable", "OK" in text, text[:40])
        except Exception as e:
            check("Anthropic API reachable", False, str(e)[:80])
    else:
        check("Anthropic API reachable", False, "No API key")

    # 5. yfinance
    print("\n[yfinance]")
    try:
        import yfinance as yf
        ticker = yf.Ticker("SPY")
        info = ticker.info
        check("yfinance reachable", "currentPrice" in info or "regularMarketPrice" in info)
    except Exception as e:
        check("yfinance reachable", False, str(e)[:80])

    # 6. Imports
    print("\n[Internal Imports]")
    try:
        from ingestion.universe import us_tickers, in_tickers
        us = us_tickers()
        check("ingestion.universe (US)", len(us) > 0, f"{len(us)} tickers")
    except Exception as e:
        check("ingestion.universe", False, str(e)[:80])

    try:
        from ingestion.macro_data import fetch_macro_snapshot
        macro = fetch_macro_snapshot()
        check("ingestion.macro_data", "vix" in macro, f"VIX={macro.get('vix')}")
    except Exception as e:
        check("ingestion.macro_data", False, str(e)[:80])

    try:
        from signals.aggregator import compute_signals
        check("signals.aggregator", True)
    except Exception as e:
        check("signals.aggregator", False, str(e)[:80])

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"FAILED: {len(errors)} check(s)")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All checks passed. Ready to start trading brain.")
        sys.exit(0)


if __name__ == "__main__":
    main()
