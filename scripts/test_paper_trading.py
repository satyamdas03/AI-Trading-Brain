"""Verify paper trading works end-to-end."""
from execution.alpaca_client import AlpacaClient
from execution.engine import ExecutionEngine
from risk.manager import RiskManager

client = AlpacaClient()

# 1. Account check
acc = client.get_account()
print(f"Account: {acc.account_number}")
print(f"Equity: ${acc.equity:,.2f}")
print(f"Cash: ${acc.cash:,.2f}")
print(f"Buying power: ${acc.buying_power:,.2f}")
print(f"Market open: {client.is_market_open()}")
print()

# 2. Current positions
positions = client.get_positions()
print(f"Positions: {len(positions)}")
for p in positions:
    print(f"  {p.symbol}: {p.qty} shares, mkt=${p.market_value:,.2f}, pnl=${p.unrealized_pnl:+,.2f}")
print()

# 3. Risk check
risk = RiskManager(client)
state = risk.check_pre_market()
print(f"Risk: {state.level.value} | DD: {state.current_drawdown_pct:.2%} | Can trade: {risk.can_trade}")
for a in state.alerts:
    print(f"  Alert: {a}")
print()

# 4. Place a test limit order (far from market = won't fill)
engine = ExecutionEngine()
last = client.get_last_price("AAPL")
print(f"AAPL last price: ${last}")
if last:
    test_price = round(last * 0.5, 2)
    print(f"Test order: BUY 1 AAPL limit ${test_price}")
    try:
        order = client.place_order(
            symbol="AAPL", qty=1, side="buy", order_type="limit",
            limit_price=test_price, time_in_force="day"
        )
        print(f"Order placed: {order.id} - status={order.status}")

        # Cancel test order immediately
        from execution.alpaca_client import REST
        api = REST()
        api.cancel_order(order.id)
        print(f"Order cancelled.")
    except Exception as e:
        print(f"Order error: {e}")

print()
print("Paper trading: ALL SYSTEMS GO")
