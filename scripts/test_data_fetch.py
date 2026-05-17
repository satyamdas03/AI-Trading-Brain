"""Test price data fetching for replay engine."""
from ingestion.market_data import fetch_prices
from ingestion.universe import us_tickers

tickers = us_tickers()[:5]
print("Tickers:", tickers)
df = fetch_prices(tickers, period="6mo")
print(f"Rows: {len(df)}")
if not df.empty:
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    print(df.head(3))
else:
    print("Empty result")
