"""Debug: trace why factor functions return NaN."""
import pandas as pd
from ingestion.universe import us_tickers
from ingestion.market_data import build_snapshot

tickers = us_tickers()[:5]
print("Tickers:", tickers)
snapshot = build_snapshot(tickers, "US")
prices_df = snapshot.prices
print(f"Prices: {len(prices_df)} rows")
print(f"Tickers in prices: {prices_df['ticker'].nunique()}")
print(f"Tickers list: {sorted(prices_df['ticker'].unique())}")

# Manual momentum check
for t in tickers:
    t_data = prices_df[prices_df["ticker"] == t].sort_values("date")
    print(f"{t}: {len(t_data)} rows, close range {t_data['close'].min():.2f}-{t_data['close'].max():.2f}")
    closes = t_data["close"].values
    if len(closes) >= 21 and len(closes) >= 200:
        p_minus_1 = closes[-21]
        p_minus_12 = closes[-200]
        mom = (p_minus_1 - p_minus_12) / p_minus_12 if p_minus_12 > 0 else 0
        print(f"  p[-21]={p_minus_1:.2f}, p[-200]={p_minus_12:.2f}, mom={mom:.4f}")
    else:
        print(f"  NOT ENOUGH DATA: {len(closes)} closes")

# Build scores dict manually
scores = {}
for t in tickers:
    t_data = prices_df[prices_df["ticker"] == t].sort_values("date")
    closes = t_data["close"].values
    if len(closes) >= 200:
        p_minus_1 = closes[-21] if len(closes) >= 21 else closes[-1]
        p_minus_12 = closes[-200]
        scores[t] = (p_minus_1 - p_minus_12) / p_minus_12 if p_minus_12 > 0 else 0.0
    else:
        scores[t] = 0.0

print("\nScores dict:", scores)
raw = pd.Series(scores)
print(f"Raw series: min={raw.min()}, max={raw.max()}, dtype={raw.dtype}")
clipped = raw.clip(lower=raw.quantile(0.01), upper=raw.quantile(0.99))
print(f"Clipped: min={clipped.min()}, max={clipped.max()}")
result = clipped.rank(pct=True).fillna(0.5)
print(f"Final: min={result.min():.3f}, max={result.max():.3f}")
print(f"Values: {result.values}")
