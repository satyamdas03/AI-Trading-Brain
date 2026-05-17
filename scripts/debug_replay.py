"""Debug replay engine factor computation."""
from learning.historical_replay import HistoricalReplay

replay = HistoricalReplay()
prices_df = replay._load_or_fetch_prices()
print(f"Prices: {len(prices_df)} rows, cols={list(prices_df.columns)}")
print(f"Tickers: {prices_df['ticker'].nunique()}")
print(f"Dates: {prices_df['date'].min()} -> {prices_df['date'].max()}")
print(f"Date dtype: {prices_df['date'].dtype}")

scored = replay._compute_factor_scores(prices_df)
print(f"\nScored: {len(scored)} rows")
print(f"Has composite_score: {'composite_score' in scored.columns}")
print(f"Has momentum_score: {'momentum_score' in scored.columns}")

if "composite_score" in scored.columns:
    valid = scored[scored["composite_score"].notna()]
    above = valid[valid["composite_score"] > 0.55]
    print(f"Non-null composite: {len(valid)}")
    print(f"Score > 0.55: {len(above)}")
    print(f"Score mean: {valid['composite_score'].mean():.3f}, max: {valid['composite_score'].max():.3f}")

    if len(above) > 0:
        print(f"\nSample tickers > 0.55:")
        sample = above[["ticker", "date", "close", "composite_score"]].head(10)
        for _, r in sample.iterrows():
            print(f"  {r['ticker']} {r['date']}: close={r['close']:.2f} composite={r['composite_score']:.3f}")
