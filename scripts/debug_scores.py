"""Debug signal scores."""
from ingestion.universe import us_tickers
from ingestion.macro_data import fetch_macro_snapshot
from ingestion.market_data import build_snapshot
from signals.aggregator import compute_signals

macro = fetch_macro_snapshot()
tickers = us_tickers()[:20]
print("Tickers:", tickers[:5], "...")

snapshot = build_snapshot(tickers, "US")
print("Prices shape:", snapshot.prices.shape)
print("Prices cols:", list(snapshot.prices.columns))
print("Tickers in prices:", snapshot.prices["ticker"].nunique())
print("Fundamentals count:", len(snapshot.fundamentals))

for t in tickers[:5]:
    t_data = snapshot.prices[snapshot.prices["ticker"] == t].sort_values("date")
    if not t_data.empty:
        print(f"  {t}: {len(t_data)} rows, {t_data['date'].min()} -> {t_data['date'].max()}")

# Debug momentum computation manually
from signals.aggregator import _compute_momentum_scores
mom_scores = _compute_momentum_scores(snapshot.prices, tickers)
print("\nRaw momentum scores:")
print(f"  min={mom_scores.min():.4f}, max={mom_scores.max():.4f}, mean={mom_scores.mean():.4f}")
print(f"  zeros: {(mom_scores == 0).sum()}, unique: {mom_scores.nunique()}")

scores = compute_signals(snapshot.prices, snapshot.fundamentals, macro, "US")
print("\nScores:")
print(scores[["ticker", "momentum_score", "low_vol_score", "value_score", "quality_score", "composite_score"]].head(10))
print("\nScore ranges:")
for c in ["momentum_score", "low_vol_score", "value_score", "quality_score", "composite_score"]:
    print(f"  {c}: {scores[c].min():.3f} - {scores[c].max():.3f} (mean={scores[c].mean():.3f})")
