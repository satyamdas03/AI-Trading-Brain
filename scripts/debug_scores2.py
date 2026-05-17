"""Debug signal scores — check normalization step."""
import numpy as np
import pandas as pd
from ingestion.universe import us_tickers
from ingestion.macro_data import fetch_macro_snapshot
from ingestion.market_data import build_snapshot
from signals.aggregator import (
    _compute_momentum_scores,
    _compute_low_vol_scores,
    _compute_value_scores,
    _compute_quality_scores,
    _detect_regime_simple,
)

macro = fetch_macro_snapshot()
tickers = us_tickers()[:10]
print("Tickers:", tickers)

snapshot = build_snapshot(tickers, "US")

scores = pd.DataFrame({"ticker": tickers})
scores["momentum_score"] = _compute_momentum_scores(snapshot.prices, tickers)
scores["low_vol_score"] = _compute_low_vol_scores(snapshot.prices, tickers)
scores["value_score"] = _compute_value_scores(snapshot.fundamentals, tickers)
scores["quality_score"] = _compute_quality_scores(snapshot.fundamentals, tickers)

print("\nBefore normalization:")
for c in ["momentum_score", "low_vol_score", "value_score", "quality_score"]:
    print(f"  {c}: min={scores[c].min():.3f}, max={scores[c].max():.3f}, nunique={scores[c].nunique()}")

factor_cols = ["momentum_score", "low_vol_score", "value_score", "quality_score"]
scores["composite_score"] = scores[factor_cols].mean(axis=1)
print(f"  composite_score: min={scores['composite_score'].min():.3f}, max={scores['composite_score'].max():.3f}")

# Normalize
print("\nNormalizing...")
for col in factor_cols + ["composite_score"]:
    col_min = scores[col].min()
    col_max = scores[col].max()
    print(f"  {col}: range=[{col_min:.4f}, {col_max:.4f}]", end="")
    if col_max > col_min:
        scores[col] = (scores[col] - col_min) / (col_max - col_min)
        print(f" -> new=[{scores[col].min():.3f}, {scores[col].max():.3f}]")
    else:
        scores[col] = 0.5
        print(" -> CONSTANT -> 0.5")

print("\nAfter normalization:")
for c in ["momentum_score", "low_vol_score", "value_score", "quality_score", "composite_score"]:
    print(f"  {c}: min={scores[c].min():.3f}, max={scores[c].max():.3f}")

print("\nTop by composite:")
print(scores.sort_values("composite_score", ascending=False)[["ticker", "composite_score", "momentum_score"]].head(5))
