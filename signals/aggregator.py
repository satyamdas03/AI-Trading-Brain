"""Signal aggregation — runs factor computations and maps to composite scores.

Phase 1 uses simple cross-sectional percentile scoring via pandas.
When NeuralQuant is installed, delegates to SignalEngine.compute().
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import SIGNAL_TOP_N, REGIME_N_STATES, TREND_FILTER_ENABLED

logger = logging.getLogger(__name__)


def compute_signals(
    prices_df: pd.DataFrame,
    fundamentals: dict[str, dict],
    macro: dict,
    market: str = "US",
) -> pd.DataFrame:
    """Compute multi-factor signals for a universe.

    Args:
        prices_df: DataFrame with columns [ticker, date, close, volume]
        fundamentals: dict of ticker -> fundamental data dict
        macro: macro snapshot dict from macro_data.fetch_macro_snapshot()
        market: "US" or "IN"

    Returns:
        DataFrame sorted by composite_score descending, with columns:
        ticker, quality_score, momentum_score, value_score,
        low_vol_score, composite_score, regime_id
    """
    if prices_df.empty:
        return pd.DataFrame()

    tickers = sorted(prices_df["ticker"].unique())
    logger.info(f"Computing signals for {len(tickers)} tickers ({market})")

    scores = pd.DataFrame({"ticker": tickers})

    # 1. Momentum (12-1 return cross-sectional)
    scores["momentum_score"] = _compute_momentum_scores(prices_df, tickers).values

    # 2. Low volatility (inverse of realized vol)
    scores["low_vol_score"] = _compute_low_vol_scores(prices_df, tickers).values

    # 3. Value (from fundamentals: P/E, P/B based)
    scores["value_score"] = _compute_value_scores(fundamentals, tickers).values

    # 4. Quality (ROE, profit margins from fundamentals)
    scores["quality_score"] = _compute_quality_scores(fundamentals, tickers).values

    # 5. Volatility Rank (current vol vs historical distribution — low rank = bullish)
    scores["vol_rank_score"] = _compute_vol_rank_scores(prices_df, tickers).values

    # 6. Composite (equal-weighted mean of available factors)
    factor_cols = ["momentum_score", "low_vol_score", "value_score", "quality_score", "vol_rank_score"]
    scores["composite_score"] = scores[factor_cols].mean(axis=1)

    # 6. Regime detection (simplified — Phase 4 adds full HMM)
    scores["regime_id"] = _detect_regime_simple(macro)
    scores["regime_confidence"] = 0.5

    # 7. Trend filter: block entries when SPX below 200MA
    trend_passed = True
    if market == "US" and TREND_FILTER_ENABLED:
        spx_vs_ma = macro.get("spx_vs_200ma", 0.02)
        if spx_vs_ma < 0:
            logger.warning(f"SPX below 200MA ({spx_vs_ma:.2%}) — blocking new entries")
            trend_passed = False
    scores["trend_filter_passed"] = trend_passed

    # 8. Normalize composites to 0-1 range
    for col in factor_cols + ["composite_score"]:
        col_min = scores[col].min()
        col_max = scores[col].max()
        if col_max > col_min:
            scores[col] = (scores[col] - col_min) / (col_max - col_min)
        else:
            scores[col] = 0.5

    # Sort by composite descending
    scores = scores.sort_values("composite_score", ascending=False).reset_index(drop=True)

    return scores


def compute_crypto_signals(
    prices_df: pd.DataFrame,
    macro: dict,
) -> pd.DataFrame:
    """Compute crypto-specific signals: momentum + low_vol only (no value/quality).

    Crypto lacks P/E, ROE, etc. Composite = 0.5×momentum + 0.5×low_vol.
    """
    if prices_df.empty:
        return pd.DataFrame()

    tickers = sorted(prices_df["ticker"].unique())
    logger.info(f"Computing crypto signals for {len(tickers)} tickers")

    scores = pd.DataFrame({"ticker": tickers})
    scores["momentum_score"] = _compute_momentum_scores(prices_df, tickers).values
    scores["low_vol_score"] = _compute_low_vol_scores(prices_df, tickers).values
    scores["value_score"] = 0.5
    scores["quality_score"] = 0.5

    scores["composite_score"] = 0.5 * scores["momentum_score"] + 0.5 * scores["low_vol_score"]

    scores["regime_id"] = _detect_crypto_regime(macro)
    scores["regime_confidence"] = 0.5
    scores["trend_filter_passed"] = True

    for col in ["momentum_score", "low_vol_score", "composite_score"]:
        col_min = scores[col].min()
        col_max = scores[col].max()
        if col_max > col_min:
            scores[col] = (scores[col] - col_min) / (col_max - col_min)
        else:
            scores[col] = 0.5

    return scores.sort_values("composite_score", ascending=False).reset_index(drop=True)


def _detect_crypto_regime(macro: dict) -> int:
    """Simplified crypto regime: use VIX as fear proxy. High VIX = risk-off for crypto."""
    vix = macro.get("vix", 18)
    if vix > 30:
        return 3
    if vix > 22:
        return 2
    if vix < 16:
        return 1
    return 4


def get_top_signals(scores: pd.DataFrame, n: int = SIGNAL_TOP_N) -> pd.DataFrame:
    """Return top-N scored tickers."""
    return scores.head(n)


def _compute_momentum_scores(prices_df: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """12-1 momentum: price return from 12mo ago to 1mo ago, cross-sectional percentile."""
    if prices_df.empty or "ticker" not in prices_df.columns:
        return pd.Series([0.5] * len(tickers), index=tickers)

    scores = {}
    for t in tickers:
        t_data = prices_df[prices_df["ticker"] == t].sort_values("date")
        closes = t_data["close"].values
        if len(closes) < 200:
            scores[t] = np.nan
            continue
        p_recent = closes[-21] if len(closes) >= 21 else closes[-1]
        p_past = closes[-200]
        if p_past > 0:
            scores[t] = (p_recent - p_past) / p_past
        else:
            scores[t] = np.nan

    raw = pd.Series(scores)
    # Remove NaN, rank, then reindex
    valid = raw.dropna()
    if valid.empty or valid.nunique() <= 1:
        return pd.Series([0.5] * len(tickers), index=tickers)

    clipped = valid.clip(lower=valid.quantile(0.01), upper=valid.quantile(0.99))
    ranked = clipped.rank(pct=True)
    # Map back to full ticker list, fill missing with 0.5
    result = ranked.reindex(tickers).fillna(0.5)
    return result


def _compute_low_vol_scores(prices_df: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """Inverse of 60-day realized volatility — lower vol = higher score."""
    if prices_df.empty:
        return pd.Series([0.5] * len(tickers), index=tickers)

    scores = {}
    for t in tickers:
        t_data = prices_df[prices_df["ticker"] == t].sort_values("date")
        closes = t_data["close"].values
        if len(closes) < 40:
            scores[t] = np.nan
            continue
        returns = np.diff(closes[-40:]) / closes[-40:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) < 10:
            scores[t] = np.nan
            continue
        vol = float(np.std(returns))
        scores[t] = 1.0 / vol if vol > 0 else np.nan

    raw = pd.Series(scores)
    valid = raw.dropna()
    if valid.empty or valid.nunique() <= 1:
        return pd.Series([0.5] * len(tickers), index=tickers)

    clipped = valid.clip(lower=valid.quantile(0.01), upper=valid.quantile(0.99))
    ranked = clipped.rank(pct=True, ascending=True)  # lower vol = higher score
    return ranked.reindex(tickers).fillna(0.5)


def _compute_value_scores(fundamentals: dict[str, dict], tickers: list[str]) -> pd.Series:
    """Value composite: low P/E + low P/B = high score."""
    if not fundamentals:
        return pd.Series([0.5] * len(tickers), index=tickers)

    scores = {}
    for t in tickers:
        f = fundamentals.get(t, {}) or {}
        pe = f.get("pe_ratio")
        pb = f.get("pb_ratio")
        score = 0.5
        if pe is not None and isinstance(pe, (int, float)) and pe > 0:
            score -= 0.25 * min(pe / 30, 1.0)
        if pb is not None and isinstance(pb, (int, float)) and pb > 0:
            score -= 0.25 * min(pb / 5, 1.0)
        scores[t] = max(0.0, min(1.0, score))

    raw = pd.Series(scores)
    if raw.nunique() <= 1:
        return pd.Series([0.5] * len(tickers), index=tickers)
    clipped = raw.clip(lower=raw.quantile(0.01), upper=raw.quantile(0.99))
    return clipped.rank(pct=True)


def _compute_quality_scores(fundamentals: dict[str, dict], tickers: list[str]) -> pd.Series:
    """Quality composite: ROE + profit margins."""
    if not fundamentals:
        return pd.Series([0.5] * len(tickers), index=tickers)

    scores = {}
    for t in tickers:
        f = fundamentals.get(t, {}) or {}
        roe = f.get("roe")
        margin = f.get("profit_margin")
        score = 0.5
        if roe is not None and isinstance(roe, (int, float)):
            score += 0.25 * min(max(roe, 0) / 0.30, 1.0)
        if margin is not None and isinstance(margin, (int, float)):
            score += 0.25 * min(max(margin, 0) / 0.25, 1.0)
        scores[t] = min(1.0, score)

    raw = pd.Series(scores)
    if raw.nunique() <= 1:
        return pd.Series([0.5] * len(tickers), index=tickers)
    clipped = raw.clip(lower=raw.quantile(0.01), upper=raw.quantile(0.99))
    return clipped.rank(pct=True)


def _compute_vol_rank_scores(prices_df: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """Volatility rank: current 60-day vol as percentile of 1-year rolling vol distribution.

    Low rank = current vol is calm relative to history (bullish).
    High rank = current vol is elevated vs history (bearish/fear spike).
    """
    if prices_df.empty:
        return pd.Series([0.5] * len(tickers), index=tickers)

    scores = {}
    for t in tickers:
        t_data = prices_df[prices_df["ticker"] == t].sort_values("date")
        closes = t_data["close"].values
        if len(closes) < 252:
            scores[t] = np.nan
            continue

        # Current 60-day vol (safe window slicing to avoid shape mismatch)
        window = closes[-61:]
        if len(window) < 2:
            scores[t] = np.nan
            continue
        recent_rets = np.diff(window) / window[:-1]
        recent_rets = recent_rets[np.isfinite(recent_rets)]
        if len(recent_rets) < 10:
            scores[t] = np.nan
            continue
        current_vol = float(np.std(recent_rets))

        # Historical rolling 60-day vol distribution (past year)
        rolling_vols = []
        for j in range(60, len(closes)):
            window_slice = closes[j - 60:j + 1]
            rets = np.diff(window_slice) / window_slice[:-1]
            rets = rets[np.isfinite(rets)]
            if len(rets) >= 10:
                rolling_vols.append(float(np.std(rets)))

        if len(rolling_vols) < 20:
            scores[t] = np.nan
            continue

        # Percentile of current vol within historical distribution
        # Low percentile = calm = bullish → score = 1 - percentile
        vol_percentile = sum(1 for v in rolling_vols if v <= current_vol) / len(rolling_vols)
        scores[t] = 1.0 - vol_percentile  # high score when vol is low

    raw = pd.Series(scores)
    valid = raw.dropna()
    if valid.empty or valid.nunique() <= 1:
        return pd.Series([0.5] * len(tickers), index=tickers)

    clipped = valid.clip(lower=valid.quantile(0.01), upper=valid.quantile(0.99))
    ranked = clipped.rank(pct=True)
    return ranked.reindex(tickers).fillna(0.5)


def _detect_regime_simple(macro: dict) -> int:
    """Simplified regime detection without trained HMM. Returns regime_id 1-4."""
    vix = macro.get("vix", 18)
    spx_vs_ma = macro.get("spx_vs_200ma", 0.02)
    hy_spread = macro.get("hy_spread_oas", 350)

    if vix > 30 or spx_vs_ma < -0.10:
        return 3  # Bear
    if vix > 22 or hy_spread > 500 or spx_vs_ma < 0.0:
        return 2  # Late-Cycle
    if spx_vs_ma > 0.03 and vix < 22:
        return 1  # Risk-On
    if spx_vs_ma > 0.0 and vix < 20:
        return 4  # Recovery
    return 2  # default Late-Cycle
