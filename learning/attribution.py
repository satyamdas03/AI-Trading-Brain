"""Factor attribution — compute which factors predicted correctly for each trade.

When a trade closes, attribute the P&L to each signal factor:
  - Entry factor scores (quality, momentum, value, low_vol)
  - Regime context at entry
  - Forward return realized

Stores rolling performance metrics for weight optimization.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import DB_PATH

logger = logging.getLogger(__name__)


def attribute_trade(
    trade: dict,
    forward_returns: dict[str, float],  # ticker → forward N-day return
    factor_history: Optional[pd.DataFrame] = None,
) -> dict:
    """Attribute a closed trade's P&L to entry factors.

    Args:
        trade: Trade record with factor scores at entry
        forward_returns: Realized forward returns for each ticker
        factor_history: Historical factor scores for IC computation (optional)

    Returns:
        Attribution dict with per-factor contribution estimates
    """
    ticker = trade["ticker"]
    pnl_pct = trade.get("pnl_pct", 0) or 0
    composite = trade.get("composite_score", 0.5) or 0.5

    factors = {
        "quality": trade.get("quality_pct", 0.5) or 0.5,
        "momentum": trade.get("momentum_pct", 0.5) or 0.5,
        "value": trade.get("value_pct", 0.5) or 0.5,
        "low_vol": trade.get("low_vol_pct", 0.5) or 0.5,
    }

    # Factor contribution: how much each factor deviated from 0.5 × P&L direction
    contributions = {}
    for name, score in factors.items():
        deviation = score - 0.5  # range: -0.5 to +0.5
        # If factor was bullish (>0.5) and P&L positive → factor was right
        # If factor was bearish (<0.5) and P&L negative → factor was right
        if pnl_pct > 0:
            contributions[name] = deviation  # positive deviation = good for winners
        else:
            contributions[name] = -deviation  # negative deviation = good for losers

    forward_return = forward_returns.get(ticker)

    return {
        "trade_id": trade.get("trade_id", ""),
        "ticker": ticker,
        "pnl_pct": pnl_pct,
        "composite_score": composite,
        "factor_contributions": contributions,
        "forward_return": forward_return,
        "regime_id": trade.get("regime_id"),
        "exit_reason": trade.get("exit_reason", "unknown"),
    }


def compute_factor_ic(
    trades: list[dict],
    factor_history: pd.DataFrame,
    factor_names: list[str] = ("quality", "momentum", "value", "low_vol"),
) -> dict[str, float]:
    """Compute Information Coefficient (Spearman rank correlation) between factor scores and realized returns.

    Args:
        trades: List of closed trades with factor scores and pnl_pct
        factor_history: DataFrame with columns [ticker, date, <factor>_pct]
        factor_names: Factor column base names

    Returns:
        Dict of factor_name → IC value
    """
    ic_results = {}

    for factor in factor_names:
        scores = []
        returns = []

        for t in trades:
            score = t.get(f"{factor}_pct", 0.5) or 0.5
            pnl = t.get("pnl_pct", 0) or 0

            if score is not None and pnl is not None:
                scores.append(score)
                returns.append(pnl)

        if len(scores) < 5:
            ic_results[factor] = 0.0
            continue

        try:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(scores, returns)
            ic_results[factor] = float(ic) if not np.isnan(ic) else 0.0
        except ImportError:
            # Fallback: Pearson correlation
            scores_arr = np.array(scores)
            returns_arr = np.array(returns)
            if scores_arr.std() > 0 and returns_arr.std() > 0:
                ic = np.corrcoef(scores_arr, returns_arr)[0, 1]
                ic_results[factor] = float(ic) if not np.isnan(ic) else 0.0
            else:
                ic_results[factor] = 0.0

    return ic_results


def compute_hit_rate(trades: list[dict]) -> dict[str, float]:
    """Compute directional hit rate per factor — did the factor predict the right direction?

    Returns dict of factor_name → hit_rate (0.0-1.0)
    """
    factor_names = ["quality", "momentum", "value", "low_vol"]
    results = {}

    for factor in factor_names:
        correct = 0
        total = 0

        for t in trades:
            score = t.get(f"{factor}_pct", 0.5) or 0.5
            pnl = t.get("pnl_pct", 0) or 0

            if score is None or pnl is None:
                continue

            total += 1
            # Factor score > 0.5 = bullish, < 0.5 = bearish
            predicted_bullish = score > 0.5
            outcome_bullish = pnl > 0

            if predicted_bullish == outcome_bullish:
                correct += 1

        results[factor] = correct / total if total > 0 else 0.5

    return results


def rolling_factor_stats(trades: list[dict], window: int = 30) -> pd.DataFrame:
    """Compute rolling factor performance stats from trade history.

    Returns DataFrame with columns: date, factor, ic_rolling, hit_rate_rolling, trade_count
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    if "exit_date" not in df.columns:
        return pd.DataFrame()

    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("exit_date")

    rows = []
    factors = ["quality", "momentum", "value", "low_vol"]

    for i in range(window, len(df) + 1):
        window_trades = df.iloc[i - window:i].to_dict(orient="records")
        ic = compute_factor_ic(window_trades, pd.DataFrame(), factors)
        hit = compute_hit_rate(window_trades)

        for f in factors:
            rows.append({
                "date": df.iloc[i - 1]["exit_date"].strftime("%Y-%m-%d"),
                "factor": f,
                "ic_rolling": ic.get(f, 0),
                "hit_rate_rolling": hit.get(f, 0.5),
                "trade_count": len(window_trades),
            })

    return pd.DataFrame(rows)
