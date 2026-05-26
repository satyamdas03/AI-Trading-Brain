"""Market data ingestion — wraps yfinance for price data and fundamentals.

Phase 1 uses yfinance directly. Phase 2+ optionally imports NeuralQuant's data_builder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """Aggregated market data for a list of tickers."""

    tickers: list[str]
    market: str  # "US" | "IN"
    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    fundamentals: dict[str, dict] = field(default_factory=dict)
    fetched_at: str = ""


def fetch_prices(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Batch download price history. Returns DataFrame with columns: ticker, date, close, high, low, volume."""
    if not tickers:
        return pd.DataFrame()

    results = []
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            data = yf.download(chunk, period=period, progress=False)
            if data.empty:
                continue

            close_col = "Close" if "Close" in data.columns else None
            vol_col = "Volume" if "Volume" in data.columns else None

            high_col = "High" if "High" in data.columns else None
            low_col = "Low" if "Low" in data.columns else None

            # Single ticker: flat columns
            if len(chunk) == 1:
                if close_col:
                    results.append(pd.DataFrame({
                        "ticker": chunk[0],
                        "date": data.index,
                        "close": data[close_col].values,
                        "high": data[high_col].values if high_col else data[close_col].values,
                        "low": data[low_col].values if low_col else data[close_col].values,
                        "volume": data[vol_col].values if vol_col else 0,
                    }))
                continue

            # Multi ticker: MultiIndex columns
            # Detect column structure: (ticker, OHLCV) or (OHLCV, ticker)
            if isinstance(data.columns, pd.MultiIndex):
                level0 = data.columns.get_level_values(0).unique()
                # If first level has tickers, use xs
                if all(t in level0 for t in chunk[:3]):
                    for t in chunk:
                        try:
                            t_data = data.xs(t, axis=1, level=0)
                            h = t_data["High"].values if "High" in t_data.columns else t_data["Close"].values
                            l = t_data["Low"].values if "Low" in t_data.columns else t_data["Close"].values
                            results.append(pd.DataFrame({
                                "ticker": t,
                                "date": t_data.index,
                                "close": t_data["Close"].values,
                                "high": h,
                                "low": l,
                                "volume": t_data["Volume"].values,
                            }))
                        except (KeyError, AttributeError):
                            continue
                else:
                    # (OHLCV, ticker) structure
                    for t in chunk:
                        try:
                            h = data["High"][t].values if "High" in data.columns else data["Close"][t].values
                            l = data["Low"][t].values if "Low" in data.columns else data["Close"][t].values
                            results.append(pd.DataFrame({
                                "ticker": t,
                                "date": data.index,
                                "close": data["Close"][t].values,
                                "high": h,
                                "low": l,
                                "volume": data["Volume"][t].values,
                            }))
                        except (KeyError, AttributeError):
                            continue
                    continue
        except Exception as e:
            logger.warning(f"Failed to fetch chunk {chunk[0]}..{chunk[-1]}: {e}")
            continue

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def compute_atr(prices_df: pd.DataFrame, period: int = 14) -> "pd.Series":
    """Compute Average True Range for a single ticker's price DataFrame.

    Args:
        prices_df: DataFrame with columns 'high', 'low', 'close'. Sorted by date ascending.
        period: ATR lookback period (default 14).

    Returns:
        Series of ATR values indexed by date, or empty Series if insufficient data.
    """
    if prices_df.empty or len(prices_df) < period + 1:
        return pd.Series(dtype=float)

    high = prices_df["high"]
    low = prices_df["low"]
    close = prices_df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(span=period, adjust=False).mean()


def get_latest_atr(ticker: str, prices_df: pd.DataFrame, period: int = 14) -> float | None:
    """Get the most recent ATR value for a ticker from price history.

    Args:
        ticker: Ticker symbol.
        prices_df: Full price history DataFrame with ticker, high, low, close columns.
        period: ATR lookback period.

    Returns:
        Latest ATR value or None if unavailable.
    """
    ticker_data = prices_df[prices_df["ticker"] == ticker].sort_values("date")
    if ticker_data.empty or len(ticker_data) < period + 1:
        return None
    atr = compute_atr(ticker_data, period)
    return atr.iloc[-1] if not atr.empty else None


def compute_all_atrs(prices_df: pd.DataFrame, period: int = 14) -> dict[str, float | None]:
    """Compute latest ATR for all tickers in a price DataFrame.

    Args:
        prices_df: Multi-ticker price DataFrame with ticker, high, low, close columns.
        period: ATR lookback period.

    Returns:
        Dict mapping ticker -> latest ATR value (or None if unavailable).
    """
    result = {}
    for ticker in prices_df["ticker"].unique():
        result[ticker] = get_latest_atr(ticker, prices_df, period)
    return result


def fetch_fundamentals(ticker: str) -> dict:
    """Fetch fundamental data for a single ticker via yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "ticker": ticker,
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
        }
    except Exception:
        return {"ticker": ticker, "sector": "Unknown"}


def fetch_fundamentals_batch(tickers: list[str]) -> dict[str, dict]:
    """Batch fetch fundamentals for multiple tickers."""
    # yfinance doesn't support true batch — fetch sequentially
    result = {}
    for t in tickers:
        result[t] = fetch_fundamentals(t)
    return result


def build_snapshot(tickers: list[str], market: str) -> MarketSnapshot:
    """Build a MarketSnapshot with prices and fundamentals for a ticker list."""
    from datetime import datetime

    prices = fetch_prices(tickers, period="2y")  # 2y for reliable factor computation
    from config import US_UNIVERSE_SIZE
    fundamentals = fetch_fundamentals_batch(tickers[:US_UNIVERSE_SIZE])

    return MarketSnapshot(
        tickers=tickers,
        market=market,
        prices=prices,
        fundamentals=fundamentals,
        fetched_at=datetime.utcnow().isoformat(),
    )
