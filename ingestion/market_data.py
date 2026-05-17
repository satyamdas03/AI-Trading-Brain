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
    """Batch download price history. Returns DataFrame with columns: ticker, date, close, volume."""
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

            # Single ticker: flat columns
            if len(chunk) == 1:
                if close_col:
                    results.append(pd.DataFrame({
                        "ticker": chunk[0],
                        "date": data.index,
                        "close": data[close_col].values,
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
                            results.append(pd.DataFrame({
                                "ticker": t,
                                "date": t_data.index,
                                "close": t_data["Close"].values,
                                "volume": t_data["Volume"].values,
                            }))
                        except (KeyError, AttributeError):
                            continue
                else:
                    # (OHLCV, ticker) structure
                    for t in chunk:
                        try:
                            results.append(pd.DataFrame({
                                "ticker": t,
                                "date": data.index,
                                "close": data["Close"][t].values,
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
