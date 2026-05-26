"""Put/Call Ratio sentiment scanner — injects options flow data into debate context.

Uses yfinance option_chain to fetch near-expiry ATM put/call volume ratios.
LIVE-ONLY — no historical options data available from yfinance at scale.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

# PCR sentiment thresholds
PCR_BULLISH = 0.7
PCR_BEARISH = 1.0


class OptionSentimentScanner:
    """Fetch put/call ratios for a list of tickers via yfinance option chain."""

    def __init__(self, max_tickers: int = 20):
        self._max_tickers = max_tickers
        self._cache: dict[str, dict] = {}

    def fetch_put_call_ratios(
        self, tickers: list[str], force_refresh: bool = False
    ) -> dict[str, dict]:
        """Fetch put/call volume ratios for a batch of tickers.

        For each ticker, fetches the nearest-expiry ATM option chain and computes:
          PCR = total put volume / total call volume

        Returns:
            Dict mapping ticker -> {
                "pcr": float or None,
                "sentiment": "bullish" | "neutral" | "bearish" | "unknown",
                "put_vol": float,
                "call_vol": float,
                "expiry": str or None,
            }
        """
        if not force_refresh and self._cache:
            cached = {t: self._cache.get(t) for t in tickers if t in self._cache}
            missing = [t for t in tickers if t not in self._cache]
            if not missing:
                return {t: self._cache[t] for t in tickers}
            tickers = missing

        results = {}
        for ticker in tickers[:self._max_tickers]:
            try:
                results[ticker] = self._fetch_single(ticker)
                time.sleep(0.15)  # rate limit yfinance
            except Exception as e:
                logger.debug(f"PCR fetch failed for {ticker}: {e}")
                results[ticker] = {
                    "pcr": None, "sentiment": "unknown",
                    "put_vol": 0, "call_vol": 0, "expiry": None,
                }

        self._cache.update(results)
        return results

    def _fetch_single(self, ticker: str) -> dict:
        """Fetch put/call ratio for a single ticker."""
        result = {
            "pcr": None, "sentiment": "unknown",
            "put_vol": 0, "call_vol": 0, "expiry": None,
        }

        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            if not expirations:
                return result

            # Use nearest expiry
            expiry = expirations[0]
            result["expiry"] = expiry

            chain = stock.option_chain(expiry)
            if chain is None:
                return result

            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                return result

            # Get current price to find ATM strikes
            try:
                price = float(stock.history(period="1d")["Close"].iloc[-1])
            except Exception:
                return result

            # Filter to near-ATM (±5% from current price)
            atm_calls = calls[
                (calls["strike"] >= price * 0.95)
                & (calls["strike"] <= price * 1.05)
            ]
            atm_puts = puts[
                (puts["strike"] >= price * 0.95)
                & (puts["strike"] <= price * 1.05)
            ]

            call_vol = atm_calls["volume"].sum() if "volume" in atm_calls.columns else 0
            put_vol = atm_puts["volume"].sum() if "volume" in atm_puts.columns else 0

            result["put_vol"] = float(call_vol)
            result["call_vol"] = float(put_vol)

            if call_vol > 0 and put_vol >= 0:
                pcr = float(put_vol / call_vol) if call_vol > 0 else None
                result["pcr"] = round(pcr, 4) if pcr is not None else None

                if pcr is not None:
                    if pcr < PCR_BULLISH:
                        result["sentiment"] = "bullish"
                    elif pcr > PCR_BEARISH:
                        result["sentiment"] = "bearish"
                    else:
                        result["sentiment"] = "neutral"
            else:
                # No volume data — treat as neutral
                result["sentiment"] = "neutral"

        except Exception as e:
            logger.debug(f"PCR error for {ticker}: {e}")

        return result

    def format_sentiment_context(self, pcr_data: dict[str, dict]) -> str:
        """Format put/call ratio data as context string for debate enrichment."""
        lines = []
        for ticker, data in pcr_data.items():
            pcr = data.get("pcr")
            sentiment = data.get("sentiment", "unknown")
            expiry = data.get("expiry", "?")

            if pcr is not None:
                lines.append(
                    f"{ticker}: PCR={pcr:.3f} ({sentiment}), "
                    f"P={data.get('put_vol', 0):.0f}/C={data.get('call_vol', 0):.0f}, exp={expiry}"
                )
            else:
                lines.append(f"{ticker}: PCR=N/A ({sentiment})")

        return " | ".join(lines) if lines else "No options data available"
