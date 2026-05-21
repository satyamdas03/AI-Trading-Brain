"""Finnhub market news scanner — free tier, 60 req/min.

Uses Finnhub /news endpoint to get latest market news with ticker tagging.
Each news item already has ticker metadata — no regex extraction needed.

Produces FinancialTweet objects for the classifier pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone

import httpx

from sentiment.x_scanner import FinancialTweet

logger = logging.getLogger(__name__)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_BASE = "https://finnhub.io/api/v1"

# Stock symbols to fetch news for — spread across sectors
_DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "GS", "BAC", "C", "WFC",
    "JNJ", "UNH", "PFE", "ABBV",
    "XOM", "CVX", "COP",
    "SPY", "QQQ", "IWM", "DIA",
    "AMD", "INTC", "AVGO", "CRM",
    "BA", "CAT", "GE", "LMT",
    "WMT", "COST", "HD", "LOW",
    "DIS", "NFLX", "CMCSA",
    "BTC", "ETH",
]

_PER_SYMBOL_LIMIT = 3  # articles per symbol


class FinnhubScanner:
    """Fetch market news from Finnhub API with ticker-level precision."""

    def fetch(
        self,
        symbols: list[str] | None = None,
        max_articles: int = 100,
        category: str = "general",
    ) -> list[FinancialTweet]:
        """Fetch market news. Returns FinancialTweet list.

        Strategy: fetch general market news (category=general) + ticker-specific
        news for top symbols to get ticker-tagged content.
        """
        if not FINNHUB_API_KEY:
            logger.warning("FINNHUB_API_KEY not set")
            return []

        if symbols is None:
            symbols = _DEFAULT_SYMBOLS

        seen_hashes: set[str] = set()
        articles: list[FinancialTweet] = []

        # 1. Fetch general market news (bulk, covers broad market)
        try:
            url = f"{FINNHUB_BASE}/news"
            params = {
                "token": FINNHUB_API_KEY,
                "category": category,
            }
            resp = httpx.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                for item in resp.json():
                    article = self._to_financial_tweet(item)
                    if article:
                        h = hashlib.md5(article.text[:100].encode()).hexdigest()
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            articles.append(article)
            else:
                logger.warning(f"Finnhub general news: HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"Finnhub general news failed: {e}")

        logger.debug(f"Finnhub general: {len(articles)} articles")

        # 2. Ticker-specific news for top symbols
        # Batch symbols to stay under 60 req/min rate limit
        ticker_articles = 0
        for symbol in symbols[:30]:  # limit to 30 symbols = 30 requests
            if len(articles) >= max_articles:
                break
            try:
                url = f"{FINNHUB_BASE}/company-news"
                params = {
                    "token": FINNHUB_API_KEY,
                    "symbol": symbol,
                    "from": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "to": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                }
                resp = httpx.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    for item in resp.json()[: _PER_SYMBOL_LIMIT]:
                        article = self._to_financial_tweet(item)
                        if article:
                            h = hashlib.md5(article.text[:100].encode()).hexdigest()
                            if h not in seen_hashes:
                                seen_hashes.add(h)
                                articles.append(article)
                                ticker_articles += 1
            except Exception:
                continue

        logger.info(
            f"Finnhub: {len(articles)} articles "
            f"(general={len(articles) - ticker_articles}, ticker={ticker_articles})"
        )
        return articles[:max_articles]

    def _to_financial_tweet(self, item: dict) -> FinancialTweet | None:
        """Convert Finnhub news item to FinancialTweet."""
        headline = item.get("headline", "")
        summary = item.get("summary", "")
        if not headline:
            return None

        # Skip super short/empty items
        if len(headline) < 15:
            return None

        text = f"{headline}\n{summary}" if summary else headline
        text = text[:600]

        article_id = str(item.get("id", hashlib.md5(text.encode()).hexdigest()[:10]))
        source = item.get("source", "finnhub")
        url = item.get("url", "")
        posted_at = datetime.fromtimestamp(
            item.get("datetime", 0), tz=timezone.utc
        ).isoformat()

        return FinancialTweet(
            tweet_id=f"fh_{article_id}",
            text=text,
            author=source,
            posted_at=posted_at,
            url=url,
        )


# ---- Convenience ----

_finnhub_scanner: FinnhubScanner | None = None


def get_finnhub_scanner() -> FinnhubScanner:
    global _finnhub_scanner
    if _finnhub_scanner is None:
        _finnhub_scanner = FinnhubScanner()
    return _finnhub_scanner


def fetch_finnhub_news(max_articles: int = 100) -> list[FinancialTweet]:
    """Convenience: fetch market news from Finnhub."""
    try:
        return get_finnhub_scanner().fetch(max_articles=max_articles)
    except Exception as e:
        logger.warning(f"Finnhub fetch failed: {e}")
        return []
