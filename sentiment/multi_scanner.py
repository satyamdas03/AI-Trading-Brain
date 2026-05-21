"""Multi-source sentiment scanner — orchestrates 3 free sources into one scan.

Sources:
  1. Reddit RSS (r/wallstreetbets, r/stocks, r/investing, r/StockMarket) — real trader sentiment, completely free, no API keys
  2. Finnhub (market news + ticker-specific news) — institutional news flow, free tier 60 req/min
  3. Financial RSS (Yahoo Finance, CNBC, MarketWatch, Seeking Alpha) — headlines, unlimited free

All produce FinancialTweet objects -> classified by same Claude Haiku pipeline
-> aggregated into SentimentScan with ticker signals.

Zero X/Twitter dependency. Zero API key requirements for Reddit (RSS). Zero cost.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from config import SIGNAL_TOP_N, SENTIMENT_SOURCES, SENTIMENT_MAX_ITEMS
from sentiment.x_scanner import (
    FinancialTweet,
    FinancialTweetClassifier,
    SentimentResult,
    SentimentScan,
    XSentimentScanner,  # reuse its _aggregate method
)

logger = logging.getLogger(__name__)

# Fallback demo tweets — used only when ALL sources return nothing
_DEMO_TWEETS: list[dict] | None = None  # lazy from x_scanner._fetch_demo_tweets


class MultiSourceScanner:
    """Unified scanner: fetch from all free sources -> classify -> aggregate -> signals."""

    def __init__(self):
        self._classifier = FinancialTweetClassifier()
        self._x_scanner = XSentimentScanner()  # reuse _aggregate

    def scan(
        self,
        max_tweets: int = 80,
        ticker_filter: list[str] | None = None,
        sources: list[str] | None = None,
    ) -> SentimentScan:
        """Run full scan across all free sources. Returns SentimentScan.

        Args:
            max_tweets: max items to classify (controls Anthropic API cost)
            ticker_filter: optional list of tickers to keep
            sources: which sources to use — ["reddit", "finnhub", "rss"]
                     default: all 3
        """
        start = time.perf_counter()
        scan = SentimentScan()

        if sources is None:
            sources = ["reddit", "finnhub", "rss"]

        # 1. Fetch from all sources in parallel (logically, sequential to respect rate limits)
        all_items: list[FinancialTweet] = []

        if "reddit" in sources:
            try:
                from sentiment.headline_scanner import fetch_reddit_headlines
                reddit_items = fetch_reddit_headlines(max_headlines=80)
                logger.info(f"Reddit RSS: {len(reddit_items)} posts")
                all_items.extend(reddit_items)
            except ImportError as e:
                logger.warning(f"Reddit RSS import failed: {e}")
            except Exception as e:
                logger.warning(f"Reddit RSS failed: {e}")

        if "finnhub" in sources:
            try:
                from sentiment.finnhub_scanner import fetch_finnhub_news
                finnhub_items = fetch_finnhub_news(max_articles=60)
                logger.info(f"Finnhub: {len(finnhub_items)} articles")
                all_items.extend(finnhub_items)
            except ImportError as e:
                logger.warning(f"Finnhub scanner import failed: {e}")
            except Exception as e:
                logger.warning(f"Finnhub scanner failed: {e}")

        if "rss" in sources:
            try:
                from sentiment.headline_scanner import fetch_headlines
                rss_items = fetch_headlines(max_headlines=80)
                logger.info(f"RSS: {len(rss_items)} headlines")
                all_items.extend(rss_items)
            except ImportError as e:
                logger.warning(f"Headline scanner import failed: {e}")
            except Exception as e:
                logger.warning(f"Headline scanner failed: {e}")

        # Deduplicate by content hash
        seen: set[str] = set()
        deduped: list[FinancialTweet] = []
        for item in all_items:
            h = f"{item.author}:{item.text[:80]}"
            if h not in seen:
                seen.add(h)
                deduped.append(item)

        logger.info(f"Multi-source: {len(all_items)} raw -> {len(deduped)} deduped")

        # 2. Apply ticker filter if specified
        if ticker_filter:
            from sentiment.x_scanner import _mentions_ticker
            ticker_set = {t.upper() for t in ticker_filter}
            deduped = [t for t in deduped if _mentions_ticker(t.text, ticker_set)]

        # 3. Cap to max_tweets
        items_to_classify = deduped[:max_tweets]
        scan.tweets_scanned = len(items_to_classify)

        if not items_to_classify:
            # Fall back to demo data if absolutely nothing
            logger.warning("No items from any source — using demo data")
            from sentiment.x_scanner import _fetch_demo_tweets
            items_to_classify = _fetch_demo_tweets(15)
            scan.tweets_scanned = len(items_to_classify)

        if not items_to_classify:
            scan.errors.append("No items from any source + demo unavailable")
            scan.latency_ms = (time.perf_counter() - start) * 1000
            return scan

        # 4. Classify using same Claude Haiku pipeline
        logger.info(f"Classifying {len(items_to_classify)} items via Claude Haiku...")
        results = self._classifier.classify_batch(items_to_classify)
        scan.results = results

        # 5. Aggregate into ticker_signals (reuse XSentimentScanner._aggregate)
        self._x_scanner._aggregate(scan, results)
        scan.latency_ms = (time.perf_counter() - start) * 1000

        source_labels = "+".join(sources)
        logger.info(
            f"Multi-source scan ({source_labels}): {scan.tweets_scanned} items -> "
            f"{scan.bullish_count}B/{scan.bearish_count}S/"
            f"{scan.tweets_scanned - scan.bullish_count - scan.bearish_count}N "
            f"({scan.overall_sentiment}), {len(scan.ticker_signals)} ticker signals, "
            f"{scan.latency_ms:.0f}ms"
        )

        return scan


# ---- Top-level convenience ----

_scanner: Optional[MultiSourceScanner] = None


def get_scanner() -> MultiSourceScanner:
    global _scanner
    if _scanner is None:
        _scanner = MultiSourceScanner()
    return _scanner


def run_scan(max_tweets: int | None = None) -> SentimentScan:
    """Convenience: run a full multi-source sentiment scan.

    Replaces sentiment.x_scanner.run_scan() — same interface, free sources.
    Sources read from SENTIMENT_SOURCES config (comma-separated: reddit,finnhub,rss).
    """
    if max_tweets is None:
        max_tweets = SENTIMENT_MAX_ITEMS
    sources = [s.strip() for s in SENTIMENT_SOURCES.split(",") if s.strip()]
    return get_scanner().scan(max_tweets=max_tweets, sources=sources)
