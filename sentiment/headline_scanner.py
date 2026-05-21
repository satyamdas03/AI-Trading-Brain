"""Financial RSS headline scanner — completely free, no API keys needed.

Parses RSS/Atom feeds from major financial news sources + Reddit subreddits.
Headlines are concise (perfect for classifier) and come pre-tagged with source credibility.

Sources:
  - Yahoo Finance, CNBC, MarketWatch, Seeking Alpha, Investing.com, Google Finance
  - Reddit: r/wallstreetbets, r/stocks, r/investing, r/StockMarket, r/trading212
  - Fallback: Yahoo Finance v2, CNBC Tech

Produces FinancialTweet objects for the classifier pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional
from xml.etree import ElementTree as ET

import httpx

from sentiment.x_scanner import FinancialTweet

logger = logging.getLogger(__name__)

# RSS feed URLs — all free, no API keys
_FINANCIAL_FEEDS = [
    # Yahoo Finance
    ("YahooFinance", "https://finance.yahoo.com/news/rssindex"),
    # CNBC
    ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    # MarketWatch
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    # Seeking Alpha
    ("SeekingAlpha", "https://seekingalpha.com/market_currents.xml"),
    # Investing.com
    ("Investing", "https://www.investing.com/rss/news.rss"),
    # Google Finance (US Markets)
    ("GoogleFin", "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en"),
]

# Reddit RSS feeds — free, no API keys, real-time trader sentiment
_REDDIT_FEEDS = [
    ("r/wallstreetbets", "https://www.reddit.com/r/wallstreetbets/new/.rss?limit=100"),
    ("r/stocks", "https://www.reddit.com/r/stocks/new/.rss?limit=100"),
    ("r/investing", "https://www.reddit.com/r/investing/new/.rss?limit=100"),
    ("r/StockMarket", "https://www.reddit.com/r/StockMarket/new/.rss?limit=100"),
    ("r/trading212", "https://www.reddit.com/r/trading212/new/.rss?limit=100"),
    ("r/options", "https://www.reddit.com/r/options/new/.rss?limit=100"),
]

# Fallback: simpler feeds if the above fail
_FALLBACK_FEEDS = [
    ("YahooFin2", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY,AAPL,MSFT,NVDA&region=US&lang=en-US"),
    ("CNBCTech", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069"),
]

# Regex to strip HTML tags from description text
_HTML_RE = re.compile(r"<[^>]+>")
# Regex to find ticker symbols in headline text
_TICKER_HINT_RE = re.compile(r'\$([A-Z]{1,5})\b|NYSE:\s*([A-Z]{1,5})\b|NASDAQ:\s*([A-Z]{1,5})\b')

# Timeout for RSS fetches
_FETCH_TIMEOUT = 15

# Browser-like User-Agent — Reddit blocks non-browser UAs
_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


class HeadlineScanner:
    """Parse financial RSS feeds into FinancialTweet objects."""

    def fetch(
        self,
        feeds: list[tuple[str, str]] | None = None,
        max_headlines: int = 120,
    ) -> list[FinancialTweet]:
        """Fetch and parse financial RSS headlines. Returns FinancialTweet list.

        Args:
            feeds: list of (source_name, rss_url) tuples
            max_headlines: cap on total headlines returned
        """
        if feeds is None:
            feeds = _FINANCIAL_FEEDS

        seen_hashes: set[str] = set()
        headlines: list[FinancialTweet] = []

        for source, url in feeds:
            if len(headlines) >= max_headlines:
                break
            try:
                items = self._fetch_feed(url, source)
                for item in items:
                    h = hashlib.md5(item.text[:100].encode()).hexdigest()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        headlines.append(item)
                        if len(headlines) >= max_headlines:
                            break
            except Exception as e:
                logger.debug(f"{source} RSS failed: {e}")
                continue

        # Fallback to simpler feeds if nothing fetched
        if len(headlines) < 10:
            logger.debug(f"Only {len(headlines)} headlines, trying fallback feeds...")
            for source, url in _FALLBACK_FEEDS:
                if len(headlines) >= max_headlines:
                    break
                try:
                    items = self._fetch_feed(url, source)
                    for item in items:
                        h = hashlib.md5(item.text[:100].encode()).hexdigest()
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            headlines.append(item)
                except Exception:
                    continue

        logger.info(f"RSS: {len(headlines)} headlines from {len(feeds)} feeds")
        return headlines[:max_headlines]

    def _fetch_feed(
        self, url: str, source: str
    ) -> list[FinancialTweet]:
        """Fetch one RSS feed and parse entries."""
        resp = httpx.get(
            url,
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
            headers=_FETCH_HEADERS,
        )
        if resp.status_code != 200:
            logger.debug(f"{source} HTTP {resp.status_code}")
            return []

        raw = resp.text
        if not raw.strip():
            return []

        return self._parse_rss(raw, source)

    def _parse_rss(self, xml_text: str, source: str) -> list[FinancialTweet]:
        """Parse RSS/Atom XML into FinancialTweet list. Handles both formats."""
        items = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.debug(f"{source} XML parse error: {e}")
            return []

        # Detect format
        # RSS 2.0: <channel><item>
        # Atom: <feed><entry>

        # RSS 2.0
        for item in root.iter("item"):
            title = self._get_text(item, "title")
            description = self._get_text(item, "description")
            link = self._get_text(item, "link")
            pub_date = self._get_text(item, "pubDate")

            if not title or len(title) < 10:
                continue

            text = f"{title}\n{self._strip_html(description)[:300]}" if description else title
            text = text[:600]

            # Clean Reddit boilerplate (submitted by /u/..., [link] [comments])
            if source.startswith("r/"):
                text = self._clean_reddit_text(text)

            # Generate stable id from link or content hash
            if link:
                item_id = f"rss_{hashlib.md5(link.encode()).hexdigest()[:10]}"
            else:
                item_id = f"rss_{hashlib.md5(text.encode()).hexdigest()[:10]}"

            posted_at = ""
            if pub_date:
                try:
                    from email.utils import parsedate_to_datetime
                    posted_at = parsedate_to_datetime(pub_date).isoformat()
                except Exception:
                    pass

            items.append(FinancialTweet(
                tweet_id=item_id,
                text=text,
                author=source,
                posted_at=posted_at,
                url=link or "",
            ))

        # Atom format
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns) or root.findall("entry"):
            title = self._get_text(entry, "title")
            summary = self._get_text(entry, "summary") or self._get_text(entry, "content")
            link_el = entry.find("link") or entry.find("atom:link", ns)
            link = link_el.get("href", "") if link_el is not None else ""
            updated = self._get_text(entry, "updated") or self._get_text(entry, "published")

            if not title or len(title) < 10:
                continue

            text = f"{title}\n{self._strip_html(summary)[:300]}" if summary else title
            text = text[:600]

            # Clean Reddit boilerplate
            if source.startswith("r/"):
                text = self._clean_reddit_text(text)

            item_id = f"atom_{hashlib.md5((link or text).encode()).hexdigest()[:10]}"
            posted_at = updated if updated else ""

            items.append(FinancialTweet(
                tweet_id=item_id,
                text=text,
                author=source,
                posted_at=posted_at,
                url=link,
            ))

        return items

    def _get_text(self, element: ET.Element, tag: str) -> str:
        """Safely get element text, handling namespaces."""
        child = element.find(tag)
        if child is None:
            # Try with common namespace prefixes
            for prefix in ("", "{http://www.w3.org/2005/Atom}", "{http://purl.org/rss/1.0/}"):
                child = element.find(f"{prefix}{tag}")
                if child is not None:
                    break
        return (child.text or "").strip() if child is not None else ""

    def _strip_html(self, html: str) -> str:
        """Remove HTML tags, decode entities."""
        if not html:
            return ""
        text = _HTML_RE.sub(" ", html)
        text = re.sub(r"\s+", " ", text).strip()
        # Decode common XML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&apos;", "'")
        return text

    def _clean_reddit_text(self, text: str) -> str:
        """Strip Reddit RSS boilerplate: 'submitted by /u/...', '[link] [comments]', etc."""
        # Remove "&#32; submitted by &#32; /u/username [link] &#32; [comments]"
        text = re.sub(r'&#32;\s*submitted by\s+&#32;\s*/u/\S+\s*\[link\]\s*&#32;\s*\[comments\]', '', text)
        # Remove bare "[link] [comments]" without user
        text = re.sub(r'\[link\]\s*\[comments\]', '', text)
        # Remove stray &#32; (space entities)
        text = text.replace('&#32;', ' ')
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# ---- Convenience ----

_headline_scanner: Optional[HeadlineScanner] = None


def get_headline_scanner() -> HeadlineScanner:
    global _headline_scanner
    if _headline_scanner is None:
        _headline_scanner = HeadlineScanner()
    return _headline_scanner


def fetch_headlines(max_headlines: int = 120) -> list[FinancialTweet]:
    """Convenience: fetch financial headlines from RSS feeds."""
    try:
        return get_headline_scanner().fetch(max_headlines=max_headlines)
    except Exception as e:
        logger.warning(f"Headline fetch failed: {e}")
        return []


def fetch_reddit_headlines(max_headlines: int = 100) -> list[FinancialTweet]:
    """Convenience: fetch posts from financial subreddits via Reddit RSS (free, no API key)."""
    try:
        return get_headline_scanner().fetch(
            feeds=_REDDIT_FEEDS, max_headlines=max_headlines
        )
    except Exception as e:
        logger.warning(f"Reddit RSS fetch failed: {e}")
        return []
