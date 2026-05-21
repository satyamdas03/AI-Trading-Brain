"""Reddit sentiment scanner — free, no rate limits beyond PRAW's 100 QPM default.

Uses PRAW to scrape hot/new posts from financial subreddits, extracts ticker
mentions with regex, and produces FinancialTweet objects for the classifier pipeline.

Subreddits: r/wallstreetbets, r/stocks, r/investing, r/StockMarket
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

from sentiment.x_scanner import FinancialTweet  # reuse the same dataclass

logger = logging.getLogger(__name__)

# Reddit API credentials (set in .env)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "AI-Trading-Brain/1.0")

# Subreddits to scan
_DEFAULT_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "StockMarket",
]

# Ticker extraction: $SYMBOL or bare ticker-like words
_TICKER_RE = re.compile(
    r'\$([A-Z]{1,5})\b|'
    r'\b([A-Z]{2,5})\b'
)

# Noise tickers to filter (common words that look like tickers)
_NOISE_TICKERS = {
    "THE", "FOR", "AND", "NOT", "ARE", "BUT", "HAS", "HAD", "WAS", "ALL",
    "NEW", "NOW", "ONE", "TWO", "CAN", "WILL", "JUST", "LIKE", "THIS",
    "THAT", "WITH", "FROM", "YOUR", "THEY", "BEEN", "VERY", "MORE",
    "SOME", "THAN", "INTO", "OVER", "ABOUT", "WOULD", "COULD", "SHOULD",
    "DD", "YOLO", "IMO", "TLDR", "EDIT", "ETF", "CEO", "CFO", "IPO",
    "NYSE", "NASDAQ", "NYSEARCA", "SEC", "FDIC", "FED", "GDP", "CPI",
    "PMI", "ITM", "OTM", "ATM", "AI", "EV", "HODL", "WSB", "LMAO",
    "FOMO", "FUD", "DYOR", "ATH", "ATL", "BTO", "STC", "ROTH", "IRA",
    "HSA", "401K", "DTE", "TLT", "GLD", "SLV", "EEM", "IWM", "DIA",
    "OXY", "APY", "EPS", "PE", "ROE", "ROI", "CAGR", "EBITDA",
}

# Noise detection: very short/low-effort posts
_MIN_TEXT_LENGTH = 25


class RedditScanner:
    """Fetch posts from financial subreddits, extract ticker mentions."""

    def __init__(self):
        self._praw = None

    @property
    def praw(self):
        """Lazy-init PRAW to avoid import at module load."""
        if self._praw is None:
            try:
                import praw
                if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
                    raise RuntimeError("Reddit API credentials not configured")
                self._praw = praw.Reddit(
                    client_id=REDDIT_CLIENT_ID,
                    client_secret=REDDIT_CLIENT_SECRET,
                    user_agent=REDDIT_USER_AGENT,
                    check_for_async=False,
                )
            except ImportError:
                raise RuntimeError("praw not installed — pip install praw")
            except Exception as e:
                raise RuntimeError(f"PRAW init failed: {e}")
        return self._praw

    def fetch(
        self,
        subreddits: list[str] | None = None,
        max_posts: int = 80,
        listing: str = "hot",
    ) -> list[FinancialTweet]:
        """Fetch posts from financial subreddits. Returns FinancialTweet list.

        Args:
            subreddits: which subreddits to scan (default: WSB, stocks, investing, StockMarket)
            max_posts: roughly how many posts to fetch total across all subs
            listing: "hot", "new", or "rising"
        """
        if subreddits is None:
            subreddits = _DEFAULT_SUBREDDITS

        total_posts = []
        per_sub = max(5, max_posts // len(subreddits))
        seen_ids: set[str] = set()

        try:
            for sub in subreddits:
                try:
                    subreddit = self.praw.subreddit(sub)
                    posts = getattr(subreddit, listing)(limit=per_sub)
                    count = 0
                    for post in posts:
                        if post.id in seen_ids:
                            continue
                        seen_ids.add(post.id)

                        # Skip stickied/pinned posts
                        if post.stickied:
                            continue

                        text = f"{post.title}\n{post.selftext}" if post.selftext else post.title
                        # Skip very short posts
                        if len(text) < _MIN_TEXT_LENGTH:
                            continue

                        url = f"https://reddit.com{post.permalink}"
                        total_posts.append(FinancialTweet(
                            tweet_id=f"rdt_{post.id}",
                            text=text[:500],
                            author=f"r/{sub}",
                            posted_at=datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
                            url=url,
                        ))
                        count += 1
                        if count >= per_sub:
                            break
                    logger.debug(f"r/{sub}: {count} posts")
                except Exception as e:
                    logger.warning(f"r/{sub} fetch failed: {e}")
                    continue
        except RuntimeError as e:
            logger.warning(f"Reddit unavailable: {e}")
            return []

        logger.info(f"Reddit: {len(total_posts)} posts from {len(subreddits)} subreddits")
        return total_posts


# ---- Convenience ----

_reddit_scanner: Optional[RedditScanner] = None


def get_reddit_scanner() -> RedditScanner:
    global _reddit_scanner
    if _reddit_scanner is None:
        _reddit_scanner = RedditScanner()
    return _reddit_scanner


def fetch_reddit_posts(max_posts: int = 80) -> list[FinancialTweet]:
    """Convenience: fetch posts from all financial subreddits."""
    try:
        return get_reddit_scanner().fetch(max_posts=max_posts)
    except Exception as e:
        logger.warning(f"Reddit fetch failed: {e}")
        return []
