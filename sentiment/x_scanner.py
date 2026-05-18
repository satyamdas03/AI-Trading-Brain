"""X (Twitter) Financial Sentiment Scanner — Phase 6 extension.

Architecture inspired by X-algorithm's Grox ContentClassifier pattern:
  _to_convo() → _sample() → _parse()

Classifies financial tweets for sentiment, ticker mentions, and materiality.
Uses Claude Haiku for fast, cost-effective classification (~$0.001/100 tweets).

Tweet sources (priority order):
  1. X API v2 recent search (needs X_BEARER_TOKEN)
  2. Public web scraping fallback (nitter.net mirrors)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from config import (
    ANTHROPIC_API_KEY, CLASSIFICATION_MODEL,
    PROJECT_ROOT, SIGNAL_TOP_N,
)

logger = logging.getLogger(__name__)

# ---- Cache ----
_CACHE_DIR = PROJECT_ROOT / "data" / "sentiment"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_TTL = 600  # 10 minutes

# ---- X API ----
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "")
X_API_KEY = os.getenv("X_API_KEY", "")
X_API_SECRET = os.getenv("X_API_SECRET", "")
X_API_BASE = "https://api.twitter.com"

# OAuth2 token cache
_x_oauth_token: Optional[str] = None
_x_oauth_expiry: float = 0.0

# ---- Financial tickers to scan ----
_FINANCE_ACCOUNTS = [
    "Bloomberg", "CNBC", "Reuters", "MarketWatch", "FinancialTimes",
    "WSJ", "TheStreet", "Benzinga", "IBDinvestors", "Stocktwits",
    "zerohedge", "elerianm", "jsblokland", "NorthmanTrader",
    "WalterDeemer", "RyanDetrick", "LizAnnSonders", "bespokeinvest",
]

_FINANCE_CASHTAGS = [
    "$SPY", "$QQQ", "$AAPL", "$MSFT", "$NVDA", "$TSLA", "$META",
    "$GOOGL", "$AMZN", "$AMD", "$BTC", "$ETH", "$SOL",
]

# ---- Data types ----

@dataclass
class FinancialTweet:
    tweet_id: str
    text: str
    author: str
    posted_at: str
    url: str = ""


@dataclass
class SentimentResult:
    """One tweet's classification output — mirrors Grox ContentCategoryResult."""
    tweet_id: str
    text: str
    sentiment: str = "neutral"  # bullish / bearish / neutral
    confidence: float = 0.5     # 0.0 - 1.0
    tickers: list[str] = field(default_factory=list)
    materiality: str = "low"    # high / medium / low — is this market-moving?
    key_points: list[str] = field(default_factory=list)
    author: str = ""
    error: str = ""


@dataclass
class SentimentScan:
    """Aggregated sentiment scan output."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tweets_scanned: int = 0
    results: list[SentimentResult] = field(default_factory=list)
    overall_sentiment: str = "neutral"
    bullish_count: int = 0
    bearish_count: int = 0
    ticker_sentiment: dict[str, dict] = field(default_factory=dict)  # ticker → {bullish, bearish, total}
    ticker_signals: list[dict] = field(default_factory=list)  # Top signals for debate
    errors: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


# ---- Core Classifier (Grox Pattern) ----

class FinancialTweetClassifier:
    """Classify financial tweets using Grox's 3-step pattern.

    Step 1: _to_convo()  — format tweet + system prompt for Claude
    Step 2: _sample()    — call Claude Haiku API
    Step 3: _parse()     — extract structured JSON from response
    """

    SYSTEM_PROMPT = """You are a financial sentiment classifier for stock/crypto trading.

Analyze the tweet and output JSON with these fields:
- sentiment: "bullish" | "bearish" | "neutral"
- confidence: 0.0 to 1.0 (how confident you are in the sentiment)
- tickers: list of stock/crypto tickers mentioned (e.g. ["AAPL", "TSLA"]). Extract from $SYMBOL, symbol mentions, or company names
- materiality: "high" | "medium" | "low" (how market-moving is this news?)
- key_points: list of 1-3 key takeaways (max 80 chars each)

Rules:
- Earnings beats, raised guidance, buybacks → bullish
- Misses, lowered guidance, layoffs, lawsuits, investigations → bearish
- Analyst upgrades/downgrades → follow the direction
- Macro/rate comments → assess impact on equities broadly
- Vague commentary with no actionable info → neutral, low materiality
- Only include tickers that are CLEARLY referenced
- Confidence above 0.7 only when the news is concrete (earnings, M&A, regulatory action)

Output ONLY valid JSON between <json></json> tags. No other text."""

    def __init__(self):
        self._client = httpx.Client(timeout=30)

    def classify(self, tweet: FinancialTweet) -> SentimentResult:
        """Classify one financial tweet. Returns SentimentResult."""
        try:
            convo = self._to_convo(tweet)
            output = self._sample(convo)
            return self._parse(tweet, output)
        except Exception as e:
            return SentimentResult(
                tweet_id=tweet.tweet_id,
                text=tweet.text,
                error=str(e)[:200],
            )

    def classify_batch(self, tweets: list[FinancialTweet]) -> list[SentimentResult]:
        """Classify multiple tweets. Falls back to sequential (Haiku is fast)."""
        return [self.classify(t) for t in tweets]

    # -- Step 1: Format conversation for Claude --

    def _to_convo(self, tweet: FinancialTweet) -> dict:
        user_msg = f"Tweet by @{tweet.author}:\n{tweet.text}"
        return {
            "model": CLASSIFICATION_MODEL,
            "max_tokens": 300,
            "temperature": 0.1,
            "system": self.SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_msg}],
        }

    # -- Step 2: Call Claude API --

    def _sample(self, convo: dict) -> str:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        resp = self._client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=convo,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    # -- Step 3: Parse structured output --

    def _parse(self, tweet: FinancialTweet, output: str) -> SentimentResult:
        # Extract JSON between <json></json> tags (exactly like Grox)
        m = re.search(r"<json>(.*?)</json>", output, re.DOTALL)
        if not m:
            # Try raw JSON fallback
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                return SentimentResult(
                    tweet_id=tweet.tweet_id, text=tweet.text,
                    error=f"Could not parse JSON from: {output[:100]}",
                )
        else:
            parsed = json.loads(m.group(1).strip())

        sentiment = str(parsed.get("sentiment", "neutral")).lower()
        if sentiment not in ("bullish", "bearish", "neutral"):
            sentiment = "neutral"

        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        materiality = str(parsed.get("materiality", "low")).lower()
        if materiality not in ("high", "medium", "low"):
            materiality = "low"

        tickers = []
        raw_tickers = parsed.get("tickers", [])
        if isinstance(raw_tickers, list):
            tickers = [str(t).upper().strip().lstrip("$") for t in raw_tickers]

        key_points = []
        raw_pts = parsed.get("key_points", [])
        if isinstance(raw_pts, list):
            key_points = [str(p)[:200] for p in raw_pts]

        return SentimentResult(
            tweet_id=tweet.tweet_id,
            text=tweet.text,
            sentiment=sentiment,
            confidence=confidence,
            tickers=tickers,
            materiality=materiality,
            key_points=key_points,
            author=tweet.author,
        )


# ---- Tweet Fetchers ----

def _get_x_bearer_token() -> str:
    """Get valid X API bearer token. Uses OAuth2 client credentials if available, falls back to env token."""
    global _x_oauth_token, _x_oauth_expiry

    # Return cached OAuth2 token if still valid (expire 1 min early for safety)
    if _x_oauth_token and time.time() < _x_oauth_expiry - 60:
        return _x_oauth_token

    # Try OAuth2 client credentials flow
    if X_API_KEY and X_API_SECRET:
        try:
            resp = httpx.post(
                "https://api.twitter.com/oauth2/token",
                data={"grant_type": "client_credentials"},
                auth=(X_API_KEY, X_API_SECRET),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                _x_oauth_token = data["access_token"]
                _x_oauth_expiry = time.time() + 7200  # 2-hour token lifetime
                return _x_oauth_token
        except Exception as e:
            logger.debug(f"OAuth2 token fetch failed: {e}")

    # Fall back to static bearer token from env
    return X_BEARER_TOKEN


def _fetch_via_x_api(query: str, max_tweets: int = 50) -> list[FinancialTweet]:
    """Fetch tweets via X API v2 recent search. Uses OAuth2 client credentials or X_BEARER_TOKEN."""
    token = _get_x_bearer_token()
    if not token:
        return []

    url = f"{X_API_BASE}/2/tweets/search/recent"
    # X API requires max_results between 10 and 100
    results_per = max(10, min(max_tweets, 100))
    params = {
        "query": f"{query} -is:retweet lang:en",
        "max_results": results_per,
        "tweet.fields": "created_at,author_id",
        "expansions": "author_id",
        "user.fields": "username",
    }

    try:
        resp = httpx.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning(f"X API returned {resp.status_code}: {resp.text[:200]}")
            return []

        data = resp.json()
        users = {u["id"]: u["username"] for u in data.get("includes", {}).get("users", [])}

        tweets = []
        for t in data.get("data", []):
            tweets.append(FinancialTweet(
                tweet_id=t["id"],
                text=t["text"],
                author=users.get(t.get("author_id", ""), "unknown"),
                posted_at=t.get("created_at", ""),
                url=f"https://twitter.com/i/web/status/{t['id']}",
            ))
        return tweets
    except Exception as e:
        logger.warning(f"X API fetch failed: {e}")
        return []


def _fetch_via_nitter(query: str, max_tweets: int = 30) -> list[FinancialTweet]:
    """Fallback: fetch tweets via Nitter public mirrors (no API key needed)."""
    # Try multiple nitter instances
    mirrors = [
        "https://nitter.net",
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
    ]

    search_query = query.replace("$", "%24").replace(" ", "%20").replace("#", "%23")

    for mirror in mirrors:
        try:
            url = f"{mirror}/search?f=tweets&q={search_query}&lang=en"
            resp = httpx.get(url, timeout=15, follow_redirects=True,
                            headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue

            # Parse tweet content from HTML
            tweets = _parse_nitter_html(resp.text, max_tweets)
            if tweets:
                logger.info(f"Fetched {len(tweets)} tweets via {mirror}")
                return tweets
        except Exception as e:
            logger.debug(f"Nitter mirror {mirror} failed: {e}")
            continue

    return []


def _parse_nitter_html(html: str, max_tweets: int) -> list[FinancialTweet]:
    """Extract tweets from Nitter HTML. Robust against DOM changes."""
    tweets = []

    # Pattern: tweet-content div contains the text, tweet-link contains the ID
    # Nitter uses class="tweet-content media-body" for tweet text
    content_pattern = re.compile(
        r'<div[^>]*class="[^"]*tweet-content[^"]*"[^>]*>(.*?)</div>\s*</div>',
        re.DOTALL,
    )
    # Tweet permalink for ID/author extraction
    link_pattern = re.compile(
        r'<a[^>]*class="[^"]*tweet-link[^"]*"[^>]*href="/(\w+)/status/(\d+)"',
    )

    contents = content_pattern.findall(html)
    links = link_pattern.findall(html)

    for i, (content_html, (author, tweet_id)) in enumerate(zip(contents, links)):
        if i >= max_tweets:
            break
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", content_html)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 20:
            continue

        tweets.append(FinancialTweet(
            tweet_id=tweet_id,
            text=text,
            author=author,
            posted_at="",
            url=f"https://twitter.com/{author}/status/{tweet_id}",
        ))

    return tweets


def _fetch_demo_tweets(max_tweets: int = 15) -> list[FinancialTweet]:
    """Generate synthetic financial tweets for testing when no real sources available."""
    demo_templates = [
        ("Bloomberg", "$AAPL earnings beat estimates by 12%, revenue up 8% YoY. iPhone sales stronger than expected in China. Board authorizes $90B buyback.", "https://twitter.com/Bloomberg/status/demo001"),
        ("CNBC", "BREAKING: Fed holds rates steady, dot plot shows 2 cuts in H2 2025. Powell: 'labor market rebalancing nicely, inflation trajectory encouraging.'", "https://twitter.com/CNBC/status/demo002"),
        ("Reuters", "$NVDA unveils next-gen AI chip with 3x performance improvement. Major cloud providers commit to adoption. Supply constraints expected to ease by Q3.", "https://twitter.com/Reuters/status/demo003"),
        ("MarketWatch", "$TSLA misses delivery estimates by 8%. Price cuts eroding margins. Analysts cutting PT. $META and $GOOGL gaining digital ad share.", "https://twitter.com/MarketWatch/status/demo004"),
        ("FinancialTimes", "China stimulus package larger than expected — $1.4T infrastructure + housing support. Emerging markets rally. Commodities surge on demand optimism.", "https://twitter.com/FinancialTimes/status/demo005"),
        ("TheStreet", "$MSFT cloud revenue accelerates to 28% growth. AI copilot adoption exceeds targets. Commercial bookings up 35% — strongest in 5 quarters.", "https://twitter.com/TheStreet/status/demo006"),
        ("Benzinga", "Whale alert: $10M call sweep in $SPY 450 strike expiring next week. Unusual options activity suggests big institutions positioning for breakout.", "https://twitter.com/Benzinga/status/demo007"),
        ("zerohedge", "Yield curve steepening rapidly — 2s10s spread widest since 2022. Bond market pricing in reflation. $BTC correlated with gold, both ripping higher.", "https://twitter.com/zerohedge/status/demo008"),
        ("Stocktwits", "$AMD breaking out above resistance at 180 on massive volume. Data center revenue guidance raised. $NVDA competitor but valuation is 1/3rd. 🚀", "https://twitter.com/Stocktwits/status/demo009"),
        ("WSJ", "Exclusive: DOJ antitrust probe into $GOOGL search monopoly expanding to AI tools. Ad business model faces existential regulatory risk over next 2 years.", "https://twitter.com/WSJ/status/demo010"),
        ("IBDinvestors", "$AMZN announces 20-for-1 stock split. Retail sentiment surging, meme stock chatter increasing. AWS reacceleration thesis intact.", "https://twitter.com/IBDinvestors/status/demo011"),
        ("elerianm", "Labor market softening faster than headline NFP suggests. Participation rate edge down, wage growth decelerating. Fed should start cutting in June not September.", "https://twitter.com/elerianm/status/demo012"),
        ("bespokeinvest", "Breadth thrust signal triggered: 90% of S&P 500 stocks above 50-day MA. Historically this precedes 6-month rally 85% of the time. $SPY $QQQ", "https://twitter.com/bespokeinvest/status/demo013"),
        ("NorthmanTrader", "Divergence alert: $SPY making new highs but $AAPL and $MSFT diverging — not confirming. This usually resolves DOWN within 2 weeks. Reduce risk.", "https://twitter.com/NorthmanTrader/status/demo014"),
        ("RyanDetrick", "Earnings season summary: 78% of S&P 500 companies beat EPS estimates. Average surprise +6.2%. This is BULLISH for equities into year-end. $SPY $QQQ $IWM", "https://twitter.com/RyanDetrick/status/demo015"),
    ]

    tweets = []
    for author, text, url in demo_templates[:max_tweets]:
        tweet_id = hashlib.md5(text.encode()).hexdigest()[:16]
        tweets.append(FinancialTweet(
            tweet_id=tweet_id,
            text=text,
            author=author,
            posted_at=datetime.now(timezone.utc).isoformat(),
            url=url,
        ))
    return tweets


def _fetch_finance_tweets(max_tweets: int = 50, use_demo: bool = True) -> list[FinancialTweet]:
    """Fetch financial tweets from multiple sources. Deduplicates by content hash.

    Falls back to demo tweets when X API and Nitter are both unavailable.
    """
    seen_hashes: set[str] = set()
    all_tweets: list[FinancialTweet] = []

    queries = [
        "($SPY OR $QQQ OR SPX) (earnings OR rally OR selloff OR fed OR inflation)",
        "stock market today breaking",
        "($AAPL OR $MSFT OR $NVDA OR $TSLA) (earnings OR upgrade OR downgrade)",
        "($BTC OR $ETH OR crypto) (breakout OR crash OR etf)",
        "financial markets macro economy",
    ]

    for query in queries:
        if len(all_tweets) >= max_tweets:
            break

        # Try X API first, fall back to nitter
        tweets = _fetch_via_x_api(query, max_tweets=20)
        if not tweets:
            tweets = _fetch_via_nitter(query, max_tweets=15)

        for t in tweets:
            content_hash = hashlib.md5(t.text[:100].encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                all_tweets.append(t)

        if len(all_tweets) >= max_tweets:
            break

    # If no real tweets, fall back to demo data
    if not all_tweets and use_demo:
        logger.info("No real tweets available — using demo data for testing")
        all_tweets = _fetch_demo_tweets(max_tweets)

    return all_tweets[:max_tweets]


# ---- Scanner Orchestrator ----

class XSentimentScanner:
    """Main scanner: fetch tweets → classify → aggregate → produce signals."""

    def __init__(self):
        self._classifier = FinancialTweetClassifier()

    def scan(self, max_tweets: int = 50, ticker_filter: list[str] | None = None) -> SentimentScan:
        """Run full scan: fetch → classify → aggregate. Returns SentimentScan."""
        start = time.perf_counter()
        scan = SentimentScan()

        # 1. Fetch tweets
        tweets = _fetch_finance_tweets(max_tweets)
        if ticker_filter:
            ticker_set = {t.upper() for t in ticker_filter}
            tweets = [t for t in tweets if _mentions_ticker(t.text, ticker_set)]
        scan.tweets_scanned = len(tweets)

        if not tweets:
            scan.errors.append("No tweets fetched — check X_BEARER_TOKEN or network")
            scan.latency_ms = (time.perf_counter() - start) * 1000
            return scan

        # 2. Classify
        results = self._classifier.classify_batch(tweets)
        scan.results = results

        # 3. Aggregate
        self._aggregate(scan, results)
        scan.latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Sentiment scan: {scan.tweets_scanned} tweets -> "
            f"{scan.bullish_count}B/{scan.bearish_count}S/{scan.tweets_scanned - scan.bullish_count - scan.bearish_count}N "
            f"({scan.overall_sentiment}), {len(scan.ticker_signals)} ticker signals, "
            f"{scan.latency_ms:.0f}ms"
        )

        return scan

    def _aggregate(self, scan: SentimentScan, results: list[SentimentResult]) -> None:
        """Aggregate individual results into scan-level stats and ticker signals."""
        ticker_map: dict[str, dict] = {}

        for r in results:
            if r.error:
                scan.errors.append(r.error)
                continue

            if r.sentiment == "bullish":
                scan.bullish_count += 1
            elif r.sentiment == "bearish":
                scan.bearish_count += 1

            for ticker in r.tickers:
                if ticker not in ticker_map:
                    ticker_map[ticker] = {"bullish": 0, "bearish": 0, "neutral": 0, "total": 0,
                                           "high_mat": 0, "tweets": [],
                                           "avg_confidence": 0.0, "confidence_sum": 0.0}
                tm = ticker_map[ticker]
                tm[r.sentiment] += 1
                tm["total"] += 1
                tm["confidence_sum"] += r.confidence
                if r.materiality == "high":
                    tm["high_mat"] += 1
                tm["tweets"].append({
                    "text": r.text[:120],
                    "sentiment": r.sentiment,
                    "confidence": r.confidence,
                    "materiality": r.materiality,
                    "author": r.author,
                })

        # Compute avg confidence
        for tm in ticker_map.values():
            if tm["total"] > 0:
                tm["avg_confidence"] = tm["confidence_sum"] / tm["total"]

        scan.ticker_sentiment = ticker_map

        # Produce ticker signals (sorted by materiality-adjusted sentiment score)
        signals = []
        for ticker, tm in ticker_map.items():
            if tm["total"] < 2:
                continue
            bull_ratio = tm["bullish"] / tm["total"] if tm["total"] > 0 else 0
            bear_ratio = tm["bearish"] / tm["total"] if tm["total"] > 0 else 0
            net_score = (bull_ratio - bear_ratio) * tm["avg_confidence"]
            # Boost by high-materiality count
            net_score += tm["high_mat"] * 0.15

            sentiment_label = "bullish" if net_score > 0.15 else ("bearish" if net_score < -0.15 else "neutral")

            signals.append({
                "ticker": ticker,
                "sentiment": sentiment_label,
                "net_score": round(net_score, 3),
                "bullish_count": tm["bullish"],
                "bearish_count": tm["bearish"],
                "total_mentions": tm["total"],
                "high_mat_count": tm["high_mat"],
                "avg_confidence": round(tm["avg_confidence"], 3),
            })

        # Sort by abs(net_score) descending
        signals.sort(key=lambda s: abs(s["net_score"]), reverse=True)
        scan.ticker_signals = signals[:SIGNAL_TOP_N]

        # Overall sentiment
        total = scan.bullish_count + scan.bearish_count
        if total == 0:
            scan.overall_sentiment = "neutral"
        else:
            ratio = scan.bullish_count / total if total > 0 else 0.5
            if ratio > 0.60:
                scan.overall_sentiment = "bullish"
            elif ratio < 0.40:
                scan.overall_sentiment = "bearish"
            else:
                scan.overall_sentiment = "neutral"


def _mentions_ticker(text: str, ticker_set: set[str]) -> bool:
    """Check if text mentions any ticker from set."""
    text_upper = text.upper()
    for t in ticker_set:
        if f"${t}" in text_upper or re.search(rf"\b{t}\b", text_upper):
            return True
    return False


# ---- Top-level convenience ----

_scanner: Optional[XSentimentScanner] = None


def get_scanner() -> XSentimentScanner:
    global _scanner
    if _scanner is None:
        _scanner = XSentimentScanner()
    return _scanner


def run_scan(max_tweets: int = 50) -> SentimentScan:
    """Convenience: run a full sentiment scan."""
    return get_scanner().scan(max_tweets=max_tweets)
