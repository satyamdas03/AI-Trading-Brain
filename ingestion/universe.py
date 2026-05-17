"""Stock universes — S&P 500 and Nifty 200 ticker lists.

Falls back to static lists if NeuralQuant universe package is unavailable.
S&P 500 fetched from Wikipedia with local JSON cache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import from NeuralQuant first
try:
    from nq_api.universe import US_DEFAULT, IN_DEFAULT, sector_of
    _HAS_NQ = True
except ImportError:
    _HAS_NQ = False

_CACHE_DIR = Path(__file__).parent.parent / "data"
_SP500_CACHE = _CACHE_DIR / "sp500_tickers.json"


def _scrape_sp500() -> list[str]:
    """Scrape S&P 500 constituents from Wikipedia, with 30-day JSON cache."""
    if _SP500_CACHE.exists():
        try:
            data = json.loads(_SP500_CACHE.read_text())
            age_days = (Path(__file__).stat().st_mtime - _SP500_CACHE.stat().st_mtime) / 86400
            if age_days < 30 and len(data) >= 400:
                return data
        except Exception:
            pass

    try:
        import pandas as pd
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = sorted(tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist())
        if len(tickers) >= 400:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _SP500_CACHE.write_text(json.dumps(tickers))
            logger.info(f"S&P 500 scraped: {len(tickers)} tickers, cached to {_SP500_CACHE}")
            return tickers
    except Exception as e:
        logger.warning(f"S&P 500 scrape failed: {e}")

    # Ultimate fallback: return whatever's in cache even if stale
    if _SP500_CACHE.exists():
        try:
            return json.loads(_SP500_CACHE.read_text())
        except Exception:
            pass
    return []


# Static S&P 500 fallback (top 100 by market cap — used only if scrape fails with no cache)
_US_TOP100 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH",
    "JNJ", "V", "XOM", "WMT", "MA", "PG", "JPM", "LLY", "HD", "CVX", "ABBV",
    "MRK", "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "CSCO", "ACN", "ABT",
    "DHR", "NFLX", "NKE", "LIN", "CRM", "VZ", "ADBE", "DIS", "PM", "TXN",
    "AMD", "WFC", "UPS", "NEE", "RTX", "MS", "QCOM", "HON", "INTC", "AMGN",
    "IBM", "INTU", "CAT", "SPGI", "BA", "GE", "AMAT", "AXP", "LOW", "DE",
    "BKNG", "PLD", "SYK", "SCHW", "MDT", "GILD", "BLK", "ADP", "CI", "TMUS",
    "SBUX", "TJX", "MMC", "ISRG", "VRTX", "LMT", "NOW", "C", "CB", "MDLZ",
    "ADI", "SO", "ZTS", "MO", "MU", "PANW", "DUK", "REGN", "ELV", "BSX",
    "KLAC", "ETN", "ANET", "SHW", "WM", "ITW", "PH", "SNPS", "EQIX", "BDX",
]

# Static Nifty 200 fallback (top 100)
_IN_TOP100 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "SBIN.NS",
    "BAJFINANCE.NS", "LT.NS", "SUNPHARMA.NS", "AXISBANK.NS", "MARUTI.NS",
    "TITAN.NS", "ASIANPAINT.NS", "HCLTECH.NS", "ADANIGREEN.NS", "ADANIPORTS.NS",
    "WIPRO.NS", "NTPC.NS", "DMART.NS", "ULTRACEMCO.NS", "POWERGRID.NS",
    "JSWSTEEL.NS", "TATASTEEL.NS", "TECHM.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS",
    "HINDZINC.NS", "INDUSINDBK.NS", "GRASIM.NS", "TRENT.NS", "NESTLEIND.NS",
    "M&M.NS", "HAL.NS", "ONGC.NS", "TATAMOTORS.NS", "DIVISLAB.NS",
    "BEL.NS", "COALINDIA.NS", "VEDL.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "IRCTC.NS", "BRITANNIA.NS", "ADANIENT.NS", "CIPLA.NS", "DRREDDY.NS",
    "APOLLOHOSP.NS", "PIDILITIND.NS", "BAJAJ-AUTO.NS", "SBILIFE.NS", "SRTRANSFIN.NS",
    "CHOLAFIN.NS", "HAVELLS.NS", "TVSMOTOR.NS", "TATAPOWER.NS", "DABUR.NS",
    "GODREJCP.NS", "SIEMENS.NS", "AMBUJACEM.NS", "VARUNBEVERAGES.NS", "BPCL.NS",
    "ZOMATO.NS", "JINDALSTEL.NS", "ABBOTINDIA.NS", "INDIGO.NS", "ICICIGI.NS",
    "BANKBARODA.NS", "DLF.NS", "BERGEPAINT.NS", "GAIL.NS", "PFC.NS",
    "MOTHERSON.NS", "TORNTPHARM.NS", "MARICO.NS", "ICICIPRULI.NS", "YESBANK.NS",
    "SRF.NS", "HDFCAMC.NS", "NAUKRI.NS", "JUBLFOOD.NS", "HINDPETRO.NS",
    "BHEL.NS", "RECLTD.NS", "SAIL.NS", "IOB.NS", "LUPIN.NS",
    "CANBK.NS", "IDBI.NS", "GODREJPROP.NS", "ASTRAL.NS", "MFSL.NS",
    "BIOCON.NS", "AUBANK.NS", "LICHSGFIN.NS", "ASHOKLEY.NS", "PERSISTENT.NS",
    "MPHASIS.NS", "LALPATHLAB.NS", "MAXHEALTH.NS", "POLICYBZR.NS", "PATANJALI.NS",
]


def us_tickers() -> list[str]:
    """Return S&P 500 ticker list."""
    if _HAS_NQ:
        return list(US_DEFAULT)
    scraped = _scrape_sp500()
    if len(scraped) >= 400:
        return scraped
    return list(_US_TOP100)


def in_tickers() -> list[str]:
    """Return Nifty 200 ticker list (with .NS suffix for yfinance)."""
    if _HAS_NQ:
        # NeuralQuant stores without suffix — add .NS for yfinance compat
        return [f"{t}.NS" if not t.endswith(".NS") else t for t in IN_DEFAULT]
    return list(_IN_TOP100)


_CRYPTO_TOP20 = [
    "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD",
    "XRP-USD", "DOT-USD", "AVAX-USD", "MATIC-USD", "LINK-USD",
    "UNI-USD", "ATOM-USD", "LTC-USD", "ETC-USD", "XLM-USD",
    "FIL-USD", "TRX-USD", "NEAR-USD", "ALGO-USD", "VET-USD",
]


def crypto_tickers() -> list[str]:
    """Return top crypto ticker list (yfinance format)."""
    return list(_CRYPTO_TOP20)


def get_tickers(market: str) -> list[str]:
    """Return ticker list for a given market ('US' or 'IN')."""
    if market == "US":
        return us_tickers()
    if market == "IN":
        return in_tickers()
    if market == "CRYPTO":
        return crypto_tickers()
    raise ValueError(f"Unknown market: {market}")
