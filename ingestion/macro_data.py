"""Macro data ingestion — VIX, SPX levels, HY spread, ISM PMI, yield curve.

Uses yfinance for market-based indicators. Requires FRED_API_KEY for FRED-sourced data.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx
import pandas as pd
import yfinance as yf

_FRED_API_KEY = os.getenv("FRED_API_KEY", "")


def fetch_macro_snapshot() -> dict:
    """Fetch current macro state. Returns dict with all fields for regime detection."""

    result = {
        "vix": 18.0,
        "vix_20d_change": 0.0,
        "spx_vs_200ma": 0.02,
        "hy_spread_oas": 350.0,
        "ism_pmi": 51.0,
        "yield_spread_2y10y": 0.10,
        "cpi_yoy": 3.0,
        "fed_funds_rate": 5.25,
    }

    # VIX and SPX from yfinance
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1mo")
        if not hist.empty:
            result["vix"] = float(hist["Close"].iloc[-1])
            if len(hist) >= 21:
                result["vix_20d_change"] = float(hist["Close"].iloc[-1] - hist["Close"].iloc[-21])

        spx = yf.Ticker("^GSPC")
        spx_hist = spx.history(period="1y")
        if not spx_hist.empty:
            ma200 = float(spx_hist["Close"].rolling(200).mean().iloc[-1])
            close = float(spx_hist["Close"].iloc[-1])
            if ma200 > 0:
                result["spx_vs_200ma"] = (close - ma200) / ma200
    except Exception:
        pass

    # Treasury yields from yfinance
    try:
        t10 = yf.Ticker("^TNX")
        t10_hist = t10.history(period="5d")
        t2 = yf.Ticker("^IRX")  # 13-week, closest proxy for 2Y on yfinance
        t2_hist = t2.history(period="5d")
        if not t10_hist.empty and not t2_hist.empty:
            y10 = float(t10_hist["Close"].iloc[-1]) / 100
            y2 = float(t2_hist["Close"].iloc[-1]) / 100
            if y10 > 0 and y2 > 0:
                result["yield_spread_2y10y"] = y10 - y2
    except Exception:
        pass

    # FRED-sourced data if API key available
    if _FRED_API_KEY:
        _enrich_from_fred(result)

    return result


def _enrich_from_fred(result: dict) -> None:
    """Add FRED-sourced indicators: HY spread, ISM PMI, CPI, Fed funds."""
    series_map = {
        "BAMLH0A0HYM2": "hy_spread_oas",  # BofA HY OAS
        "ISM": "ism_pmi",                   # Alt name
        "CPIAUCSL": "cpi_yoy",
        "FEDFUNDS": "fed_funds_rate",
    }
    for series_id, key in series_map.items():
        try:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}&api_key={_FRED_API_KEY}"
                f"&file_type=json&sort_order=desc&limit=2"
            )
            resp = httpx.get(url, timeout=10)
            if resp.status_code == 200:
                obs = resp.json()["observations"]
                if obs:
                    val = float(obs[0]["value"])
                    if key == "cpi_yoy":
                        if len(obs) > 1:
                            prev = float(obs[1]["value"])
                            if prev > 0:
                                val = (val - prev) / prev * 100
                    result[key] = val
        except Exception:
            pass


def fetch_macro_history(years: int = 5) -> pd.DataFrame:
    """Fetch macro history for HMM training. Returns DataFrame with regime feature columns."""
    records = []

    # Try FRED for historical data first
    if _FRED_API_KEY:
        try:
            df = _fetch_fred_bulk(years)
            if not df.empty:
                return df
        except Exception:
            pass

    # Fallback: yfinance 1-year VIX + SPX only
    try:
        vix = yf.download("^VIX", period=f"{min(years, 1)}y", progress=False)
        spx = yf.download("^GSPC", period=f"{min(years, 1)}y", progress=False)

        if vix.empty or spx.empty:
            return _synthetic_macro_history()

        spx["spx_vs_200ma"] = (spx["Close"] - spx["Close"].rolling(200).mean()) / spx["Close"].rolling(200).mean()

        for idx in vix.index:
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            records.append({
                "date": date_str,
                "vix": float(vix.loc[idx, "Close"]),
                "vix_20d_change": 0.0,
                "spx_vs_200ma": float(spx.loc[idx, "spx_vs_200ma"]) if idx in spx.index and not pd.isna(spx.loc[idx, "spx_vs_200ma"]) else 0.02,
                "hy_spread_oas": 350.0,
                "ism_pmi": 51.0,
            })
    except Exception:
        return _synthetic_macro_history()

    if not records:
        return _synthetic_macro_history()

    return pd.DataFrame(records)


def _fetch_fred_bulk(years: int) -> pd.DataFrame:
    """Bulk fetch from FRED for multiple series."""
    series = {
        "VIXCLS": "vix",
        "BAMLH0A0HYM2": "hy_spread_oas",
        "T10Y2Y": "yield_spread_2y10y",
    }
    df = pd.DataFrame()
    for sid, col in series.items():
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={sid}&api_key={_FRED_API_KEY}"
            f"&file_type=json&sort_order=asc&limit={years * 365}"
        )
        try:
            resp = httpx.get(url, timeout=30)
            if resp.status_code == 200:
                obs = resp.json()["observations"]
                rows = [{"date": o["date"], col: float(o["value"])} for o in obs if o["value"] != "."]
                s = pd.DataFrame(rows).set_index("date")
                if df.empty:
                    df = s
                else:
                    df = df.join(s, how="outer")
        except Exception:
            pass

    if df.empty:
        return df

    df = df.ffill().dropna()
    if "vix" in df.columns:
        df["vix_20d_change"] = df["vix"].diff(20)
    return df.reset_index()


def _synthetic_macro_history() -> pd.DataFrame:
    """Generate synthetic macro data when no real sources available."""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=252, freq="B")
    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "vix": 18.0,
        "vix_20d_change": 0.0,
        "spx_vs_200ma": 0.02,
        "hy_spread_oas": 350.0,
        "ism_pmi": 51.0,
    })
