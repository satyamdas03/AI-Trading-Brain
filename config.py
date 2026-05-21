"""Central configuration for AI Trading Brain."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env with explicit path and override to beat any pre-existing env vars
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=str(_ENV_PATH), override=True)

# Kill system-level ANTHROPIC_BASE_URL that routes to Ollama
os.environ.pop("ANTHROPIC_BASE_URL", None)

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DB_DIR = PROJECT_ROOT / "data"
DB_PATH = DB_DIR / "brain.db"

# --- Supabase ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

# --- Anthropic / Claude ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLASSIFICATION_MODEL = "claude-haiku-4-5-20251001"
SCORING_MODEL = "claude-sonnet-4-6"
DEBATE_MODEL = "claude-sonnet-4-6"

# --- Alpaca (paper trading) ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", os.getenv("APCA_API_KEY_ID", ""))
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", os.getenv("APCA_API_SECRET_KEY", ""))
ALPACA_PAPER = os.getenv("ALPACA_PAPER_TRADE", "true").lower() == "true"
ALPACA_PAPER_ACCOUNT = "PA3M6G5LMKMI"

# --- Data APIs ---
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# --- Pipeline thresholds ---
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.10"))
MAX_SECTOR_PCT = float(os.getenv("MAX_SECTOR_PCT", "0.30"))
MAX_POSITIONS_PER_SECTOR = int(os.getenv("MAX_POSITIONS_PER_SECTOR", "1"))
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "0.10"))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "100.0"))
DAILY_TRADE_CAP = int(os.getenv("DAILY_TRADE_CAP", "5"))
MAX_BET_DEFAULT = float(os.getenv("MAX_BET_DEFAULT", "5000.0"))
MAX_POSITIONS_DEFAULT = int(os.getenv("MAX_POSITIONS_DEFAULT", "10"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
TRADE_ENABLED = os.getenv("TRADE_ENABLED", "false").lower() == "true"

# --- Signal Engine ---
SIGNAL_TOP_N = int(os.getenv("TRADE_DAEMON_TOP_N", os.getenv("SIGNAL_TOP_N", "5")))
TRADE_DAEMON_MAX_SIGNALS_PER_DAY = int(os.getenv("TRADE_DAEMON_MAX_SIGNALS_PER_DAY", "20"))
TRADE_DAEMON_SCAN_INTERVAL_MINUTES = int(os.getenv("TRADE_DAEMON_SCAN_INTERVAL_MINUTES", "15"))
TRADE_DAEMON_BANKROLL = float(os.getenv("TRADE_DAEMON_BANKROLL", "10000.0"))

# --- Universe ---
US_UNIVERSE_SIZE = int(os.getenv("US_UNIVERSE_SIZE", "500"))
IN_UNIVERSE_SIZE = int(os.getenv("IN_UNIVERSE_SIZE", "200"))
CRYPTO_UNIVERSE_SIZE = int(os.getenv("CRYPTO_UNIVERSE_SIZE", "20"))

# --- Multi-Asset Toggles ---
IN_MARKET_ENABLED = os.getenv("IN_MARKET_ENABLED", "false").lower() == "true"
CRYPTO_ENABLED = os.getenv("CRYPTO_ENABLED", "false").lower() == "true"
POLYMARKET_ENABLED = os.getenv("POLYMARKET_ENABLED", "false").lower() == "true"

# --- Crypto ---
CRYPTO_TOP_N = int(os.getenv("CRYPTO_TOP_N", "3"))

# --- Polymarket ---
POLYMARKET_MAX_PER_MARKET = float(os.getenv("POLYMARKET_MAX_PER_MARKET", "500"))
POLYMARKET_SCAN_INTERVAL_MINUTES = int(os.getenv("POLYMARKET_SCAN_INTERVAL_MINUTES", "30"))

# --- Regime Detection ---
REGIME_N_STATES = int(os.getenv("REGIME_N_STATES", "4"))

# --- Online Learning (Phase 4 campaign: equal weights +141.9% vs learned +119.2%) ---
ONLINE_LEARNING_ENABLED = os.getenv("ONLINE_LEARNING_ENABLED", "false").lower() == "true"

# --- Weekend Retraining ---
RETRAIN_ENABLED = os.getenv("RETRAIN_ENABLED", "true").lower() == "true"
RETRAIN_MAX_TRIALS = int(os.getenv("RETRAIN_MAX_TRIALS", "500"))

# --- Risk-Reward (7%TP / 6%SL = +1.54% expectancy at 58% WR) ---
TAKE_PROFIT_PCT = 0.07
STOP_LOSS_PCT = 0.06

# --- Trend Filter (block entries when SPX below 200MA) ---
TREND_FILTER_ENABLED = os.getenv("TREND_FILTER_ENABLED", "true").lower() == "true"

# --- Multi-Source Sentiment Scanner (free: Reddit + Finnhub + RSS) ---
SENTIMENT_ENABLED = os.getenv("SENTIMENT_ENABLED", "false").lower() == "true"
SENTIMENT_SCAN_INTERVAL_MINUTES = int(os.getenv("SENTIMENT_SCAN_INTERVAL_MINUTES", "30"))
SENTIMENT_SOURCES = os.getenv("SENTIMENT_SOURCES", "reddit,finnhub,rss")
SENTIMENT_MAX_ITEMS = int(os.getenv("SENTIMENT_MAX_ITEMS", "80"))

# --- Dashboard ---
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8420"))
