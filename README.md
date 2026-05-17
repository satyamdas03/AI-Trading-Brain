# AI Trading Brain

Unified autonomous trading system combining quantitative factor signals, Claude-powered multi-agent reasoning (PARA-DEBATE), and automated execution via Alpaca.

**Status:** Phase 5 Complete — Pre-market scoring, market-open execution, intraday monitoring, weekend retraining, X/Twitter sentiment scanner, historical replay engine all operational. Paper trading mode (execution gate closed).

---

## Architecture

```
ai-trading-brain/
├── brain.py                  # Main daemon entry — 13 scheduled routines
├── config.py                 # Central configuration (from .env)
├── scheduler.py              # APScheduler wrapper with NY timezone
│
├── signals/                  # Phase 1: Factor Signal Engine
│   └── aggregator.py         # 4-factor scoring + cross-sectional percentiles + regime detection
│
├── strategy/                 # Phase 3: AI Reasoning
│   └── decider.py            # 7-agent PARA-DEBATE (5 specialists + adversarial + head analyst)
│
├── execution/                # Phase 2: Trade Execution
│   ├── engine.py             # Bracket-order execution (take-profit + stop-loss at entry)
│   ├── alpaca_client.py      # Alpaca Markets API wrapper
│   ├── order_manager.py      # Order lifecycle tracking
│   └── polymarket_bridge.py  # Bridge to Polymarket Pipeline V2
│
├── risk/                     # Risk Management
│   └── manager.py            # Position sizing, sector caps, drawdown limits, multi-asset detection
│
├── ingestion/                # Data Pipeline
│   ├── universe.py           # S&P 500 + Nifty 200 ticker lists
│   ├── market_data.py        # Price/volume data via yfinance
│   └── macro_data.py         # VIX, yields, spreads, ISM, CPI
│
├── learning/                 # Phase 4: Continuous Improvement
│   ├── historical_replay.py  # 20-year replay engine with trend filter
│   ├── feedback.py           # Performance tracking to Supabase
│   ├── retrainer.py          # Weekend hyperparameter optimization
│   ├── weight_optimizer.py   # Gradient-based factor weight optimization
│   ├── attribution.py        # Factor IC/ICIR attribution
│   └── decay.py              # Strategy decay detection (3-tier alerts)
│
├── sentiment/                # X/Twitter Sentiment Scanner
│   └── x_scanner.py          # Financial tweet classification via Claude Haiku
│
├── migration/                # Paper → Live Migration Gates
│   ├── live_gate.py          # 3-gate system: Consistency, Drawdown, Stability
│   └── ramp.py               # Graduated capital ramp (10%→25%→50%→100%)
│
├── paper/                    # Paper Trading
│   └── pnl.py                # Simulated P&L tracking
│
├── notifications/            # Alerting
│   ├── alerts.py             # Signal/decay alert generation
│   └── email.py              # Email notification templates
│
├── dashboard/                # Live Dashboard (FastAPI)
│   └── server.py             # REST API + WebSocket on port 8420
│
├── db/                       # Database
│   ├── client.py             # Supabase client (read/write/upsert)
│   ├── models.py             # Pydantic data models
│   └── migrations/
│       └── 001_init_trading.sql  # Full schema (7 tables)
│
├── scripts/                  # Debug and testing utilities
│   ├── run_replay_now.py     # Manually trigger replay engine
│   ├── verify.py             # System integrity check
│   └── ...
│
├── data/                     # Local data (gitignored)
│   ├── brain_state.json      # Last scoring results
│   └── replay/               # Replay engine state + checkpoints
│
├── logs/                     # Runtime logs (gitignored)
│   └── brain.log
│
├── pyproject.toml            # Python 3.12+ dependencies
└── .env.example              # Environment variables template
```

## Core Components

### 1. Factor Signal Engine (`signals/aggregator.py`)

Four equally-weighted factors ranked via cross-sectional percentiles:

| Factor | Weight | Metric |
|--------|--------|--------|
| Quality | 25% | ROE + Profit Margins (FMP data) |
| Momentum | 25% | 12-1 month price momentum |
| Value | 25% | P/E + P/B valuation |
| Low Vol | 25% | Inverse 60-day realized volatility |

Each factor scored 0-1 via percentile rank across universe. Composite = equal-weighted average (learned weights disabled — see Learning section).

Regime detection via HMM (4 states) trained on VIX, yield spreads, ISM PMI, CPI, Fed funds rate. Regime 1 = risk-on (allows entries).

### 2. PARA-DEBATE Strategy (`strategy/decider.py`)

7-agent debate system following NeuralQuant's proven architecture:

- **Phase 1:** 5 specialist agents run in parallel (Claude Haiku): Macro, Fundamental, Technical, Sentiment, Geopolitical
- **Phase 2:** Adversarial agent (enforced BEAR stance) reviews all specialist outputs
- **Phase 3:** Weighted consensus scoring (adversarial gets 1.5x weight)
- **Phase 4:** Head Analyst (Claude Sonnet) synthesizes final verdict with confidence level

X/Twitter sentiment data enriches debate contexts when available (see Sentiment Scanner).

### 3. Trade Execution (`execution/engine.py`)

- **Entry:** Market orders at 9:37 AM ET
- **Bracket orders:** Take-profit (7%) + Stop-loss (6%) placed immediately on fill
- **Time exit:** Auto-close after 21 days if neither TP nor SL hit
- **Position sizing:** Equal-weight by signal rank, capped at 10% per position, 30% per sector
- **Trend filter:** Blocks entries when SPX below 200-day moving average
- **Gates:** `DRY_RUN` and `TRADE_ENABLED` both checked before live orders

### 4. Historical Replay Engine (`learning/historical_replay.py`)

Simulates full strategy on 20-year price data during market-closed hours:

- 50-S&P-500 basket with factor scoring per historical date
- Trend filter (SPX vs 200MA) applied per historical day
- Simulated trades tracked with P&L, win rate, drawdown
- Checkpoint/resume support for long-running replays
- Latest run: 4,780 days, 1,580 trades, $219,177 equity (from $100K), 58.2% WR

### 5. Online Learning (`learning/weight_optimizer.py`)

Gradient-based factor weight optimization from replay data. **Currently disabled** (`ONLINE_LEARNING_ENABLED=false`) — 19-year replay showed equal weights (+141.9%) outperformed learned weights (+119.2%) by $22,716 on $100K seed. See learning section below.

### 6. X/Twitter Sentiment Scanner (`sentiment/x_scanner.py`)

Classifies financial tweets using Claude Haiku following Grox ContentClassifier pattern:

- 3-step pipeline: `_to_convo()` → `_sample()` → `_parse()`
- Sources: X API v2 (OAuth2) → Nitter mirrors → demo data fallback
- 18 financial accounts monitored, 13 cashtags tracked
- Output: SentimentScan with per-ticker net_score, confidence, bull/bear counts
- Enriches PARA-DEBATE contexts during market-open routine

### 7. Live Dashboard (`dashboard/server.py`)

FastAPI server on port 8420:

- **REST endpoints:** `/api/state`, `/api/equity`, `/api/trades`, `/api/signals`, `/api/regime`, `/api/alerts`, `/api/migration`
- **WebSocket:** `/ws` — real-time state push every 5 seconds
- Migration gate status, factor performance, decay alerts all exposed

### 8. Migration Gates (`migration/live_gate.py`)

Three-gate paper-to-live migration system:

| Gate | Metric | Threshold |
|------|--------|-----------|
| Consistency | Sharpe Ratio (60-day) | > 0.5 |
| Drawdown | Max Drawdown | < 10% |
| Stability | Sharpe Ratio (120-day) | > 0.3 |

Graduated capital ramp: 10% → 25% → 50% → 100% over 4 weeks when all gates green.

## Daemon Routines

The scheduler runs 13 routines on NY time:

| # | Name | Time (ET) | Action |
|---|------|-----------|--------|
| 1 | Pre-Market Scoring | 6:07 AM | Fetch universe, compute 4-factor signals, detect regime, store to Supabase + state file |
| 2 | Market Open Execution | 9:37 AM | Load top-N signals, run PARA-DEBATE, execute bracket orders |
| 3 | Intraday Monitor | Every 15 min | Check positions, stop-loss hits, trailing stops |
| 4 | Market Close | 4:03 PM | Calculate P&L, factor attribution, update equity curve |
| 5 | Weekend Retrain | Fri 4:07 PM | Hyperparameter optimization (500 trials), factor weight recalibration |
| 6 | US Data Refresh | 5:00 AM | Refresh S&P 500 universe + fundamentals |
| 7 | Macro Update | 8:30 AM | Fetch VIX, yields, spreads, ISM, CPI |
| 8 | Crypto Scan | Every 30 min | Top-20 crypto scoring (momentum + low-vol only) |
| 9 | India Scan | 9:15 AM IST | NSE 200 signal computation |
| 10 | Polymarket Scan | Every 30 min | Pipeline V2 bridge: classify → detect edge → execute |
| 11 | X Sentiment Scanner | Every 30 min | Financial tweet classification |
| 12 | Replay Engine | Market-closed hrs | 20-year historical simulation |
| 13 | Decay Monitor | Hourly | Strategy drift detection, 3-tier alerts |

## Risk-Reward Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Take-Profit | +7% | Target 1.16:1 reward/risk |
| Stop-Loss | -6% | Expected value +1.54%/trade at 58% WR |
| Max Position | 10% of equity | Single-ticker concentration cap |
| Max Sector | 30% of equity | Sector diversification |
| Max Drawdown | 10% | Circuit breaker |
| Daily Loss Limit | $100 | Stop-loss for session |
| Time Exit | 21 days | Remove stale positions |
| Max Positions | 10 | Portfolio saturation cap |

## Database (Supabase PostgreSQL)

7 tables shared with NeuralQuant project:

| Table | Purpose |
|-------|---------|
| `daily_signals` | Per-ticker factor scores with regime labels |
| `regime_history` | Macro indicators for HMM training |
| `portfolio_snapshots` | Daily equity curve |
| `trades` | Full trade lifecycle (entry→exit→P&L) |
| `debate_records` | PARA-DEBATE agent verdicts per trade |
| `factor_performance` | Factor IC/ICIR/hit rate tracking |
| `decay_alerts` | Strategy decay detection alerts |

## Getting Started

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- API keys: Anthropic, Alpaca, FMP, Finnhub, X (Twitter)

### Setup

```bash
# Clone
git clone https://github.com/satyamdas/AI-Trading-Brain.git
cd AI-Trading-Brain

# Install dependencies
uv sync

# Configure
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

See `.env.example` for full list. Critical ones:

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude API for PARA-DEBATE + sentiment |
| `SUPABASE_URL` | PostgreSQL for signal/trade storage |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase write access |
| `ALPACA_API_KEY` | Alpaca for order execution |
| `ALPACA_PAPER_TRADE` | `true` = paper, `false` = live |
| `TRADE_ENABLED` | **GATE**: `true` to execute real orders |
| `DRY_RUN` | `true` = log only, no API calls |
| `SENTIMENT_ENABLED` | `true` to activate X/Twitter scanner |
| `ONLINE_LEARNING_ENABLED` | `true` to enable weight optimization |

### Run

```bash
# Start the daemon (all routines)
python brain.py

# Start dashboard only
uvicorn dashboard.server:app --port 8420

# Run historical replay manually
python scripts/run_replay_now.py
```

## Learning: Why Online Learning Is Disabled

19-year replay (2006-2025) on 50 S&P 500 stocks:

| Strategy | Equity (from $100K) | Return | Max DD |
|----------|---------------------|--------|--------|
| Equal-Weight Factors | $241,948 | +141.9% | 32.1% |
| Learned Weights | $219,177 | +119.2% | 30.7% |

Equal weights beat learned weights by $22,716. Reason: learned weights overfit to recent regime, equal weights more robust across regime shifts. Decision: ship equal-weight first, re-test learning after 6 months live data.

## Project Phases

| Phase | Status | What |
|-------|--------|------|
| Phase 1 | Done | Factor signal engine + regime detection + Supabase schema |
| Phase 2 | Done | Alpaca execution engine with bracket orders |
| Phase 3 | Done | 7-agent PARA-DEBATE strategy decider |
| Phase 4 | Done | Online learning + decay detection + replay engine |
| Phase 5 | Done | Daemon scheduler + dashboard + migration gates + X sentiment |
| Phase 6 | Planned | Multi-asset live (India + Crypto + Polymarket) |
| Phase 7 | Planned | Next.js standalone dashboard UI |

## Key Design Decisions

- **Equal-weight factors over learned:** More robust across regime shifts, proven in 19-year replay
- **Bracket orders at entry:** No intraday monitoring gap — TP/SL set immediately
- **Haiku for specialists, Sonnet for synthesis:** Cost optimization ($0.02/debate vs $0.50 all-Sonnet)
- **Trend filter gates entries:** SPX below 200MA blocks all buys, reduces drawdown in bear markets
- **Cross-sectional percentiles over raw scores:** Eliminates regime bias in factor scoring
- **Graduated migration ramp:** 4-week capital scale prevents overconfidence on small sample

## License

MIT

---

*Built with Claude Code. Anthropic AI-powered trading agent.*
