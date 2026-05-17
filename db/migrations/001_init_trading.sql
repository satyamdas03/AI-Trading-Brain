-- Phase 1: Core schema for daily scoring without trading
-- Run against Supabase PostgreSQL (shared NeuralQuant project)

-- Daily per-ticker scores with factor breakdown + regime
CREATE TABLE IF NOT EXISTS daily_signals (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    market TEXT NOT NULL CHECK (market IN ('US', 'IN')),
    date DATE NOT NULL,
    composite_score DOUBLE PRECISION NOT NULL,
    score_1_10 INTEGER NOT NULL CHECK (score_1_10 BETWEEN 1 AND 10),
    quality_percentile DOUBLE PRECISION,
    momentum_percentile DOUBLE PRECISION,
    value_percentile DOUBLE PRECISION,
    low_vol_percentile DOUBLE PRECISION,
    short_interest_percentile DOUBLE PRECISION,
    insider_percentile DOUBLE PRECISION,
    delivery_percentile DOUBLE PRECISION,  -- India only
    regime_id INTEGER NOT NULL,
    regime_confidence DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (ticker, date)
);

CREATE INDEX idx_daily_signals_date ON daily_signals (date DESC);
CREATE INDEX idx_daily_signals_ticker ON daily_signals (ticker, date DESC);
CREATE INDEX idx_daily_signals_market_date ON daily_signals (market, date DESC);

-- Macro data for HMM regime detection training
CREATE TABLE IF NOT EXISTS regime_history (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    vix DOUBLE PRECISION NOT NULL,
    vix_20d_change DOUBLE PRECISION DEFAULT 0,
    spx_vs_200ma DOUBLE PRECISION DEFAULT 0,
    hy_spread_oas DOUBLE PRECISION DEFAULT 350,
    ism_pmi DOUBLE PRECISION DEFAULT 51,
    yield_spread_2y10y DOUBLE PRECISION,
    cpi_yoy DOUBLE PRECISION,
    fed_funds_rate DOUBLE PRECISION,
    regime_id INTEGER,  -- filled after HMM fit
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_regime_history_date ON regime_history (date DESC);

-- Daily equity curve
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    equity DOUBLE PRECISION NOT NULL,
    cash DOUBLE PRECISION NOT NULL,
    positions_value DOUBLE PRECISION DEFAULT 0,
    daily_pnl DOUBLE PRECISION DEFAULT 0,
    daily_pnl_pct DOUBLE PRECISION DEFAULT 0,
    cumulative_return_pct DOUBLE PRECISION DEFAULT 0,
    num_positions INTEGER DEFAULT 0,
    spx_close DOUBLE PRECISION,
    spx_return_pct DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Phase 2: Trades table (created now, used later)
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    trade_id TEXT NOT NULL UNIQUE,
    ticker TEXT NOT NULL,
    market TEXT NOT NULL CHECK (market IN ('US', 'IN')),
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity DOUBLE PRECISION NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_price DOUBLE PRECISION,
    entry_date DATE NOT NULL,
    exit_date DATE,
    status TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED')),
    exit_reason TEXT,
    pnl_usd DOUBLE PRECISION,
    pnl_pct DOUBLE PRECISION,
    composite_score DOUBLE PRECISION,
    quality_pct DOUBLE PRECISION,
    momentum_pct DOUBLE PRECISION,
    value_pct DOUBLE PRECISION,
    low_vol_pct DOUBLE PRECISION,
    regime_id INTEGER,
    max_favorable_excursion DOUBLE PRECISION,
    max_adverse_excursion DOUBLE PRECISION,
    stop_loss_price DOUBLE PRECISION,
    position_size_pct DOUBLE PRECISION,
    debate_json TEXT,       -- Phase 3+
    agent_verdict TEXT,     -- Phase 3+
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_trades_ticker ON trades (ticker, entry_date DESC);
CREATE INDEX idx_trades_status ON trades (status, entry_date DESC);
CREATE INDEX idx_trades_entry_date ON trades (entry_date DESC);

-- Phase 3: Debate records
CREATE TABLE IF NOT EXISTS debate_records (
    id BIGSERIAL PRIMARY KEY,
    trade_id TEXT NOT NULL REFERENCES trades(trade_id),
    ticker TEXT NOT NULL,
    market TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT now(),
    macro_verdict TEXT,
    fundamental_verdict TEXT,
    technical_verdict TEXT,
    sentiment_verdict TEXT,
    geopolitical_verdict TEXT,
    adversarial_verdict TEXT,
    head_analyst_verdict TEXT,
    head_analyst_confidence DOUBLE PRECISION,
    head_analyst_reasoning TEXT,
    raw_json TEXT NOT NULL
);

-- Phase 4: Factor performance tracking
CREATE TABLE IF NOT EXISTS factor_performance (
    id BIGSERIAL PRIMARY KEY,
    factor_name TEXT NOT NULL,
    date DATE NOT NULL,
    ic_60d DOUBLE PRECISION,
    icir_60d DOUBLE PRECISION,
    hit_rate_60d DOUBLE PRECISION,
    weight DOUBLE PRECISION NOT NULL,
    weight_target DOUBLE PRECISION,
    UNIQUE (factor_name, date)
);

-- Phase 4: Strategy decay alerts
CREATE TABLE IF NOT EXISTS decay_alerts (
    id BIGSERIAL PRIMARY KEY,
    alert_id TEXT NOT NULL UNIQUE,
    timestamp TIMESTAMPTZ DEFAULT now(),
    detector TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('GREEN', 'YELLOW', 'RED')),
    value DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    details TEXT,
    acknowledged BOOLEAN DEFAULT false
);
