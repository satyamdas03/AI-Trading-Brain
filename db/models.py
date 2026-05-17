"""Pydantic models for all database tables."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DailySignal(BaseModel):
    """Per-ticker daily score with factor breakdown + regime."""

    ticker: str
    market: str  # "US" | "IN"
    date: str  # YYYY-MM-DD
    composite_score: float
    score_1_10: int
    quality_percentile: float
    momentum_percentile: float
    value_percentile: float
    low_vol_percentile: float
    short_interest_percentile: float
    insider_percentile: float
    delivery_percentile: Optional[float] = None  # India only
    regime_id: int
    regime_confidence: float
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RegimeHistory(BaseModel):
    """Macro data snapshot for regime detection training."""

    date: str
    vix: float
    vix_20d_change: float
    spx_vs_200ma: float
    hy_spread_oas: float
    ism_pmi: float
    yield_spread_2y10y: Optional[float] = None
    cpi_yoy: Optional[float] = None
    fed_funds_rate: Optional[float] = None
    regime_id: Optional[int] = None  # filled after HMM fit


class PortfolioSnapshot(BaseModel):
    """Daily equity curve point."""

    date: str
    equity: float
    cash: float
    positions_value: float
    daily_pnl: float
    daily_pnl_pct: float
    cumulative_return_pct: float
    num_positions: int
    spx_close: Optional[float] = None
    spx_return_pct: Optional[float] = None


class Trade(BaseModel):
    """Every trade with full factor attribution."""

    trade_id: str
    ticker: str
    market: str
    side: str  # "BUY" | "SELL"
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_date: str
    exit_date: Optional[str] = None
    status: str  # "OPEN" | "CLOSED" | "CANCELLED"
    exit_reason: Optional[str] = None
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None

    # Factor scores at entry
    composite_score: float
    quality_pct: Optional[float] = None
    momentum_pct: Optional[float] = None
    value_pct: Optional[float] = None
    low_vol_pct: Optional[float] = None
    regime_id: Optional[int] = None

    # Risk
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    stop_loss_price: Optional[float] = None
    position_size_pct: Optional[float] = None

    # Agent debate (Phase 3+)
    debate_json: Optional[str] = None
    agent_verdict: Optional[str] = None

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class DebateRecord(BaseModel):
    """7-agent debate output stored per trade decision."""

    trade_id: str
    ticker: str
    market: str
    timestamp: str
    macro_verdict: str
    fundamental_verdict: str
    technical_verdict: str
    sentiment_verdict: str
    geopolitical_verdict: str
    adversarial_verdict: str
    head_analyst_verdict: str
    head_analyst_confidence: float
    head_analyst_reasoning: str
    raw_json: str  # full debate JSON


class FactorPerformance(BaseModel):
    """Rolling factor performance tracking for weight optimization."""

    factor_name: str
    date: str
    ic_60d: Optional[float] = None
    icir_60d: Optional[float] = None
    hit_rate_60d: Optional[float] = None
    weight: float
    weight_target: Optional[float] = None


class DecayAlert(BaseModel):
    """Strategy decay detection alerts."""

    alert_id: str
    timestamp: str
    detector: str  # "CUSUM" | "ROLLING_SHARPE" | "PSI"
    severity: str  # "GREEN" | "YELLOW" | "RED"
    value: float
    threshold: float
    details: Optional[str] = None
    acknowledged: bool = False
