const API_BASE = "http://localhost:8420";

// ── /api/state ──────────────────────────────────────────────────────

export type OpenPosition = {
  ticker: string;
  entry_price: number;
  entry_date: string;
  pnl_pct: number;
  composite_score: number;
  position_size_pct: number;
};

export type RecentTrade = {
  ticker: string;
  pnl_usd: number;
  pnl_pct: number;
  exit_date: string;
  exit_reason: string;
};

export type FactorPerf = {
  factor_name: string;
  ic_60d: number;
  hit_rate_60d: number;
  weight: number;
};

export type DashboardState = {
  timestamp: string;
  uptime_seconds: number;
  online_learning_enabled: boolean;
  trend_filter_enabled: boolean;
  markets_active: string[];
  equity: number;
  cash: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  cumulative_return_pct: number;
  num_positions: number;
  spx_return_pct: number;
  open_positions: OpenPosition[];
  recent_trades: RecentTrade[];
  signals: Signal[];
  alerts: unknown[];
  factor_performance: FactorPerf[];
  regime: RegimeSnapshot | null;
  weights: Record<string, number>;
  last_scoring: string | null;
  last_trades_executed: number;
};

// ── /api/equity ─────────────────────────────────────────────────────

export type EquityDatum = {
  date: string;
  equity: number;
  cash: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  cumulative_return_pct: number;
  spx_return_pct: number;
};

// ── /api/trades ─────────────────────────────────────────────────────

export type Trade = {
  trade_id: string;
  ticker: string;
  side: string;
  quantity: number;
  entry_price: number;
  exit_price: number | null;
  entry_date: string;
  exit_date: string | null;
  status: string;
  pnl_pct: number | null;
  pnl_usd: number | null;
  exit_reason: string | null;
  composite_score: number | null;
  position_size_pct: number | null;
};

// ── /api/signals ────────────────────────────────────────────────────

export type Signal = {
  ticker: string;
  composite_score: number;
  quality_score?: number;
  momentum_score?: number;
  value_score?: number;
  low_vol_score?: number;
  momentum_percentile?: number;
  low_vol_percentile?: number;
  score_1_10?: number;
  regime_id: number;
};

// ── /api/regime ─────────────────────────────────────────────────────

export type RegimeSnapshot = {
  date: string;
  vix: number;
  spx_vs_200ma: number;
  hy_spread_oas: number;
  regime_id?: number;
};

// ── Migration ───────────────────────────────────────────────────────

export type MigrationState = {
  gates: {
    consistency: string;
    drawdown: string;
    stability: string;
    ready: boolean;
    days_green: number;
  };
  ramp: Record<string, unknown>;
};

// ── fetch helpers ───────────────────────────────────────────────────

export async function fetchState(): Promise<DashboardState | null> {
  try {
    const res = await fetch(`${API_BASE}/api/state`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchEquity(days = 90): Promise<EquityDatum[]> {
  try {
    const res = await fetch(`${API_BASE}/api/equity?days=${days}`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export async function fetchTrades(limit = 20, status?: string): Promise<Trade[]> {
  try {
    const params = new URLSearchParams({ limit: String(limit) });
    if (status) params.set("status", status);
    const res = await fetch(`${API_BASE}/api/trades?${params}`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export async function fetchSignals(limit = 10): Promise<Signal[]> {
  try {
    const res = await fetch(`${API_BASE}/api/signals?limit=${limit}`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export async function fetchRegime(days = 90): Promise<RegimeSnapshot[]> {
  try {
    const res = await fetch(`${API_BASE}/api/regime?days=${days}`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export async function fetchMigration(): Promise<MigrationState | null> {
  try {
    const res = await fetch(`${API_BASE}/api/migration`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}
