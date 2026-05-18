const API_BASE = "http://localhost:8420";

export type PortfolioSnapshot = {
  equity: number;
  cash: number;
  positions_value: number;
  num_positions: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  total_pnl: number;
  total_pnl_pct: number;
  drawdown_pct: number;
  peak_equity: number;
  spx_return_pct: number;
  timestamp: string;
};

export type Position = {
  ticker: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  days_held: number;
};

export type Signal = {
  ticker: string;
  composite_score: number;
  quality_score: number;
  momentum_score: number;
  value_score: number;
  low_vol_score: number;
  regime_id: number;
};

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
};

export type ReplayStats = {
  days_processed: number;
  trades_taken: number;
  wins: number;
  losses: number;
  win_rate: number;
  current_equity: number;
  peak_equity: number;
  current_drawdown: number;
  last_processed_date: string;
};

export type RegimeData = {
  date: string;
  vix: number;
  spx_vs_200ma: number;
  hy_spread_oas: number;
};

export type DebateRecord = {
  ticker: string;
  verdict: string;
  consensus_score: number;
  investment_thesis: string;
  latency_ms: number;
};

export async function fetchPortfolio(): Promise<PortfolioSnapshot | null> {
  try {
    const res = await fetch(`${API_BASE}/api/portfolio`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchPositions(): Promise<Position[]> {
  try {
    const res = await fetch(`${API_BASE}/api/positions`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export async function fetchSignals(): Promise<Signal[]> {
  try {
    const res = await fetch(`${API_BASE}/api/signals`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export async function fetchTrades(limit = 20): Promise<Trade[]> {
  try {
    const res = await fetch(`${API_BASE}/api/trades?limit=${limit}`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export async function fetchReplayStats(): Promise<ReplayStats | null> {
  try {
    const res = await fetch(`${API_BASE}/api/replay`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchRegimeHistory(limit = 30): Promise<RegimeData[]> {
  try {
    const res = await fetch(`${API_BASE}/api/regime?limit=${limit}`);
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}
