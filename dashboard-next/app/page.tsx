"use client";

import { useEffect, useState, useCallback } from "react";
import { useWebSocket } from "@/app/lib/useWebSocket";
import {
  fetchState,
  fetchEquity,
  fetchTrades,
  fetchSignals,
  fetchRegime,
  fetchMigration,
  type DashboardState,
  type EquityDatum,
  type Trade,
  type Signal,
  type RegimeSnapshot,
  type MigrationState,
  type FactorPerf,
} from "@/app/lib/api";
import EquityChart from "@/app/components/EquityChart";
import PositionsTable from "@/app/components/PositionsTable";
import SignalsCard from "@/app/components/SignalsCard";
import TradeHistory from "@/app/components/TradeHistory";

function StatCard({ label, value, sub, accent }: {
  label: string;
  value: string;
  sub?: string;
  accent?: "green" | "red" | "yellow";
}) {
  const color =
    accent === "green" ? "text-green-400"
    : accent === "red" ? "text-red-400"
    : accent === "yellow" ? "text-yellow-400"
    : "text-white";
  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className={`text-xl font-bold ${color}`}>{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
}

function Skeleton() {
  return (
    <main className="p-6 max-w-7xl mx-auto space-y-6 animate-pulse">
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="bg-gray-800 rounded-lg p-4 h-20" />
        ))}
      </div>
      <div className="bg-gray-800 rounded-lg h-72" />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-gray-800 rounded-lg h-48" />
        <div className="bg-gray-800 rounded-lg h-48" />
      </div>
    </main>
  );
}

export default function DashboardPage() {
  const { state: wsState, connected } = useWebSocket();

  const [state, setState] = useState<DashboardState | null>(null);
  const [equity, setEquity] = useState<EquityDatum[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [regime, setRegime] = useState<RegimeSnapshot[]>([]);
  const [migration, setMigration] = useState<MigrationState | null>(null);
  const [loading, setLoading] = useState(true);

  // Initial fetch from REST API
  const loadAll = useCallback(async () => {
    const [st, eq, tr, sig, reg, mig] = await Promise.all([
      fetchState(),
      fetchEquity(90),
      fetchTrades(20),
      fetchSignals(10),
      fetchRegime(90),
      fetchMigration(),
    ]);
    setState(st);
    setEquity(eq);
    setTrades(tr);
    setSignals(sig);
    setRegime(reg);
    setMigration(mig);
    setLoading(false);
  }, []);

  useEffect(() => { loadAll(); }, [loadAll]);

  // When WebSocket pushes fresh state, merge it in
  useEffect(() => {
    if (!wsState) return;
    setState(wsState);
    // Use signals from WS state
    if (wsState.signals?.length) setSignals(wsState.signals);
  }, [wsState]);

  if (loading) return <Skeleton />;

  // Derive positions array for PositionsTable from open_positions
  const positionsForTable = state?.open_positions?.map((p) => ({
    ticker: p.ticker,
    quantity: 0, // not in WS state
    entry_price: p.entry_price,
    current_price: 0, // not in WS state
    market_value: 0,
    unrealized_pnl: 0,
    unrealized_pnl_pct: p.pnl_pct ?? 0,
    days_held: 0, // not in WS state
  })) ?? [];

  return (
    <main className="p-6 max-w-7xl mx-auto space-y-6">
      {/* status bar */}
      <div className="flex items-center gap-2 text-xs">
        <span className={`inline-block w-2 h-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`} />
        <span className="text-gray-400">{connected ? "Live" : "REST only — ws disconnected"}</span>
        {state && (
          <span className="text-gray-500 ml-2">
            Markets: {state.markets_active.join(", ")} &middot; Trend filter: {state.trend_filter_enabled ? "ON" : "OFF"}
          </span>
        )}
      </div>

      {/* backend down */}
      {!state && (
        <div className="bg-yellow-400/10 border border-yellow-400/30 rounded-lg p-4 text-yellow-400 text-sm">
          Dashboard backend not reachable. Start daemon:{" "}
          <code className="bg-gray-800 px-1.5 py-0.5 rounded">python brain.py</code>
        </div>
      )}

      {/* portfolio summary */}
      {state && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <StatCard
            label="Equity"
            value={`$${state.equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          />
          <StatCard
            label="Cash"
            value={`$${state.cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          />
          <StatCard
            label="Total Return"
            value={`${(state.cumulative_return_pct) >= 0 ? "+" : ""}${state.cumulative_return_pct.toFixed(2)}%`}
            accent={state.cumulative_return_pct >= 0 ? "green" : "red"}
          />
          <StatCard
            label="Daily P&L"
            value={`${state.daily_pnl_pct >= 0 ? "+" : ""}${state.daily_pnl_pct.toFixed(2)}%`}
            sub={`$${state.daily_pnl.toFixed(2)}`}
            accent={state.daily_pnl_pct >= 0 ? "green" : "red"}
          />
          <StatCard label="Positions" value={`${state.num_positions}`} />
          <StatCard
            label="SPX Return"
            value={`${state.spx_return_pct >= 0 ? "+" : ""}${state.spx_return_pct.toFixed(2)}%`}
            accent={state.spx_return_pct >= 0 ? "green" : "red"}
          />
        </div>
      )}

      {/* equity chart */}
      <div className="bg-gray-800 rounded-lg p-4">
        <EquityChart data={equity} />
      </div>

      {/* positions + signals */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-gray-800 rounded-lg p-4">
          <PositionsTable positions={positionsForTable} />
        </div>
        <div className="bg-gray-800 rounded-lg p-4">
          <SignalsCard signals={signals} />
        </div>
      </div>

      {/* factor performance */}
      {state && state.factor_performance && state.factor_performance.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-2">Factor Performance</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            {state.factor_performance.map((f: FactorPerf) => (
              <div key={f.factor_name}>
                <span className="text-gray-400 capitalize">{f.factor_name}:</span>{" "}
                <span className="text-white">IC {(f.ic_60d * 100).toFixed(1)}%</span>{" "}
                <span className={f.hit_rate_60d >= 0.5 ? "text-green-400" : "text-red-400"}>
                  hit {(f.hit_rate_60d * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* trade history */}
      <div className="bg-gray-800 rounded-lg p-4">
        <TradeHistory trades={trades} />
      </div>

      {/* migration gate */}
      {migration && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-2">Live Migration Gate</h3>
          <div className="grid grid-cols-3 gap-3 text-sm mb-2">
            <GateLight label="Consistency" status={migration.gates.consistency} />
            <GateLight label="Drawdown" status={migration.gates.drawdown} />
            <GateLight label="Stability" status={migration.gates.stability} />
          </div>
          <div className="text-xs text-gray-400">
            {migration.gates.ready ? `${migration.gates.days_green} days green — ready for live` : "Not ready"}
          </div>
        </div>
      )}
    </main>
  );
}

function GateLight({ label, status }: { label: string; status: string }) {
  const color =
    status === "GREEN" ? "bg-green-500" : status === "RED" ? "bg-red-500" : "bg-yellow-500";
  return (
    <div className="flex items-center gap-2">
      <span className={`w-2 h-2 rounded-full ${color}`} />
      <span className="text-gray-400">{label}</span>
      <span className="text-white">{status}</span>
    </div>
  );
}
