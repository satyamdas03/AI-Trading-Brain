"use client";

import { useEffect, useState, useCallback } from "react";
import { useWebSocket } from "@/app/lib/useWebSocket";
import {
  fetchPortfolio,
  fetchPositions,
  fetchSignals,
  fetchTrades,
  fetchReplayStats,
  type PortfolioSnapshot,
  type Position,
  type Signal,
  type Trade,
  type ReplayStats,
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
    <div className="space-y-4 animate-pulse">
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
    </div>
  );
}

export default function DashboardPage() {
  const { lastMessage, connected } = useWebSocket();

  const [portfolio, setPortfolio] = useState<PortfolioSnapshot | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [replay, setReplay] = useState<ReplayStats | null>(null);
  const [loading, setLoading] = useState(true);

  const loadAll = useCallback(async () => {
    const [p, pos, sig, tr, rp] = await Promise.all([
      fetchPortfolio(),
      fetchPositions(),
      fetchSignals(),
      fetchTrades(10),
      fetchReplayStats(),
    ]);
    setPortfolio(p);
    setPositions(pos);
    setSignals(sig);
    setTrades(tr);
    setReplay(rp);
    setLoading(false);
  }, []);

  useEffect(() => { loadAll(); }, [loadAll]);

  useEffect(() => {
    if (!lastMessage) return;
    const type = lastMessage.type;
    if (type === "portfolio_update" || type === "trade" || type === "signal" || type === "position_update") {
      loadAll();
    }
  }, [lastMessage, loadAll]);

  if (loading) {
    return (
      <main className="p-6 max-w-7xl mx-auto">
        <Skeleton />
      </main>
    );
  }

  const equityData = portfolio
    ? [{ date: new Date().toISOString().slice(0, 10), equity: portfolio.equity }]
    : [];

  return (
    <main className="p-6 max-w-7xl mx-auto space-y-6">
      {/* connection status */}
      <div className="flex items-center gap-2 text-xs">
        <span className={`inline-block w-2 h-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`} />
        <span className="text-gray-400">{connected ? "Live" : "Disconnected"}</span>
        {portfolio && (
          <span className="text-gray-500 ml-2">
            Last update: {new Date(portfolio.timestamp).toLocaleTimeString()}
          </span>
        )}
      </div>

      {/* backend unreachable */}
      {!portfolio && (
        <div className="bg-yellow-400/10 border border-yellow-400/30 rounded-lg p-4 text-yellow-400 text-sm">
          Dashboard backend not reachable. Start daemon:{" "}
          <code className="bg-gray-800 px-1.5 py-0.5 rounded">python brain.py</code>
        </div>
      )}

      {/* portfolio summary */}
      {portfolio && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <StatCard
            label="Equity"
            value={`$${portfolio.equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          />
          <StatCard
            label="Cash"
            value={`$${portfolio.cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          />
          <StatCard
            label="Total P&L"
            value={`${portfolio.total_pnl_pct >= 0 ? "+" : ""}${(portfolio.total_pnl_pct * 100).toFixed(2)}%`}
            sub={`$${portfolio.total_pnl.toFixed(2)}`}
            accent={portfolio.total_pnl_pct >= 0 ? "green" : "red"}
          />
          <StatCard
            label="Daily P&L"
            value={`${portfolio.daily_pnl_pct >= 0 ? "+" : ""}${(portfolio.daily_pnl_pct * 100).toFixed(2)}%`}
            sub={`$${portfolio.daily_pnl.toFixed(2)}`}
            accent={portfolio.daily_pnl_pct >= 0 ? "green" : "red"}
          />
          <StatCard
            label="Drawdown"
            value={`${(portfolio.drawdown_pct * 100).toFixed(2)}%`}
            sub={portfolio.drawdown_pct < -0.05 ? "⚠ Elevated" : "Normal"}
            accent={portfolio.drawdown_pct < -0.1 ? "red" : portfolio.drawdown_pct < -0.05 ? "yellow" : "green"}
          />
          <StatCard label="Positions" value={`${portfolio.num_positions}`} />
        </div>
      )}

      {/* equity chart */}
      <div className="bg-gray-800 rounded-lg p-4">
        <EquityChart data={equityData} />
      </div>

      {/* positions + signals */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-gray-800 rounded-lg p-4">
          <PositionsTable positions={positions} />
        </div>
        <div className="bg-gray-800 rounded-lg p-4">
          <SignalsCard signals={signals} />
        </div>
      </div>

      {/* trade history */}
      <div className="bg-gray-800 rounded-lg p-4">
        <TradeHistory trades={trades} />
      </div>

      {/* replay stats */}
      {replay && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-2">Historical Replay</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div><span className="text-gray-400">Days:</span> <span className="text-white">{replay.days_processed}</span></div>
            <div><span className="text-gray-400">Trades:</span> <span className="text-white">{replay.trades_taken}</span></div>
            <div><span className="text-gray-400">Win Rate:</span> <span className="text-green-400">{(replay.win_rate * 100).toFixed(1)}%</span></div>
            <div><span className="text-gray-400">Equity:</span> <span className="text-white">${replay.current_equity.toFixed(2)}</span></div>
            <div><span className="text-gray-400">Peak:</span> <span className="text-white">${replay.peak_equity.toFixed(2)}</span></div>
            <div><span className="text-gray-400">Drawdown:</span> <span className="text-red-400">{(replay.current_drawdown * 100).toFixed(2)}%</span></div>
            <div><span className="text-gray-400">Wins:</span> <span className="text-green-400">{replay.wins}</span></div>
            <div><span className="text-gray-400">Losses:</span> <span className="text-red-400">{replay.losses}</span></div>
          </div>
        </div>
      )}
    </main>
  );
}
