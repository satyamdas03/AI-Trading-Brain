"use client";

import type { Trade } from "@/app/lib/api";

export default function TradeHistory({ trades }: { trades: Trade[] }) {
  if (!trades || trades.length === 0) {
    return <div className="text-gray-500 text-sm py-4">No trades yet</div>;
  }

  return (
    <div>
      <h3 className="text-lg font-semibold text-white mb-2">Recent Trades</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-400 border-b border-gray-700">
              <th className="text-left py-2">Ticker</th>
              <th className="text-left py-2">Side</th>
              <th className="text-right py-2">Entry</th>
              <th className="text-right py-2">Exit</th>
              <th className="text-right py-2">Status</th>
              <th className="text-right py-2">P&L</th>
              <th className="text-right py-2">Reason</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => (
              <tr key={t.trade_id} className="border-b border-gray-800">
                <td className="py-2 text-white font-medium">{t.ticker}</td>
                <td className={`py-2 ${t.side === "BUY" ? "text-green-400" : "text-red-400"}`}>{t.side}</td>
                <td className="py-2 text-right text-gray-300">${t.entry_price.toFixed(2)}</td>
                <td className="py-2 text-right text-gray-300">{t.exit_price ? `$${t.exit_price.toFixed(2)}` : "—"}</td>
                <td className="py-2 text-right">
                  <span className={`px-1.5 py-0.5 rounded text-xs ${
                    t.status === "OPEN" ? "bg-blue-400/20 text-blue-400" :
                    t.status === "CLOSED" ? "bg-gray-400/20 text-gray-400" : "bg-yellow-400/20 text-yellow-400"
                  }`}>{t.status}</span>
                </td>
                <td className={`py-2 text-right ${(t.pnl_pct ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {t.pnl_pct != null ? `${t.pnl_pct >= 0 ? "+" : ""}${(t.pnl_pct * 100).toFixed(2)}%` : "—"}
                </td>
                <td className="py-2 text-right text-gray-400 text-xs">{t.exit_reason ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
