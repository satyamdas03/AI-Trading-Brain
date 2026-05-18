"use client";

import type { Position } from "@/app/lib/api";

export default function PositionsTable({ positions }: { positions: Position[] }) {
  if (!positions || positions.length === 0) {
    return (
      <div className="text-gray-500 text-sm py-4">No open positions</div>
    );
  }

  const totalPnl = positions.reduce((sum, p) => sum + p.unrealized_pnl, 0);

  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-lg font-semibold text-white">Open Positions</h3>
        <span className={`text-sm ${totalPnl >= 0 ? "text-green-400" : "text-red-400"}`}>
          ${totalPnl >= 0 ? "+" : ""}{totalPnl.toFixed(2)}
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-400 border-b border-gray-700">
              <th className="text-left py-2">Ticker</th>
              <th className="text-right py-2">Qty</th>
              <th className="text-right py-2">Entry</th>
              <th className="text-right py-2">Current</th>
              <th className="text-right py-2">P&L %</th>
              <th className="text-right py-2">Days</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p) => (
              <tr key={p.ticker} className="border-b border-gray-800">
                <td className="py-2 text-white font-medium">{p.ticker}</td>
                <td className="py-2 text-right text-gray-300">{p.quantity}</td>
                <td className="py-2 text-right text-gray-300">${p.entry_price.toFixed(2)}</td>
                <td className="py-2 text-right text-gray-300">${p.current_price.toFixed(2)}</td>
                <td className={`py-2 text-right ${p.unrealized_pnl_pct >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {p.unrealized_pnl_pct >= 0 ? "+" : ""}{(p.unrealized_pnl_pct * 100).toFixed(2)}%
                </td>
                <td className="py-2 text-right text-gray-400">{p.days_held}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
