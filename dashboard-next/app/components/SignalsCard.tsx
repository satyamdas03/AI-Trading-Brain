"use client";

import type { Signal } from "@/app/lib/api";

function scoreColor(v: number): string {
  if (v >= 0.7) return "text-green-400";
  if (v >= 0.5) return "text-yellow-400";
  return "text-red-400";
}

function scoreBg(v: number): string {
  if (v >= 0.7) return "bg-green-400/20";
  if (v >= 0.5) return "bg-yellow-400/20";
  return "bg-red-400/20";
}

function regimeLabel(id: number): string {
  const labels: Record<number, string> = { 1: "Risk-On", 2: "Late-Cycle", 3: "Bear", 4: "Recovery" };
  return labels[id] ?? "Unknown";
}

export default function SignalsCard({ signals }: { signals: Signal[] }) {
  if (!signals || signals.length === 0) {
    return <div className="text-gray-500 text-sm py-4">No signals computed yet</div>;
  }

  return (
    <div>
      <h3 className="text-lg font-semibold text-white mb-2">Top Factor Signals</h3>
      <div className="space-y-2">
        {signals.slice(0, 5).map((s, i) => (
          <div key={s.ticker} className="bg-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-white font-medium">
                #{i + 1} {s.ticker}
              </span>
              <span className={`text-xs px-2 py-0.5 rounded ${scoreBg(s.composite_score)} ${scoreColor(s.composite_score)}`}>
                {(s.composite_score * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex gap-2 text-xs">
              <span className="text-gray-400">M:{((s.momentum_score ?? s.momentum_percentile ?? 0) * 100).toFixed(0)}%</span>
              <span className="text-gray-400">Q:{((s.quality_score ?? 0) * 100).toFixed(0)}%</span>
              <span className="text-gray-400">V:{((s.value_score ?? 0) * 100).toFixed(0)}%</span>
              <span className="text-gray-400">LV:{((s.low_vol_score ?? s.low_vol_percentile ?? 0) * 100).toFixed(0)}%</span>
              <span className="text-blue-400 ml-auto">{regimeLabel(s.regime_id)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
