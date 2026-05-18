"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

type EquityDatum = {
  date: string;
  equity: number;
};

export default function EquityChart({ data }: { data: EquityDatum[] }) {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        No equity data — start paper trading to build history
      </div>
    );
  }

  const start = data[0].equity;
  const end = data[data.length - 1].equity;
  const change = ((end - start) / start) * 100;

  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-lg font-semibold text-white">Equity Curve</h3>
        <span className={`text-sm ${change >= 0 ? "text-green-400" : "text-red-400"}`}>
          {change >= 0 ? "+" : ""}{change.toFixed(1)}%
        </span>
      </div>
      <ResponsiveContainer width="100%" height={256}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="date" tick={{ fill: "#9CA3AF", fontSize: 11 }} />
          <YAxis
            tick={{ fill: "#9CA3AF", fontSize: 11 }}
            tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
          />
          <Tooltip
            contentStyle={{ backgroundColor: "#1F2937", border: "1px solid #374151", borderRadius: 8 }}
            labelStyle={{ color: "#E5E7EB" }}
            formatter={(value: unknown) => [`$${Number(value).toLocaleString()}`, "Equity"]}
          />
          <Line type="monotone" dataKey="equity" stroke="#22C55E" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
