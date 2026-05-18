import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Trading Brain — Dashboard",
  description: "Multi-asset autonomous trading system dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-gray-950 text-gray-100 min-h-screen font-mono antialiased">
        <header className="border-b border-gray-800 px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-xl font-bold text-green-400">🧠</span>
            <span className="text-lg font-semibold text-white">AI Trading Brain</span>
            <span className="text-xs px-2 py-0.5 rounded bg-green-400/10 text-green-400 border border-green-400/30">
              Paper Trading
            </span>
          </div>
          <div className="flex items-center gap-4 text-xs text-gray-400">
            <StatusDot />
          </div>
        </header>
        <main className="p-6">{children}</main>
      </body>
    </html>
  );
}

function StatusDot() {
  return (
    <div className="flex items-center gap-2">
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
        <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
      </span>
      Live
    </div>
  );
}
