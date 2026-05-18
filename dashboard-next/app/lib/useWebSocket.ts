"use client";

import { useEffect, useRef, useState } from "react";
import type { DashboardState } from "@/app/lib/api";

export function useWebSocket() {
  const [state, setState] = useState<DashboardState | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.hostname;
    const url = `${protocol}//${host}:8420/ws`;

    function connect() {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as DashboardState;
          setState(data);
        } catch {
          // ignore non-JSON
        }
      };
    }

    connect();
    return () => wsRef.current?.close();
  }, []);

  return { state, connected };
}
