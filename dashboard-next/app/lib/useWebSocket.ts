"use client";

import { useEffect, useRef, useState, useCallback } from "react";

type WSMessage = {
  type: string;
  data: unknown;
};

export function useWebSocket() {
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
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
          const msg = JSON.parse(event.data);
          setLastMessage(msg);
        } catch {
          // ignore non-JSON messages
        }
      };
    }

    connect();
    return () => wsRef.current?.close();
  }, []);

  return { lastMessage, connected };
}
