"use client";

import { useEffect, useState } from "react";
import {
  PredictionResponseSchema,
  WS_BASE,
  type PredictionResponse,
} from "@/lib/api";

/**
 * useLivePrediction - subscribe to /ws/live/{ticker} and validate each
 * payload with Zod. Invalid payloads are ignored (logged in dev).
 */
export function useLivePrediction(ticker: string): PredictionResponse | null {
  const [data, setData] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    if (!ticker) return;
    const ws = new WebSocket(
      `${WS_BASE}/ws/live/${encodeURIComponent(ticker)}`,
    );

    ws.onmessage = (event: MessageEvent<string>) => {
      try {
        const parsed = PredictionResponseSchema.safeParse(JSON.parse(event.data));
        if (parsed.success) {
          setData(parsed.data);
        } else if (process.env.NODE_ENV !== "production") {
          console.warn("ws payload invalid", parsed.error.flatten());
        }
      } catch {
        // ignore malformed messages
      }
    };

    return () => ws.close();
  }, [ticker]);

  return data;
}
