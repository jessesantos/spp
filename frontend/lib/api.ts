/**
 * Typed fetch wrapper + Zod schemas mirroring the FastAPI backend.
 * Schemas double as runtime validators (OWASP A03 - input validation).
 */
import { z } from "zod";

export const TickerInfoSchema = z.object({
  ticker: z.string(),
  name: z.string().nullable().optional(),
  currency: z.string(),
});
export type TickerInfo = z.infer<typeof TickerInfoSchema>;

export const SentimentScoreSchema = z.object({
  score: z.number().min(-1).max(1),
  confidence: z.number().min(0).max(1),
  positives: z.number().int(),
  negatives: z.number().int(),
  neutrals: z.number().int(),
});
export type SentimentScore = z.infer<typeof SentimentScoreSchema>;

export const PredictionPointSchema = z.object({
  date: z.string(),
  predicted_close: z.number(),
  direction: z.enum(["ALTA", "BAIXA", "NEUTRO"]),
});
export type PredictionPoint = z.infer<typeof PredictionPointSchema>;

export const HorizonPredictionSchema = z.object({
  horizon: z.enum(["D1", "W1", "M1"]),
  horizon_days: z.number().int(),
  target_date: z.string(),
  base_close: z.number(),
  predicted_close: z.number(),
  predicted_pct: z.number(),
  direction: z.enum(["ALTA", "BAIXA", "NEUTRO"]),
  explanation: z.string().nullable().optional(),
});
export type HorizonPrediction = z.infer<typeof HorizonPredictionSchema>;

export const MultiHorizonResponseSchema = z.object({
  ticker: z.string(),
  last_price: z.number(),
  horizons: z.array(HorizonPredictionSchema),
  sentiment: SentimentScoreSchema.nullable().optional(),
  predictions: z.array(PredictionPointSchema),
  direction_accuracy: z.number().nullable().optional(),
});
export type MultiHorizonResponse = z.infer<typeof MultiHorizonResponseSchema>;

// Legacy alias so existing callers keep compiling.
export const PredictionResponseSchema = MultiHorizonResponseSchema;
export type PredictionResponse = MultiHorizonResponse;

export const PredictionHistoryItemSchema = z.object({
  id: z.number().int(),
  ticker: z.string(),
  horizon_days: z.number().int(),
  created_at: z.string(),
  target_date: z.string(),
  base_close: z.number(),
  predicted_close: z.number(),
  predicted_pct: z.number(),
  actual_close: z.number().nullable().optional(),
  error_pct: z.number().nullable().optional(),
  resolved: z.boolean(),
  explanation: z.string().nullable().optional(),
});

export const ExplanationResponseSchema = z.object({
  ticker: z.string(),
  horizon_days: z.number().int(),
  explanation: z.string(),
});
export type ExplanationResponse = z.infer<typeof ExplanationResponseSchema>;
export type PredictionHistoryItem = z.infer<typeof PredictionHistoryItemSchema>;

export const HistoryResponseSchema = z.object({
  ticker: z.string(),
  items: z.array(PredictionHistoryItemSchema),
});
export type HistoryResponse = z.infer<typeof HistoryResponseSchema>;

export const ReconcileResponseSchema = z.object({
  checked: z.number().int(),
  resolved: z.number().int(),
});
export type ReconcileResponse = z.infer<typeof ReconcileResponseSchema>;

/**
 * Base URL used by the typed fetch wrapper.
 *
 * During SSR inside Docker the browser-visible URL (``NEXT_PUBLIC_API_URL``,
 * typically ``http://localhost:8000``) does not resolve to the backend
 * container. ``API_URL_INTERNAL`` (e.g. ``http://backend:8000``) is
 * preferred at runtime and falls back to the public URL client-side.
 */
const API_BASE =
  (typeof window === "undefined"
    ? process.env.API_URL_INTERNAL ?? process.env.NEXT_PUBLIC_API_URL
    : process.env.NEXT_PUBLIC_API_URL) ?? "http://localhost:8000";

async function request<T>(
  path: string,
  schema: z.ZodType<T>,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    cache: "no-store",
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    throw new Error(`SPP API ${path} failed: ${response.status}`);
  }
  const data = await response.json();
  return schema.parse(data);
}

export const api = {
  listTickers: () => request("/api/tickers", z.array(TickerInfoSchema)),
  predict: (ticker: string, days = 5) =>
    request(
      `/api/predict/${encodeURIComponent(ticker)}?days=${days}`,
      MultiHorizonResponseSchema,
    ),
  predictWithHorizons: (ticker: string) =>
    request(
      `/api/predict/${encodeURIComponent(ticker)}`,
      MultiHorizonResponseSchema,
    ),
  history: (ticker: string, limit = 60) =>
    request(
      `/api/predictions/${encodeURIComponent(ticker)}/history?limit=${limit}`,
      HistoryResponseSchema,
    ),
  reconcile: () =>
    request("/api/predictions/reconcile", ReconcileResponseSchema, {
      method: "POST",
    }),
  train: (ticker: string) =>
    request(
      `/api/train/${encodeURIComponent(ticker)}`,
      z.object({ status: z.string(), ticker: z.string() }),
      { method: "POST" },
    ),
  sentiment: (ticker: string) =>
    request(`/api/sentiment/${encodeURIComponent(ticker)}`, SentimentScoreSchema),
  explanation: (ticker: string, horizonDays: number) =>
    request(
      `/api/predictions/${encodeURIComponent(ticker)}/horizon/${horizonDays}/explanation`,
      ExplanationResponseSchema,
    ),
};

export const WS_BASE =
  process.env.NEXT_PUBLIC_WS_URL ??
  (API_BASE.startsWith("https")
    ? API_BASE.replace("https", "wss")
    : API_BASE.replace("http", "ws"));
