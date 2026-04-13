/**
 * PredictionCard - Tailwind-styled (no shadcn/ui dependency).
 * Shows ticker, current price, predicted price and trend badge.
 */
import type { JSX } from "react";

export interface PredictionCardProps {
  ticker: string;
  currentPrice: number;
  predictedPrice: number;
  directionAccuracy?: number | null;
  trend: "ALTA" | "BAIXA" | "NEUTRO";
}

function formatBRL(value: number): string {
  return new Intl.NumberFormat("pt-BR", {
    style: "currency",
    currency: "BRL",
  }).format(value);
}

export function PredictionCard({
  ticker,
  currentPrice,
  predictedPrice,
  directionAccuracy,
  trend,
}: PredictionCardProps): JSX.Element {
  const variation = currentPrice
    ? ((predictedPrice - currentPrice) / currentPrice) * 100
    : 0;
  const trendColor =
    trend === "ALTA"
      ? "text-emerald-400"
      : trend === "BAIXA"
        ? "text-red-400"
        : "text-yellow-400";
  const badgeTone =
    trend === "ALTA"
      ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
      : trend === "BAIXA"
        ? "bg-red-500/20 text-red-300 border-red-500/40"
        : "bg-yellow-500/20 text-yellow-200 border-yellow-500/40";
  const arrow = trend === "ALTA" ? "↑" : trend === "BAIXA" ? "↓" : "→";

  return (
    <section className="rounded-xl border border-neutral-800 bg-neutral-900 p-6 shadow-sm">
      <header className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-neutral-100">{ticker}</h2>
        <span
          className={`rounded-full border px-3 py-1 text-xs font-semibold ${badgeTone}`}
          aria-label={`Tendência ${trend}`}
        >
          {trend} {arrow}
        </span>
      </header>
      <div className="mt-4">
        <p className="text-sm text-neutral-400">Preço atual</p>
        <p className="text-2xl font-bold text-neutral-50">{formatBRL(currentPrice)}</p>
        <p className={`mt-2 text-lg font-semibold ${trendColor}`}>
          Predição: {formatBRL(predictedPrice)}
          <span className="ml-2 text-sm">
            ({variation > 0 ? "+" : ""}
            {variation.toFixed(2)}%)
          </span>
        </p>
        {directionAccuracy !== null && directionAccuracy !== undefined && (
          <p className="mt-1 text-xs text-neutral-500">
            Acurácia de direção: {(directionAccuracy * 100).toFixed(1)}%
          </p>
        )}
      </div>
    </section>
  );
}
