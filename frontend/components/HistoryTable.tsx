/**
 * HistoryTable: shows past predictions for analysis.
 */
import type { JSX } from "react";

import type { PredictionHistoryItem } from "@/lib/api";
import { formatDate, formatMoney, formatPct } from "@/lib/format";

export interface HistoryTableProps {
  items: PredictionHistoryItem[];
}

function horizonLabel(days: number): string {
  if (days === 1) return "D1";
  if (days === 7) return "W1";
  if (days === 30) return "M1";
  return `${days}d`;
}

function statusBadge(resolved: boolean): JSX.Element {
  const tone = resolved
    ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
    : "bg-yellow-500/20 text-yellow-200 border-yellow-500/40";
  return (
    <span className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${tone}`}>
      {resolved ? "resolvido" : "pendente"}
    </span>
  );
}

export function HistoryTable({ items }: HistoryTableProps): JSX.Element {
  return (
    <section className="rounded-xl border border-neutral-800 bg-neutral-900 p-6">
      <h3 className="mb-4 text-lg font-semibold text-neutral-100">
        Histórico de previsões
      </h3>
      {items.length === 0 ? (
        <p className="text-sm text-neutral-400">
          Ainda não há previsões persistidas para este ticker.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase text-neutral-500">
              <tr>
                <th className="py-2 pr-3">Feito em</th>
                <th className="py-2 pr-3">Horizonte</th>
                <th className="py-2 pr-3">Target</th>
                <th className="py-2 pr-3">Base</th>
                <th className="py-2 pr-3">Previsto</th>
                <th className="py-2 pr-3">Real</th>
                <th className="py-2 pr-3">Erro %</th>
                <th className="py-2 pr-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800 text-neutral-200">
              {items.map((item) => (
                <tr key={item.id}>
                  <td className="py-2 pr-3 font-mono text-xs">
                    {formatDate(item.created_at)}
                  </td>
                  <td className="py-2 pr-3">{horizonLabel(item.horizon_days)}</td>
                  <td className="py-2 pr-3 font-mono text-xs">
                    {formatDate(item.target_date)}
                  </td>
                  <td className="py-2 pr-3 font-mono">
                    {formatMoney(item.base_close)}
                  </td>
                  <td className="py-2 pr-3 font-mono">
                    {formatMoney(item.predicted_close)}
                  </td>
                  <td className="py-2 pr-3 font-mono">
                    {item.actual_close !== null && item.actual_close !== undefined
                      ? formatMoney(item.actual_close)
                      : "-"}
                  </td>
                  <td className="py-2 pr-3 font-mono">
                    {item.error_pct !== null && item.error_pct !== undefined
                      ? formatPct(item.error_pct)
                      : "-"}
                  </td>
                  <td className="py-2 pr-3">{statusBadge(item.resolved)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
