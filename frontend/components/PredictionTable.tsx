/**
 * PredictionTable - renders the predicted price path as a simple table.
 */
import type { JSX } from "react";
import type { PredictionPoint } from "@/lib/api";

export interface PredictionTableProps {
  predictions: PredictionPoint[];
}

function directionClass(direction: PredictionPoint["direction"]): string {
  if (direction === "ALTA") return "text-emerald-400";
  if (direction === "BAIXA") return "text-red-400";
  return "text-yellow-400";
}

export function PredictionTable({ predictions }: PredictionTableProps): JSX.Element {
  return (
    <div className="overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900">
      <table className="w-full text-sm">
        <thead className="bg-neutral-950/60 text-neutral-400">
          <tr>
            <th className="px-4 py-3 text-left font-medium">Data</th>
            <th className="px-4 py-3 text-right font-medium">Preço previsto</th>
            <th className="px-4 py-3 text-right font-medium">Direção</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-neutral-800">
          {predictions.map((p) => (
            <tr key={p.date} className="text-neutral-200">
              <td className="px-4 py-3">{p.date}</td>
              <td className="px-4 py-3 text-right font-mono">
                R$ {p.predicted_close.toFixed(2)}
              </td>
              <td
                className={`px-4 py-3 text-right font-semibold ${directionClass(p.direction)}`}
              >
                {p.direction}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
