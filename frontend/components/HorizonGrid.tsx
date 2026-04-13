/**
 * HorizonGrid: renders the three horizon cards (D1, W1, M1) in a row.
 */
import type { JSX } from "react";

import { HorizonCard } from "@/components/HorizonCard";
import type { HorizonPrediction } from "@/lib/api";

export interface HorizonGridProps {
  horizons: HorizonPrediction[];
}

export function HorizonGrid({ horizons }: HorizonGridProps): JSX.Element {
  if (horizons.length === 0) {
    return (
      <p className="text-sm text-neutral-400">
        Sem horizontes disponíveis no momento.
      </p>
    );
  }
  return (
    <section className="grid grid-cols-1 gap-4 md:grid-cols-3">
      {horizons.map((h) => (
        <HorizonCard
          key={h.horizon}
          horizon={h.horizon}
          targetDate={h.target_date}
          baseClose={h.base_close}
          predictedClose={h.predicted_close}
          predictedPct={h.predicted_pct}
          direction={h.direction}
          explanation={h.explanation ?? null}
        />
      ))}
    </section>
  );
}
