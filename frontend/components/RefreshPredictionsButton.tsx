"use client";

import { useState } from "react";
import type { JSX } from "react";

export interface RefreshPredictionsButtonProps {
  ticker: string;
}

export function RefreshPredictionsButton({
  ticker,
}: RefreshPredictionsButtonProps): JSX.Element {
  const [busy, setBusy] = useState(false);

  async function onClick(): Promise<void> {
    setBusy(true);
    try {
      const apiBase =
        process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
      await fetch(
        `${apiBase}/api/predict/${encodeURIComponent(ticker)}`,
        { cache: "no-store" },
      );
    } catch {
      // Silent fail: the reload still shows current state.
    } finally {
      window.location.reload();
    }
  }

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy}
      className="rounded-lg border border-neutral-700 bg-neutral-800 px-3 py-1.5 text-sm font-semibold text-neutral-100 transition hover:bg-neutral-700 disabled:opacity-50"
    >
      {busy ? "Atualizando..." : "Atualizar previsões"}
    </button>
  );
}
