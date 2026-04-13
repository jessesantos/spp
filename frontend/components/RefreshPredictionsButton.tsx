"use client";

import { useState } from "react";
import type { JSX } from "react";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  ArrowReloadHorizontalIcon,
  RefreshIcon,
} from "@hugeicons/core-free-icons";

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
      aria-busy={busy}
      className="inline-flex items-center gap-2 rounded-lg border border-neutral-700 bg-neutral-800 px-3 py-1.5 text-sm font-semibold text-neutral-100 transition hover:bg-neutral-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/60 focus-visible:ring-offset-2 focus-visible:ring-offset-neutral-950 disabled:cursor-wait disabled:opacity-60"
    >
      <HugeiconsIcon
        icon={busy ? ArrowReloadHorizontalIcon : RefreshIcon}
        size={16}
        strokeWidth={1.75}
        className={busy ? "animate-spin" : undefined}
        aria-hidden
      />
      {busy ? "Atualizando..." : "Atualizar previsões"}
    </button>
  );
}
