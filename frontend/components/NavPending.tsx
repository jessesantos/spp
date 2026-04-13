"use client";

/**
 * Feedback visual durante transicao de rota via Next.js App Router.
 *
 * Usa ``useLinkStatus`` (Next 15.3+): so renderiza algo quando o
 * ``<Link>`` ancestor esta com navegacao pendente. Exige ser descendent
 * direto/indireto de um ``next/link``.
 */
import type { JSX } from "react";
import { useLinkStatus } from "next/link";

/**
 * Barra de progresso sutil no topo do card-link.
 * Aparece apenas enquanto a rota esta pendente.
 */
export function LinkTopProgress({
  className = "",
}: {
  className?: string;
}): JSX.Element | null {
  const { pending } = useLinkStatus();
  if (!pending) return null;
  return (
    <span
      role="status"
      aria-live="polite"
      aria-label="Navegando..."
      className={`pointer-events-none absolute inset-x-0 top-0 z-10 h-0.5 overflow-hidden bg-emerald-500/20 ${className}`}
    >
      <span className="block h-full w-1/3 animate-[shimmer_1.1s_ease-in-out_infinite] bg-gradient-to-r from-transparent via-emerald-400 to-transparent" />
    </span>
  );
}

/**
 * Spinner inline circular para botoes CTA.
 */
export function LinkSpinner({
  size = 14,
  className = "",
}: {
  size?: number;
  className?: string;
}): JSX.Element | null {
  const { pending } = useLinkStatus();
  if (!pending) return null;
  return (
    <span
      aria-hidden
      className={`inline-block animate-spin rounded-full border-2 border-current/30 border-t-current ${className}`}
      style={{ width: size, height: size }}
    />
  );
}
