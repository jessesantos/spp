/**
 * Loading UI for the ticker dashboard.
 *
 * Renderizado automaticamente pelo Next.js App Router enquanto
 * `page.tsx` resolve as promises de predicao + historico. Evita a
 * pagina anterior "travada" durante transicoes.
 */
import type { JSX } from "react";
import { AppHeader } from "@/components/AppHeader";
import { Skeleton } from "@/components/Skeleton";

export default function DashboardLoading(): JSX.Element {
  return (
    <>
      <AppHeader />
      <main className="mx-auto max-w-6xl px-6 py-10">
        <div className="mb-6 flex items-center gap-2">
          <Skeleton className="h-3 w-16" />
          <span aria-hidden className="text-neutral-700">
            /
          </span>
          <Skeleton className="h-3 w-14" />
        </div>

        <section className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div className="space-y-2">
            <Skeleton className="h-10 w-40" />
            <Skeleton className="h-4 w-96 max-w-full" />
          </div>
          <div className="flex items-center gap-3">
            <Skeleton className="h-4 w-28" />
            <Skeleton className="h-9 w-40 rounded-lg" />
          </div>
        </section>

        <div className="grid gap-6 lg:grid-cols-5">
          <div className="lg:col-span-3">
            <div className="rounded-xl border border-neutral-800 bg-neutral-900 p-6">
              <div className="flex items-center justify-between">
                <Skeleton className="h-6 w-24" />
                <Skeleton className="h-6 w-20 rounded-full" />
              </div>
              <div className="mt-5 space-y-3">
                <Skeleton className="h-4 w-24" />
                <Skeleton className="h-9 w-44" />
                <Skeleton className="h-5 w-56" />
              </div>
            </div>
          </div>
          <div className="rounded-xl border border-neutral-800 bg-neutral-900 p-6 lg:col-span-2">
            <Skeleton className="h-3 w-32" />
            <Skeleton className="mt-3 h-10 w-24" />
            <Skeleton className="mt-1 h-3 w-24" />
            <div className="mt-4 grid grid-cols-3 gap-2">
              {[0, 1, 2].map((i) => (
                <div key={i} className="rounded-lg bg-neutral-950/60 p-2">
                  <Skeleton className="h-2 w-14" />
                  <Skeleton className="mt-2 h-6 w-8" />
                </div>
              ))}
            </div>
          </div>
        </div>

        <section className="mt-10 space-y-4">
          <div>
            <Skeleton className="h-6 w-48" />
            <Skeleton className="mt-2 h-3 w-72" />
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="rounded-xl border border-neutral-800 bg-neutral-900 p-5"
              >
                <div className="flex items-center justify-between">
                  <Skeleton className="h-4 w-20" />
                  <Skeleton className="h-5 w-16 rounded-full" />
                </div>
                <Skeleton className="mt-3 h-8 w-32" />
                <Skeleton className="mt-2 h-4 w-20" />
                <div className="mt-4 grid grid-cols-2 gap-2">
                  <Skeleton className="h-8" />
                  <Skeleton className="h-8" />
                </div>
              </div>
            ))}
          </div>
        </section>

        <div
          role="status"
          aria-live="polite"
          className="mt-10 flex items-center justify-center gap-2 text-xs text-neutral-400"
        >
          <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-emerald-500/40 border-t-emerald-400" />
          Carregando predicao, sentimento e historico...
        </div>
      </main>
    </>
  );
}
