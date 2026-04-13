/**
 * Route-level loading UI for the home page.
 *
 * Next.js App Router renderiza este componente automaticamente enquanto
 * o `page.tsx` ainda esta resolvendo seus fetches assincronos (api.listTickers,
 * predictWithHorizons...). Garante feedback visual imediato em todo clique
 * que leva para "/".
 */
import type { JSX } from "react";
import { AppHeader } from "@/components/AppHeader";
import { SkeletonCard } from "@/components/Skeleton";

export default function HomeLoading(): JSX.Element {
  return (
    <>
      <AppHeader />
      <main className="relative overflow-hidden">
        <div
          aria-hidden
          className="pointer-events-none absolute -top-40 left-1/2 h-[540px] w-[1000px] -translate-x-1/2 rounded-full bg-gradient-to-br from-emerald-500/15 via-cyan-500/5 to-transparent blur-3xl"
        />
        <div className="relative mx-auto max-w-6xl px-6 py-14">
          <section className="mb-12 grid gap-10 lg:grid-cols-5 lg:items-center">
            <div className="space-y-4 lg:col-span-3">
              <div className="h-6 w-56 animate-pulse rounded-full bg-emerald-500/10" />
              <div className="space-y-3">
                <div className="h-10 w-4/5 animate-pulse rounded-md bg-neutral-800/70" />
                <div className="h-10 w-3/5 animate-pulse rounded-md bg-neutral-800/70" />
              </div>
              <div className="space-y-2 pt-2">
                <div className="h-4 w-full animate-pulse rounded bg-neutral-800/60" />
                <div className="h-4 w-11/12 animate-pulse rounded bg-neutral-800/60" />
                <div className="h-4 w-4/6 animate-pulse rounded bg-neutral-800/60" />
              </div>
              <div className="flex gap-3 pt-3">
                <div className="h-9 w-44 animate-pulse rounded-lg bg-emerald-500/30" />
                <div className="h-9 w-24 animate-pulse rounded-lg bg-neutral-800/70" />
              </div>
            </div>
            <aside className="lg:col-span-2">
              <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5">
                <div className="h-3 w-28 animate-pulse rounded bg-neutral-800" />
                <div className="mt-4 grid grid-cols-3 gap-3">
                  {[0, 1, 2].map((i) => (
                    <div
                      key={i}
                      className="rounded-lg bg-neutral-950/50 p-3"
                    >
                      <div className="h-2 w-16 animate-pulse rounded bg-neutral-800" />
                      <div className="mt-2 h-7 w-10 animate-pulse rounded bg-neutral-800/80" />
                    </div>
                  ))}
                </div>
                <div className="mt-5 border-t border-neutral-800 pt-4">
                  <div className="h-3 w-40 animate-pulse rounded bg-neutral-800" />
                  <div className="mt-2 h-7 w-24 animate-pulse rounded bg-neutral-800/80" />
                </div>
              </div>
            </aside>
          </section>

          <section>
            <div className="mb-5 flex items-end justify-between">
              <div>
                <div className="h-5 w-44 animate-pulse rounded bg-neutral-800" />
                <div className="mt-2 h-3 w-56 animate-pulse rounded bg-neutral-800/70" />
              </div>
              <div className="h-6 w-14 animate-pulse rounded-full bg-neutral-800" />
            </div>
            <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {[0, 1, 2, 3, 4, 5].map((i) => (
                <li key={i}>
                  <SkeletonCard />
                </li>
              ))}
            </ul>
          </section>

          <div
            role="status"
            aria-live="polite"
            className="mt-10 flex items-center justify-center gap-2 text-xs text-neutral-400"
          >
            <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-emerald-500/40 border-t-emerald-400" />
            Carregando panorama e predicoes...
          </div>
        </div>
      </main>
    </>
  );
}
