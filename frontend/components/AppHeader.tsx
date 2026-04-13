/**
 * Shared top navigation: logo + breadcrumb.
 * Renders on every page via app/layout.tsx.
 */
import Link from "next/link";
import type { JSX } from "react";

export interface AppHeaderProps {
  ticker?: string;
}

export function AppHeader({ ticker }: AppHeaderProps): JSX.Element {
  return (
    <header className="sticky top-0 z-20 border-b border-neutral-800/80 bg-neutral-950/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-3">
        <Link
          href="/"
          className="group flex items-center gap-2 text-neutral-100"
          aria-label="Voltar para a tela inicial"
        >
          <span
            aria-hidden
            className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-emerald-500 to-cyan-500 text-sm font-black text-neutral-950 shadow-md"
          >
            S
          </span>
          <span className="flex flex-col leading-tight">
            <span className="text-sm font-semibold tracking-tight group-hover:text-emerald-300">
              SPP
            </span>
            <span className="text-[10px] uppercase tracking-wider text-neutral-500">
              Stock Price Predictor
            </span>
          </span>
        </Link>

        {ticker ? (
          <nav aria-label="Ticker atual" className="flex items-center text-sm">
            <span className="rounded-md bg-neutral-900 px-2 py-1 font-mono text-xs font-semibold text-emerald-300">
              {ticker}
            </span>
          </nav>
        ) : (
          <span aria-hidden />
        )}

        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="hidden rounded-md border border-neutral-800 px-3 py-1.5 text-xs font-semibold text-neutral-300 transition hover:border-emerald-500/50 hover:text-emerald-300 sm:inline-block"
        >
          API docs
        </a>
      </div>
    </header>
  );
}
