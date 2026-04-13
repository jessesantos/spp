/**
 * Shared top navigation: logo + breadcrumb.
 * Renders on every page via app/layout.tsx.
 */
import Image from "next/image";
import Link from "next/link";
import type { JSX } from "react";
import { HugeiconsIcon } from "@hugeicons/react";
import { ApiIcon } from "@hugeicons/core-free-icons";
import { LinkTopProgress } from "@/components/NavPending";

export interface AppHeaderProps {
  ticker?: string;
}

export function AppHeader({ ticker }: AppHeaderProps): JSX.Element {
  return (
    <header className="sticky top-0 z-20 border-b border-neutral-800/80 bg-neutral-950/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-3">
        <Link
          href="/"
          title="Stock Price Predictor"
          className="group relative flex items-center gap-3 text-neutral-100"
          aria-label="Stock Price Predictor - voltar para a tela inicial"
        >
          <LinkTopProgress className="-top-3" />
          <Image
            src="/spp.png"
            alt="SPP logo"
            width={120}
            height={58}
            priority
            className="h-10 w-auto select-none"
          />
          <span className="sr-only">Stock Price Predictor</span>
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
          className="hidden items-center gap-1.5 rounded-md border border-neutral-800 px-3 py-1.5 text-xs font-semibold text-neutral-300 transition hover:border-emerald-500/50 hover:text-emerald-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/60 sm:inline-flex"
        >
          <HugeiconsIcon icon={ApiIcon} size={14} strokeWidth={1.75} aria-hidden />
          API docs
        </a>
      </div>
    </header>
  );
}
