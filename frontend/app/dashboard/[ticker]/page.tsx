import type { JSX } from "react";
import Link from "next/link";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  Alert01Icon,
  ArrowLeft01Icon,
  CheckmarkCircle01Icon,
  Clock01Icon,
} from "@hugeicons/core-free-icons";
import { AppHeader } from "@/components/AppHeader";
import { LinkSpinner } from "@/components/NavPending";
import {
  api,
  type HistoryResponse,
  type MultiHorizonResponse,
} from "@/lib/api";

// ISR curto: LSTM rollout e chamadas BrAPI sao pesadas; manter pagina
// cacheada por 60s reduz drasticamente a latencia de clique em clique.
export const revalidate = 60;
import { HistoryTable } from "@/components/HistoryTable";
import { HorizonGrid } from "@/components/HorizonGrid";
import { PredictionCard } from "@/components/PredictionCard";
import { PredictionTable } from "@/components/PredictionTable";
import { RefreshPredictionsButton } from "@/components/RefreshPredictionsButton";

async function loadPrediction(
  ticker: string,
): Promise<MultiHorizonResponse | null> {
  try {
    return await api.predictWithHorizons(ticker);
  } catch {
    return null;
  }
}

async function loadHistory(ticker: string): Promise<HistoryResponse | null> {
  try {
    return await api.history(ticker, 60);
  } catch {
    return null;
  }
}

export default async function Dashboard({
  params,
}: {
  params: Promise<{ ticker: string }>;
}): Promise<JSX.Element> {
  const { ticker } = await params;
  const [data, history] = await Promise.all([
    loadPrediction(ticker),
    loadHistory(ticker),
  ]);

  if (!data) {
    return (
      <>
        <AppHeader ticker={ticker} />
        <main className="mx-auto max-w-3xl px-6 py-16">
          <div className="flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/5 p-6">
            <HugeiconsIcon
              icon={Alert01Icon}
              size={24}
              strokeWidth={1.75}
              className="mt-0.5 shrink-0 text-red-300"
              aria-hidden
            />
            <div>
              <h2 className="text-lg font-semibold text-red-300">
                Falha ao carregar predição para {ticker}
              </h2>
              <p className="mt-2 text-sm text-red-200/80">
                Verifique se o backend está saudável em{" "}
                <code>GET /api/predict/{ticker}</code> e tente novamente.
              </p>
              <Link
                href="/"
                className="mt-4 inline-flex items-center gap-2 rounded-md border border-neutral-700 px-3 py-1.5 text-sm text-neutral-100 transition hover:bg-neutral-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/60"
              >
                <HugeiconsIcon
                  icon={ArrowLeft01Icon}
                  size={14}
                  strokeWidth={2}
                  aria-hidden
                />
                Voltar para tickers
              </Link>
            </div>
          </div>
        </main>
      </>
    );
  }

  // Fonte canonica: o horizonte D1 persistido no DB. Home e dashboard
  // leem do mesmo registro, garantindo consistencia visual entre telas
  // (antes: dashboard usava predictions[0] que podia divergir por timing
  // de cache Next). Fallback seguro para predictions[0] em payload legado.
  const d1 =
    data.horizons.find((h) => h.horizon === "D1") ??
    null;
  const next = d1
    ? {
        predicted_close: d1.predicted_close,
        direction: d1.direction,
      }
    : data.predictions[0] ?? null;
  const trend = next?.direction ?? "NEUTRO";

  const resolved = history?.items.filter((i) => i.resolved).length ?? 0;
  const pending = (history?.items.length ?? 0) - resolved;

  return (
    <>
      <AppHeader ticker={data.ticker} />
      <main className="mx-auto max-w-6xl px-6 py-10">
        <nav
          aria-label="Breadcrumb"
          className="mb-6 flex items-center gap-2 text-xs text-neutral-400"
        >
          <Link
            href="/"
            className="inline-flex items-center gap-1 rounded transition hover:text-emerald-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/60"
          >
            <HugeiconsIcon
              icon={ArrowLeft01Icon}
              size={12}
              strokeWidth={2}
              aria-hidden
            />
            Tickers
            <LinkSpinner size={10} className="text-emerald-300" />
          </Link>
          <span aria-hidden>/</span>
          <span className="font-mono font-semibold text-neutral-200">
            {data.ticker}
          </span>
        </nav>

        <section className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="font-mono text-4xl font-bold tracking-tight">
              {data.ticker}
            </h1>
            <p className="mt-1 text-sm text-neutral-300">
              LSTM 3 anos · sentimento Claude (SKILL Austríaco) · macro · FX · mercados de previsão
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-3 text-xs text-neutral-400">
              <span className="inline-flex items-center gap-1">
                <HugeiconsIcon
                  icon={CheckmarkCircle01Icon}
                  size={12}
                  strokeWidth={2}
                  className="text-emerald-400"
                  aria-hidden
                />
                <span className="text-emerald-400">{resolved}</span> resolvidas
              </span>
              <span className="inline-flex items-center gap-1">
                <HugeiconsIcon
                  icon={Clock01Icon}
                  size={12}
                  strokeWidth={2}
                  className="text-yellow-300"
                  aria-hidden
                />
                <span className="text-yellow-300">{pending}</span> pendentes
              </span>
            </div>
            <RefreshPredictionsButton ticker={data.ticker} />
          </div>
        </section>

        <div className="grid gap-6 lg:grid-cols-5">
          <div className="lg:col-span-3">
            <PredictionCard
              ticker={data.ticker}
              currentPrice={data.last_price}
              predictedPrice={next?.predicted_close ?? data.last_price}
              directionAccuracy={data.direction_accuracy ?? null}
              trend={trend}
            />
          </div>
          {data.sentiment && (
            <section className="rounded-xl border border-neutral-800 bg-neutral-900 p-6 lg:col-span-2">
              <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-neutral-400">
                Sentimento agregado
              </h3>
              <p className="text-3xl font-bold text-neutral-50">
                {data.sentiment.score.toFixed(2)}
              </p>
              <p className="text-xs text-neutral-400">
                confiança {(data.sentiment.confidence * 100).toFixed(0)}%
              </p>
              <dl className="mt-4 grid grid-cols-3 gap-2 text-xs">
                <SentimentStat
                  label="Positivas"
                  value={data.sentiment.positives}
                  tone="emerald"
                />
                <SentimentStat
                  label="Neutras"
                  value={data.sentiment.neutrals}
                  tone="yellow"
                />
                <SentimentStat
                  label="Negativas"
                  value={data.sentiment.negatives}
                  tone="red"
                />
              </dl>
            </section>
          )}
        </div>

        <section className="mt-10 space-y-4">
          <div>
            <h2 className="text-xl font-semibold">Horizontes de previsão</h2>
            <p className="mt-1 text-sm text-neutral-400">
              Valores preditos com variação sobre o preço base atual
            </p>
          </div>
          <HorizonGrid horizons={data.horizons} />
        </section>

        <section className="mt-10 space-y-4">
          <div>
            <h2 className="text-xl font-semibold">Próximos 5 dias</h2>
            <p className="mt-1 text-sm text-neutral-400">
              Projeção sequencial derivada do mesmo rollout
            </p>
          </div>
          <PredictionTable predictions={data.predictions} />
        </section>

        <section className="mt-10">
          <HistoryTable items={history?.items ?? []} />
        </section>
      </main>
    </>
  );
}

function SentimentStat({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "emerald" | "yellow" | "red";
}): JSX.Element {
  const color =
    tone === "emerald"
      ? "text-emerald-400"
      : tone === "red"
        ? "text-red-400"
        : "text-yellow-300";
  return (
    <div className="rounded-lg bg-neutral-950/60 p-2">
      <dt className="text-[10px] uppercase tracking-wider text-neutral-400">
        {label}
      </dt>
      <dd className={`mt-0.5 font-mono text-lg font-bold ${color}`}>{value}</dd>
    </div>
  );
}
