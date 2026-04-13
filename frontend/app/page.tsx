import type { JSX } from "react";
import Link from "next/link";
import { AppHeader } from "@/components/AppHeader";
import {
  api,
  type MultiHorizonResponse,
  type TickerInfo,
} from "@/lib/api";

export const dynamic = "force-dynamic";

interface TickerPreview {
  info: TickerInfo;
  prediction: MultiHorizonResponse | null;
}

async function loadTickers(): Promise<TickerInfo[]> {
  try {
    return await api.listTickers();
  } catch {
    return [];
  }
}

async function loadPreviews(tickers: TickerInfo[]): Promise<TickerPreview[]> {
  const results = await Promise.all(
    tickers.map(async (info) => {
      try {
        const prediction = await api.predictWithHorizons(info.ticker);
        return { info, prediction };
      } catch {
        return { info, prediction: null };
      }
    }),
  );
  return results;
}

export default async function Home(): Promise<JSX.Element> {
  const tickers = await loadTickers();
  const previews = await loadPreviews(tickers);

  const { up, down, neutral, averagePct } = summarize(previews);

  return (
    <>
      <AppHeader />
      <main className="relative overflow-hidden">
        <div
          aria-hidden
          className="pointer-events-none absolute -top-40 left-1/2 h-[540px] w-[1000px] -translate-x-1/2 rounded-full bg-gradient-to-br from-emerald-500/25 via-cyan-500/10 to-transparent blur-3xl"
        />
        <div
          aria-hidden
          className="pointer-events-none absolute -bottom-32 right-0 h-[400px] w-[600px] rounded-full bg-gradient-to-tr from-purple-500/15 via-transparent to-transparent blur-3xl"
        />

        <div className="relative mx-auto max-w-6xl px-6 py-14">
          <section className="mb-12 grid gap-10 lg:grid-cols-5 lg:items-center">
            <div className="lg:col-span-3">
              <span className="inline-flex items-center gap-2 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-emerald-300">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
                Análise preditiva em tempo real
              </span>
              <h1 className="mt-4 text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
                Previsões da{" "}
                <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                  B3
                </span>{" "}
                com IA e sentimento global
              </h1>
              <p className="mt-5 max-w-xl text-base text-neutral-400 sm:text-lg">
                Rede LSTM treinada em 1 ano de histórico, cruzada com notícias
                nacionais e feeds internacionais (Reuters, BBC, FT, Bloomberg)
                para capturar política monetária, geopolítica e commodities.
              </p>
              <div className="mt-6 flex flex-wrap gap-3">
                <Link
                  href={previews[0]?.info.ticker ? `/dashboard/${previews[0].info.ticker}` : "#tickers"}
                  className="inline-flex items-center gap-2 rounded-lg bg-emerald-500 px-4 py-2 text-sm font-semibold text-neutral-950 transition hover:bg-emerald-400"
                >
                  Explorar primeiro ticker
                  <span aria-hidden>→</span>
                </Link>
                <a
                  href="http://localhost:8000/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 rounded-lg border border-neutral-700 px-4 py-2 text-sm font-semibold text-neutral-200 transition hover:border-emerald-500/50 hover:text-emerald-300"
                >
                  API docs
                </a>
              </div>
            </div>

            <aside className="lg:col-span-2">
              <div className="rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5 backdrop-blur">
                <h2 className="text-xs font-semibold uppercase tracking-widest text-neutral-400">
                  Panorama do dia
                </h2>
                <dl className="mt-4 grid grid-cols-3 gap-3">
                  <Stat label="Em alta" value={up} tone="emerald" />
                  <Stat label="Em baixa" value={down} tone="red" />
                  <Stat label="Neutras" value={neutral} tone="yellow" />
                </dl>
                <div className="mt-5 border-t border-neutral-800 pt-4">
                  <p className="text-xs text-neutral-500">
                    Variação média prevista (próximo pregão)
                  </p>
                  <p
                    className={`mt-1 font-mono text-2xl font-bold ${
                      averagePct > 0.1
                        ? "text-emerald-400"
                        : averagePct < -0.1
                          ? "text-red-400"
                          : "text-yellow-300"
                    }`}
                  >
                    {averagePct > 0 ? "+" : ""}
                    {averagePct.toFixed(2)}%
                  </p>
                </div>
              </div>
            </aside>
          </section>

          <section id="tickers">
            <div className="mb-5 flex items-end justify-between">
              <div>
                <h2 className="text-xl font-semibold">Tickers monitorados</h2>
                <p className="text-xs text-neutral-500">
                  Clique para abrir o dashboard completo
                </p>
              </div>
              <span className="rounded-full border border-neutral-800 bg-neutral-900 px-3 py-1 text-xs text-neutral-400">
                {previews.length} ativo{previews.length === 1 ? "" : "s"}
              </span>
            </div>

            {previews.length === 0 ? (
              <div className="rounded-xl border border-yellow-500/30 bg-yellow-500/5 p-6 text-sm text-yellow-200">
                <p className="font-semibold">Backend indisponível</p>
                <p className="mt-1 text-yellow-200/80">
                  Verifique se o container <code>spp-backend</code> responde em{" "}
                  <code>/api/tickers</code>.
                </p>
              </div>
            ) : (
              <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {previews.map(({ info, prediction }) => (
                  <li key={info.ticker}>
                    <TickerCard info={info} prediction={prediction} />
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="mt-16 grid gap-4 sm:grid-cols-3">
            <FeatureCard
              icon="⏱"
              title="Multi-horizonte"
              body="Previsões para amanhã, 7 dias e 30 dias, com valor em R$ e variação percentual."
            />
            <FeatureCard
              icon="📊"
              title="Histórico rastreável"
              body="Toda previsão é persistida. O sistema compara com o preço real no vencimento e calcula o erro."
            />
            <FeatureCard
              icon="🌐"
              title="Contexto macro global"
              body="Sentimento cruzado de notícias nacionais e feeds internacionais captura guerras, Fed, Selic e commodities."
            />
          </section>
        </div>
      </main>
    </>
  );
}

function TickerCard({
  info,
  prediction,
}: {
  info: TickerInfo;
  prediction: MultiHorizonResponse | null;
}): JSX.Element {
  const d1 = prediction?.horizons.find((h) => h.horizon === "D1") ?? null;
  const w1 = prediction?.horizons.find((h) => h.horizon === "W1") ?? null;
  const pct = d1?.predicted_pct ?? 0;
  const tone =
    pct > 0.5
      ? {
          ring: "hover:border-emerald-500/60",
          bar: "from-emerald-500 to-cyan-400",
          pct: "text-emerald-400",
        }
      : pct < -0.5
        ? {
            ring: "hover:border-red-500/60",
            bar: "from-red-500 to-orange-400",
            pct: "text-red-400",
          }
        : {
            ring: "hover:border-yellow-500/60",
            bar: "from-yellow-400 to-neutral-500",
            pct: "text-yellow-300",
          };

  return (
    <Link
      href={`/dashboard/${info.ticker}`}
      className={`group relative block overflow-hidden rounded-2xl border border-neutral-800 bg-neutral-900/60 p-5 transition hover:-translate-y-0.5 hover:bg-neutral-900 ${tone.ring}`}
    >
      <div
        aria-hidden
        className={`absolute inset-x-0 top-0 h-1 bg-gradient-to-r ${tone.bar} opacity-0 transition-opacity group-hover:opacity-100`}
      />
      <div className="flex items-start justify-between">
        <div>
          <p className="font-mono text-xl font-bold text-neutral-50">
            {info.ticker}
          </p>
          <p className="mt-0.5 text-xs text-neutral-500">{info.name ?? "-"}</p>
        </div>
        <span className="rounded-full border border-neutral-800 px-2 py-0.5 text-[10px] font-semibold text-neutral-400">
          {info.currency}
        </span>
      </div>

      {prediction ? (
        <>
          <div className="mt-5 flex items-baseline gap-2">
            <span className="font-mono text-2xl font-bold text-neutral-50">
              {formatMoney(prediction.last_price)}
            </span>
            <span className={`font-mono text-sm font-semibold ${tone.pct}`}>
              {pct > 0 ? "+" : ""}
              {pct.toFixed(2)}%
            </span>
          </div>
          <p className="mt-1 text-[10px] uppercase tracking-wider text-neutral-500">
            Amanhã · previsto {d1 ? formatMoney(d1.predicted_close) : "-"}
          </p>

          {w1 && (
            <div className="mt-4 flex items-center justify-between rounded-lg border border-neutral-800 bg-neutral-950/60 px-3 py-2 text-xs">
              <span className="text-neutral-500">+7 dias</span>
              <span className="font-mono font-semibold text-neutral-200">
                {formatMoney(w1.predicted_close)}
              </span>
              <span
                className={`font-mono ${w1.predicted_pct > 0 ? "text-emerald-400" : w1.predicted_pct < 0 ? "text-red-400" : "text-yellow-300"}`}
              >
                {w1.predicted_pct > 0 ? "+" : ""}
                {w1.predicted_pct.toFixed(2)}%
              </span>
            </div>
          )}
        </>
      ) : (
        <p className="mt-6 text-xs text-neutral-500">Sem dados recentes.</p>
      )}

      <p className="mt-5 flex items-center gap-1 text-xs font-semibold text-neutral-500 transition group-hover:text-emerald-300">
        Abrir dashboard
        <span aria-hidden className="transition-transform group-hover:translate-x-0.5">
          →
        </span>
      </p>
    </Link>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "emerald" | "red" | "yellow";
}): JSX.Element {
  const color =
    tone === "emerald"
      ? "text-emerald-400"
      : tone === "red"
        ? "text-red-400"
        : "text-yellow-300";
  return (
    <div className="rounded-lg bg-neutral-950/50 p-3">
      <dt className="text-[10px] uppercase tracking-wider text-neutral-500">
        {label}
      </dt>
      <dd className={`mt-1 font-mono text-2xl font-bold ${color}`}>{value}</dd>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  body,
}: {
  icon: string;
  title: string;
  body: string;
}): JSX.Element {
  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-5 transition hover:border-emerald-500/30">
      <span
        aria-hidden
        className="grid h-9 w-9 place-items-center rounded-lg bg-neutral-950 text-lg"
      >
        {icon}
      </span>
      <h3 className="mt-3 text-sm font-semibold text-neutral-100">{title}</h3>
      <p className="mt-2 text-xs leading-relaxed text-neutral-400">{body}</p>
    </div>
  );
}

function summarize(previews: TickerPreview[]): {
  up: number;
  down: number;
  neutral: number;
  averagePct: number;
} {
  let up = 0;
  let down = 0;
  let neutral = 0;
  let total = 0;
  let count = 0;
  for (const p of previews) {
    const d1 = p.prediction?.horizons.find((h) => h.horizon === "D1");
    if (!d1) continue;
    count += 1;
    total += d1.predicted_pct;
    if (d1.predicted_pct > 0.5) up += 1;
    else if (d1.predicted_pct < -0.5) down += 1;
    else neutral += 1;
  }
  return {
    up,
    down,
    neutral,
    averagePct: count > 0 ? total / count : 0,
  };
}

function formatMoney(value: number): string {
  return new Intl.NumberFormat("pt-BR", {
    style: "currency",
    currency: "BRL",
  }).format(value);
}
