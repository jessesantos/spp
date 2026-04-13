# ADR 0003 - Frontend stack: Next.js 15 + Tailwind (shadcn/ui deferred)

- Status: Accepted
- Date: 2026-04-13
- Deciders: @architect, @dev, @ux-design-expert

## Context

The legacy SPP had **no frontend** - results were printed to the
terminal and saved to disk. To ship the v3 product we need a dashboard
that lists monitored tickers, shows LSTM predictions, sentiment summary,
and live WebSocket updates.

Constraints:

- SSR so the dashboard is useful even without client JS enabled.
- TypeScript end-to-end with Zod validation mirroring the Pydantic
  schemas exposed by FastAPI.
- Must build and run in a Docker container with a slim image.
- Small, composable component surface - the UX scope for v3 is limited
  to a handful of cards, a table and a chart.

## Decision

- **Framework**: Next.js 15 with the App Router and Turbopack.
- **Language**: TypeScript (strict).
- **Styling**: Tailwind CSS v4.
- **Validation**: Zod schemas colocated with a typed `fetch` wrapper in
  `frontend/lib/api.ts`.
- **Charts (future)**: `lightweight-charts` added in a later phase; v3
  MVP uses plain Tailwind-styled tables/cards to keep the bundle small.
- **shadcn/ui**: **deferred**. We initially planned to integrate it
  (see `SPP_RECOMENDACAO_MELHORIA.md` §1) but for the first shippable
  slice we use plain Tailwind components with the same visual contract.
  This avoids the `npx shadcn init` step inside CI and keeps the image
  deterministic. shadcn/ui can be adopted incrementally without breaking
  the component API.

## Consequences

Positive
- Next.js `output: "standalone"` produces a ~150 MB Docker image.
- Tailwind + App Router give us RSC data fetching directly against
  FastAPI, so the initial dashboard render is HTML.
- Zod schemas act as a runtime contract test between frontend and
  backend - type drift surfaces at parse time, not runtime crash.

Negative / trade-offs
- We hand-roll the card/badge/table visuals instead of using shadcn.
  Cost is ~50 LOC of Tailwind and no design system primitives yet.
- No TanStack Query / Zustand in the MVP; we fetch in RSC and use a
  minimal custom hook for WebSocket. Acceptable while screen count is
  low; revisit once we have ≥5 pages sharing data.

## Alternatives considered

- **Remix**: great DX but smaller ecosystem for SSR + WebSocket demos.
- **SvelteKit**: smaller bundles, but team familiarity is lower.
- **Plain React + Vite**: loses SSR, making SEO and first-paint worse.
