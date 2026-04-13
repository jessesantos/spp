# ADR 0004 - Data sources: BrAPI + Yahoo Finance + RSS replace CSVs

- Status: Accepted
- Date: 2026-04-13
- Deciders: @architect, @data-engineer, @dev

## Context

The legacy pipeline read OHLCV and news from CSV files committed to the
repository. This was acceptable for offline experiments but not for a
production-grade service:

- Data became stale the moment the repo was cloned.
- Storing news CSVs risked copyright issues.
- No separation between *source* (a quote) and *derived feature* (an RSI).

The new system must pull live data for B3-listed tickers with a free
tier acceptable for a self-hosted demo.

## Decision

Three adapters live under `backend/app/data/`:

1. **BrAPI (`brapi.py`)** - primary source for B3 quotes. Async
   `httpx.AsyncClient`, 10 s timeout, 2 retries with exponential backoff,
   responses cached in Redis for 5 minutes keyed by
   `brapi:quote:{ticker}:{range}`.
2. **Yahoo Finance (`yahoo.py`)** - fallback via `yfinance`. Sync library
   is wrapped in `asyncio.to_thread` to keep the event loop unblocked.
3. **RSS news (`news.py`)** - `feedparser` polling InfoMoney, MoneyTimes
   and Valor Econômico. Filtering by ticker substring (case-insensitive)
   on title and summary.

The `PriceRepository` protocol in `prices_repository.py` wraps the two
price adapters with a *BrAPI-first, Yahoo-fallback* strategy. Consumers
depend on the protocol, not the concrete clients (DIP).

## Consequences

Positive
- Fresh data on every call; no CSVs tracked in git.
- Clear separation of concerns - data adapters are dumb fetchers; feature
  engineering lives in `app/ml/features.py`.
- Fallback keeps the service up when BrAPI has an outage.
- Redis cache keeps BrAPI well within its generous-but-undocumented rate
  limit.

Negative / trade-offs
- BrAPI is a third-party service with no SLA; we rely on Yahoo as a
  safety net.
- RSS feeds change unannounced; we log and skip unparseable entries
  instead of crashing.

## Alternatives considered

- **Alpha Vantage**: 5 req/min free tier is too tight for multi-ticker
  dashboards.
- **Paid feed (e.g. StatusInvest, CEDRO)**: out of budget for v3.
- **Scraping B3 directly**: fragile and likely against ToS.
