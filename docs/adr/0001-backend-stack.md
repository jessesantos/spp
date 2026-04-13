# ADR 0001 - Backend stack: FastAPI + Celery + Redis + PostgreSQL

- Status: Accepted
- Date: 2026-04-13
- Deciders: @architect, @dev, @devops
- Supersedes: legacy monolithic Python script (`main.py`, `predictor.py`)

## Context

The legacy SPP system ran as a single Python script reading CSVs, calling
Gemini synchronously and writing model files locally. This had several
structural problems:

- No HTTP surface - impossible to consume from a web UI or external system.
- Blocking calls to LLMs and TensorFlow inside the request path.
- No durable storage for predictions, ticker metadata or model runs.
- Configuration scattered across module-level globals.

The PLAN mandates turning SPP into a modular, containerized, full-stack
platform executable via `docker compose up`. We therefore need a web
framework, a task queue, a cache/broker and a relational store.

## Decision

We adopt the following backend stack:

| Concern | Choice | Version |
|---|---|---|
| HTTP framework | **FastAPI** | 0.115.x |
| Data validation | Pydantic v2 | 2.9.x |
| ORM | SQLAlchemy 2.0 (async) + asyncpg | 2.0.36 |
| Migrations | Alembic | 1.13.x |
| Task queue | **Celery** | 5.4.x |
| Broker & cache | **Redis 7** | 7-alpine |
| Relational store | **PostgreSQL 16** | 16-alpine |
| ASGI server | uvicorn[standard] | 0.32.x |

Synchronous inference (`/api/predict/{ticker}`) runs inside FastAPI with a
short Redis cache (5 min). Long-running training (`/api/train/{ticker}`)
is dispatched to the Celery `training` queue backed by Redis.

## Consequences

Positive
- OpenAPI schema free of charge (Swagger at `/docs`); Next.js can generate
  typed clients from it.
- Async I/O model aligned with the external APIs we depend on
  (BrAPI, Anthropic, feedparser via `asyncio.to_thread`).
- Redis serves triple duty (cache, Celery broker, Celery result backend),
  keeping the infra surface small for a self-hostable demo.
- PostgreSQL gives durable history for `ohlcv`, predictions and model runs,
  enabling audit and backtesting.

Negative / trade-offs
- Celery is heavyweight compared to alternatives like Arq or RQ, but it is
  battle-tested and well understood by the team.
- Running Postgres + Redis + worker pushes baseline compose memory above
  ~1 GB; acceptable for a dev demo, to be revisited for prod sizing.

## Alternatives considered

- **Flask + RQ**: simpler, but lacks native async and Pydantic integration.
- **Django + Celery**: brings an ORM/admin but too opinionated for an ML
  service; heavier startup and more templating than we need.
- **Litestar**: modern and fast, but smaller ecosystem for observability
  plugins (Prometheus, OpenTelemetry).
- **Arq instead of Celery**: excellent async fit, but we may want chord /
  chain semantics for training pipelines and Celery is safer there.
