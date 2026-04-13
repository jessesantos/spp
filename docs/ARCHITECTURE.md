# Arquitetura - SPP v3

> Visão geral da arquitetura em camadas. Complementa os ADRs em `adr/`.

---

## Diagrama geral

```
+--------------------------------------------------------------+
|                       Browser (user)                         |
|          http://localhost:3000   ws://localhost:8000         |
+---------------------------+----------------------------------+
                            |
                            v
+--------------------------------------------------------------+
|            Frontend Next.js 15 (App Router, SSR)             |
|    app/ page.tsx -> AppHeader + panorama + TickerCards       |
|    app/dashboard/[ticker]/page.tsx                           |
|    components/ (Prediction, Horizon, History, Refresh)       |
|    lib/api.ts (Zod runtime validation)                       |
|    hooks/useLivePrediction.ts (WebSocket)                    |
+---------------------------+----------------------------------+
                            |  REST (JSON) + WebSocket
                            v
+--------------------------------------------------------------+
|           Backend FastAPI - 4 camadas                        |
|                                                              |
|  api/         controllers + schemas Pydantic + errors        |
|    |          register_exception_handlers, rate_limit        |
|    v                                                         |
|  services/    casos de uso (PredictionService,               |
|    |          SentimentService, ReconciliationService)       |
|    v                                                         |
|  repositories/ SqlAlchemyPredictionsRepo                     |
|  ml/          features, LSTM, ClaudeSentiment, MacroContext  |
|  data/        BrAPIClient, YahooClient, RSSNewsClient        |
|    v                                                         |
|  db/          SQLAlchemy 2.0 models + session factory        |
|  infra/       config, logging, middleware, celery, DI        |
+---------------------------+----------------------------------+
                            |
      +---------------------+---------------------+
      |                     |                     |
      v                     v                     v
+------------+       +------------+        +-------------+
| PostgreSQL |       |   Redis    |        |   Celery    |
|     16     |       |     7      |        |   worker    |
|            |       | cache+broker        | training Q  |
+------------+       +------------+        +-------------+
                            |
                            v
+--------------------------------------------------------------+
|   External APIs:  BrAPI (B3) · yfinance · Anthropic Claude   |
|                   RSS feeds (PT-BR + internacionais macro)   |
+--------------------------------------------------------------+
```

---

## Camadas do backend

Seguindo MVC moderno (detalhe em `AGENTS.md §1`):

### `api/` - Controllers

Routers FastAPI. Validação Pydantic. Sem regra de negócio. Delegam para `services/` via `Depends`.

- `routes.py` - REST endpoints REST (`/api/tickers`, `/api/predict/{ticker}`, `/api/predictions/{ticker}/history`, `/api/predictions/reconcile`, `/api/sentiment/{ticker}`, `/api/train/{ticker}`)
- `websocket.py` - `/ws/live/{ticker}` com tick a cada 60 s
- `schemas.py` - DTOs Pydantic
- `errors.py` - `DomainError`, `NotFoundError`, exception handlers

### `services/` - Casos de uso

Orquestram repositories + ml + data. Recebem dependências via construtor (DIP).

- `PredictionService.predict_with_horizons()` - rollout 30-step do LSTM, pega índices 0/6/29 para D1/W1/M1, persiste em `predictions_repo`, devolve `MultiHorizonResponse`.
- `ReconciliationService.reconcile()` - varre previsões vencidas sem `actual_close`, busca preço real via BrAPI/Yahoo, preenche erro.
- `SentimentService.aggregate()` - fetch RSS -> Claude -> `aggregate_sentiment`.

### `repositories/` - Persistência

Abstraídos via `Protocol`. Implementação SQLAlchemy 2.0 async.

- `PredictionsRepo` / `SqlAlchemyPredictionsRepo` - `upsert` idempotente, `list_history`, `list_unresolved`, `update_actual`.

### `ml/` - Modelos e sinais

- `features.py` - indicadores técnicos puros sobre pandas DataFrame.
- `lstm_model.py` - `LSTMPricePredictor` (build / train / predict / save / load) com rollout recursivo.
- `claude_sentiment.py` - `ClaudeSentimentAnalyzer` com prompt-injection guard.
- `macro_context.py` - feeds internacionais + nacionais macro, filtro por keywords geopolíticas, `MacroContextBuilder`.
- `aggregator.py` - pondera sentimentos por confiança.
- `explanation.py` - `ExplanationGenerator` (Protocol) com `HeuristicExplanationGenerator` determinístico e `ClaudeExplanationGenerator` (prompt-injection guard + fallback). O texto (100..500 palavras) e gerado sincronamente durante `predict_with_horizons` e persistido em `predictions.explanation` para uso no modal do frontend.
- `train.py` - CLI de treinamento (1 ano OHLCV + notícias + macro).

### `data/` - Fontes externas

- `brapi.py` - cotações B3 com retry, backoff, Redis cache opcional.
- `yahoo.py` - fallback via yfinance (sync wrapped com `asyncio.to_thread`).
- `news.py` - RSS PT-BR de notícias do ticker.
- `prices_repository.py` - `BrAPIYahooRepository` escolhendo fonte primária.

### `infra/` - Plataforma

- `config.py` - `Settings` Pydantic (12-Factor via env vars).
- `logging.py` - structlog JSON.
- `middleware.py` - `RequestIDMiddleware`, `SecureHeadersMiddleware`.
- `rate_limit.py` - slowapi limiter.
- `celery_app.py` + `tasks.py` - fila `training`.
- `dependencies.py` - composition root (factories `prediction_service`, `sentiment_service`, `prediction_repo`, `reconciliation_service`).

### `db/` - SQLAlchemy

- `session.py` - engine + async session factory.
- `models.py` - `Ticker`, `OHLCV`, `NewsArticle`, `SentimentCache`, `PredictionRecord` (com colunas de multi-horizonte e reconciliação), `ModelRun`.
- Alembic migrations: `0001_initial`, `0002_multi_horizon`. `upgrade head` rodado automaticamente no `lifespan` do FastAPI.

---

## Camadas do frontend

- `app/` - rotas (Next.js App Router)
  - `/` página home com Hero + panorama + TickerCards
  - `/dashboard/[ticker]` com PredictionCard + Sentimento + HorizonGrid + PredictionTable + HistoryTable
- `components/` - UI pura (AppHeader, PredictionCard, HorizonCard, HorizonGrid, HistoryTable, PredictionTable, RefreshPredictionsButton)
- `hooks/` - `useLivePrediction` WebSocket
- `lib/` - `api.ts` (TanStack-free client + Zod), `format.ts` (pt-BR formatters)
- `next.config.ts` com `output: "standalone"` para imagem Docker slim.

---

## Persistência (Postgres)

Tabelas principais:

- `tickers` - símbolo, nome, moeda
- `ohlcv` - histórico de preços (uso futuro pelo backfill)
- `news_articles` - notícias coletadas
- `sentiment_cache` - cache de pontuação por `cache_key`
- `predictions` - **central**: `(ticker_symbol, horizon_days, target_date)` unique; colunas `base_close`, `predicted_close`, `predicted_pct`, `actual_close`, `error_pct`, `resolved_at`
- `model_runs` - metadados de treinos

---

## Runtime e fluxos

### Startup backend (`lifespan`)

1. `configure_logging`
2. `_run_alembic_upgrade` programático (best-effort)
3. `asyncio.create_task(_reconcile_loop)` executa `ReconciliationService.reconcile()` a cada 3600 s

### `GET /api/predict/{ticker}`

1. Validação de ticker (`_validate_ticker` - alnum, 1..10 chars)
2. `PredictionService.predict_with_horizons(ticker)`:
   - `_compute_rollout` - carrega histórico via `PriceRepository`, carrega `.keras` se existir, senão fallback stub determinístico
   - `_build_horizons` - pega índices 0/6/29 do rollout de 30 passos
   - `_safe_sentiment` - chama `SentimentService` sem propagar erros
   - `_persist_horizons` - upsert 3 linhas em `predictions`
3. Retorna `MultiHorizonResponse` (horizontes + 5 pontos de preview + sentimento)

### Reconciliação

- Trigger manual: `POST /api/predictions/reconcile`
- Trigger automático: loop a cada 1 h
- `ReconciliationService` agrupa previsões não-resolvidas por ticker, busca preço real via BrAPI/Yahoo, preenche `actual_close` + `error_pct` + `resolved_at`.

---

## Segurança (OWASP Top 10)

Resumo - detalhe em `AGENTS.md §2`:

- **A01** ticker validado server-side; CORS restrito a `FRONTEND_URL`.
- **A03** SQLAlchemy parametrizado; input Pydantic/Zod; prompt-injection guard em `ClaudeSentimentAnalyzer` (tags `<article>`, redação explícita).
- **A05** Secure headers middleware; `DEBUG=False` em prod; sem banners.
- **A06** requirements.txt e package.json pinados; `trivy image` + `pip-audit` + `npm audit` no CI.
- **A09** structlog JSON com request-id; 5xx loga exception; PII redigida.
- **A10** httpx com timeout + limite de redirect; BrAPI / Yahoo / RSS são allowlist implícita.

---

## Observabilidade

- Logs JSON (structlog) em stdout.
- `/metrics` via `prometheus-fastapi-instrumentator` (histogramas de latência, counters por endpoint).
- Reconcile loop loga `spp.reconcile checked=X resolved=Y` a cada hora.
- Docker healthchecks em todos os serviços principais.

Grafana + Prometheus stack opcional em `docker-compose.observability.yml` (roadmap).

---

## Pipeline de treinamento inteligente

Ver [ADR 0007](adr/0007-intelligent-training.md) para justificativa das escolhas.

```
+--------------------+   crontab (America/Sao_Paulo)
| beat (celery beat) |---> 22:00 todo dia     -> spp.daily_ohlcv_sync
|                    |---> 23:00 domingo      -> spp.weekly_retrain
+--------------------+
          |
          v
+--------------------+    queue: training
| worker (celery)    |
+--------------------+
          |
          +--> PriceRepository      (BrAPI -> Yahoo)
          +--> SqlAlchemyOhlcvRepo  (upsert por ticker+data)
          +--> TrainingOrchestrator
                +--> ClaudeSentimentAnalyzer (opcional)
                +--> MacroContextBuilder     (opcional)
                +--> LSTMPricePredictor.train
                +--> artifact .keras em MODELS_DIR
                +--> SqlAlchemyModelRunsRepo.create
                      status: running -> done | failed
                      loss, direction_accuracy, artifact_path
```

Componentes:

- `backend/app/repositories/ohlcv_repository.py` - `OhlcvRepo` protocol + `SqlAlchemyOhlcvRepo` com `ensure_ticker`, `upsert_many`, `count`, `latest_trade_date`.
- `backend/app/repositories/model_runs_repository.py` - `ModelRunsRepo` protocol + `SqlAlchemyModelRunsRepo` com `create`, `update_status`, `list_recent`.
- `backend/app/ml/training_orchestrator.py` - `TrainingOrchestrator` com DI de `prices_repo`, `ohlcv_repo`, `news_source`, `macro_builder`, `model_runs_repo`. CLI `app/ml/train.py` é thin wrapper.
- `backend/app/infra/celery_beat.py` - schedule via `celery.schedules.crontab`.
- `backend/app/infra/tasks.py` - `spp.train_model`, `spp.daily_ohlcv_sync`, `spp.weekly_retrain`.
- Migração `0003_ohlcv_unique_date.py` adiciona `UNIQUE (ticker_id, trade_date)` no `ohlcv` e índice em `model_runs.ticker_symbol`.
- Endpoint read-only `GET /api/models/{ticker}/runs` expõe o histórico.

Auto-populate: em `PredictionService._compute_rollout`, antes de chamar BrAPI live, consulta o repo OHLCV local. Com menos de `MIN_HISTORY_DAYS=120` linhas, pulls 1 ano via BrAPI e persiste. A próxima predição já usa o banco.

---

## Referências

- `AGENTS.md` - padrões de engenharia obrigatórios
- `PLAN.md` - fases e aceites
- `adr/0001..0007` - decisões arquiteturais
- `security.md` - threat model e checklist OWASP aplicado
