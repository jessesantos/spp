# SPP v3 - Plano de Modernização

> Este documento é o **plano mestre** que guia todos os agentes (aiox-core + Claude Code) na modernização do SPP (Sistema de Predição de Preços de Ações). É **canônico** - em caso de conflito com outras fontes, este arquivo prevalece, exceto onde `SPP_RECOMENDACAO_MELHORIA.md` especificar detalhes técnicos mais precisos (stack, APIs, trechos de código).

---

## 1. Objetivo

Transformar o SPP de um script Python monolítico baseado em CSVs para uma plataforma modular full-stack rodando em **containers Docker**, com:

- **Backend**: FastAPI + Python 3.12 + TensorFlow (LSTM) + Claude SDK (sentimento) + PostgreSQL + Redis + Celery.
- **Frontend**: Next.js 15 + TypeScript + shadcn/ui + TradingView Lightweight Charts.
- **Dados em tempo real**: BrAPI / Yahoo Finance / RSS (sem CSVs manuais).
- **Orquestração agêntica**: `aiox-core` (SynkraAI) como core principal - agentes `@architect`, `@dev`, `@qa`, `@devops`, `@data-engineer`, `@analyst`, `@pm`, `@sm`.

O resultado final deve ser **funcional e executável** com `docker compose up`.

---

## 2. Fonte da verdade técnica

A stack detalhada, endpoints, trechos de código, APIs externas, docker-compose e estrutura de pastas estão em **`SPP_RECOMENDACAO_MELHORIA.md`**. Agentes devem seguir aquele documento para decisões concretas de implementação e este `PLAN.md` para sequenciamento, metodologia e critérios de aceitação.

---

## 3. Fases de execução

As fases devem ser executadas pelos agentes do aiox-core na ordem abaixo. Cada fase tem um **agente dono** e **critérios de aceitação** objetivos.

### Fase 0 - Preparação (já feito por Claude Code antes do handoff)
- [x] `PLAN.md` criado (este arquivo)
- [x] `AGENTS.md` com diretrizes (MVC moderno, SOLID, Clean Code, OWASP, TDD)
- [x] `aiox-core` instalado via `npx aiox-core install`
- [x] Esqueleto de diretórios `backend/`, `frontend/`, `models/`, `.env.example`

### Fase 1 - Arquitetura (`@architect`) ✅ done
Produzir ADRs curtos em `docs/adr/` justificando:
- Escolha FastAPI + Celery + Redis + PostgreSQL
- Substituição Gemini -> Claude SDK
- Next.js 15 App Router + shadcn/ui
- Cloudflare Tunnel como substituto do ngrok
- Fontes de dados (BrAPI + yfinance + RSS)
- Contexto macro global (ADR 0006)

**Aceite**: `docs/adr/0001..0006` criados.

### Fase 2 - Data Engineering (`@data-engineer`) ✅ done
- Modelar schema PostgreSQL: `tickers`, `ohlcv`, `news`, `sentiment_cache`, `predictions`, `model_runs`.
- Migrations via Alembic.
- Scripts de ingest (`backend/app/data/brapi.py`, `yahoo.py`, `news.py`) com retry + backoff exponencial.
- Cache de sentimento em Redis (TTL 24h) substituindo `sentiment_cache.json`.

**Aceite**: `alembic upgrade head` aplica schema; `pytest backend/tests/test_data_*.py` passa.

### Fase 3 - ML Core migration (`@dev`) ✅ done
Migrar arquivos atuais (`model.py`, `feature_engine.py`, `sentiment_analyzer.py`, `predictor.py`, `gpu_manager.py`) para `backend/app/ml/`:
- Trocar `requests` + Gemini por `anthropic` SDK (Claude Sonnet 4.5).
- Refatorar seguindo SOLID - cada módulo com responsabilidade única; injetar dependências (cliente HTTP, cliente Claude, repositórios) via construtor.
- Manter compatibilidade com modelos `.keras` já treinados em `models/`.
- Features técnicas idênticas ao legado (MAs, RSI, MACD, Bollinger, volume, sentimento).

**Aceite**: teste de regressão compara predição nova × legado com tolerância < 1e-4.

### Fase 4 - Backend API (`@dev`) ✅ done
Implementar endpoints conforme `SPP_RECOMENDACAO_MELHORIA.md` §2:
- `GET /api/predict/{ticker}` (sync com cache Redis)
- `POST /api/train/{ticker}` (assíncrono via Celery)
- `GET /api/history/{ticker}`
- `GET /api/sentiment/{ticker}`
- `WebSocket /ws/live/{ticker}` (tick a cada 60 s)
- Healthcheck `GET /health` (liveness + readiness)
- Middleware: CORS restrito por env, rate-limit (slowapi), request-id, structured logging (JSON).

**Aceite**: Swagger em `/docs` com todos endpoints; `pytest` cobertura ≥ 70 %.

### Fase 5 - Frontend (`@dev`) ✅ done v1 (lightweight-charts diferido)
Conforme `SPP_RECOMENDACAO_MELHORIA.md` §1:
- `npx create-next-app@latest frontend --typescript --tailwind --app`
- shadcn/ui + lightweight-charts + TanStack Query + Zustand + Zod.
- Rotas: `/` (lista tickers), `/dashboard/[ticker]`.
- Componentes: `PredictionCard`, `CandlestickChart`, `LSTMChart`, `SentimentGauge`, `TechnicalTable`, `AlertBanner`.
- Hook `useLivePrediction` via WebSocket.
- Schemas Zod para todas respostas de API.

**Aceite**: `npm run build` sem erros; página dashboard renderiza dados mockados.

### Fase 6 - Docker & CI/CD (`@devops`) 🟢 quase (GitLab CI e tunnel de deploy pendentes)
- `backend/Dockerfile` (multi-stage, `python:3.12-slim`, user não-root).
- `frontend/Dockerfile` (multi-stage, `node:22-alpine`, standalone output).
- `docker-compose.yml` completo (frontend, backend, worker Celery, postgres:16-alpine, redis:7-alpine).
- `.env.example` com todas variáveis.
- `.gitlab-ci.yml` (test → build → deploy) + opcional `.github/workflows/ci.yml`.
- Health checks nos serviços.

**Aceite**: `docker compose up -d` sobe todos serviços; `curl localhost:8000/health` → 200; `curl localhost:3000` → 200.

### Fase 7 - QA & Segurança (`@qa`) 🟢 quase (E2E Playwright pendente)
- Testes unitários (pytest + vitest/jest).
- Testes de integração (httpx AsyncClient + testcontainers-python).
- Testes E2E Playwright para golden path do dashboard.
- Varredura OWASP Top 10 (ver `AGENTS.md` §OWASP).
- `bandit`, `ruff`, `mypy --strict` no backend; `eslint`, `tsc --noEmit` no frontend.
- `trivy image` nas imagens Docker.

**Aceite**: CI verde; cobertura ≥ 70 % backend / ≥ 60 % frontend; zero vulnerabilidades críticas Trivy.

### Fase 8 - Observabilidade (`@devops`) 🟢 quase (Grafana diferido)
- Structured logging JSON (structlog) no backend.
- `/metrics` Prometheus (prometheus-fastapi-instrumentator).
- Opcional: `docker-compose.observability.yml` com Prometheus + Grafana.

**Aceite**: `curl localhost:8000/metrics` retorna métricas.

### Fase 9 - Previsões multi-horizonte + reconciliação (`@dev`) ✅ done
- `GET /api/predict/{ticker}` retorna horizontes D1 (+1), W1 (+7), M1 (+30) com `base_close`, `predicted_close`, `predicted_pct`, `direction`.
- Persistência idempotente via unique `(ticker, horizon_days, target_date)` e upsert explícito.
- `GET /api/predictions/{ticker}/history` lista histórico com `actual_close`, `error_pct`, `resolved`.
- `POST /api/predictions/reconcile` + loop asyncio 1 h preenchem preços reais.
- Alembic 0002 cria as colunas novas; `lifespan` roda `upgrade head` programático.
- Testes: `test_horizons.py`, `test_prediction_repo.py` (SQLite in-memory).

**Aceite**: 3 linhas em `predictions` após um `GET /api/predict/PETR4`; `/dashboard/PETR4` renderiza 3 HorizonCards + HistoryTable.

### Fase 10 - Contexto macro global (`@dev`) ✅ done
- `backend/app/ml/macro_context.py` com `RSSMacroNewsSource` cobrindo Reuters, BBC World/Business, FT, Bloomberg, AP, Valor, InfoMoney, G1.
- Filtro por `MACRO_KEYWORDS` (war, sanction, Fed, Selic, OPEC, tarifa, etc.).
- `ClaudeMacroAnalyzer` com prompt-injection guard + fallback heurístico.
- Coluna `macro_score` injetada no feature engineering.
- CLI de treino passa macro score automaticamente (`--no-macro` pula).

**Aceite**: `python -m app.ml.train --ticker PETR4 --period 1y` roda e salva modelo em `models/`; ADR 0006 descreve decisão.

### Fase 12 - Treinamento inteligente auto-populado (`@dev`) ✅ done
- Backfill automático de OHLCV: `PredictionService._compute_rollout` popula o DB na primeira consulta (`MIN_HISTORY_DAYS=120`).
- `SqlAlchemyOhlcvRepo` + `SqlAlchemyModelRunsRepo` com upsert idempotente e listagem recente.
- `TrainingOrchestrator` centraliza ingest + sentimento + macro + fit + persistência de `ModelRun`; CLI é thin wrapper.
- Celery beat: `daily_ohlcv_sync` (22:00 BRT) e `weekly_retrain` (dom 23:00 BRT); schedule em `infra/celery_beat.py`.
- Novo serviço `beat` no `docker-compose.yml` compartilhando a imagem do backend.
- Endpoint `GET /api/models/{ticker}/runs` retorna últimos 20 runs.
- Migração Alembic `0003_ohlcv_unique_date.py` (unique `(ticker_id, trade_date)` + índice em `model_runs.ticker_symbol`).
- Testes `test_ohlcv_repository.py` e `test_training_orchestrator.py`.
- ADR 0007, `docs/security.md` (fecha gap de QA), docs refrescadas.

**Aceite**: `docker compose up -d` sobe `beat`; primeira chamada a `/api/predict/PETR4` popula OHLCV; `GET /api/models/PETR4/runs` responde (vazio ou com execuções); testes passam.

### Fase 11 - UI/UX + navegação (`@dev`) ✅ done
- `AppHeader` sticky com logo linkado à home e badge do ticker ativo.
- Home com hero, painel "Panorama do dia" (alta/baixa/neutras + variação média) e ticker cards ricos com preview D1 + W1.
- Dashboard com breadcrumb, stats resolvidas/pendentes, layout 3+2, seções com subtítulo.

**Aceite**: voltar para a home funciona em qualquer página; cards da home exibem preço + variação prevista.

### Fase 13 - Explicacao narrativa por horizonte (`@dev`) ✅ done
- `backend/app/ml/explanation.py` com `HeuristicExplanationGenerator` + `ClaudeExplanationGenerator` (prompt-injection guard + fallback), injetados via constructor em `PredictionService`.
- Migration Alembic `0004_prediction_explanation` adiciona coluna `predictions.explanation TEXT NULL`; `PredictionUpsert` preserva texto existente quando novo ciclo nao gerar.
- `GET /api/predict/{ticker}` devolve `horizons[i].explanation` (100..500 palavras); endpoint dedicado `GET /api/predictions/{ticker}/horizon/{horizon_days}/explanation` le o ultimo registro persistido (404 se nao houver).
- Frontend: `ExplanationModal` (Tailwind puro, acessivel, ESC + backdrop) e novo icone de info em `HorizonCard`; `HorizonGrid` repassa `explanation` vindo do backend.
- Testes: `test_explanation.py` cobre heuristica, fallback do Claude e `_clamp_words`; `test_horizons.py` passa a exigir explicacao valida.
- ADR `docs/adr/0008-horizon-explanation.md` justifica pre-geracao persistida + fallback.

**Aceite**: `alembic upgrade head` aplica 0004; `/api/predict/PETR4` inclui explicacao em cada horizonte; modal abre pelo icone no dashboard e mostra o texto persistido.

---

## 4. Critério de pronto (Definition of Done)

Um merge request só é aprovado quando:

1. Código segue `AGENTS.md` (SOLID, Clean Code, nomes claros, sem comentários inúteis).
2. Testes automatizados passam e cobertura mínima atingida.
3. Lints passam (`ruff`, `mypy`, `eslint`, `tsc`).
4. Checklist OWASP Top 10 revisado (ver `AGENTS.md`).
5. Swagger / tipos TypeScript atualizados.
6. `docker compose up -d` continua funcional.
7. ADR criado se a mudança for arquitetural.

---

## 5. Handoff para o orquestrador aiox-core

Após Fase 0 pelo Claude Code, invocar em sequência:

```
@pm       - lê PLAN.md e quebra em épicos/histórias
@architect - produz ADRs (Fase 1)
@data-engineer - Fase 2
@dev       - Fases 3, 4, 5
@devops    - Fases 6, 8
@qa        - Fase 7
```

Cada agente deve:
- Ler `PLAN.md`, `AGENTS.md` e `SPP_RECOMENDACAO_MELHORIA.md` antes de agir.
- Atualizar o status da fase em `docs/status.md` ao terminar.
- Abrir PRs pequenos e focados (≤ 400 LOC diff quando possível).

---

## 6. Riscos conhecidos

| Risco | Mitigação |
|---|---|
| BrAPI rate-limit | Cache Redis 5 min + fallback yfinance |
| Claude API custo | Cache 24 h + batch de notícias |
| TensorFlow imagem Docker grande | Base `tensorflow/tensorflow:2.16.1` ou usar `tf-cpu` no worker |
| Modelo `.keras` incompatível após refactor | Teste de regressão na Fase 3 |
| Time zone em séries temporais | Padronizar UTC no banco, America/Sao_Paulo no frontend |

---

*Plano mestre - SPP v3. Atualize este arquivo se o escopo mudar.*
