# SPP - Stock Price Predictor

Plataforma full-stack de análise preditiva para a B3, combinando rede neural LSTM, análise de sentimento via Claude (Anthropic) e contexto macro global (notícias nacionais + internacionais) para antecipar movimentos de preço em múltiplos horizontes temporais.

Inspirado em arquiteturas de sistemas de investimento como o Aladdin da BlackRock, o projeto integra histórico de preços, volume, indicadores técnicos e sinais de notícias para projetar preços em **D+1**, **D+7** e **D+30**, com reconciliação automática do erro contra o preço real.

> **Status:** v3.1 produção demo em Docker com fallback stub determinístico. Inclui mercados de previsão (Kalshi + Polymarket), analisador de impacto cambial BRL/USD, volatilidade condicional EWMA (RiskMetrics) e SKILL.md com Escola Austríaca aplicado ao Claude. Para previsões reais, treine seguindo `docs/TRAINING_WINDOWS.md`.

---

<div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
  <img src="https://codde.dev/spp/print-ssp.png" alt="SPP logo">
</div>

---

## Stack

| Camada | Tecnologias |
|---|---|
| Frontend | Next.js 15 (App Router) · TypeScript · Tailwind CSS · Zod · React 19 · [Hugeicons](https://hugeicons.com/) (`@hugeicons/react` + `@hugeicons/core-free-icons`) |
| Backend | FastAPI · Python 3.12 · Pydantic v2 · SQLAlchemy 2.0 async · Alembic · structlog · slowapi · Prometheus instrumentator |
| Machine Learning | TensorFlow 2.16 / Keras (LSTM 128-64-32) · scikit-learn · pandas · numpy |
| Dados | BrAPI (B3) · yfinance (fallback) · feedparser (RSS) · Claude Sonnet 4.5 (Anthropic SDK) |
| Persistência | PostgreSQL 16 · Redis 7 (cache + broker) |
| Async jobs | Celery · asyncio reconcile loop (1 h) |
| Orquestração agêntica | `aiox-core` (SynkraAI) com squad `@architect` · `@dev` · `@qa` · `@devops` · `@data-engineer` · `@pm` · `@analyst` |
| Infra | Docker Compose · multi-stage Dockerfiles · non-root users · healthchecks |

---

## Quick start

Pré-requisitos: Docker + Docker Compose v2. Para treinamento com GPU (RTX 5070 Blackwell), veja [`docs/TRAINING_WINDOWS.md`](docs/TRAINING_WINDOWS.md).

```bash
git clone https://github.com/jessesantos/spp.git
cd spp
cp .env.example .env
# (opcional) edite .env para colocar ANTHROPIC_API_KEY

docker compose up -d
```

Serviços:

- Frontend: <http://localhost:3000>
- API FastAPI: <http://localhost:8000>
- Swagger: <http://localhost:8000/docs>
- Métricas Prometheus: <http://localhost:8000/metrics>
- Postgres: `localhost:5432` (user `spp`, db `spp`)
- Redis: `localhost:6379`

O Alembic roda `upgrade head` automaticamente na subida do backend. Sem `ANTHROPIC_API_KEY` e sem arquivos `.keras` treinados, o sistema continua funcional via fallback stub determinístico (valores por seed do ticker), útil para demos.

---

## Funcionalidades principais

### 1. Previsões multi-horizonte

Para cada ticker, o endpoint `GET /api/predict/{ticker}` retorna três projeções:

- **D1** próximo pregão
- **W1** +7 dias
- **M1** +30 dias

Cada horizonte contém `base_close`, `predicted_close`, `predicted_pct` (variação assinada) e `direction` (`ALTA` / `BAIXA` / `NEUTRO`, limiar configurável).

Junto com cada horizonte vem `explanation`, uma narrativa executiva e dinâmica (100-500 palavras heurística, 120-350 palavras Claude) integrando **todos** os sinais do motor v3.1: leitura técnica (LSTM + médias/RSI/MACD), sentimento, macro global, **impacto cambial BRL/USD**, **mercados de previsão (Kalshi + Polymarket)** e **regime de volatilidade condicional** (EWMA). Sinais neutros são omitidos para evitar redundância. O texto é gerado no mesmo ciclo (via `ClaudeExplanationGenerator` com `SKILL.md` injetado como system prompt quando `ANTHROPIC_API_KEY` está presente, ou `HeuristicExplanationGenerator` como fallback determinístico) e persistido na coluna `predictions.explanation`. No dashboard, cada card D1/W1/M1 tem um ícone de info que abre um modal com o texto; a rota dedicada `GET /api/predictions/{ticker}/horizon/{n}/explanation` também recupera o texto sem disparar nova previsão.

### 2. Histórico rastreável e reconciliação

Toda previsão é persistida na tabela `predictions` (unique em `ticker + horizon_days + target_date`, upsert idempotente). Um loop asyncio executado a cada hora chama `ReconciliationService`, busca o preço real via BrAPI quando `target_date <= today` e preenche `actual_close` + `error_pct` + `resolved_at`. O endpoint `GET /api/predictions/{ticker}/history` expõe tudo com status `resolvido | pendente`.

### 3. Sentimento cruzado (ticker + macro + mercados de previsão + câmbio)

Quatro camadas:

- **Sentimento do ticker** - `RSSNewsClient` coleta feeds PT-BR (InfoMoney, MoneyTimes, Estadão/E-Investidor), filtra por menção ao ticker, `ClaudeSentimentAnalyzer` pontua cada artigo com prompt-injection guard (OWASP LLM01). O Claude recebe `SKILL.md` (Escola Austríaca, Graham/Buffett, Soros) como system prompt para contextualizar o scoring.
- **Contexto macro global** - `RSSMacroNewsSource` + `MacroContextBuilder` cruzam Reuters, BBC World/Business, FT, Bloomberg, AP, Valor, InfoMoney, G1 com filtro por palavras-chave geopolíticas. Score entra como feature `macro_score` no LSTM.
- **Mercados de previsão** (v3.1) - `KalshiClient` + `PolymarketClient` puxam probabilidades precificadas para eventos macro (Fed cuts, recession, Brent > $100). `PredictionMarketAggregator` filtra por tópicos relevantes ao ticker e gera `market_signal_score`. Snapshots persistidos em `prediction_market_signals`.
- **Impacto cambial BRL/USD** (v3.1) - `CurrencyImpactAnalyzer` combina heurística setorial (PETR4/VALE3 exportadoras, MGLU3/AZUL4 importadoras) com correlação empírica ticker vs `USDBRL=X` nos últimos 90 dias, produzindo feature `fx_score`.

### 4. Treinamento inteligente

O sistema se auto-alimenta via Celery beat (serviço `beat` no `docker-compose.yml`):

- **Backfill sob demanda** - ao acessar um ticker pela primeira vez, `PredictionService` puxa 3 anos de OHLCV via BrAPI (fallback Yahoo) e persiste idempotentemente em `ohlcv` (unique `ticker_id + trade_date`).
- **Daily OHLCV sync** - todo dia às 22:00 BRT, o beat agenda `spp.daily_ohlcv_sync`: para cada ticker, sincroniza apenas o delta desde `latest_trade_date`.
- **Weekly retrain** - domingo 23:00 BRT, o beat agenda `spp.weekly_retrain`: para cada ticker com dados, `TrainingOrchestrator.train` constrói features, pontua sentimento + contexto macro, treina o LSTM e salva em `models/{TICKER}.keras`.
- **Auditoria** - cada execução cria uma linha em `model_runs` (`status`, `loss`, `direction_accuracy`, `artifact_path`). Exposta em `GET /api/models/{ticker}/runs`.

Treino manual continua disponível:

```bash
docker compose exec backend python -m app.ml.train --ticker PETR4 --period 3y --epochs 50
# ou dispare via API (vai pro Celery)
curl -X POST http://localhost:8000/api/train/PETR4
```

Ver [`docs/adr/0007-intelligent-training.md`](docs/adr/0007-intelligent-training.md) para a decisão arquitetural.

---

<div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
  <img src="https://codde.dev/spp/how-it-works.png" alt="SPP logo">
</div>

---

### 5. Dashboard

- Home (`/`) com hero atualizado v3.1, painel "Panorama do dia" (alta / baixa / neutras + variação média prevista), grid de tickers e seção "Como o motor funciona" com 9 feature cards cobrindo todos os sinais.
- `/dashboard/[ticker]` com PredictionCard principal, sentimento agregado, grid de horizontes, projeção sequencial de 5 dias e tabela de histórico com status resolvido/pendente e erro %.
- `AppHeader` sticky com logo PNG (tooltip "Stock Price Predictor") + badge do ticker atual + link direto pra Swagger.
- **Ícones**: [Hugeicons](https://hugeicons.com/) via `@hugeicons/react` (pacote `core-free-icons`), substituindo emojis e SVGs inline em todo o frontend para consistência visual.
- **Acessibilidade**: focus-visible rings em todos os botões/links interativos, `aria-label`/`aria-busy`/`role="dialog"`, contraste revisado para AA (body copy em `text-neutral-300`, metadata em `text-neutral-400`).
- **Performance de navegação**: ISR `revalidate=60` no backend fetch + `loading.tsx` por rota (skeleton instantâneo via Next.js App Router) + `useLinkStatus` (Next 15.3+) com barra de progresso e spinners inline nos links. Navegação quente < 30ms, feedback visual imediato mesmo no cold path.

---

<div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
  <img src="https://codde.dev/spp/ticker-dashboard.png" alt="SPP logo">
</div>

---

## Arquitetura em camadas

Ver [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) para diagrama completo.

```
frontend/                          backend/app/
├── app/         (rotas)           ├── api/          (controllers + schemas + errors)
├── components/  (UI)              ├── services/     (casos de uso)
├── hooks/       (WebSocket)       ├── repositories/ (predictions_repository)
├── lib/         (api + format)    ├── ml/           (features, LSTM, Claude, macro)
└── ...                            ├── data/         (BrAPI, yfinance, RSS)
                                   ├── db/           (SQLAlchemy models + session)
                                   ├── infra/        (config, logging, middleware, celery)
                                   └── migrations/   (Alembic)
```

Padrões seguidos (detalhe em [`AGENTS.md`](AGENTS.md)): MVC moderno em 4 camadas, SOLID, Clean Code, Twelve-Factor, TDD, OWASP Top 10.

---

## Treinamento dos modelos

O sistema funciona sem treinamento via fallback stub. Para previsões reais:

```bash
# Dentro do container backend (CPU)
docker compose exec backend python -m app.ml.train --ticker PETR4 --period 3y --epochs 50

# Ou no Windows nativo com GPU RTX (Blackwell sm_120 exige TF 2.18+)
# Ver docs/TRAINING_WINDOWS.md
```

O CLI:

1. Baixa 3 anos de OHLCV via BrAPI (fallback yfinance)
2. Coleta notícias RSS do ticker + feeds macro globais
3. Agrega sentimento via Claude com SKILL.md (Escola Austríaca) ou heurística
4. Puxa sinais de Kalshi/Polymarket relevantes ao ticker
5. Calcula impacto cambial (correlação + heurística setorial BRL/USD)
6. Monta features técnicas + `cond_vol` (EWMA RiskMetrics), `sentiment`, `macro_score`, `fx_score`, `market_signal_score`
7. Treina LSTM e salva em `./models/{TICKER}.keras`

O volume `./models` é montado em `/app/models` no container; nova predição passa a usar o modelo sem restart.

---

## Testes

```bash
# 41 testes, todos passando
docker compose exec -u root backend \
  sh -c "pip install -q pytest pytest-asyncio httpx aiosqlite && python -m pytest tests/ -q"
```

Cobertura:

- `test_features.py` - indicadores técnicos (puros)
- `test_claude_sentiment.py` - Claude mockado, parsing robusto, prompt-injection guard
- `test_horizons.py` - service multi-horizonte com fake repo
- `test_prediction_repo.py` - upsert idempotente + listagens (SQLite in-memory)
- `test_predict.py` / `test_sentiment.py` / `test_health.py` - integração via FastAPI TestClient
- `test_macro_context.py` - agregador macro + heurística

Backend CI-ready via `.github/workflows/ci.yml` (pytest + ruff + mypy + trivy + Next build).

---

## Variáveis de ambiente

Principais (ver `.env.example` completo):

```env
ENV=development
LOG_LEVEL=INFO

POSTGRES_PASSWORD=change_me_in_prod
DATABASE_URL=postgresql+asyncpg://spp:change_me_in_prod@postgres/spp
REDIS_URL=redis://redis:6379/0

ANTHROPIC_API_KEY=           # opcional: sem ela o fallback stub mantem o demo vivo
CLAUDE_MODEL=claude-sonnet-4-5-20250929

CORS_ORIGINS=["http://localhost:3000"]
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

MODEL_FALLBACK_STUB=true
MODELS_DIR=/app/models
RATE_LIMIT_DEFAULT=60/minute
RATE_LIMIT_PREDICT=10/minute

# v3.1 (ADR 0009)
TRAINING_PERIOD_DEFAULT=3y
BACKFILL_LOOKBACK_DAYS=1095
KALSHI_BASE_URL=https://api.elections.kalshi.com/trade-api/v2
KALSHI_API_KEY=                 # opcional, somente eleva rate-limit
POLYMARKET_BASE_URL=https://gamma-api.polymarket.com
PREDICTION_MARKETS_ENABLED=true
SENTIMENT_SKILL_PATH=/app/app/ml/SKILL.md
```

---

## Agentes (aiox-core)

O projeto tem `aiox-core` instalado como core agêntico. Os prompts de agente ficam em `.claude/commands/AIOX/agents/` e podem ser invocados no Claude Code via `@<agente>`:

| Agente | Papel |
|---|---|
| `@aiox-master` (Orion) | orquestrador principal |
| `@architect` | decisões arquiteturais e ADRs |
| `@dev` | implementação |
| `@qa` | testes e validação |
| `@devops` | Docker, CI/CD, observabilidade |
| `@data-engineer` | schema, migrations, pipelines |
| `@pm` | épicos / histórias |
| `@po` | priorização |
| `@sm` | coordenação entre squads |
| `@analyst` | requisitos, PRD, brainstorming |
| `@ux-design-expert` | experiência do usuário |

Antes de agir, todo agente lê obrigatoriamente [`AGENTS.md`](AGENTS.md) (regras + §0 de estilo), [`PLAN.md`](PLAN.md) (roadmap) e [`SPP_RECOMENDACAO_MELHORIA.md`](SPP_RECOMENDACAO_MELHORIA.md) (stack detalhada).

---

## Documentação

| Arquivo | Finalidade |
|---|---|
| [`PLAN.md`](PLAN.md) | Plano mestre, fases, critérios de aceite |
| [`AGENTS.md`](AGENTS.md) | Diretrizes de engenharia (SOLID, Clean Code, OWASP Top 10, TDD, 12-Factor) |
| [`SPP_RECOMENDACAO_MELHORIA.md`](SPP_RECOMENDACAO_MELHORIA.md) | Stack e trechos de código concretos |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Arquitetura em camadas |
| [`docs/API.md`](docs/API.md) | Catálogo de endpoints |
| [`docs/status.md`](docs/status.md) | Status por fase |
| [`docs/security.md`](docs/security.md) | Threat model + checklist OWASP Top 10 e LLM Top 10 |
| [`docs/TRAINING_WINDOWS.md`](docs/TRAINING_WINDOWS.md) | Treinar LSTM no Windows com GPU |
| [`docs/adr/`](docs/adr/) | Architecture Decision Records (0001..0011) |
| [`backend/app/ml/SKILL.md`](backend/app/ml/SKILL.md) | System prompt econômico do Claude (Austrian school, value, macro) |

---

## Licença

MIT. Este projeto é didático / pesquisa. **Não constitui recomendação financeira.**
