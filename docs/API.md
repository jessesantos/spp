# API Reference - SPP v3

> Catálogo completo dos endpoints HTTP e WebSocket expostos pelo FastAPI. Fonte da verdade: `/docs` (Swagger UI gerado automaticamente).

Base URL local: `http://localhost:8000`. WebSocket: `ws://localhost:8000`.

---

## Meta

### `GET /health`

Liveness probe. Sempre 200 com:

```json
{ "status": "ok", "version": "3.0.0" }
```

### `GET /`

Metadados da API.

```json
{ "name": "SPP API", "docs": "/docs" }
```

### `GET /metrics`

Prometheus counters/histograms. Não inclui `include_in_schema` (oculto do Swagger).

---

## Tickers

### `GET /api/tickers`

Lista dos tickers monitorados.

```json
[
  { "ticker": "PETR4", "name": "Petrobras PN", "currency": "BRL" },
  { "ticker": "VALE3", "name": "Vale ON",      "currency": "BRL" },
  { "ticker": "ITUB4", "name": "Itaú Unibanco PN", "currency": "BRL" }
]
```

---

## Previsões

### `GET /api/predict/{ticker}`

Query params: `days` (1..30, default 5, controla o preview sequencial).

Produz os três horizontes (D1, W1, M1) e persiste cada um em `predictions` (upsert idempotente). Resposta:

```json
{
  "ticker": "PETR4",
  "last_price": 49.03,
  "horizons": [
    { "horizon": "D1", "horizon_days":  1, "target_date": "2026-04-14",
      "base_close": 49.03, "predicted_close": 49.19,
      "predicted_pct":  0.33, "direction": "NEUTRO",
      "explanation": "Para o ticker PETR4 no horizonte Amanha ..." },
    { "horizon": "W1", "horizon_days":  7, "target_date": "2026-04-20",
      "base_close": 49.03, "predicted_close": 49.47,
      "predicted_pct":  0.90, "direction": "ALTA",
      "explanation": "Para o ticker PETR4 no horizonte +7 dias ..." },
    { "horizon": "M1", "horizon_days": 30, "target_date": "2026-05-13",
      "base_close": 49.03, "predicted_close": 48.95,
      "predicted_pct": -0.16, "direction": "NEUTRO",
      "explanation": "Para o ticker PETR4 no horizonte +30 dias ..." }
  ],
  "sentiment": {
    "score": 0.15, "confidence": 0.6,
    "positives": 4, "negatives": 2, "neutrals": 4
  },
  "predictions": [
    { "date": "2026-04-14", "predicted_close": 49.19, "direction": "NEUTRO" }
  ],
  "direction_accuracy": null
}
```

Limiar de direção: `|predicted_pct| < 0.5` -> `NEUTRO`. Rate-limit: `RATE_LIMIT_PREDICT` (default 10/minuto).

Códigos:

- `200` sucesso
- `400` ticker inválido (`DomainError`)
- `404` ticker sem dados e `MODEL_FALLBACK_STUB=false`
- `429` rate-limited

### `GET /api/predictions/{ticker}/history`

Query params: `limit` (1..500, default 60).

Lista previsões persistidas mais recentes por `target_date DESC`. Status `resolved = actual_close IS NOT NULL`.

```json
{
  "ticker": "PETR4",
  "items": [
    {
      "id": 3, "ticker": "PETR4",
      "horizon_days": 30, "created_at": "2026-04-13T07:12:22Z",
      "target_date": "2026-05-13",
      "base_close": 49.03, "predicted_close": 48.95, "predicted_pct": -0.16,
      "actual_close": null, "error_pct": null, "resolved": false,
      "explanation": "Para o ticker PETR4 no horizonte +30 dias ..."
    }
  ]
}
```

### `GET /api/predictions/{ticker}/horizon/{horizon_days}/explanation`

Retorna a explicacao narrativa persistida (100..500 palavras) para o ultimo registro de predicao daquele par `(ticker, horizon_days)`. `horizon_days` aceita apenas `1`, `7` ou `30`.

```json
{
  "ticker": "PETR4",
  "horizon_days": 7,
  "explanation": "Para o ticker PETR4 no horizonte +7 dias (7 dias), o modelo LSTM projeta ..."
}
```

Codigos:

- `200` explicacao encontrada
- `400` `horizon_days` fora do conjunto `{1, 7, 30}` ou ticker invalido
- `404` nenhuma explicacao armazenada ainda para o par (ex: previsao antes da Fase 13)

O texto tambem vem inline em `horizons[i].explanation` de `GET /api/predict/{ticker}`; este endpoint existe para recuperar o texto sem disparar nova previsao.

### `POST /api/predictions/reconcile`

Trigger manual do `ReconciliationService`. Varre previsões com `target_date <= today` e sem `actual_close`, busca preço real via BrAPI (fallback Yahoo) e preenche `actual_close` + `error_pct` + `resolved_at`.

```json
{ "checked": 12, "resolved": 8 }
```

Também roda automaticamente a cada 1 h via `asyncio` loop no `lifespan` do FastAPI.

---

## Sentimento

### `GET /api/sentiment/{ticker}`

Sentimento agregado do ticker. Se `ANTHROPIC_API_KEY` não estiver configurada, retorna um score stub determinístico (útil para demos).

```json
{
  "score": 0.15,
  "confidence": 0.6,
  "positives": 4,
  "negatives": 2,
  "neutrals": 4
}
```

---

## Treinamento

### `POST /api/train/{ticker}`

Dispara `celery_app.tasks.train_model` na fila `training`. O worker Celery chama `TrainingOrchestrator.train(ticker)` e persiste um `ModelRun` com status + métricas.

```json
{ "status": "queued", "ticker": "PETR4" }
```

Se o broker Redis estiver inacessível retorna `{"status": "queue_unavailable", ...}` (degrada sem quebrar o fluxo).

Treinos automáticos (Celery beat):

- `daily_ohlcv_sync`: todo dia às 22:00 BRT, sincroniza o delta de OHLCV de cada ticker para o Postgres.
- `weekly_retrain`: domingo 23:00 BRT, retreina todos os tickers headlessly (CPU no worker).

Treinar localmente com CLI (1 ano de preços + notícias + macro):

```bash
docker compose exec backend python -m app.ml.train --ticker PETR4 --period 1y --epochs 50
```

### `GET /api/models/{ticker}/runs`

Query params: `limit` (1..100, default 20).

Lista os últimos `ModelRun` registrados pelo pipeline de treinamento, ordenados por `created_at DESC`.

```json
{
  "ticker": "PETR4",
  "items": [
    {
      "id": 7,
      "ticker": "PETR4",
      "status": "done",
      "epochs": 50,
      "loss": 0.0042,
      "direction_accuracy": 0.72,
      "artifact_path": "/app/models/PETR4.keras",
      "created_at": "2026-04-12T23:00:05Z",
      "finished_at": "2026-04-12T23:04:18Z"
    }
  ]
}
```

Status possíveis: `running`, `done`, `failed`.

---

## WebSocket

### `WS /ws/live/{ticker}`

Stream de predições. A cada 60 s emite um `PredictionResponse` (1 ponto) como JSON. Encerrado em `WebSocketDisconnect` ou em caso de erro interno (close code `1011`).

Exemplo de mensagem:

```json
{
  "ticker": "PETR4",
  "last_price": 49.03,
  "predictions": [
    { "date": "2026-04-14", "predicted_close": 49.19, "direction": "NEUTRO" }
  ],
  "sentiment": { "score": 0.15, "confidence": 0.6,
                 "positives": 4, "negatives": 2, "neutrals": 4 },
  "direction_accuracy": null
}
```

---

## Tratamento de erros

Qualquer 4xx/5xx de rota vem em formato estruturado:

```json
{
  "error": {
    "code": "NotFoundError",
    "message": "no data or trained model for XYZ99",
    "status": 404,
    "request_id": "<uuid>"
  }
}
```

- `DomainError` (generic 4xx)
- `NotFoundError` (404)
- `RateLimitExceeded` (429)

---

## Headers obrigatórios

- `Content-Type: application/json` em POST
- Respostas incluem `X-Request-ID` (gerado pelo `RequestIDMiddleware`) para correlação em logs.
- CORS só aceita origens em `CORS_ORIGINS` (default `http://localhost:3000`).

---

## Versionamento

`X.Y.Z` seguindo SemVer. Mudanças breaking exigem novo caminho `/api/v2/...` (não há v2 ainda).
