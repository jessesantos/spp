# ADR 0007: Treinamento inteligente com Celery beat e auto-backfill

## Status

Aceito (2026-04-13)

## Contexto

O treinamento inicial era 100% manual: `python -m app.ml.train --ticker X`. Isso tem tres problemas:

1. Depende de intervencao humana para manter modelos atualizados contra
   dados recentes (drift).
2. Nao persiste historico de execucoes, entao nao da para comparar
   versoes do modelo ao longo do tempo.
3. Exige que o usuario ja tenha OHLCV em mao, enquanto o proprio
   backend ja sabe buscar via BrAPI.

O requisito novo: o sistema deve se auto-alimentar. Assim que um ticker
e acessado pela primeira vez, seu historico de 1 ano e populado; e a
cada semana o modelo e retreinado automaticamente com os dados mais
recentes.

## Decisao

Escolhemos **Celery beat no mesmo codebase** em vez de alternativas:

- **Celery beat** (escolhido): reusa a infra Redis ja existente, roda
  como um novo servico Docker (`beat`) que publica tarefas na mesma
  fila `training` consumida pelo `worker`. Configuracao declarativa em
  `app/infra/celery_beat.py` com `crontab(...)`.
- **Cron no host**: quebra o isolamento Docker, exige setup fora do
  compose, nao conhece o virtualenv do container.
- **APScheduler in-process**: morre junto com o FastAPI e duplica
  trabalho se houver mais de um worker/replica da API.
- **Scheduler externo (Airflow, Dagster)**: overkill para tres jobs
  simples em uma stack single-node.

Arquitetura do pipeline:

```
+-----------------------------+    every 1 day @ 22:00 BRT
| beat (celery-beat)          |---> spp.daily_ohlcv_sync
|                             |    every Sunday @ 23:00 BRT
|                             |---> spp.weekly_retrain
+-----------------------------+
              |
              v
+-----------------------------+
| worker (celery-worker)      |
|   queue: training           |
+-----------------------------+
              |
              +--> PriceRepository (BrAPI -> Yahoo)
              +--> SqlAlchemyOhlcvRepo (upsert idempotente)
              +--> TrainingOrchestrator
                    +--> ClaudeSentimentAnalyzer (opcional)
                    +--> MacroContextBuilder (opcional)
                    +--> LSTMPricePredictor.train
                    +--> artifact .keras em MODELS_DIR
                    +--> ModelRunsRepo.create(status, metrics)
```

Componentes SOLID:

- `SqlAlchemyOhlcvRepo` abstrai persistencia OHLCV; unique constraint
  `(ticker_id, trade_date)` garante idempotencia.
- `SqlAlchemyModelRunsRepo` persiste cada tentativa (status,
  loss, direction_accuracy, artifact_path, finished_at).
- `TrainingOrchestrator` recebe via construtor: `prices_repo`,
  `ohlcv_repo`, `news_source`, `macro_builder`, `model_runs_repo`.
  Nenhuma instanciacao concreta dentro; wiring em
  `infra/dependencies.build_training_orchestrator`.
- CLI `app/ml/train.py` vira thin wrapper que pega o orchestrator do
  composition root.
- `GET /api/models/{ticker}/runs` expoe os registros (read-only).

## Consequencias

Positivas:

- Modelos ficam frescos automaticamente sem acao humana.
- Historico auditavel de execucoes em `model_runs`.
- Mesma imagem Docker roda `backend`, `worker` e `beat` (menor
  complexidade de pipeline e menor superficie de ataque).
- Testavel: TrainingOrchestrator isolado recebe fakes em
  `tests/test_training_orchestrator.py`.

Negativas:

- `celery-beat` precisa de persistencia de schedule state (arquivo
  `celerybeat-schedule`) se a instancia reiniciar no meio de um tick.
  Aceitavel com volume ou beat stateless em containers efemeros.
- Treinamento dentro do container e CPU-only (limitacao TensorFlow +
  Blackwell). GPU segue o fluxo Windows do
  [TRAINING_WINDOWS.md](../TRAINING_WINDOWS.md). O retreino semanal em
  CPU e OK para demo; para prod com GPU, apontar o beat para um
  worker externo GPU.
- Single-point-of-failure: se o beat cair, schedules param. Para
  prod real, usar Celery beat HA via RedBeat ou Kubernetes CronJob.

## Rastreabilidade

- Migracoes: `backend/migrations/versions/0003_ohlcv_unique_date.py`
- Repos: `backend/app/repositories/ohlcv_repository.py`,
  `backend/app/repositories/model_runs_repository.py`
- Orchestrator: `backend/app/ml/training_orchestrator.py`
- Beat schedule: `backend/app/infra/celery_beat.py`
- Tasks: `backend/app/infra/tasks.py` (`train_model`,
  `daily_ohlcv_sync`, `weekly_retrain`)
- API: `GET /api/models/{ticker}/runs` em `backend/app/api/routes.py`
- Docker: servico `beat` em `docker-compose.yml`
- Testes: `backend/tests/test_ohlcv_repository.py`,
  `backend/tests/test_training_orchestrator.py`
