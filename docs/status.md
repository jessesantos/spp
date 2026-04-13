# Status de Modernização SPP v3

Atualizado em 2026-04-13.

| Fase | Agente | Status | Notas |
|------|--------|--------|-------|
| 0. Preparação (docs, scaffold, aiox-core) | claude-code | ✅ done | PLAN.md, AGENTS.md, scaffold backend/frontend, docker-compose.yml |
| 1. Arquitetura (ADRs) | @architect | ✅ done | `docs/adr/0001..0007` (0006 macro context, 0007 treinamento inteligente) |
| 2. Data Engineering (schema, migrations, ingest) | @data-engineer | ✅ done | SQLAlchemy 2.0 + Alembic 0001/0002/0003 + BrAPI/Yahoo/RSS + `OhlcvRepository` |
| 3. ML Core migration (Gemini -> Claude, SOLID) | @dev | ✅ done | `backend/app/ml/` com features, LSTM wrapper, Claude sentiment, aggregator, macro_context, training_orchestrator |
| 4. Backend API (rotas + middleware) | @dev | ✅ done | Multi-horizonte, history, reconcile, `/api/models/{ticker}/runs`, request-id, secure headers, slowapi, `/metrics` |
| 5. Frontend Next.js | @dev | ✅ done (v1) | AppHeader sticky + home com panorama + dashboard com HorizonGrid + HistoryTable. lightweight-charts diferido (ADR 0003) |
| 6. Docker & CI/CD | @devops | ✅ done | Dockerfiles multi-stage, serviços backend/worker/**beat**/frontend/postgres/redis, `.github/workflows/ci.yml`. GitLab CI deixado como backlog opcional |
| 7. QA & Segurança | @qa | ✅ done | Testes passando (unit + integração in-memory via SQLite), `docs/security.md` com checklist OWASP Top 10 + LLM Top 10 aplicado ao código. Playwright E2E deixado em backlog |
| 8. Observabilidade | @devops | ✅ done (core) | structlog JSON + `/metrics` Prometheus + healthchecks. Grafana pre-provisionado em backlog |
| 9. Previsões multi-horizonte + reconciliação | @dev | ✅ done | 3 horizontes persistidos por chamada, loop asyncio 1h reconcilia |
| 10. Contexto macro/internacional | @dev | ✅ done | Reuters/BBC/FT/Bloomberg/AP + Valor/InfoMoney/G1, `MacroContextBuilder`, coluna `macro_score` no LSTM |
| 11. UI/UX (navegação + panorama) | @dev | ✅ done | Header sticky com logo + ticker badge; home com hero, panorama, ticker cards ricos; dashboard com stats |
| 12. Treinamento inteligente auto-populado | @dev | ✅ done | `TrainingOrchestrator` + `SqlAlchemyOhlcvRepo` + `SqlAlchemyModelRunsRepo`; Celery beat com `daily_ohlcv_sync` 22h e `weekly_retrain` dom 23h; endpoint `/api/models/{ticker}/runs`; ADR 0007 |
| 13. Explicacao narrativa por horizonte | @dev | ✅ done | `ExplanationGenerator` (heuristico + Claude), coluna `predictions.explanation` (migration 0004), endpoint `/api/predictions/{ticker}/horizon/{n}/explanation`, modal no dashboard. ADR 0008 |

## Artefatos-chave

- Backend: `backend/app/` em 4 camadas (api, services, repositories, ml, data, db, infra)
- Frontend: `frontend/app/` + `components/` + `lib/`
- ADRs: `docs/adr/0001..0008`
- API: [`docs/API.md`](API.md)
- Arquitetura: [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- Segurança: [`docs/security.md`](security.md)
- Treinamento GPU Windows: [`TRAINING_WINDOWS.md`](TRAINING_WINDOWS.md)

## Backlog (não bloqueia a entrega)

1. Integrar `lightweight-charts` no dashboard para candlestick + overlay de predição.
2. Teste de regressão numérica legado vs. novo (< 1e-4) após retreino de PETR4 com modelo `.keras` persistido.
3. E2E Playwright cobrindo `/` e `/dashboard/[ticker]`.
4. `docker-compose.observability.yml` com Prometheus + Grafana + dashboard provisionado.
5. Autenticação JWT (A01/A07 do OWASP) antes de expor publicamente.
6. Cache Redis nos endpoints de predição com TTL curto (60 s) para reduzir custo BrAPI.
7. Dependabot ou Renovate para A06.
8. Assinatura de imagens Docker (cosign) para A08.
9. GitLab CI espelho do GitHub Actions caso o projeto migre.
