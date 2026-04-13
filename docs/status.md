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
| 14. v3.1 - técnicas avançadas | @architect + @dev | ✅ done | EWMA conditional volatility (RiskMetrics), Kalshi + Polymarket clients, tabela `prediction_market_signals` (migration 0005), `CurrencyImpactAnalyzer` BRL/USD, `SKILL.md` (Austrian + value + macro + behavioral) injetado via `system` prompt em `ClaudeSentimentAnalyzer` e `ClaudeExplanationGenerator`, janela default 3y. ADR 0009 |
| 15. Explicação multi-sinal executiva | @dev | ✅ done | Heurística reescrita em narrativa dinâmica não-repetitiva; `ExplanationInput` estendido com `fx_score`, `fx_exposure_label`, `market_signal_score`, `market_signal_confidence`, `market_signal_topics`, `cond_vol`; `PredictionService` faz best-effort fetch (`_safe_fx`, `_safe_market`, `_safe_cond_vol`); prompt Claude upgrade (executivo, 120-350 palavras, usa SKILL); word count removido do modal. ADR 0008 atualizado |
| 16. Frontend UX pass | @dev | ✅ done | Logo PNG + tooltip; Hugeicons substituindo emojis; focus-visible rings; contraste AA; `loading.tsx` skeletons; `useLinkStatus` + `LinkTopProgress`/`LinkSpinner`; ISR `revalidate=60` em paginas + `next.revalidate` em `lib/api.ts` (navegação quente <30ms) |
| 17. Treinamento real validado | @dev | ✅ done | RTX 5070 Blackwell sm_120 detectada mas PTX JIT falha (TF 2.19); fallback CPU entrega PETR4/VALE3/ITUB4 em 20-30s cada. Bugs corrigidos no caminho: defaults BrAPI/Yahoo `3y→5y` (APIs rejeitam 3y), `LSTMPricePredictor.save/load` agora persiste scalers+colunas em `{TICKER}.keras.aux.joblib`, `predict` tolera features externas ausentes via preenchimento zero. Pipeline end-to-end confirmado com predições reais (não mais stub) e explicação multi-sinal integrada. ADR 0010 |
| 18. Refatorações pós-revisão | @architect + @dev | ✅ done | Novo `backend/app/domain/` com `TickerSymbol` (value object com allowlist regex que bloqueia SSRF/path-traversal) e `horizons.py` (constantes movidas do `prediction_service`). `PredictionService._safe_*` consolidados via helper `_safe_await`. Celery `train_model` protegido por `fcntl.flock` em `{MODELS_DIR}/{TICKER}.lock` (serializa treinos concorrentes para o mesmo ticker, evita corrida na escrita de `.keras` + aux.joblib). Novos testes: `test_domain.py` (8), `test_training_lock.py` (3). `_sanitize()` em `explanation.py`/`claude_sentiment.py` também remove `\r\n\t` (mitigação extra OWASP LLM01). Total: **101 testes passando** |
| 19. Fix matemático: log-retorno como alvo do LSTM | @architect + @dev | ✅ done | **Bug crítico:** LSTM predizia preço absoluto via MinMaxScaler, saturava quando ticker fazia topo histórico (PETR4 last=R$49.97 vs scaler.max=49.67 → D1=-18%). Rollout recursivo compunha erro exponencial. **Fix (ADR 0011):** target trocado para `log(close[t+1]/close[t])` com scaler `feature_range=(-1,1)`; predict aplica `next = last * exp(pred)` + guard-rail ±30%/passo. EarlyStopping patience 10→15. Nova métrica `direction_accuracy` no treino. Retreino 200 épocas: PETR4 D1=-0.03%, VALE3 M1=-4.43%, ITUB4 M1=-2.90% (range realista). `DELETE FROM predictions` limpou 12 linhas com valores absurdos; novo predict repopula consistente com histórico |
| 20. Consistência home ↔ dashboard | @architect + @dev | ✅ done | **Bug visual:** valores na home divergiam dos cards do dashboard. Causa raiz: cache Next ISR expirava entre renderizações em momentos diferentes, backend recomputava com histórico BrAPI levemente atualizado entre uma chamada e outra → rollouts diferentes. **Fix (a):** `PredictionService` curto-circuita para predições persistidas <60s no DB (`_try_cached_response` + `CACHE_WINDOW_SECONDS`), garantindo resposta determinística por janela; preview sintetizada a partir dos 3 horizontes quando servida do cache. **Fix (b):** dashboard `PredictionCard` passa a ler `horizons[0]` (D1) em vez de `predictions[0]`, mesma fonte canônica que a home. Resultado: chamadas back-to-back e renderizações cross-page mostram valores idênticos |

## Artefatos-chave

- Backend: `backend/app/` em 4 camadas (api, services, repositories, ml, data, db, infra)
- Frontend: `frontend/app/` + `components/` + `lib/`
- ADRs: `docs/adr/0001..0009`
- SKILL econômico do Claude: `backend/app/ml/SKILL.md`
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
