# Threat model e checklist OWASP Top 10 - SPP v3

Revisao aplicada ao codigo presente no repositorio em 2026-04-13.
Complementa o `AGENTS.md §2`. Revisitar a cada PR que toque autenticacao,
autorizacao, ingest externo ou prompts LLM.

---

## Superficie de ataque

```
Browser -> Frontend Next.js (SSR) -> FastAPI -> {Postgres, Redis, BrAPI, Yahoo, RSS, Claude}
                                    ^
                                    +--- Celery worker + beat (fila training)
```

Entradas nao-confiaveis:

1. Paths da API (`ticker` em URL, `days`, `limit`).
2. Headers HTTP (rate-limit usa IP origem).
3. Conteudo de feeds RSS (usado como prompt do Claude).
4. Payload JSON do BrAPI e Yahoo (deserializado sem validacao rigida).

---

## Checklist OWASP Top 10 (2021)

### A01 - Broken Access Control

- [x] Tickers validados em `_validate_ticker` (regex alnum, 1..10 chars) em todo endpoint.
- [x] Nao ha conceito de "dono" de recurso; ainda assim, todas as rotas sao read-only ou idempotentes (upsert por target_date).
- [ ] Autenticacao JWT ausente. Aceitavel enquanto o servico roda `localhost` + CORS restrito, mas obrigatorio antes de expor publicamente.

**Acao:** Adicionar JWT (OAuth2 password flow do FastAPI) antes de deploy externo.

### A02 - Cryptographic Failures

- [x] Segredos em `.env` (nao versionado), `.env.example` e placeholder.
- [x] Cloudflare Tunnel (ADR 0005) termina TLS antes de bater na API.
- [x] Sem hashes proprios; Claude/BrAPI/Yahoo falam HTTPS.
- [ ] Nao ha assinatura de payload do BrAPI. Mitigacao: validar campos.

### A03 - Injection

- [x] SQL: SQLAlchemy parametrizado em todas as queries.
- [x] HTTP input: Pydantic v2 valida tipo e range (`Query(..., ge=1, le=30)`).
- [x] Frontend: Zod valida toda resposta antes de renderizar.
- [x] **Prompt injection (LLM01):** `ClaudeSentimentAnalyzer` encapsula o artigo em tags `<article>...</article>` e orienta explicitamente Claude a tratar o conteudo como dado, nao instrucao. `_sanitize` strip de `<`, `>`, `` ` `` no ticker antes do prompt.
- [x] `MacroAnalyzer` usa a mesma via segura.

### A04 - Insecure Design

- [x] Threat model documentado aqui.
- [x] Rate-limit por IP via slowapi (`RATE_LIMIT_DEFAULT=60/min`, predict 10/min).
- [x] Circuit breaker implicito: BrAPI falha -> fallback Yahoo -> fallback stub. Nenhum endpoint retorna 500 por causa de fonte externa caida.
- [ ] Quota por tenant ausente (nao aplicavel ainda, nao ha tenant).

### A05 - Security Misconfiguration

- [x] `CORSMiddleware` com `allow_origins` explicito via env.
- [x] `SecureHeadersMiddleware` adiciona `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy` (ver `backend/app/infra/middleware.py`).
- [x] `DEBUG` nao e exposto; FastAPI em producao usa `uvicorn` sem reload.
- [x] Containers rodam com usuario `app` nao-root (Dockerfile).
- [x] Postgres nao expoe password default fora do dev compose.
- [ ] Imagens Docker: `trivy image` roda no CI mas nao bloqueia build ainda.

### A06 - Vulnerable and Outdated Components

- [x] Todas as deps pinadas em `requirements.txt` (backend) e `package-lock.json` (frontend).
- [x] `.github/workflows/ci.yml` roda `pip-audit`, `npm audit`, `trivy image`.
- [ ] Dependabot/Renovate nao configurado. Backlog.

### A07 - Identification and Authentication

- [ ] Nao implementado. Justificado acima (A01). Plano: OAuth2 password + refresh token quando publicar.

### A08 - Software and Data Integrity

- [x] Migrations via Alembic versionadas no repo; startup roda `upgrade head`.
- [x] `predictions` tem unique `(ticker, horizon_days, target_date)`; upsert preserva consistencia.
- [x] Modelos `.keras` nao sao aceitos de fonte externa; treino persistido + `artifact_path` registrado em `model_runs`.
- [ ] Imagens Docker nao sao assinadas (cosign). Backlog.

### A09 - Security Logging and Monitoring

- [x] structlog JSON em stdout com `request_id` em toda request (gerado por `RequestIDMiddleware`).
- [x] Erros 5xx logam stack trace pela `register_exception_handlers`.
- [x] Prometheus `/metrics` com histogramas por rota e contadores de 4xx/5xx.
- [x] Nao logamos tokens, ANTHROPIC_API_KEY, senhas de Postgres.
- [ ] Alerta automatizado em 5xx > 1% nao configurado (Grafana alerting no backlog).

### A10 - Server-Side Request Forgery

- [x] URLs externas sao uma allowlist estatica: `BRAPI_BASE_URL`, `RSS_FEEDS`, Yahoo (constante em yfinance), Anthropic API. Nenhum input do usuario vira URL.
- [x] `httpx.AsyncClient` com `timeout=10.0` e `retries=2`, sem seguir redirect para IP privado manualmente (e aceitavel porque todos os hosts sao conhecidos).
- [x] Treinamento nao aceita `url` como arg do CLI.

---

## Pontos de atencao especificos da pipeline de IA (OWASP LLM Top 10)

- **LLM01 Prompt injection:** mitigado (A03).
- **LLM02 Insecure output handling:** resposta do Claude e parseada com `json.loads` + best-effort regex; retorno degradado para neutro em caso de falha. Nao executamos nada que venha do modelo.
- **LLM03 Training data poisoning:** dados vem de BrAPI + Yahoo + feeds publicos. Nao aceitamos upload de usuario para treino. Backlog: checksum de OHLCV historico.
- **LLM04 Model DoS:** rate-limit em predict (10/min); treino so via fila Celery interna.
- **LLM05 Supply chain:** deps pinadas; auditoria via `pip-audit` + `trivy`.
- **LLM06 Sensitive info disclosure:** sem PII em logs; ticker e dado publico.
- **LLM07 Insecure plugin design:** nao aplicavel (nao ha plugin store).
- **LLM08 Excessive agency:** Claude so pontua sentimento; nao executa acoes.
- **LLM09 Overreliance:** disclaimers em README e AGENTS.md: "nao constitui recomendacao financeira".
- **LLM10 Model theft:** artefatos `.keras` sao montados via volume local, sem endpoint de download.

---

## Plano de acao (curto prazo)

1. Implementar JWT para fechar A01/A07 antes de expor publicamente.
2. Adicionar Dependabot para reduzir janela de A06.
3. Ligar alerta Grafana em 5xx > 1% e em Claude error-rate.
4. Rodar `trivy image` como gate obrigatorio no CI.

---

## Revisao

A cada PR que toque autenticacao, autorizacao, prompts LLM ou fontes
externas de dados, o reviewer (`@qa`) deve reavaliar os itens
relevantes deste checklist e anotar no docs/status.md.
