# AGENTS.md - Diretrizes para Agentes (aiox-core + Claude Code)

## 0. Regra de estilo obrigatória (leia antes de qualquer coisa)

**PROIBIDO** usar o caractere `-` (em-dash U+2014) ou `-` (en-dash U+2013) em qualquer saída gerada neste projeto: código, comentários, docstrings, documentação (`.md`), nomes, mensagens de commit, descrições de PR, respostas ao usuário e prompts para subagentes.

Substitua por:
- `-` (hífen) quando separando cláusulas curtas
- `:` ou `,` quando introduzindo explicação
- Reformule a frase quando nenhum dos anteriores couber

Essa regra vale para **todos** os agentes do squad aiox (`@architect`, `@dev`, `@qa`, `@devops`, `@data-engineer`, `@pm`, `@po`, `@sm`, `@analyst`, `@ux-design-expert`) e para o orquestrador `@aiox-master`. Ao delegar para outro agente, inclua esta regra no prompt.

Se encontrar `-` ou `-` em arquivos pre-existentes ao editar, substitua pelo equivalente ASCII.

---

> Este arquivo e **leitura obrigatoria** para qualquer agente antes de escrever codigo no projeto SPP.

---

## 1. Metodologias e princípios

### MVC moderno (camadas)

Backend (FastAPI) deve seguir separação em 4 camadas - não misturar responsabilidades:

```
backend/app/
├── api/          # Controllers - routers FastAPI, validação Pydantic, sem regra de negócio
├── services/     # Use cases / lógica de aplicação - orquestra repositories + ml
├── repositories/ # Acesso a dados - SQLAlchemy, Redis, APIs externas
├── domain/       # Entidades + value objects puros - zero dependência de framework
├── ml/           # Modelos, feature engineering, predição
└── infra/        # Config, logging, middleware, db engine, celery app
```

Frontend (Next.js) segue o análogo:

```
frontend/
├── app/           # Routing (App Router)
├── components/    # UI (apresentação, sem fetch direto)
├── hooks/         # Lógica reutilizável (fetch, state)
├── lib/api/       # Cliente HTTP tipado (TanStack Query)
├── lib/schemas/   # Schemas Zod
└── stores/        # Zustand (estado global)
```

### SOLID

- **S - Single Responsibility**: cada classe/função faz **uma** coisa. Se o nome contém "and"/"e", divida.
- **O - Open/Closed**: estenda via polimorfismo/injeção; não edite classes estáveis para adicionar variantes.
- **L - Liskov**: subclasses respeitam o contrato da base (parâmetros, exceções, invariantes).
- **I - Interface Segregation**: interfaces pequenas e focadas (`SentimentProvider`, `PriceProvider`, `NewsFetcher`) em vez de um mega-`IService`.
- **D - Dependency Inversion**: `services/` recebe abstrações no construtor (`def __init__(self, prices: PriceProvider, ...)`), nunca instancia clientes concretos. Wire em `infra/container.py` ou via FastAPI `Depends`.

### Clean Code (regras curtas)

- Nomes revelam intenção - `calculate_rsi(prices)`, não `calc(p)`.
- Funções **curtas** (≤ 20 linhas idealmente) e **um nível de abstração** por função.
- Zero numeros/strings magicos - constantes em modulo de configuracao (`backend/app/infra/config.py`) ou enums.
- **Sem comentários redundantes**. Comentário só para explicar **por quê**, nunca **o quê**.
- `None`/`null` defensivo apenas nas bordas (entrada do usuário, APIs externas). Dentro do domínio, valores válidos são garantidos por tipos.
- Early return em vez de `if/else` aninhado.
- Imutabilidade por padrão (`dataclass(frozen=True)`, `readonly`, `const`).

### TDD / testes

- Pirâmide: muitos unit, alguns integração, poucos E2E.
- **Não** mockar o banco em testes de integração - usar `testcontainers-python` com Postgres real.
- Testes são código de produção: nomes descritivos (`test_predict_returns_4xx_when_ticker_unknown`), sem lógica condicional dentro do teste.
- Cobertura alvo: backend ≥ 70 %, frontend ≥ 60 %, ML core ≥ 80 %.

### 12-Factor App

- Configuração **exclusivamente** via variáveis de ambiente (`pydantic-settings`). Nada hardcoded.
- Logs em `stdout` (structured JSON), nunca em arquivo dentro do container.
- Processos stateless - estado vai para Postgres / Redis / volume.
- Parity dev/prod - mesma imagem Docker, só muda env.

### Git / PRs

- Commits no imperativo: "add rate limiting to predict endpoint".
- PRs pequenos (≤ 400 LOC diff). Se maior, quebrar.
- Conventional Commits opcional mas encorajado (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`).

---

## 2. OWASP Top 10 (2021) - Checklist obrigatório

Antes de fechar qualquer PR, o agente **deve** validar cada item aplicável:

| # | Categoria | Ação no SPP |
|---|-----------|-------------|
| A01 | Broken Access Control | Autenticação JWT nas rotas sensíveis; middleware valida escopo; nunca confiar em IDs do cliente - sempre re-validar ownership no servidor. |
| A02 | Cryptographic Failures | TLS em todas as pontas (Cloudflare Tunnel); segredos em `.env`/vault, **nunca** no repositório; hashes fortes (`argon2`/`bcrypt`) se houver senhas. |
| A03 | Injection | **Zero** f-strings/concatenação em SQL - usar SQLAlchemy parametrizado. Validar entrada com Pydantic/Zod. Sanitizar entrada do usuário antes de passar ao Claude SDK (prompt injection). |
| A04 | Insecure Design | Threat-model documentado em `docs/security.md` (local, fora do git); rate-limit por IP + por user; quotas por ticker. |
| A05 | Security Misconfiguration | CORS restrito (`allow_origins=[FRONTEND_URL]`); `DEBUG=False` em produção; headers de segurança (helmet/secure-headers); remover banners de versão. |
| A06 | Vulnerable Components | `pip-audit`, `npm audit`, `trivy image` no CI. Dependabot/Renovate habilitado. Fixar versões em lockfile. |
| A07 | Identification & Auth | JWT com expiração curta (15 min) + refresh token; rate-limit em `/login`; 2FA opcional. |
| A08 | Software/Data Integrity | Verificar checksums em downloads; assinar imagens Docker (cosign, opcional); CI bloqueia merge sem code review. |
| A09 | Logging & Monitoring | Log estruturado JSON; request-id em toda requisição; alertas em `5xx > 1%`; nunca logar PII, tokens ou chaves. |
| A10 | SSRF | URLs externas (BrAPI, RSS) em allowlist; httpx com timeout + limite de redirect; bloquear IPs privados (169.254.x.x, 10.x, etc.) em qualquer fetch iniciado por input do usuário. |

### Regras extras específicas de IA

- **Prompt injection**: notícias vindas de RSS são **dados não confiáveis**. Ao montar prompt para Claude, delimitar com tags (`<article>...</article>`) e instruir o modelo a tratar o conteúdo como dado, nunca como instrução.
- **PII em logs do Claude**: não enviar dados pessoais; redact emails/CPFs se aparecerem em notícias (regex).
- **Rate-limit custo**: limitar chamadas Claude por minuto/usuário para evitar bill explosion.

---

## 3. Padrões específicos do SPP

### Python (backend)

- Python **3.12+**, `ruff` (lint + format), `mypy --strict`, `pytest`, `httpx`, `pydantic-settings`.
- Async em endpoints e I/O externo. Código síncrono (TensorFlow) roda em `asyncio.to_thread` ou no worker Celery.
- `structlog` para logs.
- Exceções do domínio explícitas (`TickerNotFoundError`, `InsufficientDataError`) - nada de `raise Exception("...")`.

### TypeScript (frontend)

- `strict: true` no `tsconfig`.
- Zero `any` - use `unknown` + narrowing.
- Schemas Zod em `lib/schemas/` **espelham** o Swagger do backend. Gerar types com `openapi-typescript` quando possível.
- Componentes shadcn/ui ficam em `components/ui/` e não são editados manualmente (use variants).

### Docker

- Multi-stage sempre.
- Usuário não-root (`USER app`).
- `HEALTHCHECK` em todas as imagens.
- `.dockerignore` cobre `node_modules`, `.git`, `__pycache__`, `.venv`, `*.keras`, `*.csv`.

---

## 4. Como usar o aiox-core

O projeto usa `aiox-core` como core agêntico. Comandos úteis dentro do Claude Code:

- `/aiox-menu` - menu interativo
- `@analyst` - levantamento de requisitos, PRD
- `@pm` - quebra de épicos em histórias
- `@architect` - decisões arquiteturais, ADRs
- `@dev` - implementação
- `@qa` - testes e validação
- `@devops` - Docker, CI/CD, deploy
- `@data-engineer` - schema, migrations, pipelines
- `@sm` - coordenação entre squads

**Fluxo recomendado** para qualquer nova funcionalidade:

1. `@analyst` → requisito claro
2. `@architect` → ADR se impacto arquitetural
3. `@pm` → história com critérios de aceite
4. `@dev` → implementação em PR pequeno
5. `@qa` → testes + OWASP checklist
6. `@devops` → só se tocar infra

---

## 5. Anti-padrões proibidos neste projeto

- Salvar segredos no código ou no `git`.
- Queries SQL com concatenação de strings.
- `except Exception: pass` (engolir erros).
- Commit de arquivos `.csv`, `.keras`, `.env` (exceto `.env.example`).
- Funções com mais de 3 níveis de indentação.
- Comentários que apenas reescrevem o que o código já diz.
- Mockar o banco em testes de integração.
- Lógica de negócio dentro de routers/controllers.
- `fetch` direto em componentes React - passar por `lib/api/` + TanStack Query.
- `console.log` / `print` em produção (usar logger estruturado).

---

## 6. Referencias rapidas

> A pasta `docs/` e mantida apenas localmente (fora do controle de versao).

- `docs/adr/` - decisoes arquiteturais (local)
- `docs/security.md` - threat model (local)
- `docs/status.md` - status das fases (local)

---

*Siga este documento. Em caso de dúvida, perguntar ao `@architect` antes de desviar.*
