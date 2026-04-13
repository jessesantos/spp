# Treinamento LSTM no Windows (com GPU via WSL2)

> Este documento descreve como treinar o **modelo LSTM** do SPP aproveitando a **NVIDIA GeForce RTX 5070 12 GB** (arquitetura Blackwell, compute capability 12.0 / sm_120) rodando em WSL2 a partir de uma máquina Windows. O backend FastAPI continua rodando no mesmo WSL2 via `docker compose`; o treinamento pesado executa **dentro de uma venv WSL2 nativa** que recebe a GPU via driver NVIDIA Windows (sem NVIDIA Container Toolkit).

> **v3.1 (ADR 0009):** a janela de treino passou de 1 para **3+ anos**. Importante: BrAPI e yfinance aceitam apenas `2y`, `5y` ou `max` como ranges nativos; o CLI usa `--period 5y` (cobre 5 anos, mais do que suficiente para ciclo Copom completo). Features adicionais: `cond_vol` (EWMA RiskMetrics), `fx_score` (correlação + heurística BRL/USD), `market_signal_score` (Kalshi + Polymarket).

> **Blackwell sm_120 (RTX 5070):** o wheel oficial `tensorflow[and-cuda]>=2.18,<2.20` **ainda não traz kernels pre-compilados** para compute capability 12.0. O fallback PTX JIT falha com `CUDA_ERROR_INVALID_PTX`. Caminhos válidos hoje: (a) `CUDA_VISIBLE_DEVICES=-1` e treinar em CPU (20-30s por ticker, 50 épocas, 5y daily), ou (b) `pip install --upgrade "tf-nightly[and-cuda]"` que periodicamente recebe suporte Blackwell. Sem impacto em produção: treinar em CPU, servir em qualquer host.

> ⚠️ **Windows nativo não é mais suportado pelo TensorFlow com GPU.** Desde TF 2.11, o Google parou de publicar wheels CUDA para Windows; o extra `pip install tensorflow[and-cuda]` **só existe no PyPI Linux**. Em Windows puro você obterá `ERROR: Could not find a version that satisfies the requirement tensorflow[and-cuda]`. A rota abaixo usa WSL2, onde as wheels Linux do TF reconhecem a RTX 5070 via driver NVIDIA do host.

---

## Por que treinar em WSL2 e não em Docker nem Windows nativo?

| Limitação | Motivo |
|---|---|
| TensorFlow no Windows nativo sem GPU | Google descontinuou wheels CUDA para Windows a partir de TF 2.11. `pip install tensorflow[and-cuda]` retorna `No matching distribution`; só CPU roda. |
| TensorFlow 2.16 em container não reconhece sm_120 | Blackwell só é suportado a partir de **TF 2.18+** com **CUDA ≥ 12.5** ou wheels nightly. O container do projeto usa TF 2.16 (CPU), adequado ao fallback mas não para GPU. |
| Docker sem NVIDIA Container Toolkit no WSL2 | `docker info` não lista runtime `nvidia`; erro `failed to discover GPU vendor from CDI`. |
| CUDA 13.2 do driver 595.97 é recente demais | Imagens oficiais `nvidia/cuda` estáveis ainda estão em 12.x. |

Solução pragmática: **treinar em uma venv WSL2 nativa** (não dentro do container), usando as wheels Linux com `tensorflow[and-cuda]`. A GPU é entregue via driver NVIDIA Windows + WSL2 sem NVIDIA Container Toolkit. Depois **servir** via `docker compose` no mesmo WSL2 com o `.keras` resultante montado em `./models/`.

---

## Pré-requisitos

1. **Windows 11** com WSL2 ativo e distribuição **Ubuntu 22.04 ou 24.04** instalada (`wsl --install -d Ubuntu-24.04`).
2. **NVIDIA Driver ≥ 571.xx** instalado no Windows (o 595.97 já habilita passthrough WSL2 sem config extra).
3. **Python 3.12** dentro do WSL2 (`sudo apt install python3.12 python3.12-venv`).
4. **Git** no WSL2 (`sudo apt install git`).
5. (opcional) Docker + Docker Compose dentro do WSL2 para servir o modelo.

> Não instale Python/CUDA/cuDNN no Windows nativo para esse fluxo. Toda a stack de treino vive no WSL2.

### Verificar que a GPU está visível no WSL2

No terminal **Ubuntu** (WSL2):

```bash
nvidia-smi
```

Deve listar `NVIDIA GeForce RTX 5070` com driver 595.x+ e CUDA 13.x, servido pelo driver Windows. Se o comando não existe dentro do WSL2, atualize o driver NVIDIA do Windows (>= 555) e reabra o shell.

---

## 1. Clonar e preparar ambiente (dentro do WSL2)

```bash
# Fora de /mnt/c para evitar I/O lento
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/jessesantos/spp.git
cd spp/backend
```

### Criar venv e instalar dependências

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# TensorFlow com suporte CUDA (wheel Linux; necessario TF 2.18+ para Blackwell sm_120)
pip install "tensorflow[and-cuda]>=2.18,<2.20"

# Demais libs usadas pelo treinamento
pip install pandas==2.2.3 numpy==1.26.4 scikit-learn==1.5.2 \
  anthropic==0.39.0 httpx==0.27.2 feedparser==6.0.11 \
  pydantic==2.9.2 pydantic-settings==2.6.0 python-dotenv==1.0.1 \
  yfinance==0.2.48 structlog==24.4.0
```

> **Se o wheel `2.18.x` ainda não reconhecer sm_120 quando você rodar**, caia nas nightly (mesmo pacote, mesmo extra):
> ```bash
> pip install --upgrade "tf-nightly[and-cuda]"
> ```

### Verificar que o TF enxerga a RTX 5070

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Saída esperada:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Se aparecer `[]`, o TF ainda não suporta sm_120 no wheel atual - use `tf-nightly[and-cuda]` conforme acima, confirme `nvidia-smi` dentro do WSL2 e garanta que o driver Windows foi atualizado.

---

## 2. Variáveis de ambiente

Crie `~/projects/spp/backend/.env` (no WSL2) com o mínimo:

> ⚠️ A senha do Postgres no `docker-compose.yml` é `${POSTGRES_PASSWORD:-spp}`. Em um ambiente de desenvolvimento limpo (sem `POSTGRES_PASSWORD` definido no `.env` da raiz), o valor efetivo e `spp`. Use isto no `DATABASE_URL` abaixo ate configurar algo mais seguro em producao.

```env
ENV=development
LOG_LEVEL=INFO
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-sonnet-4-5-20250929

# Para o script de treinamento local (sem Docker), apontamos para servicos
# disponiveis no WSL2 host via localhost (docker compose expoe 5432/6379).
# Senha default do docker-compose.yml e `spp` (POSTGRES_PASSWORD:-spp).
DATABASE_URL=postgresql+asyncpg://spp:spp@localhost/spp
REDIS_URL=redis://localhost:6379/0

# v3.1 (ADR 0009): janela default 3 anos e sinais externos opcionais
TRAINING_PERIOD_DEFAULT=3y
BACKFILL_LOOKBACK_DAYS=1095
KALSHI_BASE_URL=https://api.elections.kalshi.com/trade-api/v2
KALSHI_API_KEY=
POLYMARKET_BASE_URL=https://gamma-api.polymarket.com
PREDICTION_MARKETS_ENABLED=true

# Caminho do SKILL.md (Escola Austríaca + value + macro) injetado como
# system prompt no Claude. Em treino local WSL2, ajuste para o path real
# dentro do repositório clonado.
SENTIMENT_SKILL_PATH=/home/USUARIO/projects/spp/backend/app/ml/SKILL.md
```

> Como treino e serviço rodam no mesmo WSL2, `localhost` já resolve para os containers expostos.

---

## 3. Rodar a stack de serviço (à parte da venv de treino)

Em outro terminal **Ubuntu** (WSL2), fora da venv:

```bash
cd ~/projects/spp
cp .env.example .env         # se ainda não existir
docker compose up -d postgres redis backend frontend
```

Valide:

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"3.0.0"}
```

Frontend: http://localhost:3000

---

## 4. Treinar o modelo no WSL2

Com a venv ativada em `~/projects/spp/backend`:

```bash
# Treino padrao v3.1: 5 anos de historico, 50 epocas, sequence_length=5
# (BrAPI/yfinance nao aceitam "3y"; usar 5y cobre o periodo alvo)
python -m app.ml.train --ticker PETR4 --period 5y --epochs 50 --sequence-length 5

# Blackwell sm_120 sem kernels validos hoje: force CPU:
CUDA_VISIBLE_DEVICES=-1 python -m app.ml.train --ticker PETR4 --period 5y --epochs 50

# Desabilitar sinais externos quando sem ANTHROPIC_API_KEY / offline:
CUDA_VISIBLE_DEVICES=-1 python -m app.ml.train --ticker PETR4 --period 5y --no-sentiment --no-macro

# Parametros relevantes (ver `app/ml/train.py`):
#   --period            janela BrAPI/Yahoo. Valores aceitos: 1mo, 3mo, 6mo, 1y, 2y, 5y, max. Default: 5y
#   --epochs            iteracoes de treino. Default: 50
#   --sequence-length   comprimento da janela LSTM. Default: 5
#   --batch-size        mini-batch size. Default: 16
#   --no-sentiment      pula sentimento Claude (util sem API key)
#   --no-macro          pula contexto macro global (Reuters/BBC/FT/Bloomberg)
#   --output-dir        destino do artefato .keras (default: settings.models_dir)
```

Saida para cada ticker em `~/projects/spp/models/`:

- `{TICKER}.keras` (~1.7 MB) - rede Keras 128-64-32 com pesos
- `{TICKER}.keras.aux.joblib` (~2.5 KB) - scalers MinMax + lista de colunas de feature

Os dois arquivos sao **obrigatorios** em producao. `LSTMPricePredictor.save()` grava ambos; `load()` reconstitui `_scaler_features`, `_scaler_target` e `_feature_columns` a partir do `.aux.joblib`. Sem o aux, a predict crasha com `Model is not trained or loaded` mesmo com o `.keras` presente.

### Monitorar GPU durante treino

Em outro terminal WSL2:

```bash
watch -n 2 nvidia-smi
```

Você deve ver utilização em torno de 40-80 % e ~2-4 GB de VRAM durante o treino.

---

## 5. Publicar o modelo treinado no serviço

Como o treino ocorre no mesmo WSL2 onde o `docker compose` roda, o artefato já está em `~/projects/spp/models/` (a pasta é montada em `/app/models` dentro do container do backend via bind mount declarado no `docker-compose.yml`). Nenhuma cópia é necessária; a próxima predição reconhece o arquivo automaticamente.

Caso queira forçar um reload imediato:

```bash
docker compose restart backend
```

---

## 6. Workflow típico (dia-a-dia)

```text
┌─────────────── WSL2 / Ubuntu (mesma máquina Windows) ─────────────────┐
│                                                                       │
│  Terminal 1 (venv de treino)       Terminal 2 (stack de serviço)      │
│  ────────────────────────────      ──────────────────────────────     │
│  source .venv/bin/activate          docker compose up -d              │
│  nvidia-smi                         (postgres, redis, backend,        │
│  python -m app.ml.train \            frontend, worker, beat)          │
│    --ticker PETR4 --period 3y                                         │
│    --epochs 50                                                        │
│  ↓                                                                    │
│  ~/projects/spp/models/PETR4.keras                                    │
│  ↓ (bind mount ./models -> /app/models)                               │
│  docker compose restart backend (opcional)                            │
│  ↓                                                                    │
│  http://localhost:3000                                                │
└───────────────────────────────────────────────────────────────────────┘
```

> Sinais v3.1 (sentimento Claude com SKILL, macro, FX, Kalshi/Polymarket) são **opcionais** no treino offline. Sem `ANTHROPIC_API_KEY`, passe `--no-sentiment --no-macro`; sem rede, `PREDICTION_MARKETS_ENABLED=false` no `.env`. Features `cond_vol` e `fx_score` continuam sendo computadas localmente (não exigem API externa).

---

## 7. Troubleshooting

| Sintoma | Causa provável | Solução |
|---|---|---|
| `ERROR: Could not find a version that satisfies the requirement tensorflow[and-cuda]` no Windows nativo | TF nao publica wheel CUDA no PyPI Windows desde 2.11 | Rode o comando **dentro do WSL2 Ubuntu**, nao em PowerShell/CMD do Windows. |
| `CUDA_ERROR_INVALID_PTX` / `cuLaunchKernel failed` no Blackwell sm_120 | Wheel TF 2.19 sem kernels nativos sm_120 e PTX JIT incompativel | `CUDA_VISIBLE_DEVICES=-1 python -m app.ml.train ...` (CPU, 20-30s/ticker) OU `pip install --upgrade "tf-nightly[and-cuda]"`. |
| `password authentication failed for user "spp"` no treino | `.env` do backend tem senha diferente da do container Postgres | Use `DATABASE_URL=postgresql+asyncpg://spp:spp@localhost/spp` (docker-compose default) ou defina `POSTGRES_PASSWORD` no `.env` da raiz antes de subir. |
| BrAPI `400 Bad Request` no fetch com `range=3y` | BrAPI so aceita `1mo,3mo,6mo,1y,2y,5y,max` | Use `--period 5y`. O default do `BrAPIClient` ja foi atualizado para `5y`. |
| `RuntimeError: Model is not trained or loaded` apos carregar .keras | O companion `.aux.joblib` (scalers + colunas) nao existe ao lado | Garanta que `.keras` e `.keras.aux.joblib` estao ambos no `models_dir`. Retreine se faltar. |
| `KeyError: ['sentiment', ...] not in index` durante predict | Model treinado com features externas que o DataFrame de inferencia nao tem | `LSTMPricePredictor.predict()` ja preenche com 0 features ausentes; se ainda falhar, verifique que `feature_columns` em `.aux.joblib` bate com os nomes produzidos por `build_features`. |
| `list_physical_devices('GPU')` retorna `[]` dentro do WSL2 | Wheel TF sem sm_120 ou driver Windows antigo | `pip install --upgrade "tf-nightly[and-cuda]"`; atualize driver NVIDIA Windows (>=555) e rode `nvidia-smi` dentro do WSL2 para confirmar passthrough. |
| `Could not load dynamic library 'libcudart.so.*'` | CUDA runtime nao encontrado pelo TF | Reinstalar com `pip install "tensorflow[and-cuda]"`; `[and-cuda]` traz as bibliotecas. |
| `OOM when allocating tensor with shape [...]` | VRAM insuficiente | Reduza `--batch-size` (default 16 -> 8) ou `--sequence-length`. |
| Backend container nao ve modelo novo | Volume nao montado ou caminho errado | `docker compose config \| grep models` deve mostrar `./models:/app/models`. |
| `Unable to register cuDNN factory` (apenas warning) | Inofensivo na 2.18+ | Ignorar. |
| `nvidia-smi` retorna "command not found" dentro do WSL2 | Driver NVIDIA Windows antigo ou instalacao incompleta | Atualize o driver Windows (>=555); WSL2 expoe `libnvidia-ml` automaticamente. |
| `ModuleNotFoundError: joblib` ao carregar modelo | `joblib` nao instalado no backend (dep transitiva de sklearn mas explicitada em v3.1) | `pip install joblib==1.4.2`; ja listado em `backend/requirements.txt`. |

---

## 8. Alternativa A: habilitar GPU no Docker/WSL2 (avançado)

Se você quiser rodar TF com GPU dentro do container:

1. Instale o **NVIDIA Container Toolkit** no WSL2 Ubuntu:
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt update && sudo apt install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo service docker restart
   ```
2. Valide: `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`
3. Adicione no `docker-compose.yml` ao serviço `worker` (não ao API, que pode rodar CPU):
   ```yaml
   worker:
     # ...
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: all
               capabilities: [gpu]
   ```
4. Troque a base do `backend/Dockerfile` para `tensorflow/tensorflow:2.18.0-gpu` e remova a linha `tensorflow==2.16.1` do `requirements.txt`.

Essa rota funciona, mas exige **TF 2.18+ com wheel sm_120** (ainda nightly em alguns casos para Blackwell) **e** o NVIDIA Container Toolkit no WSL2. A rota venv-WSL2 acima é mais estável.

---

## 9. Alternativa B: Windows nativo CPU-only (sem GPU)

Se precisar treinar temporariamente sem GPU (máquina sem NVIDIA, ou para validar pipeline):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install tensorflow==2.18.0    # sem [and-cuda]: CPU only
pip install pandas==2.2.3 numpy==1.26.4 scikit-learn==1.5.2 ...
python -m app.ml.train --ticker PETR4 --period 3y --epochs 10
```

Esperado: ~10-30x mais lento que a RTX 5070. Útil apenas para smoke tests.

---

*Documento mantido em `docs/TRAINING_WINDOWS.md`. Atualize quando versões do TensorFlow estabilizarem suporte oficial a sm_120.*
