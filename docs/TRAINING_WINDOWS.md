# Treinamento LSTM no Windows (host nativo)

> Este documento descreve como executar o **treinamento do modelo LSTM** do SPP diretamente no Windows, aproveitando a **NVIDIA GeForce RTX 5070 12 GB** (arquitetura Blackwell, compute capability 12.0 / sm_120). O backend FastAPI continua rodando no WSL2 via `docker compose`; apenas o treinamento pesado é executado do lado Windows para usar a GPU.

---

## Por que treinar fora do Docker/WSL?

| Limitação | Motivo |
|---|---|
| TensorFlow 2.16 não reconhece sm_120 | Blackwell só é suportado a partir de **TF 2.18+** com **CUDA ≥ 12.5** ou wheels nightly |
| Docker sem NVIDIA Container Toolkit no WSL2 | `docker info` não lista runtime `nvidia`; erro `failed to discover GPU vendor from CDI` |
| CUDA 13.2 do driver 595.97 é recente demais | Imagens oficiais `nvidia/cuda` estáveis ainda estão em 12.x |

Solução pragmática: **treinar no Windows** (onde o driver + CUDA oficial rodam sem ginástica) e **servir** via Docker/WSL2 usando o artefato `.keras` resultante montado em `./models/`.

---

## Pré-requisitos no Windows

1. **Python 3.12** 64-bit (https://www.python.org/downloads/windows/). Marque "Add to PATH".
2. **NVIDIA Driver ≥ 571.xx** (o 595.97 já instalado é mais do que suficiente).
3. **CUDA Toolkit 12.5** ou superior - instale o runtime via `pip install tensorflow[and-cuda]` (abaixo); nenhuma instalação manual de CUDA/cuDNN é necessária nas versões recentes.
4. **Git for Windows** (opcional, para clonar o repo).
5. **PowerShell** (vem pronto).

### Verificar GPU no Windows

Em um PowerShell:

```powershell
nvidia-smi
```

Deve listar `NVIDIA GeForce RTX 5070` com driver 595.x+ e CUDA 13.x.

---

## 1. Clonar e preparar ambiente

```powershell
# Em C:\dev (ou onde preferir)
git clone https://github.com/jessesantos/spp.git
cd spp\backend
```

Se você não quer clonar: copie a pasta `backend\` deste repositório para o Windows (ex.: `C:\dev\spp\backend`).

### Criar venv e instalar dependências

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# TensorFlow com suporte CUDA (Blackwell sm_120 a partir da 2.18):
pip install "tensorflow[and-cuda]>=2.18,<2.20"

# Demais libs usadas pelo treinamento
pip install pandas==2.2.3 numpy==1.26.4 scikit-learn==1.5.2 anthropic==0.39.0 httpx==0.27.2 feedparser==6.0.11 pydantic==2.9.2 pydantic-settings==2.6.0 python-dotenv==1.0.1 yfinance==0.2.48 structlog==24.4.0
```

> **Se `tensorflow[and-cuda]` ainda não trouxer wheel sm_120 quando você executar**, caia nas nightly:
> ```powershell
> pip install --upgrade tf-nightly[and-cuda]
> ```

### Verificar que o TF enxerga a RTX 5070

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Saída esperada:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Se aparecer `[]`, a wheel do TF ainda não suporta sm_120 - use `tf-nightly` conforme acima.

---

## 2. Variáveis de ambiente

Crie `C:\dev\spp\.env` (ou `backend\.env`) com o mínimo:

```env
ENV=development
LOG_LEVEL=INFO
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-sonnet-4-5-20250929

# Para o script de treinamento local (sem Docker), apontamos para serviços
# disponíveis no WSL2 host. O WSL2 é alcançável do Windows via localhost
# quando docker compose expõe as portas 5432/6379:
DATABASE_URL=postgresql+asyncpg://spp:change_me_in_prod@localhost/spp
REDIS_URL=redis://localhost:6379/0
```

> O `docker compose` no WSL2 publica `5432` e `6379` em `localhost` do Windows automaticamente (WSL2 port forwarding).

---

## 3. Rodar a stack de serviço no WSL2 (à parte)

No **WSL2 Ubuntu**:

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

## 4. Treinar o modelo no Windows

Com a venv ativada em `C:\dev\spp\backend`:

```powershell
# Treinar PETR4 com 50 épocas (ajuste conforme seu dataset)
python -m app.ml.train --ticker PETR4 --epochs 50 --sequence-length 5

# Ou usando o script legado migrado:
python -m app.ml.lstm_model --train --ticker PETR4
```

O artefato é salvo em `C:\dev\spp\models\petr4_lstm_model.keras`.

### Monitorar GPU durante treino

Em outro PowerShell:

```powershell
nvidia-smi -l 2
```

Você deve ver utilização em torno de 40-80 % e ~2-4 GB de VRAM durante o treino.

---

## 5. Publicar o modelo treinado no serviço

Copie o artefato para a pasta `models/` do WSL2 (montada no container):

```powershell
# Do Windows, a pasta do WSL2 é acessível via \\wsl$\Ubuntu
Copy-Item .\..\models\petr4_lstm_model.keras `
  -Destination "\\wsl$\Ubuntu\home\jesse\projects\spp\models\"
```

Ou, rodando no WSL2:

```bash
cp /mnt/c/dev/spp/models/petr4_lstm_model.keras ~/projects/spp/models/
```

O `docker-compose.yml` monta `./models` em `/app/models` dentro do container do backend - o arquivo é reconhecido na próxima predição sem restart. Caso queira forçar:

```bash
docker compose restart backend
```

---

## 6. Workflow típico (dia-a-dia)

```text
┌─────────────── Windows ────────────────┐   ┌──────────── WSL2 ─────────────┐
│ 1. pull CSV/dados                      │   │                               │
│ 2. python -m app.ml.train --ticker X   │──▶│ models/  (bind mount)         │
│ 3. commit do modelo (opcional)         │   │ docker compose restart backend│
└────────────────────────────────────────┘   │ http://localhost:3000         │
                                             └───────────────────────────────┘
```

---

## 7. Troubleshooting

| Sintoma | Causa provável | Solução |
|---|---|---|
| `list_physical_devices('GPU')` retorna `[]` | Wheel TF sem sm_120 | `pip install --upgrade tf-nightly[and-cuda]` |
| `Could not load dynamic library 'cudart64_*.dll'` | CUDA runtime não encontrado | Reinstalar com `pip install "tensorflow[and-cuda]"` |
| `OOM when allocating tensor with shape [...]` | VRAM insuficiente | Reduza `BATCH_SIZE` em `config.py` ou use `SEQUENCE_LENGTH=5` |
| Backend (WSL) não vê modelo novo | Volume não montado ou caminho errado | Verifique `docker compose config` e o `-v ./models:/app/models` |
| `Unable to register cuDNN factory` (apenas warning) | Inofensivo na 2.18+ | Ignorar |

---

## 8. Alternativa: habilitar GPU no Docker/WSL2 (avançado)

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

Essa rota funciona, mas exige **TF 2.18+ com wheel sm_120** (ainda nightly em alguns casos para Blackwell). A rota Windows nativa acima é mais estável.

---

*Documento mantido em `docs/TRAINING_WINDOWS.md`. Atualize quando versões do TensorFlow estabilizarem suporte oficial a sm_120.*
