# SPP - Sistema de Predição Preços

![Diagrama do sistema](https://codde.dev/spp.png 'Arquitetura do Sistema')

É uma plataforma de análise preditiva que combina modelos de séries temporais (LSTM) com análise de sentimento de notícias financeiras para antecipar movimentos de mercado e variações no preço de ações.

Inspirado em arquiteturas avançadas de sistemas de investimento, como o Aladdin da BlackRock, o projeto busca simular estratégias inteligentes de decisão financeira, integrando múltiplas fontes de dados, histórico de preços, volume e conteúdo noticioso e para aprimorar a acurácia das previsões.

Este sistema pode servir como base modular para soluções mais complexas de trading automatizado, gestão de risco e análise de portfólio com integração de IA.

## Stacks e Tecnologias

Python 3.12+ | TensorFlow 2.16+ | LSTM + NLP (Gemini API)

---

## ⚡ Quick Start

```bash
# 1. Ambiente
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Instalar
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configurar
echo "GEMINI_API_KEY=sua_chave" > .env
# Obter chave: https://makersuite.google.com/app/apikey

# 4. Verificar
python check_environment.py

# 5. Executar
python main.py
```

---

## 📋 Requisitos

- **Python 3.12+**
- **15+ dias** de dados históricos
- **Gemini API Key** (gratuita)
- Arquivos CSV: preços + notícias

---

## 🏗️ Arquitetura

```
CSV Preços + CSV Notícias
         ↓
    Gemini API (Sentimento)
         ↓
  Feature Engineering (35+ features)
   RSI, MACD, Bollinger, MAs
         ↓
    Modelo LSTM (3 camadas)
    128 → 64 → 32 → Dense
         ↓
   Predição Preço Futuro
```

---

## 📦 Estrutura

```
├── main.py                  # Pipeline completo
├── predictor.py             # Predições rápidas
├── check_environment.py     # Verificar instalação
├── test_gpu.py             # Testar GPUs
│
├── config.py               # Configurações
├── data_loader.py          # Carregar dados
├── sentiment_analyzer.py   # Gemini API
├── feature_engine.py       # Indicadores técnicos
├── model.py                # LSTM
├── gpu_manager.py          # Gerenciar GPU
│
├── requirements.txt        # Dependências
├── .env                    # API keys
└── .gitignore
```

---

## ⚙️ Configuração

### config.py

```python
# APIs
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

# Dados
STOCK_DATA_FILE: str = "PETR4_2025_10_01_to_2025_10_25.csv"
NEWS_DATA_FILE: str = "noticias_PETR4_setembro_2025.csv"

# Modelo
SEQUENCE_LENGTH: int = 5      # Dias de histórico
TRAIN_TEST_SPLIT: float = 0.7 # 70% treino
BATCH_SIZE: int = 4
EPOCHS: int = 50
LEARNING_RATE: float = 0.001

# GPU
GPU_ID: int = -1  # -1=CPU, 0=GPU0, 1=GPU1, None=todas
GPU_MEMORY_GROWTH: bool = True
```

---

## 🚀 Uso

### Pipeline Completo

```bash
python main.py
```

**Faz:**

1. Carrega dados (preços + notícias)
2. Analisa sentimento (Gemini API com cache)
3. Cria 35+ features técnicas
4. Treina modelo LSTM
5. Gera predição

**Tempo:** ~1-2 min (com cache) | ~5-10 min (primeira vez)

### Predições Rápidas

```bash
# Prever próximos 5 dias
python predictor.py --mode predict --days 5

# Analisar performance
python predictor.py --mode analyze
```

### Verificar Ambiente

```bash
python check_environment.py
```

### Testar GPU

```bash
python test_gpu.py
```

---

## 📊 Features Técnicas

| Categoria        | Features                      |
| ---------------- | ----------------------------- |
| **Tendência**    | MA5, MA10, MA20, MACD         |
| **Momentum**     | RSI, Momentum                 |
| **Volatilidade** | Bollinger Bands, Volatilidade |
| **Volume**       | Volume Ratio, Volume MA       |
| **Sentimento**   | Score (-1/0/1), Confidence    |

Total: **35+ features**

---

## 🔧 Resolução de Problemas

### Rate Limit 429 (Gemini API)

**Solução:** Sistema usa cache automático em `sentiment_cache.json`

```bash
# Executar novamente usa cache
python main.py
```

Ou pular sentimento:

```python
# main.py linha 252
df = system.run_pipeline(skip_sentiment=True)
```

### Dataset Pequeno

**Mínimo:** 15 dias (SEQUENCE_LENGTH × 3)

**Solução:** Reduzir em config.py:

```python
SEQUENCE_LENGTH: int = 3
```

### GPU não detectada

**Verificar:**

```bash
nvidia-smi
nvcc --version
python test_gpu.py
```

**Usar CPU:**

```python
# config.py
GPU_ID: int = -1
```

### Erro batch_outputs

**Já corrigido na v2.1** - Batch size adaptativo

---

## 🎮 GPU

### Configurar GPU

```python
# config.py
GPU_ID: int = 0     # Usar GPU 0
GPU_ID: int = 1     # Usar GPU 1
GPU_ID: int = None  # Usar todas
GPU_ID: int = -1    # Usar CPU
```

### Instalar CUDA (se necessário)

```bash
# Verificar driver
nvidia-smi

# Instalar bibliotecas CUDA
pip install tensorflow[and-cuda]

# Testar
python test_gpu.py
```

---

## 📈 Outputs

### Modelo Treinado

- `petr4_lstm_model.keras` - Modelo salvo

### Datasets

- `dataset_PETR4_normalizado_completo.csv` - Features processadas
- `sentiment_cache.json` - Cache de sentimentos

### Predições

- `predictions.csv` - Predições multi-dia

### Métricas

- RMSE, MAE, MAPE
- Direction Accuracy (% acerto de direção)

---

## 🔑 Variáveis de Ambiente

```bash
# .env
GEMINI_API_KEY=sua_chave_aqui
NEWS_API_KEY=sua_chave_newsapi  # opcional
```

---

## 📝 Comandos Úteis

```bash
# Verificar instalação
python check_environment.py

# Pipeline completo
python main.py

# Predições (modelo treinado)
python predictor.py --days 5

# Testar GPUs
python test_gpu.py

# Limpar cache
rm sentiment_cache.json

# Desativar venv
deactivate
```

---

## 🎯 Hiperparâmetros

### Datasets Pequenos (<20 dias)

```python
SEQUENCE_LENGTH: int = 3
TRAIN_TEST_SPLIT: float = 0.6
BATCH_SIZE: int = 2
```

### Datasets Médios (20-50 dias)

```python
SEQUENCE_LENGTH: int = 5  # padrão
TRAIN_TEST_SPLIT: float = 0.7
BATCH_SIZE: int = 4
```

### Datasets Grandes (50+ dias)

```python
SEQUENCE_LENGTH: int = 10
TRAIN_TEST_SPLIT: float = 0.8
BATCH_SIZE: int = 16
```

---

## 📊 Performance

| Métrica       | Bom  | Aceitável | Ruim |
| ------------- | ---- | --------- | ---- |
| MAPE          | <2%  | 2-5%      | >5%  |
| Direction Acc | >70% | 60-70%    | <60% |
| RMSE          | <0.5 | 0.5-1.0   | >1.0 |

---

## 🔄 Workflow

### Primeira Execução

```bash
python main.py
# → Análise completa (5-10 min)
# → Cria cache
# → Treina modelo
```

### Execuções Seguintes

```bash
python main.py
# → Usa cache (1-2 min)
# → Retreina se necessário
```

### Apenas Predições

```bash
python predictor.py --days 5
# → Usa modelo existente (<1 min)
```

---

## 📚 Dependências

```
pandas >= 2.2.0
numpy >= 1.26.0
tensorflow >= 2.16.1
scikit-learn >= 1.4.0
requests >= 2.31.0
python-dotenv >= 1.0.0
```

---

## ⚠️ Notas Importantes

- **Não é recomendação financeira**
- Sistema para **fins educacionais**
- **Rate limit:** 15 req/min (Gemini free tier)
- **Cache** economiza quota da API
- **GPU opcional** mas acelera treino

---

## 🆘 Suporte

### Checklist

- [ ] Python 3.12+?
- [ ] requirements.txt instalado?
- [ ] .env com GEMINI_API_KEY?
- [ ] CSVs de dados presentes?
- [ ] check_environment.py passou?

### Logs

```bash
# Ver erros
python main.py 2>&1 | tee error.log
```

---

## 📞 Links

- Gemini API: https://makersuite.google.com/app/apikey
- TensorFlow GPU: https://www.tensorflow.org/install/gpu
- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

---

**Versão:** 2.1  
**Python:** 3.12+  
**Status:** ✅ Production Ready  
**Licença:** MIT

## 📋 Visão Geral

Este sistema combina:

- **Análise técnica**: Indicadores como RSI, MACD, Bandas de Bollinger, médias móveis
- **Análise de sentimento**: Processamento de notícias financeiras usando Google Gemini API
- **Deep Learning**: Rede LSTM (Long Short-Term Memory) para predição temporal
- **Pipeline automatizado**: Da coleta de dados até a predição final

## 🏗️ Arquitetura

```
┌──────────────────┐
│  Dados Históricos│
│   (CSV Preços)   │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  Dados Notícias │─────▶│  Gemini API      │
│   (CSV News)    │      │  (Sentiment)     │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         ▼                        ▼
┌────────────────────────────────────┐
│      Feature Engineering           │
│  • Indicadores Técnicos            │
│  • Scores de Sentimento            │
│  • Médias Móveis                   │
│  • RSI, MACD, Bollinger            │
└────────────────┬───────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  Modelo LSTM  │
         │  (TensorFlow) │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │   Predição    │
         │  Preço Futuro │
         └───────────────┘
```

## 📦 Estrutura do Projeto

```
.
├── main.py                 # Orquestrador principal
├── config.py               # Configurações do sistema
├── data_loader.py          # Carregamento de dados
├── sentiment_analyzer.py   # Análise de sentimento (Gemini)
├── feature_engine.py       # Feature engineering
├── model.py                # Modelo LSTM
├── predictor.py            # Script de predição
├── newsApi.py              # Coleta de notícias (opcional)
├── requirements.txt        # Dependências Python
├── .env                    # Variáveis de ambiente
└── README.md               # Esta documentação
```

## 🚀 Instalação

### 1. Clonar repositório (ou baixar arquivos)

### 2. Criar ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

Criar arquivo `.env`:

```env
GEMINI_API_KEY=sua_chave_gemini_aqui
NEWS_API_KEY=sua_chave_newsapi_aqui  # opcional
```

**Como obter chave Gemini:**

1. Acesse: https://makersuite.google.com/app/apikey
2. Crie uma API key gratuita
3. Cole no arquivo `.env`

### 5. Adicionar arquivos de dados

Coloque na raiz do projeto:

- `PETR4_2025_10_01_to_2025_10_25.csv` (dados históricos)
- `noticias_PETR4_setembro_2025.csv` (notícias)

## 💻 Uso

### Modo 1: Pipeline Completo (Treinar + Prever)

```bash
python main.py
```

**O que acontece:**

1. ✅ Carrega dados históricos e notícias
2. 🧠 Analisa sentimento com Gemini API
3. ⚙️ Cria features técnicas
4. 🎯 Treina modelo LSTM
5. 🔮 Faz predição para próximo dia
6. 💾 Salva modelo treinado

**Output esperado:**

```
============================================================
🚀 SISTEMA DE PREDIÇÃO DE INVESTIMENTOS - PETR4
============================================================

📂 ETAPA 1: Carregamento de Dados
------------------------------------------------------------
✅ 18 dias de dados carregados
✅ 25 notícias carregadas
✅ Dataset mesclado: 18 registros

🧠 ETAPA 2: Análise de Sentimento (Gemini API)
------------------------------------------------------------
Analisando sentimento 1/18
...
✅ Sentimentos analisados: 18
   Positivos: 5
   Neutros: 8
   Negativos: 5

⚙️  ETAPA 3: Feature Engineering
------------------------------------------------------------
✅ Médias móveis adicionadas: [5, 10, 20]
✅ RSI adicionado
✅ MACD adicionado
✅ Bandas de Bollinger adicionadas
...

🎯 TREINAMENTO DO MODELO
------------------------------------------------------------
Época 1/50: loss: 0.0234 - val_loss: 0.0198
...

📊 Métricas de Avaliação:
  RMSE: 0.42
  MAE: 0.35
  Direction_Accuracy: 75.00%

🔮 PREDIÇÃO PARA PRÓXIMO DIA
------------------------------------------------------------
📅 Última data: 2025-10-24
💰 Último preço: R$ 29.84

🔮 Predição para 2025-10-25:
   Preço: R$ 30.15
   Variação: R$ 0.31 (+1.04%)
   Tendência: ALTA ⬆️
```

### Modo 2: Apenas Predição (Modelo já treinado)

```bash
# Prever próximos 5 dias
python predictor.py --mode predict --days 5

# Analisar performance do modelo
python predictor.py --mode analyze
```

### Modo 3: Treino Rápido (sem análise de sentimento)

Editar `main.py`:

```python
# Linha ~181
df = system.run_pipeline(skip_sentiment=True)  # Pular Gemini API
```

## 📊 Features Técnicas Implementadas

| Feature             | Descrição                                        |
| ------------------- | ------------------------------------------------ |
| **MA5, MA10, MA20** | Médias móveis simples                            |
| **RSI**             | Relative Strength Index (sobrecompra/sobrevenda) |
| **MACD**            | Moving Average Convergence Divergence            |
| **Bollinger Bands** | Bandas superior/inferior e %B                    |
| **Volatilidade**    | Desvio padrão dos retornos                       |
| **Momentum**        | Taxa de mudança de preço                         |
| **Volume Ratio**    | Volume atual vs. média                           |
| **Sentiment**       | Score de sentimento das notícias (-1, 0, 1)      |

## 🧠 Modelo LSTM

**Arquitetura:**

```
LSTM(128) → Dropout(0.2)
    ↓
LSTM(64) → Dropout(0.2)
    ↓
LSTM(32) → Dropout(0.2)
    ↓
Dense(16, relu) → Dropout(0.2)
    ↓
Dense(1) [Saída]
```

**Parâmetros (config.py):**

- Sequência de entrada: 10 dias
- Épocas: 50 (com early stopping)
- Batch size: 16
- Learning rate: 0.001

## 📈 Métricas de Avaliação

- **RMSE**: Erro quadrático médio
- **MAE**: Erro absoluto médio
- **MAPE**: Erro percentual médio
- **Direction Accuracy**: Acurácia na previsão de direção (subida/descida)

## 🔧 Configuração Avançada

Editar `config.py`:

```python
@dataclass
class Config:
    # Modelo
    SEQUENCE_LENGTH: int = 10      # Dias de histórico
    EPOCHS: int = 50               # Épocas de treino
    BATCH_SIZE: int = 16           # Tamanho do batch
    LEARNING_RATE: float = 0.001   # Taxa de aprendizado

    # Features
    MA_PERIODS: list = [5, 10, 20] # Períodos de MA
    RSI_PERIOD: int = 14           # Período do RSI
    BOLLINGER_PERIOD: int = 20     # Período Bollinger
```

## 🐛 Troubleshooting

### Erro: "GEMINI_API_KEY não encontrada"

**Solução:** Criar arquivo `.env` com sua chave da API

### Erro: "Arquivo de ações não encontrado"

**Solução:** Verificar se os CSVs estão na raiz do projeto

### Predições sempre neutras

**Solução:** Modelo precisa de mais dados de treino ou ajuste de hiperparâmetros

### Erro de memória no TensorFlow

**Solução:** Reduzir `BATCH_SIZE` ou `SEQUENCE_LENGTH` em `config.py`

## 📝 Coleta de Novas Notícias (Opcional)

Para coletar notícias atualizadas:

```bash
# 1. Obter chave NewsAPI: https://newsapi.org/register
# 2. Adicionar ao .env: NEWS_API_KEY=sua_chave
# 3. Executar:
python newsApi.py
```

## ⚠️ Avisos Importantes

1. **Não é recomendação financeira**: Sistema para fins educacionais
2. **Rate limits**: Gemini API tem limites de requisições
3. **Dados históricos**: Performance depende da qualidade dos dados
4. **Validação temporal**: Nunca usar dados futuros no treino

## 📄 Licença

MIT License - Uso livre para fins educacionais
