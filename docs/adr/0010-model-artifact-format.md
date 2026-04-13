# ADR 0010: Formato de artefato do modelo LSTM (.keras + .aux.joblib)

Status: accepted
Date: 2026-04-13
Related: 0001 (backend stack), 0009 (tecnicas avancadas)

## Contexto

Descoberto ao final da v3.1, durante a primeira sessao real de treinamento: carregar um `.keras` treinado via `LSTMPricePredictor.load()` e chamar `predict()` crashava com `RuntimeError: Model is not trained or loaded`.

Causa raiz: o formato nativo `.keras` do Keras salva apenas **arquitetura + pesos**. O `LSTMPricePredictor` precisa tambem de:

- `_scaler_features` (`MinMaxScaler` do sklearn) ajustado ao range das features tecnicas no treino.
- `_scaler_target` (`MinMaxScaler`) ajustado ao range do preco alvo.
- `_feature_columns` (lista de strings) para garantir que a ordem das colunas no `predict` bate com a ordem vista no `train`.

Sem esses tres artefatos o modelo carregado conhece a topologia mas nao sabe como normalizar entrada nem como inverter a saida.

Alternativas consideradas:

1. **Embutir scalers como camadas Keras** (`Normalization` layer). Rejeitada: exige refatorar o pipeline, quebra compatibilidade com modelos ja treinados, e `MinMaxScaler` do sklearn nao tem equivalente 1:1 em Keras.
2. **Recomputar scalers on-the-fly a cada predict** a partir do historico. Rejeitada: resultado nao-determinstico (depende da janela carregada), introduz drift entre treino e inferencia.
3. **Serializar tudo num unico pickle do `LSTMPricePredictor`**. Rejeitada: perde a portabilidade do `.keras` (ferramentas externas, Netron, TF Serving).
4. **Arquivo companion joblib ao lado do `.keras`** (adotado).

## Decisao

Adicionar ao `LSTMPricePredictor`:

```python
def save(self, path):
    self.model.save(str(path))                     # ~1.7 MB
    joblib.dump(                                    # ~2.5 KB
        {
            "feature_columns": self._feature_columns,
            "scaler_features": self._scaler_features,
            "scaler_target":   self._scaler_target,
            "config":          self.config,
            "feature_config":  self.feature_config,
        },
        Path(str(path) + ".aux.joblib"),
    )

def load(self, path):
    self.model = keras.models.load_model(str(path))
    aux_path = Path(str(path) + ".aux.joblib")
    if aux_path.exists():
        aux = joblib.load(aux_path)
        self._feature_columns  = aux["feature_columns"]
        self._scaler_features  = aux["scaler_features"]
        self._scaler_target    = aux["scaler_target"]
        # ... config/feature_config opcionais
```

`predict()` tambem foi endurecido para **tolerar features externas ausentes** no DataFrame de inferencia (sentimento, macro, fx, market) preenchendo com zero - a mesma convencao usada no `TrainingOrchestrator` quando `_safe_signal` devolve 0.0.

## Consequencias

Positivas:
- Um treino produz dois arquivos ao lado: `PETR4.keras` + `PETR4.keras.aux.joblib`. Deploy simples via volume Docker.
- Retrocompatibilidade: se o `.aux.joblib` nao existe (modelo pre-v3.1), `load()` ainda carrega a rede; `predict()` falhara explicitamente com `Model is not trained or loaded` ate o proximo treino regenerar o aux.
- Nenhuma dependencia nova: `joblib` ja vem como transitiva do sklearn e agora esta listado explicitamente em `requirements.txt`.
- Formato `.keras` continua inspetivel por ferramentas padrao.

Negativas:
- Dois arquivos em vez de um - precisa copiar os dois juntos em qualquer pipeline de deploy.
- `joblib.dump` usa pickle internamente: nao carregar `.aux.joblib` de fontes nao confiaveis (mesma regra de qualquer artefato sklearn persistido).

## Rastreabilidade

- Backend: `backend/app/ml/lstm_model.py` (`save`, `load`, `predict` endurecido)
- Deps: `backend/requirements.txt` (`joblib==1.4.2` explicito)
- Docs: `docs/TRAINING_WINDOWS.md` (secao 5 "Publicar o modelo" e troubleshooting)
- Rastreamento da correcao: fase 17 em `docs/status.md`
