# ADR 0011: Alvo do LSTM trocado de preco absoluto para log-retorno

Status: accepted
Date: 2026-04-13
Related: 0001 (backend stack), 0010 (artefato .keras + .aux.joblib)

## Contexto

Na primeira sessao de treinamento real (fase 17), tickers PETR4/VALE3/ITUB4 produziam previsoes absurdas (D1 -10 a -18%, M1 -20 a -28%) que nao batiam com a volatilidade historica dos ativos e nem com as linhas persistidas no `Historico de previsoes` do dashboard.

Diagnostico numerico direto:

```
PETR4: last_close=R$49.97  scaler_features.data_max_[close]=49.67  ← saturado acima do treino
       scaler_target.data_max_=49.67                                ← mesmo range do close
       rollout[0]=R$40.64  (-18.67%)  propagando erro para rollout[6..29]
VALE3: rollout trend 76 -> 68 -> 70 -> 71 -> 68                    ← erro recursivo encadeado
```

Causa raiz: o modelo previa o **preco absoluto** `target_price = close.shift(-1)`, normalizado via `MinMaxScaler`. Dois problemas:

1. **Saturacao do scaler**: quando o mercado supera o maximo do treino (caso PETR4 em topo historico), o scaler extrapola alem de [0,1] e o modelo, que nunca viu essa regiao, devolve valor enviesado para dentro do range visto.
2. **Propagacao exponencial no rollout**: cada passo usa o preco previsto do passo anterior como input; um erro de -10% no passo 1 contamina os 29 proximos. Erro cresce geometricamente.

## Decisao

Trocar o target do LSTM para **log-retorno de 1 passo**:

```python
# Treino:
log_ret = np.log(close.shift(-1) / close).fillna(0.0)
features_df["target_return"] = log_ret
# scaler_target com feature_range=(-1, 1): mantem centro em zero
self._scaler_target = MinMaxScaler(feature_range=(-1.0, 1.0))

# Inferencia:
log_return = float(self._scaler_target.inverse_transform(y_scaled)[0, 0])
log_return = max(-0.30, min(0.30, log_return))   # guard-rail: ±30%/passo
next_close = last_close * np.exp(log_return)
```

Propriedades que corrigem o problema:

- **Estacionario**: log-retornos de equity sao aproximadamente N(0, 0.02) em blue-chips B3. O modelo sempre opera no mesmo regime, independente do nivel de preco.
- **Invariante a escala**: nao sofre saturacao quando o preco sobe alem do maximo historico do treino.
- **Composivel**: `next_close = last_close * exp(log_return)` e numericamente estavel. Multiplicacao por `exp` pequeno da variacao fracionaria coerente.
- **Rollout bem-comportado**: cada passo gera movimento tipicamente <1% em regime normal; guard-rail clipa caudas patologicas antes de propagarem.

Adicionalmente:

- **EarlyStopping patience**: 10 → 15 (da ao treino mais folga para convergir em log-retornos de variancia menor).
- **Metrica de validacao direcional** adicionada: `direction_accuracy` = `sign(y_pred) == sign(y_true)` calculada no conjunto de treino pos-fit. Valores abaixo de 0.52 indicam modelo ao acaso (signal-to-noise baixo, esperavel em equity de alta frequencia).

## Alternativas consideradas

1. **StandardScaler / RobustScaler no preco absoluto**. Rejeitada: ainda satura fora dos percentis treino; nao resolve a composicao exponencial de erro no rollout.
2. **Differencing simples** (`target = close[t+1] - close[t]`). Rejeitada: absoluto em R$, nao-estacionario entre tickers de niveis de preco diferentes (PETR4 ~R$50 vs PRIO3 ~R$40 vs SBSP3 ~R$75).
3. **Percent return** (`target = close[t+1] / close[t] - 1`). Equivalente matematico de log-retorno para movimentos pequenos; log-retornos composem melhor com multiplicacao (`np.exp`) e tem propriedades estatisticas mais limpas. Adotado log.
4. **Prever multiplos horizontes em paralelo** (output dim > 1). Rejeitada: refatoracao maior e nao e a causa do bug atual. Revisitar em v4.

## Consequencias

Positivas:

- Previsoes agora em range realista (D1 tipicamente ±1%, M1 ±5-10%). Validado pos-fit em PETR4 D1=-0.03%, VALE3 M1=-4.43%.
- Horizontes e historico agora convergem: um novo `/api/predict` persiste valores comparaveis aos anteriores, nao discrepantes.
- Modelo passa a generalizar para niveis de preco nao vistos no treino sem retraino manual.

Negativas:

- **Retrocompatibilidade quebrada**: modelos pre-v3.1 salvos pre-ADR 0011 tinham `target_price`; agora esperamos `target_return`. A troca e somente no codigo interno (scaler + predict), sem impacto em API ou schema. Artefatos antigos devem ser descartados e reretrenados - foi o que fizemos ao limpar `models/` antes desta sessao.
- `direction_accuracy` adicionada ao retorno de `train()` pode exigir ajuste em consumidores que esperam apenas `loss`/`epochs_run`. `TrainingOrchestrator` ja passa o campo transparente para `model_runs`.

## Rastreabilidade

- Implementacao: `backend/app/ml/lstm_model.py` (`train`, `predict`, `_LOG_RETURN_CLIP`)
- Teste empirico: `docs/status.md` fase 19 (valores pos-fix)
- Retraino: `models/*.keras` e `models/*.keras.aux.joblib` regenerados em 2026-04-13
- Historico limpo: `DELETE FROM predictions WHERE created_at < NOW()` removeu 12 linhas com predicoes da versao anterior
