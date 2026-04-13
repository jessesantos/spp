# ADR 0009: Tecnicas ML avancadas agregadas ao LSTM SPP v3.1

Status: accepted
Date: 2026-04-13
Related: 0001 (backend stack), 0006 (macro context), 0007 (intelligent training)

## Contexto

O SPP v3 usa LSTM 128-64-32 com features tecnicas classicas, sentimento (Claude) e contexto macro (RSS). Falta capturar:

1. **Clustering de volatilidade** (efeito ARCH/GARCH empirico em series financeiras: periodos de alta vol seguem periodos de alta vol).
2. **Sinais de mercados de previsao** (Kalshi / Polymarket precificam probabilidade de eventos macro - Fed rate, guerras, eleicoes - que movem a B3).
3. **Exposicao cambial** (PETR4/VALE3 beneficiam alta do USD; varejo importador sofre).
4. **Viesamento economico qualitativo no LLM** (Claude precisa de arcabouco teorico para avaliar noticias: ciclo Mises-Hayek, margin of safety, reflexividade).

Pesquisa avaliou: Transformer para series temporais (Informer, PatchTST, Temporal Fusion Transformer), N-BEATS, DeepAR, Prophet, GARCH(1,1), modelos de difusao para series, ensembles stacking LSTM + XGBoost.

## Decisao

Adotar **aditivamente**, sem refatorar o core LSTM:

| Tecnica | Motivo de adocao | Implementacao |
|---|---|---|
| **EWMA conditional volatility (RiskMetrics, lambda=0.94)** | Captura clustering sem dep extra; GARCH(1,1) com lambda fixo em serie diaria. Padrao JP Morgan desde 1996. | Feature `cond_vol` adicionada em `features.py`. |
| **Prediction markets (Kalshi + Polymarket)** | Unica fonte publica de probabilidade precificada para eventos macro; "wisdom of crowds" com skin in the game. | Clients em `app/data/prediction_markets.py`, tabela `prediction_market_signals`, feature `market_signal_score`. |
| **Currency exposure analyzer (BRL/USD)** | Correlacao + heuristica setorial. | Modulo `app/ml/fx_impact.py`, feature `fx_score`. |
| **Economic SKILL.md para Claude** | Sentimento puro ignora marco teorico; com Austrian school + Graham/Buffett + Soros o modelo pontua malinvestment, bubbles, moat. | `backend/app/ml/SKILL.md` injetado como `system` prompt em `claude_sentiment.py` e `explanation.py`. |
| **Janela de treino 3 anos** | 1 ano perdia ciclo Copom completo e nao capturava regime shifts. 3 anos cobre ao menos um ciclo de alta + corte de juros. | Defaults em `training_orchestrator`, CLI, Celery beat. |

### Rejeitado (future work)

| Tecnica | Motivo de recusa agora |
|---|---|
| Temporal Fusion Transformer | Requer PyTorch + refatoracao do pipeline. Ganho marginal em series de ~750 pontos (3y diario). Revisitar em v4. |
| Informer / PatchTST | Mesmo bloqueador. Alem disso, series B3 nao tem granularidade (minutos) que justifique o custo. |
| N-BEATS puro | Excelente em M-competitions mas sem alavancar features exogenas (sentimento, macro, FX). Stacking seria necessario. |
| Prophet | Otimo para sazonalidade/feriados, mas nao para cauda financeira. Nao agrega onde LSTM ja funciona. |
| GARCH(1,1) estimado por MLE | Dependencia `arch` pesa e fit por ticker a cada run e caro. EWMA e 95% do sinal por 5% do custo. |
| XGBoost ensemble | Plausivel (stacking), mas exige duplicar pipeline de features. Revisitar se direction_accuracy estagnar. |

### Integracao com o LSTM

As novas features entram como colunas numericas no mesmo `build_features`, portanto o `LSTMPricePredictor` as consome automaticamente via `select_feature_columns`. Nenhuma mudanca no shape do modelo alem do crescimento de `n_features`.

## Consequencias

+ Modelo passa a reagir a regime shifts (volatilidade), eventos precificados (Kalshi/Polymarket), choque cambial e teoria economica no LLM.
+ Nenhuma dependencia Python nova obrigatoria (EWMA e numpy puro).
+ Opcional: `KALSHI_API_KEY` (publico tambem serve para leitura).
- Aumento do tempo de treino proporcional a 3x mais dados.
- Mais pontos de falha externa (prediction markets podem ficar fora do ar): todos os clients sao best-effort com fallback neutro.

## Referencias

- RiskMetrics Technical Document, J.P. Morgan, 1996
- Engle, R. (1982). Autoregressive Conditional Heteroskedasticity...
- Kalshi API docs: https://trading-api.readme.io/
- Polymarket Gamma API: https://docs.polymarket.com/
- Mises, L. von. Human Action (1949) - teoria da acao e ciclo
- Hayek, F. Prices and Production (1931) - estrutura de capital
- Graham, B. Security Analysis (1934), The Intelligent Investor (1949)
- Soros, G. The Alchemy of Finance (1987) - reflexividade
