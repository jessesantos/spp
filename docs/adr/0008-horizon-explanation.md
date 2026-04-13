# ADR 0008: Explicacao narrativa por horizonte de previsao

## Status

Aceito (2026-04-13)

## Contexto

As previsoes multi-horizonte (D1, W1, M1) mostram valor, percentual e
direcao, mas o usuario nao ve a razao do modelo: quais sinais pesaram,
se foi tecnico ou sentimento, se houve contexto macro favoravel. Sem
esse "por que" a ferramenta vira caixa-preta e perde valor como suporte
a decisao.

## Decisao

Introduzir uma camada de **explicacao narrativa persistida** por
horizonte de previsao:

1. **Geracao no momento da predicao**. O texto e produzido dentro de
   `PredictionService._build_horizons` logo apos o calculo numerico, a
   partir de um `ExplanationInput` que empacota ticker, horizonte,
   base, predito, pct, direcao, sentimento (score + contagens) e
   contexto macro disponivel.
2. **Dois geradores intercambiaveis (SOLID/DIP)**:
   - `HeuristicExplanationGenerator` monta texto deterministico em 6
     blocos (abertura, tendencia, sentimento, macro, horizonte, risco)
     e garante 100 a 500 palavras via `_clamp_words`.
   - `ClaudeExplanationGenerator` usa Anthropic Claude Sonnet com
     prompt-injection guard: sinais dentro de `<signal>`, instrucao
     explicita para tratar como dado, sem markdown. Se a resposta tem
     menos de 100 palavras ou falha, cai no heuristico.
   Escolha acontece em `infra/dependencies._build_explanation_generator`
   com base em `settings.anthropic_api_key`.
3. **Persistencia**. Nova coluna `predictions.explanation TEXT NULL`
   (migration `0004_prediction_explanation`). O `PredictionUpsert`
   carrega o texto; o repo preserva a explicacao existente quando um
   novo ciclo nao gera nova (`item.explanation is None`).
4. **Endpoint dedicado**. `GET /api/predictions/{ticker}/horizon/{n}/explanation`
   faz `latest_for(ticker, horizon_days)` no repo e devolve o texto
   persistido. Usado pelo modal do frontend para lazy load se desejado;
   o payload de `/api/predict` ja inclui o campo inline.
5. **UI acessivel**. `HorizonCard` ganha icone "information-circle"
   com `aria-label="Por que essa tendencia?"`. Clique abre
   `ExplanationModal` (role dialog, ESC, backdrop click, word count
   no rodape). Sem dependencia nova.

## Alternativas consideradas

- **Gerar sob demanda, so quando o modal abre**: reduz custo Claude
  mas causa latencia perceptivel no primeiro clique e obriga ter chave
  ativa sempre. Rejeitado.
- **Cache apenas em Redis (sem banco)**: perde rastreabilidade
  historica. Queremos auditar o que o modelo "disse" quando fez a
  previsao. Rejeitado.
- **Sem fallback heuristico**: quebra a demo sem `ANTHROPIC_API_KEY`.
  Rejeitado.
- **Texto livre sem limite de palavras**: risco de respostas gigantes
  custosas e ruim para UX. Adotamos 100..500 palavras com `_clamp_words`.

## Consequencias

Positivas:

- Usuario entende a tendencia.
- Audit trail: a cada chamada de `/api/predict`, o texto do momento e
  persistido; reconciliacao posterior compara preco real x previsto x
  explicacao original.
- Prompt-injection mitigado via `<signal>` tag + instrucoes.
- Degrada sem Claude: heuristica mantem a feature visivel.

Negativas:

- Custo extra de tokens Claude por chamada quando API key esta ativa.
  Mitigado porque `PredictionService` so gera explicacao no fluxo de
  predicao persistida (nao em WebSocket ticks) e o upsert reaproveita
  texto previamente salvo.
- Heuristica e repetitiva entre tickers; tradeoff aceito para o modo
  sem LLM.
- Janela de 100..500 palavras e um compromisso: explicacoes muito
  tecnicas podem ficar curtas; cenarios complexos podem ser truncados.

## Rastreabilidade

- Modulo: `backend/app/ml/explanation.py`
- Modelo: `backend/app/db/models.py` (coluna `explanation`)
- Migration: `backend/migrations/versions/0004_prediction_explanation.py`
- Repo: `backend/app/repositories/predictions_repository.py`
  (`PredictionUpsert.explanation`, `latest_for`)
- Service: `backend/app/services/prediction_service.py`
- API: `backend/app/api/routes.py` (endpoint `/horizon/{n}/explanation`)
- Wiring: `backend/app/infra/dependencies.py`
  (`_build_explanation_generator`)
- Frontend: `frontend/components/ExplanationModal.tsx`,
  `frontend/components/HorizonCard.tsx`,
  `frontend/lib/api.ts` (`api.explanation`)
- Testes: `backend/tests/test_explanation.py`, atualizacoes em
  `backend/tests/test_horizons.py`
