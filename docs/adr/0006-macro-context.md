# ADR 0006: Contexto macro/geopolitico como feature de predicao

## Status

Aceito (2026-04-13)

## Contexto

Movimentos da B3 nao dependem apenas de fundamentos da empresa. Guerras,
decisoes do Fed/BCE, tarifas, sancoes, choques de commodities (petroleo,
minerio) e politica nacional explicam boa parte da volatilidade de curto
prazo de tickers como PETR4, VALE3, ITUB4. O SPP original so analisava
noticias relacionadas diretamente ao ticker, perdendo esse contexto.

## Decisao

Introduzir um modulo `app.ml.macro_context` que:

1. Coleta noticias de feeds **internacionais** (Reuters, BBC World, FT,
   Bloomberg, AP) e **nacionais macro** (Valor, InfoMoney Mercados, G1
   Economia/Politica).
2. Filtra apenas artigos com pelo menos uma `MACRO_KEYWORDS` (war,
   sanction, Fed, rate hike, OPEC, Selic, Copom, etc.) para limitar custo
   de analise.
3. Pontua cada artigo via `ClaudeMacroAnalyzer` (reusa o prompt-safe do
   `ClaudeSentimentAnalyzer` com pseudo-ticker `GLOBAL`) ou via heuristica
   por palavras-chave se nao houver `ANTHROPIC_API_KEY`.
4. Agrega em um `MacroContext { score, confidence, count, top_keywords,
   high_impact_titles }`.
5. Injeta `macro_score` como **coluna adicional** no feature engineering
   do LSTM (ao lado de `sentiment`), passando para o modelo um sinal
   diario do clima global.

## Alternativas consideradas

- **Ignorar**: mantem simplicidade, mas perde acuracia em semanas de
  choque geopolitico.
- **Usar API paga (NewsAPI, GDELT)**: mais cobertura, mas custo e
  dependencia. Evitado para manter o tier gratuito.
- **Fine-tune LLM dedicado**: overkill para o escopo atual.

## Consequencias

Positivas:
- Feature nova sem trocar a arquitetura LSTM.
- Desacopla fontes via `MacroNewsSource` Protocol (SOLID/I, D).
- Degrada gracefully: sem Claude, usa heuristica; sem rede, retorna
  contexto neutro.

Negativas:
- Custo extra de tokens Claude (mitigado por filtro de keywords e cache
  Redis no futuro).
- Feeds internacionais podem mudar URL; cobertura precisa de monitoramento
  periodico.
- Correlacao `macro_score` -> preco e ruidosa; pode piorar em periodos
  calmos. Mitigacao: pesar por `confidence` no agregador.

## Rastreabilidade

- Modulo: `backend/app/ml/macro_context.py`
- Testes: `backend/tests/test_macro_context.py`
- Feature engineering: `backend/app/ml/features.py` (coluna `macro_score`)
- CLI de treino: `backend/app/ml/train.py` passa contexto macro via
  `--with-macro` (default: on).
