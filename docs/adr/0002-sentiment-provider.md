# ADR 0002 - Sentiment provider: Claude SDK replaces Gemini

- Status: Accepted
- Date: 2026-04-13
- Deciders: @architect, @dev, @analyst

## Context

The legacy `sentiment_analyzer.py` called Google Gemini via raw
`requests` to classify Portuguese financial news. This had three issues:

1. **Prompt format drift**: Gemini frequently wrapped JSON in markdown
   fences, forcing brittle string cleanups.
2. **PT-BR quality**: in A/B evaluations on InfoMoney headlines, Claude
   produced more consistent reasoning strings and impact classifications
   than Gemini 1.5 Flash.
3. **Workflow coherence**: the rest of the agentic workflow (aiox-core,
   Claude Code) is already Claude-centric; consolidating on one SDK
   reduces credential sprawl and simplifies observability.

## Decision

Replace Gemini with **Anthropic Claude** via the official
`anthropic` Python SDK. Default model: `claude-sonnet-4-5-20250929`
(overridable through `CLAUDE_MODEL`).

Implementation rules (enforced in `backend/app/ml/claude_sentiment.py`):

- The `anthropic.Anthropic` client is **injected** into the analyzer via
  the constructor - no module-level singleton.
- Each article is wrapped in `<article>...</article>` tags and the prompt
  explicitly instructs Claude to **treat the content as untrusted data**
  (prompt-injection mitigation, OWASP LLM01).
- The response is required to be a bare JSON object; on parse failure we
  return a neutral `SentimentResult` (`score=0`, `confidence=0`) and log
  the failure instead of raising, so one bad article does not break the
  batch.
- Results are cached in Redis keyed by `sha256(ticker + article_url)`
  with TTL 24 h to bound API cost.

## Consequences

Positive
- Structured, reliable JSON output; cheaper to parse and test.
- Single LLM vendor → one key, one SDK, one latency tail to monitor.
- Prompt-injection guardrails are first-class in the adapter.

Negative / trade-offs
- Vendor lock-in on Anthropic. Mitigated by keeping the analyzer behind
  a narrow `SentimentAnalyzer` interface so we can swap providers later.
- Claude Sonnet is more expensive per token than Gemini Flash; offset by
  aggressive 24 h Redis caching and batching of headlines.

## Alternatives considered

- **Keep Gemini**: loses PT-BR quality, keeps parsing fragility.
- **OpenAI GPT-4o**: competitive, but extra credential and second
  vendor relationship for no clear quality gain on our corpus.
- **Local model (e.g. Llama-3 via Ollama)**: attractive for cost, but
  inference hardware requirements conflict with our "runs on
  `docker compose up` on a laptop" target.
