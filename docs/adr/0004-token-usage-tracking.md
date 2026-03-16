ADR: Token Usage Tracking via LangFuse

Status: Accepted
Date: 2026-03-16
Owner: Yasuhiro Okamoto

## Context

The requirements specify "Token usage tracking for literature review automation." Options considered:

1. **Track in API response** — return token counts in every `/ask` and `/analyze` response
2. **Track via observability platform** — use LangFuse to capture all LLM calls with full telemetry
3. **Both**

## Decision

Use LangFuse (option 2) as the sole token tracking mechanism. Do not embed token counts in API responses.

## Rationale

- **LiteLLM has native LangFuse integration.** Setting three env vars (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`) enables automatic tracing of every LLM call — zero code changes.
- **Richer data.** LangFuse captures prompt tokens, completion tokens, total tokens, latency, cost, model name, and full request/response content per call. An API response field would only show aggregated totals.
- **Per-call granularity.** A single `/ask` request triggers 3+ LLM calls (query expansion, answer generation, guardrail validation). LangFuse traces each individually with parent-child relationships. Exposing this in the API response would require significant plumbing for limited benefit.
- **Dashboard and analytics.** LangFuse provides time-series dashboards, cost analysis, and trace exploration out of the box — functionality that would be expensive to replicate in custom code.
- **Separation of concerns.** Token tracking is an operational concern, not a user-facing feature. Keeping it in the observability layer avoids coupling the API contract to internal implementation details.

## Implementation

Already in place:

- `shared/llm.py` — logs `tokens` per call via `logger.debug`
- `shared/llm.py` — enables `litellm.success_callback = ["langfuse"]` when `LANGFUSE_PUBLIC_KEY` is set
- `shared/config.py` — `langfuse_public_key`, `langfuse_secret_key`, `langfuse_host` settings

## Consequences

- Token usage is not visible in API responses. Consumers needing token data must query LangFuse.
- LangFuse is optional — when not configured, token counts are still logged to application logs at DEBUG level.
