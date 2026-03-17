ADR: Retry and Circuit Breaker Strategy

Status: Accepted
Date: 2026-03-17
Owner: Yasuhiro Okamoto

## Context

LLM API calls are subject to transient failures: rate limiting (429), server errors (5xx), and network timeouts. Without retry logic, a single transient error causes the entire request to fail.

The capstone evaluation checklist (§6 Reliability) asks for: "Retry logic, local fallbacks, and circuit breakers."

## Decision

### Retry: LiteLLM built-in `num_retries`

Use LiteLLM's native retry mechanism via the `num_retries` parameter on all `litellm.completion()` calls. LiteLLM internally handles:

- Exponential backoff between retries
- Retries on: Timeout, RateLimitError (429), ServiceUnavailableError (503), InternalServerError (500)
- No retry on: AuthenticationError (401), BadRequestError (400), NotFoundError (404)

Configuration: `LLMClient(num_retries=3)` (default). Configurable per-instance.

### Circuit Breaker: Deferred to production

Not implemented in the PoC. Rationale:

- The system has a single LLM provider. If the circuit opens, there is no fallback target — the result is the same as returning an error immediately.
- LiteLLM's retry with backoff already prevents excessive calls during transient outages.
- Circuit breaker adds value when there are multiple downstream services or fallback providers configured via LiteLLM Router.

**Production recommendation:** Adopt `pybreaker` with a failure threshold of 5 and a recovery timeout of 30s, wrapping `LLMClient.complete()`. Combine with LiteLLM Router for automatic provider failover (see ADR-0013).

## Alternatives Considered

### tenacity decorator

- General-purpose retry library with rich configuration
- Adds an explicit dependency for functionality LiteLLM already provides
- Would be needed if we wanted circuit breaker in the same library, but tenacity does not provide true circuit breaker semantics

**Rejected because:** LiteLLM's built-in retry is sufficient and avoids an additional dependency.

### Manual retry loop (matching embedder.py pattern)

- The existing `embedder.py` uses a hand-written retry loop with `time.sleep()`
- Duplicates logic that LiteLLM provides natively

**Not adopted for LLM calls** because LiteLLM handles this internally. The embedder's manual retry remains because it calls the OpenAI embeddings API directly (not through LiteLLM).

## Consequences

- **Positive:** 3 automatic retries with backoff on transient LLM failures. No additional dependencies.
- **Positive:** `num_retries` is configurable per `LLMClient` instance for different use cases (e.g., guardrails may use fewer retries to avoid long waits).
- **Trade-off:** No circuit breaker in PoC — acceptable given single-provider architecture.
