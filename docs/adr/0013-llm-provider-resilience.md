ADR: LLM Provider Resilience — LiteLLM Provider Switch

Status: Accepted
Date: 2026-03-17
Owner: Yasuhiro Okamoto

## Context

The system relies on LLM calls for answer generation, query expansion, guardrail validation, and multi-agent analysis. If the primary LLM provider (OpenAI) experiences an outage or rate-limiting, all core functionality would be unavailable.

The capstone evaluation checklist (§4 ML Resiliency) asks: "Does the system have a local fallback model (e.g., Flan-T5) if OpenAI times out?"

## Decision

Use **LiteLLM's provider abstraction** as the resilience strategy instead of bundling a local fallback model. All LLM calls go through `LLMClient` which wraps LiteLLM. Switching providers requires only changing the `LLM_MODEL` environment variable — no code changes.

Supported failover targets:

| Provider | Example `LLM_MODEL` value | Use case |
|----------|---------------------------|----------|
| OpenAI | `gpt-4o-mini` | Default (lowest latency, best quality/cost ratio) |
| Azure OpenAI | `azure/gpt-4o-mini` | Enterprise failover (SLA-backed, regional) |
| Ollama (self-hosted) | `ollama/llama3.1` | Air-gapped / offline / cost-free fallback |
| Anthropic | `claude-3-5-haiku-20241022` | Alternative commercial provider |

## Alternatives Considered

### Embedded local model (Flan-T5, LLaMA via Transformers)

- Bundling a local model adds significant dependencies (~2GB+ model weights, GPU/CPU inference overhead)
- Quality gap is substantial: Flan-T5 produces much lower quality medical summaries than GPT-4o-mini
- Cold start time for loading model weights (10-30s) degrades user experience
- Not practical for a PoC deployment

**Rejected because:** The quality degradation makes the fallback output unreliable for medical research queries. LiteLLM already provides the same code-change-free provider switching that a local model would, but with better quality options (Azure AOAI, Ollama with larger models).

### LiteLLM Router with automatic fallback chain

- LiteLLM supports a `Router` class that can auto-failover across a list of models
- Would provide zero-downtime failover without manual intervention

**Deferred:** Viable for production but adds configuration complexity. For the PoC, manual env-var switch is sufficient. The Router can be adopted later without code changes to `LLMClient`.

## Consequences

- **Positive:** Zero code changes for provider switch. Ollama provides a true local/offline option when needed.
- **Positive:** Each provider can be tested independently by changing one env var.
- **Trade-off:** Manual intervention required to switch providers (acceptable for PoC; Router pattern addresses this for production).
- **Trade-off:** Ollama-hosted models (e.g., LLaMA 3.1 8B) will have lower quality than GPT-4o-mini for medical domain tasks, but significantly better than Flan-T5.
