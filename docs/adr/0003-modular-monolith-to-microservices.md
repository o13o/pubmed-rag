ADR: Modular Monolith Architecture with Microservice Migration Path

Status: Accepted
Date: 2026-03-16
Owner: Yasuhiro Okamoto

## Context

The PubMed RAG system is built as a **modular monolith** — a single deployable unit where internal modules (`ingestion`, `retrieval`, `rag`, `agents`, `guardrails`, `shared`) are cleanly separated with well-defined boundaries. This was a deliberate choice for the PoC/capstone scope, but the architecture is designed so that each module can be extracted into an independent microservice with minimal refactoring.

This ADR documents the current module boundaries, inter-module dependencies, and the concrete steps required to migrate to a microservice architecture.

## Decision

Deploy as a modular monolith for the capstone. Document the microservice migration path so the system can be decomposed when scaling requirements demand it.

## Implementation Status

The Protocol-based abstraction layer has been **implemented** — the codebase supports both deployment modes today:

- `SearchClient` Protocol with `LocalSearchClient` (direct Milvus) and `RemoteSearchClient` (HTTP)
- `GuardrailClient` Protocol with `LocalGuardrailClient` (in-process)
- `DEPLOY_MODE=monolith` (default) uses local clients; `DEPLOY_MODE=microservice` uses remote clients
- Switch is controlled by a single env var — no code changes needed

## Current Module Dependency Map

```
ingestion ──→ shared (models)
retrieval ──→ shared (models, config)
rag ────────→ retrieval, guardrails, shared (models, llm, mesh_db)
agents ─────→ shared (models, llm)
guardrails ──→ shared (models, llm, mesh_db)
api ────────→ all modules (thin orchestration layer)
```

Key observation: **all modules depend on `shared`, but no module depends on another peer module** except `rag`, which orchestrates `retrieval` + `guardrails`. This is the primary coupling point.

## Microservice Decomposition

### Proposed Services

```
┌──────────────────────────────────────────────────────┐
│                    API Gateway                        │
│              (nginx / Kong / Envoy)                   │
└──────────┬──────────┬──────────┬─────────────────────┘
           │          │          │
     ┌─────▼─────┐ ┌──▼───┐ ┌───▼────┐
     │ RAG       │ │Search│ │Agent   │
     │ Service   │ │Svc   │ │Service │
     │ /ask      │ │/search│ │/analyze│
     └─────┬─────┘ └──┬───┘ └───┬────┘
           │          │          │
     ┌─────▼──────────▼──────────▼─────┐
     │         Shared Infrastructure    │
     │   Milvus │ MeSH DB │ LLM API    │
     └────────────────────────────────┘
```

| Service | Modules Included | Endpoints | External Dependencies |
|---------|-----------------|-----------|----------------------|
| **Search Service** | `retrieval`, `shared` | `POST /search` | Milvus, OpenAI (embeddings) |
| **RAG Service** | `rag`, `guardrails`, `shared` | `POST /ask` | Search Service (HTTP), LLM API, MeSH DB |
| **Agent Service** | `agents`, `shared` | `POST /analyze` | LLM API |
| **Ingestion Service** | `ingestion`, `shared` | CLI / batch job | Milvus, OpenAI (embeddings) |

### What Makes This Migration Feasible

1. **Pydantic models as API contracts.** All inter-module communication uses Pydantic models in `shared/models.py` (`SearchResult`, `RAGResponse`, `AgentResult`, etc.). These models serialize directly to JSON, making them natural REST API request/response schemas. No translation layer is needed.

2. **Dependency injection via FastAPI Depends.** Services (LLM, Milvus, MeSH DB, Reranker) are injected through `api/dependencies.py`, not imported directly in route handlers. Each microservice would simply provide its own dependency wiring.

3. **Agents are stateless and independent.** Each agent receives `(query, list[SearchResult])` and returns `AgentResult`. No inter-agent state, no shared memory. The Agent Service is the simplest to extract — it only needs `shared/models.py` and `shared/llm.py`.

4. **Search is self-contained.** The `retrieval` module only depends on Milvus and `shared`. It can be deployed as a standalone service immediately.

### What Requires Refactoring

1. **~~`rag/chain.py` direct imports.~~** ✅ **Done.** `chain.py` now depends on `SearchClient` and `GuardrailClient` protocols. Switching between local (direct Milvus) and remote (HTTP) is handled via `DEPLOY_MODE` env var.

2. **`shared/` as a published package.** The `shared` module (models, config, LLM client) would need to be extracted into a shared Python package (e.g., `pubmed-rag-shared`) installed by each service. Alternatively, each service copies the subset of models it needs.

   **Effort:** Create a `shared/` package with `pyproject.toml`, publish to a private registry or use path dependencies.

3. **MeSH DB access.** Currently, `retrieval/query_expander.py` and `guardrails/output.py` both access MeSH DB directly. Options:
   - Each service bundles its own MeSH DuckDB file (simple, ~50MB)
   - Create a MeSH lookup microservice (overkill for this dataset size)

   **Recommendation:** Bundle the DuckDB file per service. It's read-only and small.

4. **Milvus connection management.** Currently managed in `api/main.py` lifespan. Each service with Milvus access would manage its own connection.

   **Effort:** Copy the connection setup (~10 lines) to each service that needs it.

## Migration Steps (Ordered)

If migrating incrementally, extract services in this order:

| Step | Service | Reason | Difficulty |
|------|---------|--------|------------|
| 1 | **Agent Service** | Zero coupling to other modules. Only needs `shared/models.py` + `shared/llm.py`. | Low |
| 2 | **Search Service** | Self-contained. Only needs Milvus + `shared`. | Low |
| 3 | **Ingestion Service** | Already runs as a standalone script (`scripts/ingest_bulk.py`). Wrap in a service or keep as a batch job. | Low |
| 4 | **RAG Service** | Requires refactoring `chain.py` to call Search Service via HTTP instead of direct import. | Medium |

## Scaling Considerations

| Concern | Monolith | Microservice |
|---------|----------|-------------|
| Search latency | In-process, ~50ms | +5-10ms network overhead |
| LLM calls | Sequential in `/ask` | RAG + Agent services can parallelize |
| Agent parallelism | Sequential `for agent in agents` | Deploy Agent Service with workers, run agents concurrently |
| Independent scaling | Scale the whole app | Scale Search (CPU-bound) separately from RAG (LLM-bound) |
| Deployment | Single Docker image | Per-service images, orchestrated via Compose/K8s |

## Consequences

### Positive

- Current monolith is simple to deploy, debug, and demonstrate for the capstone
- Module boundaries are clean enough that migration is mechanical, not architectural
- Pydantic models serve as both internal types and API contracts

### Trade-offs

- `rag/chain.py` is the one coupling point that requires refactoring for microservice extraction
- `shared/` needs packaging strategy (shared library vs. copy-per-service)
- Network latency between services adds ~5-10ms per hop

### Future Extensions

- Add API Gateway (nginx/Kong) for routing, rate limiting, and authentication
- Use message queue (Redis/RabbitMQ) for async agent execution
- Deploy on Kubernetes with horizontal pod autoscaling per service
