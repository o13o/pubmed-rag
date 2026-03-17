ADR: Connection Pooling — Not Required for PoC

Status: Accepted
Date: 2026-03-17
Owner: Yasuhiro Okamoto

## Context

The capstone evaluation checklist (§3, §6) mentions connection pooling as a production-grade practice to eliminate per-request TCP connection overhead for database and downstream service calls.

The system has two external connections:
1. **Milvus** — vector database for search
2. **LLM API** — OpenAI / Azure via LiteLLM (HTTPS)

## Decision

Do not implement explicit connection pooling in the PoC. The existing connection management is sufficient.

### Milvus

PyMilvus `connections.connect()` establishes a **persistent gRPC channel** that is reused across all operations within the process. This is not a per-request connection — it is created once at startup (in FastAPI's `lifespan`) and shared for the lifetime of the application. gRPC channels internally multiplex requests over a single TCP connection via HTTP/2, which provides connection-pool-like behavior by default.

Explicit pool configuration (e.g., `pool_size`, `max_overflow`) would only be needed if running multiple Milvus clusters or partitioning connections across shards — neither of which applies to our single-node Standalone deployment.

### LLM API (LiteLLM / httpx)

LiteLLM uses `httpx` under the hood, which maintains a default connection pool (`max_connections=100`, `max_keepalive_connections=20`). This is already connection-pooled without explicit configuration.

## Production Considerations

For multi-pod Kubernetes deployment:
- Each pod maintains its own persistent gRPC channel to Milvus — this scales linearly with pod count
- If Milvus Distributed is used with multiple query nodes, PyMilvus supports multi-address connection (`uri="http://lb:19530"`) with the load balancer handling distribution
- LiteLLM's httpx pool settings can be tuned via environment variables if needed

## Consequences

- **Positive:** No unnecessary configuration complexity in the PoC.
- **Positive:** Both Milvus (gRPC persistent channel) and LLM (httpx pool) already reuse connections by default.
- **Trade-off:** No explicit pool-size tuning — acceptable for single-pod deployment.
