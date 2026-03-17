ADR: Retrieval Resilience — Milvus-Native Fault Tolerance

Status: Accepted
Date: 2026-03-17
Owner: Yasuhiro Okamoto

## Context

The capstone evaluation checklist (§4 ML Resiliency) asks: "Does the system return fallback results (e.g., keyword-only) if vector indexing fails?"

Traditional RAG systems that use separate services for dense retrieval (e.g., FAISS) and keyword retrieval (e.g., Elasticsearch) can fall back to keyword-only search if the vector index is unavailable. Our system uses Milvus for both dense and BM25 retrieval.

## Decision

Rely on **Milvus's built-in fault tolerance** rather than implementing a separate keyword search fallback.

Since both dense vector search and BM25 keyword search run inside the same Milvus instance, there is no scenario where "vector indexing fails but keyword search still works" — if Milvus is down, both are unavailable. Therefore, the correct resilience strategy is ensuring Milvus itself is reliable.

### Milvus reliability features (current PoC deployment)

- **etcd**: Metadata store with write-ahead log, tolerates restarts
- **MinIO**: Object storage for segment data, provides data durability
- **Milvus Standalone**: Automatic segment recovery on restart, WAL-based crash recovery
- **Docker Compose health checks**: `milvus` service depends on `etcd` and `minio` being healthy

### Production path (not implemented in PoC)

- Milvus Distributed mode with multiple query nodes for horizontal scaling
- etcd cluster (3+ nodes) for metadata HA
- MinIO cluster or S3 for storage-layer redundancy
- Kubernetes liveness/readiness probes

## Alternatives Considered

### Separate Elasticsearch for keyword fallback

- Run Elasticsearch alongside Milvus; fall back to ES-based BM25 if Milvus is down
- Adds significant infrastructure complexity (another stateful service to manage)
- Data synchronization between Milvus and Elasticsearch creates consistency risk
- BM25 quality would differ between Milvus and Elasticsearch implementations

**Rejected because:** The added operational complexity outweighs the benefit. Milvus already provides BM25 natively, and the failure mode this addresses (Milvus vector index down, but a separate keyword service still up) does not exist in our architecture.

### In-memory keyword index (e.g., Whoosh)

- Lightweight Python-based keyword search as fallback
- No external dependency
- Would need to load all 100k abstracts into memory on startup

**Rejected because:** Duplicates data already in Milvus, adds memory overhead, and the in-memory index would also be lost on process restart — providing only marginally better resilience.

## Consequences

- **Positive:** No additional infrastructure to maintain. Single source of truth for both dense and sparse retrieval.
- **Positive:** Milvus's native WAL and segment recovery handle crash scenarios.
- **Trade-off:** If Milvus is completely unavailable, the system returns an error rather than degraded results. This is acceptable for a PoC; the health endpoint (`/health`) reports Milvus status so monitoring can detect this.
- **Future:** Production deployment should use Milvus Distributed mode for HA.
