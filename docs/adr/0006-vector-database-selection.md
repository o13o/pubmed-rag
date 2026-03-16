ADR: Vector Database Selection — Milvus 2.5

Status: Accepted
Date: 2026-03-16
Owner: Yasuhiro Okamoto

## Context

The system requires a vector database that supports:

1. Dense vector search (cosine similarity over 1536-dim OpenAI embeddings)
2. Keyword-based retrieval (BM25) for hybrid search
3. Metadata filtering (publication year, journal)
4. Scalability to 100k+ records for the PoC, with a path to 36M+ (full PubMed)

The choice of vector database is a foundational decision that affects retrieval quality, operational complexity, and future scalability.

## Decision

Use **Milvus 2.5 (Standalone)** deployed via Docker Compose with etcd and MinIO as backing services.

## Alternatives Considered

### ChromaDB

- Lightweight, single-binary deployment
- Python-native, easy to integrate
- **No native BM25 support** — hybrid search would require a separate keyword index (e.g., Elasticsearch) or a two-pass approach
- Limited scaling options (single-node only in OSS)
- Best for: small prototypes, <50k vectors

**Rejected because:** Hybrid search is a core requirement (Requirement 2). Running a separate BM25 engine would double the infrastructure complexity, negating ChromaDB's simplicity advantage.

### Pinecone

- Fully managed SaaS — zero infrastructure
- Native hybrid search (dense + sparse)
- **No local deployment** — requires internet access and API key
- Cost scales with usage; less suitable for development iteration
- Vendor lock-in concerns

**Rejected because:** The capstone requires a self-contained, locally runnable system. Docker Compose must bring up the full stack without external SaaS dependencies.

### Qdrant

- Strong OSS option, Rust-based, performant
- Supports dense search with metadata filtering
- **BM25/sparse support is newer** and requires separate sparse vector configuration
- Growing ecosystem but smaller community than Milvus

**Considered viable**, but Milvus's mature hybrid search (dense + BM25 in a single `hybrid_search()` call with RRF fusion) was more straightforward for this use case.

### Weaviate

- Integrated vectorizer modules (can embed at ingestion time)
- Hybrid search support (BM25 + dense)
- Heavier resource footprint; Java-based modules
- GraphQL-centric API differs from standard vector DB patterns

**Rejected because:** The system already manages embeddings via OpenAI API. Weaviate's integrated vectorizer adds complexity without benefit. The GraphQL API is less familiar for the team.

## Rationale for Milvus

| Factor | Milvus 2.5 |
|--------|-----------|
| Hybrid search | Native `hybrid_search()` with RRF fusion — dense + BM25 in a single call |
| BM25 support | Built-in `Function.BM25` on VARCHAR fields; no separate index needed |
| Metadata filtering | Boolean expressions on scalar fields (`year >= 2022 and journal in [...]`) |
| Deployment | Docker Compose (etcd + MinIO + Milvus); well-documented standalone mode |
| SDK | `pymilvus` — mature Python client with typed APIs |
| Scale path | Standalone → Cluster mode; supports billions of vectors in production |
| Community | Large OSS community, LF AI & Data foundation project |

The key differentiator is that Milvus 2.5 handles both dense and sparse retrieval within the same engine. This means:

- One collection schema defines both `embedding` (FloatVector) and `chunk_text_sparse` (BM25 function)
- `hybrid_search()` fuses results from both fields using `RRFRanker(k=60)`
- No need for a separate Elasticsearch/OpenSearch instance

## Implementation

```yaml
# docker-compose.yml
services:
  etcd:       # Metadata store for Milvus
  minio:      # Object storage for Milvus segments
  milvus:     # Vector database (standalone mode)
```

Collection schema (`ingestion/milvus_setup.py`):

- `pmid` (VARCHAR, primary key)
- `embedding` (FLOAT_VECTOR, dim=1536, HNSW index, COSINE metric)
- `chunk_text_sparse` (SPARSE_FLOAT_VECTOR, BM25 function on `chunk_text`)
- Scalar fields: `title`, `abstract_text`, `year`, `journal`, `mesh_terms`

## Consequences

### Positive

- Single engine for both dense and sparse retrieval — simpler operations
- `search_mode` config switches between dense-only and hybrid without code changes
- Milvus Standalone is sufficient for 100k records; cluster mode available if needed

### Trade-offs

- Heavier infrastructure than ChromaDB (3 containers: etcd + MinIO + Milvus)
- First startup takes ~90s for health checks
- Milvus-specific API (not portable to other vector DBs without abstraction)

### Mitigation

- Infrastructure weight is acceptable for Docker Compose deployment
- `SearchClient` Protocol in `retrieval/client.py` abstracts the Milvus dependency — a different vector DB can be swapped by implementing `LocalSearchClient` against a new backend
