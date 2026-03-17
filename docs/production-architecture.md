# Production Architecture

This document describes the current PoC architecture and the recommended production architecture, highlighting what would change and why.

## PoC Architecture (Current)

The current system runs as a **modular monolith on a single machine** via Docker Compose:

```
┌─────────────────────────────────────────────────────┐
│  Docker Compose (single host)                       │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  etcd     │  │  MinIO   │  │  Milvus 2.5      │  │
│  │  (meta)   │  │  (store) │  │  Standalone       │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  Backend (FastAPI + Uvicorn)                  │   │
│  │  - RAG pipeline, agents, guardrails           │   │
│  │  - Cross-encoder reranker (in-process)        │   │
│  │  - MeSH DB (DuckDB, in-process)               │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  Frontend (React, Vite dev server)            │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**Characteristics:**
- Single Milvus node (no replication)
- Single backend process (Uvicorn with default workers)
- Cross-encoder model loaded in the API process
- DuckDB for MeSH lookups (single-writer, file lock)
- No caching layer
- No API gateway or load balancer
- CORS allows all origins

## Production Architecture (Recommended)

```
                    ┌──────────┐
                    │  CDN     │
                    │ (static) │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
         ┌─────────│  API GW   │─────────┐
         │         │ (Kong/ALB) │         │
         │         └────┬──────┘         │
         │              │                │
    ┌────▼────┐   ┌────▼────┐   ┌──────▼──────┐
    │ Backend  │   │ Backend  │   │  Reranker    │
    │ Pod (1)  │   │ Pod (N)  │   │  Service     │
    │ FastAPI  │   │ FastAPI  │   │ (dedicated)  │
    └────┬─────┘   └────┬─────┘   └──────────────┘
         │              │
    ┌────▼──────────────▼────┐
    │       Redis Cluster     │
    │  (query cache, session) │
    └────────────┬────────────┘
                 │
    ┌────────────▼─────────────────────┐
    │        Milvus Distributed         │
    │  ┌──────┐ ┌──────┐ ┌──────┐     │
    │  │Query │ │Query │ │Query │     │
    │  │Node 1│ │Node 2│ │Node N│     │
    │  └──────┘ └──────┘ └──────┘     │
    │  ┌──────┐ ┌──────────────────┐  │
    │  │etcd  │ │  S3 / MinIO HA   │  │
    │  │cluster│ │  (segment store) │  │
    │  └──────┘ └──────────────────┘  │
    └──────────────────────────────────┘
                 │
    ┌────────────▼────────────┐
    │    LiteLLM Router       │
    │  ┌───────┐ ┌─────────┐ │
    │  │OpenAI │ │Azure    │ │
    │  │       │ │AOAI     │ │
    │  └───────┘ └─────────┘ │
    │  ┌───────────────────┐ │
    │  │Ollama (fallback)  │ │
    │  └───────────────────┘ │
    └─────────────────────────┘
```

## POC vs Production Comparison

| Layer | PoC (Current) | Production | Why |
|-------|---------------|------------|-----|
| **API Gateway** | None (direct access) | Kong / AWS ALB | Rate limiting, authentication, TLS termination, request routing |
| **Load Balancing** | Single process | K8s Service + HPA | Horizontal scaling based on CPU/request latency |
| **Backend** | 1x Uvicorn (default workers) | N x Uvicorn pods (K8s Deployment) | Stateless design (ADR-0003) already supports this; scale out on demand |
| **Reranker** | In-process (cross-encoder loaded in API) | Dedicated service / GPU pod | Cross-encoder model loading (~200MB) should not compete with API memory; GPU inference for lower latency |
| **MeSH Lookup** | DuckDB (file lock, single writer) | Redis or PostgreSQL | DuckDB's single-writer lock prevents concurrent access across pods |
| **Caching** | None | Redis Cluster | Cache frequent queries (TTL-based); cache embedding vectors for repeated terms |
| **Vector DB** | Milvus Standalone (1 node) | Milvus Distributed (N query nodes) | Horizontal read scaling for 36M+ abstracts; etcd cluster (3 nodes) for metadata HA |
| **Object Storage** | MinIO (single node) | S3 or MinIO HA | Durability and availability for Milvus segment data |
| **LLM Provider** | Single provider (OpenAI) | LiteLLM Router (multi-provider) | Automatic failover across OpenAI, Azure AOAI, Ollama (see ADR-0013) |
| **Retry** | LiteLLM `num_retries=3` | + Circuit breaker (`pybreaker`) | Prevent cascading failures during provider outages (see ADR-0015) |
| **Frontend** | Vite dev server | Static build → CDN (CloudFront / Vercel) | Global edge caching, no server-side rendering needed |
| **Observability** | LangFuse (LLM tracing) | + Prometheus metrics, Grafana dashboards, structured log aggregation (ELK/Loki) | Full observability stack: metrics, logs, traces |
| **Health Checks** | `GET /health` (basic) | K8s liveness + readiness probes with dependency checks | Readiness probe checks Milvus + LLM connectivity before accepting traffic |
| **Secrets** | `.env` file | K8s Secrets / AWS Secrets Manager | No secrets in environment files or container images |
| **CORS** | `allow_origins=["*"]` | Restricted to known frontend domains | Prevent unauthorized cross-origin access |
| **Connection Pooling** | Milvus default connection | Explicit pool configuration (pool_size, max_overflow) | Reduce TCP connection overhead under concurrent load |

## Scaling Considerations

### Scaling to 36M+ Abstracts (Full PubMed)

The current system indexes 100k abstracts. Scaling to the full PubMed corpus requires:

1. **Ingestion pipeline**: The existing `ingest_bulk.py` streams JSONL in batches with checkpointing. For 36M records, run ingestion in parallel across multiple workers partitioned by PMID range. Estimated time: ~12 hours with 4 parallel workers.

2. **Milvus capacity**: Milvus Distributed with multiple query nodes. HNSW index memory grows linearly with vector count (~55GB for 36M x 1536-dim float32). Segment-level loading allows query nodes to share the load.

3. **Embedding cost**: 36M abstracts at ~300 tokens each ≈ 10.8B tokens. With `text-embedding-3-small` at $0.02/1M tokens ≈ $216 one-time cost. Batch processing with checkpointing handles failures gracefully.

### Stateless Backend

The backend is already designed for horizontal scaling (ADR-0003):
- No in-process state beyond loaded models (reranker)
- All search state is in Milvus
- MeSH lookups are read-only
- LLM calls are stateless API requests

The only change needed for multi-pod deployment is moving the reranker model to a dedicated service and replacing DuckDB with a shared read store (Redis or PostgreSQL).

### Kubernetes Deployment (Example)

```yaml
# backend-deployment.yaml (illustrative)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pubmed-rag-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pubmed-rag-backend
  template:
    spec:
      containers:
      - name: backend
        image: pubmed-rag-backend:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: openai-api-key
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pubmed-rag-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pubmed-rag-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```
