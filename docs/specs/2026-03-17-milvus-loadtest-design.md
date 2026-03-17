# Milvus Load Test Design

## Problem

The existing load test (`loadtest/locustfile.py`) targets the HTTP API layer, which includes LLM calls, embedding generation, and other overhead. There is no way to isolate and measure Milvus search performance directly, which is critical for validating scalability at 30M+ abstracts.

## Goals

1. Measure Milvus search latency (p50/p95/p99) and throughput (RPS) independently from API/LLM overhead
2. Provide a ready-to-run script for scalability validation when data volume increases
3. Organize load tests by target (API vs Milvus) with clear folder structure

## Design

### Folder Structure

```
capstone/loadtest/
├── api/
│   ├── locustfile.py    # existing API load test (moved)
│   └── README.md        # existing README (moved)
├── milvus/
│   ├── locustfile.py    # Milvus direct load test (new)
│   └── README.md
└── README.md            # top-level overview (updated)
```

### Milvus Locust User

Uses `pymilvus` directly via a custom Locust User (not HttpUser). Reports timing via `self.environment.events.request.fire()` for Locust dashboard integration.

### Test Scenarios

| Task | Weight | Description |
|------|--------|-------------|
| Dense search | 10 | HNSW search with random 1536-dim vector |
| Dense search + filter | 5 | Dense search with year range filter |
| Hybrid search | 5 | Dense + BM25 RRF fusion |
| Hybrid search + filter | 3 | Hybrid search with year range filter |

### Data Generation

- Query vectors: `numpy.random.rand(1536)` normalized — no OpenAI API dependency
- BM25 query strings: sample medical queries from a static list
- Year filters: random range within 2015-2024

### Configuration

Environment variables with sensible defaults:

- `MILVUS_HOST` (default: `localhost`)
- `MILVUS_PORT` (default: `19530`)
- `MILVUS_COLLECTION` (default: `pubmed_abstracts`)
- `TOP_K` (default: `10`)

### Measurements

Locust standard metrics apply automatically:
- Per-request latency (min/avg/median/p95/p99/max)
- Requests per second
- Error rate
- All available in WebUI and CSV export
