# Load Testing — Milvus Direct

Performance testing for Milvus search operations, bypassing the API and LLM layers.

## Setup

```bash
pip install locust numpy pymilvus
```

## Usage

### Web UI mode

```bash
cd capstone/loadtest/milvus
locust
```

Open http://localhost:8089, configure users and ramp-up. No host needed (connects to Milvus directly).

### Headless mode

```bash
# 10 concurrent users, ramp 2 users/sec, run for 60 seconds
locust --headless -u 10 -r 2 -t 60s

# Higher concurrency stress test
locust --headless -u 50 -r 5 -t 120s

# Export results to CSV
locust --headless -u 20 -r 5 -t 60s --csv=results
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_HOST` | `localhost` | Milvus server host |
| `MILVUS_PORT` | `19530` | Milvus server port |
| `MILVUS_COLLECTION` | `pubmed_abstracts` | Collection to search |
| `TOP_K` | `10` | Number of results per search |

Example with remote Milvus:

```bash
MILVUS_HOST=milvus.example.com MILVUS_PORT=19530 locust --headless -u 20 -r 5 -t 60s
```

## Test Scenarios

| Task | Weight | Description |
|------|--------|-------------|
| `dense_search` | 10 | HNSW vector search (random 1536-dim vector) |
| `dense_search [filtered]` | 5 | Dense search with year range filter |
| `hybrid_search` | 5 | Dense + BM25 via RRF fusion |
| `hybrid_search [filtered]` | 3 | Hybrid search with year range filter |

## What This Measures

- **Latency**: p50/p95/p99 per search type — how fast Milvus responds under load
- **Throughput**: requests per second — how many concurrent queries Milvus can handle
- **Error rate**: connection failures, timeouts under high concurrency
- **Filter overhead**: performance difference between filtered and unfiltered searches

Query vectors are random (no OpenAI API needed). This isolates Milvus engine performance from embedding generation and LLM overhead.

## Prerequisites

- Milvus running with data ingested into `pubmed_abstracts` collection
- No OpenAI API key needed (uses random vectors)

## Scalability Testing

To test at different data scales, ingest varying amounts of data and compare results:

```bash
# After ingesting 100K records
locust --headless -u 20 -r 5 -t 60s --csv=results_100k

# After ingesting 1M records
locust --headless -u 20 -r 5 -t 60s --csv=results_1m

# After ingesting 30M records
locust --headless -u 20 -r 5 -t 60s --csv=results_30m
```

Compare the CSV outputs to see how latency and throughput change with data volume.
