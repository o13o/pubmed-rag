# Load Testing — Locust

Performance testing for the PubMed RAG API.

## Setup

```bash
pip install locust
```

## Usage

### Web UI mode

```bash
cd capstone/loadtest
locust
```

Open http://localhost:8089, set host to `http://localhost:8000`, configure users and ramp-up.

### Headless mode

```bash
# 10 concurrent users, ramp 2 users/sec, run for 60 seconds
locust --headless -u 10 -r 2 -t 60s --host http://localhost:8000

# Light load (search only)
locust --headless -u 5 -r 1 -t 30s --host http://localhost:8000
```

## Test Scenarios

| Endpoint | Weight | Description |
|----------|--------|-------------|
| `GET /health` | 5 | Lightweight health check |
| `POST /search` | 10 | Vector search (dense/hybrid, no LLM) |
| `POST /search` (filtered) | 3 | Search with year filter |
| `POST /ask` | 2 | Full RAG pipeline (LLM call) |
| `POST /analyze` | 1 | Multi-agent analysis (multiple LLM calls) |

Weights reflect realistic usage: search-heavy with occasional RAG and rare agent analysis.

## Prerequisites

- Backend running on `http://localhost:8000`
- Milvus running with ingested data
- `OPENAI_API_KEY` set (for `/ask` and `/analyze` tasks)
