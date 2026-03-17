# Load Testing

Performance and scalability testing for the PubMed RAG system.

## Structure

```
loadtest/
├── api/          # HTTP API load test (Locust)
│   └── README.md
├── milvus/       # Milvus direct search load test (Locust)
│   └── README.md
└── README.md     # This file
```

## Tests

| Directory | Target | Dependencies |
|-----------|--------|--------------|
| `api/` | FastAPI endpoints (`/search`, `/ask`, `/analyze`) | Running backend, OpenAI API key |
| `milvus/` | Milvus search engine directly (pymilvus) | Running Milvus with ingested data |

## Quick Start

```bash
pip install locust numpy pymilvus

# API load test
cd loadtest/api && locust

# Milvus load test
cd loadtest/milvus && locust
```

See each subdirectory's README for detailed usage.
