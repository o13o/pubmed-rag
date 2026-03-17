# Phase A-1: Project Scaffold & Milvus Setup

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the project structure, dependency management, Docker Compose with Milvus, and verify Milvus connectivity.

**Architecture:** Python project managed by uv with pyproject.toml. Milvus 2.5+ runs via Docker Compose. Backend code lives in `backend/src/`.

**Tech Stack:** uv, Docker Compose, Milvus 2.5+, pymilvus, pytest

**Spec:** [2026-03-14-pubmed-rag-system-design.md](../specs/2026-03-14-pubmed-rag-system-design.md)

---

## Chunk 1: Project Scaffold & Milvus

### Task 1: Initialize Python Project with uv

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/src/__init__.py`
- Create: `backend/src/shared/__init__.py`
- Create: `backend/src/ingestion/__init__.py`
- Create: `backend/src/retrieval/__init__.py`
- Create: `backend/src/rag/__init__.py`
- Create: `backend/src/guardrails/__init__.py`
- Create: `backend/src/api/__init__.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/unit/__init__.py`
- Create: `backend/tests/integration/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "pubmed-rag"
version = "0.1.0"
description = "PubMed RAG System - Medical research abstract retrieval and analysis"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pymilvus>=2.5.0",
    "litellm>=1.0",
    "openai>=1.0",
    "duckdb>=1.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Create all module `__init__.py` files**

Create empty `__init__.py` in each directory listed above:
```bash
cd backend
mkdir -p src/shared src/ingestion src/retrieval src/rag src/guardrails src/api tests/unit tests/integration
touch src/__init__.py src/shared/__init__.py src/ingestion/__init__.py src/retrieval/__init__.py src/rag/__init__.py src/guardrails/__init__.py src/api/__init__.py tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py
```

- [ ] **Step 3: Initialize uv and install dependencies**

```bash
cd backend
uv sync
```

Expected: `.venv` created, all dependencies installed.

- [ ] **Step 4: Create .gitignore**

Create: `backend/.gitignore`

```gitignore
.venv/
.env
__pycache__/
*.pyc
data/
*.duckdb
.pytest_cache/
```

- [ ] **Step 5: Create .env.example**

Create: `backend/.env.example`

```env
# OpenAI (for embeddings and default LLM)
OPENAI_API_KEY=sk-...

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# LLM (via LiteLLM)
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

- [ ] **Step 6: Commit**

```bash
git add backend/
git commit -m "feat(scaffold): initialize backend project with uv and module structure"
```

---

### Task 2: Docker Compose with Milvus

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Create docker-compose.yml**

```yaml
services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.18
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.5.4
    command: ["milvus", "run", "standalone"]
    security_opt:
      - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      etcd:
        condition: service_healthy
      minio:
        condition: service_healthy

volumes:
  etcd_data:
  minio_data:
  milvus_data:
```

- [ ] **Step 2: Start Milvus and verify it's running**

```bash
# cd to repository root
docker compose up -d
docker compose ps
```

Expected: All 3 services (etcd, minio, milvus) running and healthy.

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "infra: add Docker Compose for Milvus 2.5 standalone"
```

---

### Task 3: Milvus Collection Setup Script + Connectivity Test

**Files:**
- Create: `backend/src/ingestion/milvus_setup.py`
- Create: `backend/tests/integration/test_milvus_connection.py`

- [ ] **Step 1: Write integration test for Milvus connectivity**

```python
# tests/integration/test_milvus_connection.py
"""Test Milvus connectivity and collection setup."""

import pytest
from pymilvus import connections, utility


@pytest.fixture(autouse=True)
def milvus_connection():
    connections.connect("default", host="localhost", port="19530")
    yield
    connections.disconnect("default")


def test_milvus_is_reachable():
    """Milvus should be reachable on localhost:19530."""
    assert connections.has_connection("default")


def test_create_and_drop_collection():
    """Should be able to create and drop a test collection."""
    from pymilvus import CollectionSchema, DataType, FieldSchema

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=20),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4),
    ]
    schema = CollectionSchema(fields, description="test collection")

    from pymilvus import Collection

    col = Collection("test_connectivity", schema)
    assert utility.has_collection("test_connectivity")
    col.drop()
    assert not utility.has_collection("test_connectivity")
```

- [ ] **Step 2: Run test to verify it fails (Milvus not yet configured in test path)**

```bash
cd backend
uv run pytest tests/integration/test_milvus_connection.py -v
```

Expected: PASS if Milvus is running from Task 2. If Milvus is not running, FAIL with connection error.

- [ ] **Step 3: Write Milvus collection setup script**

```python
# src/ingestion/milvus_setup.py
"""Create the pubmed_abstracts collection in Milvus."""

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

COLLECTION_NAME = "pubmed_abstracts"
EMBEDDING_DIM = 1536


def get_schema() -> CollectionSchema:
    """Define the pubmed_abstracts collection schema per spec Section 5."""
    fields = [
        FieldSchema(name="pmid", dtype=DataType.VARCHAR, is_primary=True, max_length=20),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="abstract_text", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=12000),
        FieldSchema(name="year", dtype=DataType.INT16),
        FieldSchema(name="journal", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="authors", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="mesh_terms", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="publication_types", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=5000),
    ]
    return CollectionSchema(fields, description="PubMed abstracts for RAG")


def create_collection(host: str = "localhost", port: str = "19530") -> Collection:
    """Create the collection and HNSW index. Idempotent."""
    connections.connect("default", host=host, port=port)

    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

    schema = get_schema()
    collection = Collection(COLLECTION_NAME, schema)

    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 256},
    }
    collection.create_index("embedding", index_params)

    return collection


if __name__ == "__main__":
    col = create_collection()
    print(f"Collection '{col.name}' ready. Fields: {[f.name for f in col.schema.fields]}")
```

- [ ] **Step 4: Write test for collection setup**

Add to `tests/integration/test_milvus_connection.py`:

```python
def test_pubmed_collection_setup():
    """Should create the pubmed_abstracts collection with correct schema."""
    from src.ingestion.milvus_setup import COLLECTION_NAME, create_collection

    # Clean up if exists
    if utility.has_collection(COLLECTION_NAME):
        Collection(COLLECTION_NAME).drop()

    col = create_collection()
    assert utility.has_collection(COLLECTION_NAME)

    field_names = [f.name for f in col.schema.fields]
    assert "pmid" in field_names
    assert "embedding" in field_names
    assert "title" in field_names
    assert "year" in field_names
    assert "mesh_terms" in field_names

    # Cleanup
    col.drop()
```

- [ ] **Step 5: Run all integration tests**

```bash
cd backend
uv run pytest tests/integration/test_milvus_connection.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/src/ingestion/milvus_setup.py backend/tests/integration/test_milvus_connection.py
git commit -m "feat(milvus): add collection schema setup and connectivity tests"
```
