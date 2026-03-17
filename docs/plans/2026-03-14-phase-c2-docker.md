# Phase C-2: Docker Compose (Backend Service)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Containerize the FastAPI backend and add it to the existing Docker Compose alongside Milvus. One `docker compose up` starts the entire stack.

**Architecture:** Dockerfile for backend (`backend/Dockerfile`). Docker Compose adds `backend` service that depends on `milvus` healthy. Backend connects to Milvus via Docker network (`milvus:19530`).

**Tech Stack:** Docker, Docker Compose, uv (in-container build)

**Prerequisites:** Phase C-1 (FastAPI app factory + endpoints) merged. Specifically, `src/api/main:app` must exist as the uvicorn entrypoint.

**Parallelism:** This plan can be started as soon as C-1 Task 2 (app factory) is committed. The Dockerfile only needs to know the entrypoint (`src.api.main:app`).

---

## Chunk 1: Dockerfile + Compose

### Task 1: Backend Dockerfile

**Files:**
- Create: `backend/Dockerfile`
- Create: `backend/.dockerignore`

- [ ] **Step 1: Create .dockerignore**

```dockerignore
.venv/
.env
__pycache__/
*.pyc
.pytest_cache/
tests/
data/
*.duckdb
.git/
```

- [ ] **Step 2: Create Dockerfile**

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Test Docker build**

```bash
cd backend
docker build -t pubmed-rag-backend .
```

Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add backend/Dockerfile backend/.dockerignore
git commit -m "infra: add backend Dockerfile with uv"
```

---

### Task 2: Update Docker Compose

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Add backend service to docker-compose.yml**

Add the following `backend` service after the `milvus` service:

```yaml
  backend:
    container_name: pubmed-rag-backend
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      MILVUS_HOST: milvus
      MILVUS_PORT: "19530"
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      LLM_MODEL: ${LLM_MODEL:-gpt-4o-mini}
      EMBEDDING_MODEL: ${EMBEDDING_MODEL:-text-embedding-3-small}
      MESH_DB_PATH: /app/data/mesh.duckdb
      RERANKER_TYPE: ${RERANKER_TYPE:-cross_encoder}
    volumes:
      - ./backend/data:/app/data
    depends_on:
      milvus:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
```

- [ ] **Step 2: Test full stack startup**

```bash
# cd to repository root
docker compose up -d --build
docker compose ps
```

Expected: All services (etcd, minio, milvus, backend) running and healthy.

- [ ] **Step 3: Verify backend connectivity**

```bash
curl http://localhost:8000/health
```

Expected: `{"status": "ok", "milvus_connected": true, ...}`

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml
git commit -m "infra: add backend service to Docker Compose"
```

---

### Task 3: Compose .env.example

**Files:**
- Create: `.env.example`

- [ ] **Step 1: Create .env.example for docker-compose**

```env
# Required
OPENAI_API_KEY=sk-...

# Optional overrides
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
RERANKER_TYPE=cross_encoder
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add .env.example for Docker Compose"
```
