# PubMed RAG — AI-Powered Medical Research Abstract Finder

Semantic search and RAG system for medical research abstracts powered by PubMed data, hybrid retrieval (dense + BM25), cross-encoder reranking, and LLM-based answer generation with output guardrails.

## Architecture

![Architecture Diagram](docs/architecture.png)

**Key components:**

- **Data Ingestion** — PubMed JSONL → parse → chunk (title + abstract + MeSH) → embed (text-embedding-3-small) → Milvus
- **Hybrid Search** — Dense cosine similarity + BM25 keyword matching with RRF fusion
- **Cross-Encoder Reranker** — ms-marco-MiniLM-L-6-v2 for improved relevance ranking
- **RAG Chain** — Query expansion (MeSH terms) → retrieval → prompt building → LLM (GPT-4o-mini)
- **Output Guardrails** — Citation grounding, hallucination detection, MeSH term validation, medical disclaimer
- **API** — FastAPI with `/ask` (RAG pipeline) and `/search` (vector search) endpoints
- **Frontend** — React + TypeScript + Tailwind CSS

## Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose
- OpenAI API key

## Quick Start

### 1. Start infrastructure (Milvus + Backend)

```bash
cd capstone

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Start all services
docker compose up -d
```

This starts:
- **Milvus** (vector DB) on port 19530
- **etcd** + **MinIO** (Milvus dependencies)
- **Backend API** on port 8000

### 2. Local development (without Docker)

```bash
# Backend
cd capstone/backend
cp .env.example .env  # Edit with your OPENAI_API_KEY
uv sync
uv run uvicorn src.api.main:app --reload --port 8000

# Frontend (separate terminal)
cd capstone/frontend
npm install
npm run dev
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key for embeddings and LLM |
| `MILVUS_HOST` | `localhost` | Milvus server host |
| `MILVUS_PORT` | `19530` | Milvus server port |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model for answer generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (1536 dim) |
| `SEARCH_MODE` | `dense` | Search mode: `dense` or `hybrid` |
| `RERANKER_TYPE` | `cross_encoder` | Reranker: `none`, `cross_encoder`, or `llm` |
| `MESH_DB_PATH` | `data/mesh.duckdb` | Path to MeSH DuckDB database |

## Data Ingestion

### Download PubMed data

```bash
cd capstone/playground/pubmed_pipeline
uv sync
uv run python download_hf.py    # Download from HuggingFace
uv run python sample.py          # Sample 100k records
```

### Ingest into Milvus

```bash
cd capstone/backend
uv run python scripts/ingest_bulk.py \
    ../playground/pubmed_pipeline/data/processed/sampled.jsonl
```

Features:
- Streams JSONL in batches (100 records per batch)
- Progress bar with ETA
- Checkpoint file for resumption on failure
- Automatically creates the Milvus collection with BM25 schema

## Usage

### CLI

```bash
cd capstone/backend

# Basic query
uv run python -m src.cli "What are the latest treatments for breast cancer?"

# With filters
uv run python -m src.cli "knee pain treatment" --year-min 2023 --top-k 5

# Hybrid search + reranker
uv run python -m src.cli "mRNA vaccine efficacy" --search-mode hybrid --reranker cross_encoder

# JSON output
uv run python -m src.cli "pancreatic cancer therapy" --json
```

### API

```bash
# Health check
curl http://localhost:8000/health

# Ask (RAG pipeline)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest treatments for breast cancer?", "top_k": 5}'

# Search (vector search only)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "CRISPR gene therapy", "year_min": 2020, "search_mode": "hybrid"}'
```

### Frontend

Open http://localhost:5173 (dev) or http://localhost:8000 (production, served by FastAPI).

## Example Output

**Query:** "What are the latest treatments for early-stage pancreatic cancer?"

```
============================================================
Query: What are the latest treatments for early-stage pancreatic cancer?
============================================================

Based on the retrieved research, several treatment approaches are being
investigated for early-stage pancreatic cancer:

1. **Neoadjuvant chemotherapy** — FOLFIRINOX-based regimens have shown
   improved resectability and survival in borderline resectable cases [1][2].

2. **Immunotherapy combinations** — Checkpoint inhibitors combined with
   chemotherapy are under active clinical trials [3].

3. **Stereotactic body radiation therapy (SBRT)** — Used as a bridge to
   surgery, showing promising local control rates [4].

============================================================
Citations (5):
============================================================
  PMID: 34567890 | Neoadjuvant FOLFIRINOX for Pancreatic Cancer
       J Clin Oncol (2023) | Score: 0.892
  PMID: 35678901 | Modified FOLFIRINOX in Borderline Resectable Disease
       Ann Surg Oncol (2023) | Score: 0.856
  ...

[warning] citation_grounding: Claim about "improved resectability" is
          partially supported — source mentions "increased R0 rates"

DISCLAIMER: This information is for research purposes only and should not
be used as medical advice. Always consult qualified healthcare professionals.
```

## Testing

```bash
cd capstone/backend

# Unit tests
uv run pytest tests/unit/ -v

# Integration tests (requires running Milvus)
uv run pytest tests/integration/ -v

# Evaluation suite (requires running system + DeepEval)
uv pip install -e ".[eval]"
uv run pytest tests/eval/ -v
```

## Project Structure

```
capstone/
├── docker-compose.yml          # Milvus + Backend orchestration
├── docs/
│   ├── architecture.mmd        # Architecture diagram (Mermaid source)
│   ├── architecture.png        # Architecture diagram (rendered)
│   ├── architecture.pdf        # Architecture diagram (PDF)
│   ├── adr/                    # Architecture Decision Records
│   ├── specs/                  # Design specifications
│   └── plans/                  # Implementation plans
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── src/
│   │   ├── api/                # FastAPI routes (/ask, /search, /health)
│   │   ├── ingestion/          # Data loading, chunking, embedding, Milvus setup
│   │   ├── retrieval/          # Hybrid search, reranker
│   │   ├── rag/                # RAG chain, prompt templates
│   │   ├── guardrails/         # Input/output validation
│   │   ├── shared/             # Config, models, LLM client, MeSH DB
│   │   └── cli.py              # CLI entry point
│   ├── scripts/
│   │   └── ingest_bulk.py      # Bulk ingestion script
│   ├── data/                   # MeSH DuckDB, runtime data
│   └── tests/
│       ├── unit/               # Unit tests
│       ├── integration/        # Integration tests
│       └── eval/               # DeepEval evaluation suite
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.tsx
│       ├── components/         # ChatPanel, FilterPanel, ResultsPanel
│       ├── lib/api.ts          # API client
│       └── types/              # TypeScript types
└── playground/
    └── pubmed_pipeline/        # Data download & sampling scripts
```

## TODO

- [ ] Multi-Agent analysis layer (Retrieval, Methodology Critic, Statistical Reviewer, Clinical Applicability, Summarization agents)
- [ ] Update architecture diagram after multi-agent implementation (change dashed lines to solid in `docs/architecture.mmd`)
- [ ] Streaming responses (SSE) for `/ask` endpoint
- [ ] Token usage tracking
