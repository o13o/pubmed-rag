# PubMed RAG System - Design Spec

**Date:** 2026-03-14
**Owner:** Yasuhiro Okamoto
**Status:** Approved

## 1. Goal

Build an AI-powered medical research abstract retrieval and analysis system using PubMed/MEDLINE abstracts. The system allows clinicians and researchers to explore publications via natural language queries, retrieving relevant abstracts with structured insights and supporting evidence.

This spec covers Requirement 1 (Basic) from `statements.md`, with architectural decisions that preserve extensibility toward Requirement 2 (Advanced).

## 2. Technical Stack

| Layer | Choice | Notes |
|---|---|---|
| Vector DB | Milvus 2.5+ (Docker Standalone) | Hybrid search (BM25 + Dense) in Phase B |
| Embedding | text-embedding-3-small (1536 dim) | ADR-0001 |
| LLM | LiteLLM wrapper, default GPT-4o-mini | Swap to Claude/GPT-4o without code changes |
| API Framework | FastAPI | Phase C |
| Frontend | TBD | Phase D / Requirement 2 |
| Data | PubMed 2021-2025, English, abstract-present, 100k cap | ADR-0002 |
| MeSH Lookup DB | DuckDB (in-memory / file) | MeSH hierarchy traversal, synonym resolution, terminology validation |
| Package Mgmt | uv (pyproject.toml) | |
| Container | Docker Compose (Milvus + backend) | |

## 3. Architecture

### 3.1 Modular Monolith

The system is a single deployable unit composed of loosely-coupled modules. Each module communicates exclusively through Pydantic models defined in `shared/models.py`. This makes future microservice extraction straightforward: replace in-process function calls with HTTP/gRPC boundaries while keeping the same Pydantic schemas as API contracts.

### 3.2 Directory Structure

```
capstone/
├── backend/
│   ├── src/
│   │   ├── shared/            # Cross-cutting concerns
│   │   │   ├── models.py      # Pydantic models (Article, Chunk, SearchResult, RAGResponse, etc.)
│   │   │   ├── config.py      # Settings via pydantic-settings (env vars)
│   │   │   ├── llm.py         # LiteLLM wrapper
│   │   │   └── mesh_db.py     # DuckDB-backed MeSH lookup (hierarchy, synonyms, validation)
│   │   ├── ingestion/         # Phase A: data pipeline
│   │   │   ├── loader.py      # Read pre-sampled PubMed JSONL (from playground pipeline), emit Article models
│   │   │   ├── chunker.py     # title + abstract → Chunk (1 abstract = 1 chunk, per ADR-0001)
│   │   │   └── embedder.py    # text-embedding-3-small → Milvus upsert
│   │   ├── retrieval/         # Phase A: search
│   │   │   ├── search.py      # Milvus ANN search + metadata filters (search_mode: dense|hybrid)
│   │   │   ├── query_expander.py  # MeSH term query expansion (LLM keyword extraction + DuckDB MeSH lookup)
│   │   │   └── reranker.py    # Phase B stub → cross-encoder
│   │   ├── rag/               # Phase A: answer generation
│   │   │   ├── chain.py       # retrieve → prompt → LLM → RAGResponse
│   │   │   └── prompts.py     # Prompt templates (system + user)
│   │   ├── guardrails/        # Phase B: output validation
│   │   │   ├── output.py      # Citation check, disclaimer, hallucination detection
│   │   │   └── input.py       # Lightweight medical-topic classifier
│   │   └── api/               # Phase C: REST API
│   │       ├── main.py        # FastAPI app factory
│   │       └── routes/
│   │           ├── search.py  # GET/POST /search
│   │           └── ask.py     # POST /ask (RAG endpoint)
│   ├── tests/
│   │   ├── unit/
│   │   └── integration/
│   ├── pyproject.toml
│   └── Dockerfile
├── frontend/                  # Phase D / Requirement 2
├── docker-compose.yml         # Milvus + backend (+ frontend later)
├── playground/                # Existing experiment code (pubmed_pipeline, etc.)
└── docs/
    ├── adr/                   # Existing ADRs
    └── specs/                 # This document
```

### 3.3 Module Boundaries (Microservice-Ready)

Each module exposes a public interface via `__init__.py`. Inter-module calls go through these interfaces only.

| Module | Public Interface | Future Service Boundary |
|---|---|---|
| `ingestion` | `ingest(source_path: Path) -> IngestReport` | Batch job / worker |
| `retrieval` | `search(query: str, filters: SearchFilters) -> list[SearchResult]` | Search service |
| `rag` | `ask(query: str, filters: SearchFilters) -> RAGResponse` | RAG service (calls retrieval internally) |
| `guardrails` | `validate_output(response: RAGResponse) -> ValidatedResponse` | Sidecar / middleware |
| `api` | FastAPI app | API gateway |

## 4. Data Flow

### 4.1 Ingestion (Phase A)

```
PubMed JSONL (from playground/pubmed_pipeline)
  → loader.py: parse → Article(pmid, title, abstract, authors, year, journal, mesh_terms)
  → chunker.py: "Title: {title}\nAbstract: {abstract}\nMeSH: {term1}; {term2}; ..." → Chunk
  → embedder.py: text-embedding-3-small → 1536-dim vector
  → Milvus upsert (vector + all scalar fields: pmid, title, abstract_text, chunk_text,
      year, journal, authors, mesh_terms, publication_types, keywords)
```

### 4.2 Query (Phase A)

```
User Query (text)
  → query_expander.py:
      1. LLM: extract medical keywords from natural language (e.g., "knee pain" → ["knee pain", "treatment"])
      2. DuckDB MeSH lookup: resolve keywords → MeSH descriptors + synonyms + child terms
         (e.g., "knee pain" → MeSH: "Knee/physiopathology", children: "Osteoarthritis, Knee", "Patellofemoral Pain Syndrome")
      3. Build expanded query (original query + MeSH terms)
  → search.py: embed expanded query → Milvus ANN search (cosine similarity ranking)
      + optional metadata filters (year range, journal, MeSH category)
  → top-k SearchResult list (ranked by cosine similarity score)
  → chain.py: build prompt with query + retrieved abstracts
  → LiteLLM (GPT-4o-mini) → RAGResponse (answer + citations)

Phase B additions:
  → search.py: search_mode=hybrid enables BM25 + Dense via Milvus 2.5 RRF fusion
  → reranker.py: cross-encoder reranking on top-k candidates
```

### 4.3 Output Guardrails (Phase B addition)

```
RAGResponse
  → output.py:
      1. Citation grounding: every claim maps to a retrieved abstract (PMID)
      2. Hallucination detection: flag facts not grounded in any SearchResult
      3. Medical terminology validation: verify entities against MeSH/controlled vocabulary
      4. Treatment recommendation guard: detect unqualified treatment recommendations
      5. Disclaimer: append medical disclaimer automatically
  → ValidatedResponse (answer + citations + warnings + disclaimer)
```

## 5. Milvus Schema

**Collection:** `pubmed_abstracts`

| Field | Type | Description |
|---|---|---|
| pmid | VARCHAR (PK) | PubMed ID |
| embedding | FLOAT_VECTOR(1536) | text-embedding-3-small output |
| title | VARCHAR | Article title |
| abstract_text | VARCHAR | Abstract body |
| chunk_text | VARCHAR | Indexed text (title + abstract + MeSH) |
| year | INT16 | Publication year (filterable) |
| journal | VARCHAR | Journal name (filterable) |
| authors | VARCHAR (JSON array) | Author list |
| mesh_terms | VARCHAR (JSON array) | MeSH terms (filterable) |
| publication_types | VARCHAR (JSON array) | Study types: ["Journal Article", "RCT"], etc. (filterable) |
| keywords | VARCHAR (JSON array) | Author-supplied keywords (filterable, supplementary to MeSH) |

**Indexes:**
- Vector: IVF_FLAT or HNSW on `embedding` (start with HNSW for 100k scale)
- Scalar: indexes on `year`, `journal`, `mesh_terms` for metadata filtering
- Phase B: add BM25 index on `chunk_text` for hybrid search

## 6. MeSH Lookup Database (DuckDB)

### 6.1 Purpose

MeSH (Medical Subject Headings) is a hierarchical controlled vocabulary of ~30,000 descriptors maintained by NLM. Using DuckDB as an in-memory lookup database enables:
- **Query expansion**: traverse the MeSH tree to find child/sibling concepts (faster and more accurate than LLM-only expansion)
- **Synonym resolution**: map Entry Terms (synonyms) to preferred MeSH descriptors
- **Terminology validation**: verify medical entities in LLM output against the controlled vocabulary (used by guardrails)

### 6.2 Schema

**Table: `mesh_descriptors`**

| Column | Type | Description |
|---|---|---|
| descriptor_ui | VARCHAR (PK) | MeSH unique identifier (e.g., "D010003") |
| name | VARCHAR | Preferred name (e.g., "Osteoarthritis, Knee") |
| tree_numbers | VARCHAR[] | Hierarchical positions (e.g., ["C05.550.114.606"]) |

**Table: `mesh_synonyms`**

| Column | Type | Description |
|---|---|---|
| synonym | VARCHAR | Entry term / synonym |
| descriptor_ui | VARCHAR (FK) | Maps to mesh_descriptors |

### 6.3 Key Operations

| Operation | SQL Pattern | Used By |
|---|---|---|
| Synonym → Descriptor | `SELECT d.name FROM mesh_synonyms s JOIN mesh_descriptors d ON s.descriptor_ui = d.descriptor_ui WHERE s.synonym ILIKE ?` | query_expander.py |
| Child term expansion | `SELECT name FROM mesh_descriptors WHERE tree_numbers LIKE 'C14.280%'` (tree number prefix match) | query_expander.py |
| Term validation | `SELECT EXISTS(SELECT 1 FROM mesh_descriptors WHERE name ILIKE ?) OR EXISTS(SELECT 1 FROM mesh_synonyms WHERE synonym ILIKE ?)` | guardrails/output.py |

### 6.4 Data Source and Setup

- Source: NLM MeSH XML (`desc2025.xml` from [NLM MeSH download](https://www.nlm.nih.gov/databases/download/mesh.html))
- One-time setup script parses XML → DuckDB file (`data/mesh.duckdb`)
- Loaded at application startup; ~30k descriptors + ~200k synonyms fits easily in memory (~50MB)
- Annual update: replace the `.duckdb` file when NLM releases new MeSH version

## 7. Guardrails Strategy

### 7.1 Output (Primary Focus)

Note: Medical terminology validation (row 3 below) uses DuckDB MeSH lookup (Section 6) for entity verification.

| Check | Method | Severity |
|---|---|---|
| Citation grounding | Compare each claim in LLM output against retrieved abstracts. Flag ungrounded claims. | Hard block: ungrounded claims are removed or flagged |
| Hallucination detection | Verify factual statements (drug names, statistics, outcomes) exist in source material | Warning + annotation |
| Medical terminology validation | Verify drug names, disease names, procedures against MeSH/controlled vocabulary. Detect confusions (e.g., similar drug names). | Warning + annotation |
| Medical disclaimer | Append standard disclaimer: "This is not medical advice..." | Always appended |
| Treatment recommendation guard | Detect definitive treatment recommendations without qualifying language | Rewrite to add hedging language |

### 7.2 Input (Lightweight)

| Check | Method | Severity |
|---|---|---|
| Medical topic relevance | LLM-based one-shot classification: is the query medical/biomedical? | Soft warning (still process, but note low relevance) |

## 8. Phase Plan

### Phase A: Data Foundation + Basic RAG (MVP Core)

```
A-1: Project scaffold & Milvus setup       ─┐
A-2: Ingestion pipeline (JSONL → Milvus)    ├─ Parallel
A-3: shared/ (models, config, LiteLLM, MeSH DuckDB setup) ─┘
         ↓ A-1,2,3 done
A-4: Retrieval (vector search + metadata filters + MeSH query expansion)
A-5: RAG Chain (retrieve → prompt → LLM → response)
         ↓
A-6: CLI / notebook E2E validation
```

**Deliverables:** CLI that takes a natural language query and returns a cited answer from PubMed abstracts.

### Phase B: Guardrails + Search Quality

```
B-1: Output guardrails (citation check, disclaimer)  ─┐ Parallel
B-2: Hybrid search (BM25 + Dense via Milvus 2.5)      ─┘
B-3: Reranker (cross-encoder)
B-4: Evaluation pipeline (retrieval precision, answer quality)
```

**Deliverables:** Guardrailed RAG with hybrid search, measurable quality metrics.

### Phase C: API

```
C-1: FastAPI endpoints (/search, /ask)
C-2: Streaming response (SSE)
C-3: Docker Compose (backend + Milvus, one command startup)
```

**Deliverables:** Dockerized API service, `docker compose up` to run everything.

### Phase D: Frontend + Requirement 2

```
D-1: Frontend (chat UI)
D-2: Requirement 2 advanced features (multi-agent, evaluation, etc.)
D-3: Production deployment config
```

**Deliverables:** Full-stack demo with UI, advanced analysis capabilities.

### Parallelization Summary

| Phase | Parallel Tasks | Sequential Dependencies |
|---|---|---|
| A | A-1, A-2, A-3 run concurrently | A-4 needs A-1+A-3; A-5 needs A-4; A-6 needs A-5 |
| B | B-1, B-2 run concurrently | B-3 needs B-2; B-4 needs B-1+B-3 |
| C | C-1, C-2 can partially overlap | C-3 needs C-1 |
| D | D-1, D-2 can partially overlap | D-3 needs D-1+D-2 |

## 9. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| 1 abstract = 1 chunk | Yes | ADR-0001: abstracts are 200-300 words, semantically self-contained |
| Milvus 2.5+ | Docker Standalone | Hybrid search support (BM25 + Dense) built-in from 2.5 |
| LiteLLM | Wrap all LLM calls | Swap models without code changes; default GPT-4o-mini |
| Guardrails: output-heavy | Citation check + hallucination detection + disclaimer | Medical domain demands verified outputs over input filtering |
| Modular monolith | Single process, module boundaries via Pydantic | Deploy as one unit now; split into services later if needed |
| Metadata in Milvus scalar fields | year, journal, mesh_terms, publication_types, keywords | Enable filtering at the vector DB level (no post-retrieval filtering) |
| Relevance ranking | Phase A: cosine similarity; Phase B: cross-encoder reranker | Simple ranking first, upgrade path clear |
| Query expansion | LLM keyword extraction + DuckDB MeSH hierarchy lookup | LLM handles NL→keywords; DuckDB handles precise MeSH traversal. Hybrid approach. |
| MeSH lookup | DuckDB (in-memory) | ~30k descriptors + ~200k synonyms; sub-ms queries; no external service dependency; reusable for guardrails validation |

## 10. Requirement 1 Traceability

| Requirement 1 Item | Where Addressed |
|---|---|
| Basic RAG with medical abstract embeddings | Phase A: ingestion + rag modules |
| Semantic search across research papers | Phase A: retrieval/search.py (ANN search) |
| Simple relevance ranking | Phase A: cosine similarity ranking; Phase B: cross-encoder upgrade |
| Medical terminology validation guardrails | Phase B: guardrails/output.py (validation via DuckDB MeSH lookup) |
| Citation extraction | Phase A: rag/chain.py (PMID citations in RAGResponse) |
| Metadata filtering (publication year, journal, study type) | Phase A: retrieval/search.py (Milvus scalar filters) |
| Query expansion using medical terminology (MeSH terms) | Phase A: retrieval/query_expander.py (LLM + DuckDB MeSH lookup) |
| API endpoint | Phase C: api/ module |

## 11. Error Handling and Observability

| Concern | Approach |
|---|---|
| Embedding API failures | Batch ingestion with configurable batch size (default 100). Retry with exponential backoff (max 3 retries). Idempotent upsert by PMID. |
| LLM call failures | Timeout (30s default). Graceful fallback: return retrieved abstracts without LLM summary. |
| Milvus unavailability | Health check at startup. Return 503 at API level. |
| Logging | Python `logging` with structured JSON format. Log levels: DEBUG for dev, INFO for prod. Key events: ingestion progress, query latency, LLM token usage. |

## 12. Testing Strategy

| Level | Scope | Examples |
|---|---|---|
| Unit | Pure logic per module | chunker format, prompt template rendering, guardrail checks, query expansion parsing |
| Integration | Module boundaries + external services | Milvus round-trip (insert → search), LLM call via LiteLLM (mocked in CI, live in local) |
| E2E | Full pipeline | Query → retrieve → generate → validate (Phase B-4 evaluation pipeline) |

## 13. References

- [ADR-0001: Chunking and Embedding](../adr/0001-chunking-and-embedding.md)
- [ADR-0002: Data Scope](../adr/0002-data-scope.md)
- [Requirement Statements](../../statements.md)
- [Milvus 2.5 Full-Text Search](https://milvus.io/docs/full-text-search.md)

## Appendix: Architecture Diagram

A visual architecture diagram (JPEG/PDF) is a required deliverable per `statements.md`. To be created after spec finalization, covering: data ingestion pipeline, embedding generation, hybrid retrieval, guardrails layer, and response generation.
