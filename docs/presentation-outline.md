# PubMed RAG — Presentation Outline

## Overview

- Format: 20 minutes total
  - Presentation Part 1 (3 min): Problem & Approach
  - Live Demo (5 min)
  - Presentation Part 2 (7 min): Architecture, Design & Retrospective
  - Q&A (5 min) + Appendix slides
- Audience: Training evaluators (business + technical mix)
- Strategy: Demo-sandwich — set up "why" and "what", show it live, then explain the "how"

---

## Presentation Part 1 (3 min) — Problem & Approach

### Slide 1: Title (15 sec)

- "PubMed RAG: AI-Powered Medical Research Retrieval & Analysis"
- Name, date

### Slide 2: Problem Statement (1 min 15 sec)

- PubMed: 36M+ articles, 1M+ added per year
- Researcher reality:
  - Keyword search misses semantically related papers
  - Quality assessment (methodology, statistics, clinical applicability) is manual and time-consuming
  - Modern workflows involve multimodal inputs (voice notes, images, documents)
- Concrete example: "Latest treatments for early-stage pancreatic cancer" — what happens with keyword search?

### Slide 3: Our Approach (1 min 30 sec)

- Three pillars:
  1. **Semantic + Hybrid Search** — understand meaning, not just keywords
  2. **Multi-Agent Analysis** — 8 specialized agents evaluate papers from different angles
  3. **Safety** — 2-layer guardrails + MeSH terminology validation
- Before/After comparison diagram: traditional keyword search vs this system
- Transition: "Let me show you this in action."

---

## Live Demo (5 min)

### Scenario 1: Text Query + SSE Streaming (1 min 30 sec)

- Input: "Latest treatments for early-stage pancreatic cancer"
- Show:
  - SSE streaming — answer appears token by token in real-time
  - Citations panel populates immediately (before LLM finishes)
  - PMID, relevance score, journal, year visible
- Key point: Real-time experience, evidence-based response with citations

### Scenario 2: Multi-Agent Analysis (1 min 30 sec)

- Click "Analyze with Agents" button
- Show 8 agents returning results in batches of 3:
  - Methodology Critic: study design evaluation
  - Statistical Reviewer: statistical validity
  - Clinical Applicability: real-world relevance
  - Summarization: cross-study synthesis
  - Conflicting Findings: inconsistencies between studies
  - Trend Analysis: emerging patterns
  - Knowledge Graph: disease-treatment-outcome connections
  - Retrieval: relevance assessment
- Key point: Multi-perspective analysis that would take a researcher hours

### Scenario 3: Voice Input (1 min)

- Upload an audio file with a medical query
- Whisper transcribes -> text appears in input box
- Submit -> full RAG pipeline executes
- Key point: Multimodal input — same pipeline, different entry point

### Scenario 4: Guardrails (1 min)

- Non-medical query (e.g., "best pizza in Tokyo") -> input guardrail warning
- Show medical disclaimer on valid responses
- Key point: Safety-first design for medical domain

---

## Presentation Part 2 (7 min) — Architecture, Design & Retrospective

### Slide 5: Architecture Overview (1 min 30 sec)

- Single architecture diagram showing end-to-end flow:
  - Data Pipeline (HuggingFace -> sampling -> ingestion)
  - Milvus 2.5 (Dense + BM25 vectors)
  - Hybrid Retrieval (Dense + BM25 + RRF fusion)
  - MeSH Query Expansion
  - CrossEncoder Reranker
  - RAG Chain (prompt + LLM + SSE streaming)
  - 8 Analysis Agents
  - 2-layer Guardrails (input + output)
  - React Frontend
- Now that we've seen the demo, walk through the architecture connecting what was shown to the components
- Infrastructure: Docker Compose (5 services: etcd + MinIO + Milvus + Backend + Frontend)

### Slide 6: Design Decision — Hybrid Search (1 min 30 sec)

- Problem: Dense-only misses exact medical terms; BM25-only misses semantic similarity
- Solution: Milvus 2.5 native BM25 + Dense + RRF fusion
- MeSH Query Expansion pipeline:
  - LLM extracts medical keywords from natural language
  - DuckDB MeSH hierarchy lookup (descriptor -> synonyms -> child terms)
  - Expanded query sent to hybrid search
- Trade-off: Latency increase (~2x) justified by recall improvement
- Reference: ADR-0007 (Hybrid Retrieval), ADR-0010 (MeSH Query Expansion)

### Slide 7: Design Decision — 8 Agents + Protocol DI (1 min)

- Agent roster (table):
  | Agent | Role |
  |-------|------|
  | Retrieval | Relevance assessment |
  | Methodology Critic | Study design & bias evaluation |
  | Statistical Reviewer | Statistical validity & significance |
  | Clinical Applicability | Real-world medical relevance |
  | Summarization | Cross-study synthesis |
  | Conflicting Findings | Inconsistency detection |
  | Trend Analysis | Emerging treatment patterns |
  | Knowledge Graph | Disease-treatment-outcome mapping |
- Architecture: BaseAgent Protocol -> all agents share same interface
- Unique point: Agents reused as DeepEval custom metrics (MethodologyQualityMetric, StatisticalValidityMetric, ClinicalRelevanceMetric)
- Protocol-based DI throughout: SearchClient, BaseReranker, GuardrailClient, BaseAgent
  - Enables: easy testing (mock injection), Monolith <-> Microservice switch
- Reference: ADR-0003, ADR-0005

### Slide 8: Design Decision — Guardrails + Prompt Management (1 min)

- Input guardrail: LLM-as-judge medical relevance classification (soft warning, non-blocking)
- Output guardrail (4 checks):
  1. Citation grounding verification
  2. Hallucination detection
  3. MeSH terminology validation
  4. Medical disclaimer auto-attachment
- Reference: ADR-0008
- Prompt externalization: 16 YAML templates (agents x9, guardrails x2, retrieval x2, RAG x1, transcribe x2)
  - Change prompts without code changes
  - Version-controllable, reviewable

### Slide 9: Evaluation & Quality (1 min)

- DeepEval — 8 metrics:
  - Standard: Faithfulness, Answer Relevancy, Contextual Relevancy
  - Custom: Citation Presence, Medical Disclaimer, Methodology Quality, Statistical Validity, Clinical Relevance
- Testing:
  - Unit tests: 38 files covering all modules
  - Integration tests: Milvus connection
  - Eval tests: End-to-end RAG quality
- Documentation:
  - 16 ADRs (Architecture Decision Records)
  - 5 design specs
- Observability: LangFuse integration for token tracking, cost analysis, latency monitoring

### Slide 10: Scope Decisions & Future Work (30 sec)

- Intentionally out of scope (Advanced/Nice-to-have):
  - Full-text retrieval handoff -> abstract-level search is the validated core
  - Full PubMed corpus (36M+) -> 100k stratified sample sufficient for PoC (ADR-0002)
- Delivered beyond initial scope:
  - A2A agent pipeline (ReviewPipeline: search → 6 parallel agents → synthesis)
  - Automated literature review generation (/review endpoint)
  - Multimodal input (audio/image/document via /transcribe)
- Message: Delivered core + advanced features with focus on quality

### Slide 11: Summary (30 sec)

- Built: AI-powered medical research retrieval & analysis system
- Technical highlights:
  1. Hybrid search (Milvus 2.5 native BM25 + Dense + RRF)
  2. 8 specialized agents, reused as evaluation metrics
  3. 2-layer guardrails for medical safety
- Infrastructure: Docker Compose (5 services), Protocol-based DI, 16 externalized prompts
- Thank you -> Q&A

---

## Appendix (for Q&A)

### A1: ADR Summary Table

| ADR | Decision | Key Trade-off |
|-----|----------|---------------|
| 0001 | Chunking: 1 abstract = 1 chunk + embedding: text-embedding-3-small | Simplicity vs granularity |
| 0002 | Data scope: 100k sampled abstracts | Dev speed vs full corpus |
| 0003 | Modular monolith with microservice escape hatch | Simplicity vs scalability |
| 0004 | Token tracking via LangFuse (not API response) | Separation of concerns |
| 0005 | Agent orchestration: independent parallel execution | Simplicity vs A2A collaboration |
| 0006 | Vector DB: Milvus 2.5 (native BM25 + hybrid) | Feature set vs managed service ease |
| 0007 | Hybrid retrieval: Dense + BM25 + RRF | Recall improvement vs latency |
| 0008 | Guardrails: 2-layer (input classification + output validation) | Safety vs latency |
| 0009 | Reranking: CrossEncoder (ms-marco-MiniLM) with LLM fallback | Accuracy vs cost |
| 0010 | MeSH query expansion via DuckDB | Domain precision vs complexity |
| 0011 | Multimodal: Whisper (audio) + GPT-4o-mini Vision (image) + PDF/DOCX extraction | Coverage vs scope |
| 0012 | Evaluation: DeepEval + agent-based custom metrics | Depth vs standard-only |
| 0013 | LLM provider resilience: LiteLLM + retry + fallback | Reliability vs simplicity |
| 0014 | Retrieval resilience: graceful degradation | Availability vs consistency |
| 0015 | Retry and circuit breaker strategy | Fault tolerance vs latency |
| 0016 | Connection pooling for external services | Throughput vs resource usage |

### A2: Technology Comparison

| Category | Chosen | Alternatives Considered | Reason |
|----------|--------|------------------------|--------|
| Vector DB | Milvus 2.5 | Pinecone, Qdrant, Weaviate | Native BM25 + hybrid search, self-hosted, no vendor lock-in |
| Reranker | CrossEncoder (CPU) | LLM reranker, Cohere Rerank | Zero API cost, low latency (~80MB model), LLM as fallback |
| MeSH DB | DuckDB | PostgreSQL, SQLite | Serverless, fast analytical queries, zero infrastructure |
| LLM | gpt-4o-mini (via LiteLLM) | gpt-4o, Claude | ~1/30 cost of gpt-4o, sufficient quality, LiteLLM enables model swap |
| Embedding | text-embedding-3-small | text-embedding-3-large, BGE | Cost-effective, 1536 dims sufficient for abstract-level search |

### A3: Operational Cost Analysis

#### Per-Query Cost Breakdown

| Component | LLM Calls | Model | Notes |
|-----------|-----------|-------|-------|
| Query Expansion | 1 | gpt-4o-mini | Keyword extraction |
| Answer Generation | 1 | gpt-4o-mini | Main RAG response (streaming) |
| Input Guardrail | 1 | gpt-4o-mini | Medical relevance classification |
| Output Guardrail | 1 | gpt-4o-mini | Grounding + hallucination check |
| **Subtotal (Ask)** | **4 calls** | | |
| Agent Analysis (optional) | 8 | gpt-4o-mini | All 8 agents |
| **Subtotal (Ask + Analyze)** | **12 calls** | | |

#### Cost Estimates

| Item | Cost |
|------|------|
| 1 query (Ask only) | ~$0.002-0.005 |
| 1 query (Ask + all Agents) | ~$0.005-0.01 |
| Embedding (per query) | ~$0.0001 |
| Whisper (voice input) | ~$0.006/min |
| CrossEncoder reranker | $0 (local CPU inference, ~80MB model) |
| MeSH DB (DuckDB) | $0 (file-based, no server) |

#### Cost Optimization Decisions

- gpt-4o-mini across all components (~1/30 cost of gpt-4o)
- CrossEncoder reranker runs locally on CPU -> zero API cost
- DuckDB for MeSH -> no external DB service needed
- LangFuse for real-time token/cost visibility -> early anomaly detection
- LiteLLM abstraction -> can switch to cheaper models without code changes

#### Infrastructure Cost

| Service | Resource | Notes |
|---------|----------|-------|
| Milvus 2.5 (standalone) | etcd + MinIO + Milvus | Self-hosted, Docker Compose |
| Backend (FastAPI) | Single container | Lightweight |
| Frontend (React) | Static build served by nginx | Minimal resources |
| LangFuse | Cloud free tier or self-hosted | Optional |

### A4: Sample Data Construction

#### Why Not 36M Records?

- PubMed full corpus: 36M+ records, 1.5-1.7M new records/year
- Full 5-year window (2021-2025): ~7-8M records
- Constraints that make full ingestion impractical for this PoC:
  - **Embedding API cost**: text-embedding-3-small at 7M records ≈ $700+ (one-time)
  - **Embedding API rate limits**: OpenAI rate limits make bulk embedding slow
  - **Storage**: 1536-dim float32 vectors × 7M records ≈ 42GB raw vectors; with Milvus index + metadata, ~100GB+
  - **Machine specs**: Development on local machine with limited RAM/disk
  - **Ingestion time**: Even at batch 100, 7M records × embedding latency = days of processing

#### Sampling Strategy (ADR-0002)

Designed a principled sampling approach instead of arbitrary truncation:

```
Full PubMed (36M+)
  → Filter: 5-year window (2021-2025)         ... ~7-8M records
  → Filter: English only                       ... reduced
  → Filter: Abstract present (non-empty)       ... reduced further
  → Year-stratified sampling (20k/year × 5)    ... 100,000 records
  → MeSH minimum coverage (500/category/year)  ... topic diversity guaranteed
```

Key design decisions:
- **Year-stratified equal allocation**: 20,000 records per year — prevents year imbalance, keeps time filters meaningful
- **MeSH minimum coverage**: 500 records per category per year across 10 disease categories (Neoplasms, Cardiovascular, Infectious, etc.) — prevents topic collapse from pure random sampling
- **Fixed seed (42)**: Reproducible results; same sample on every run
- **Audit log**: Records per-year population, per-category coverage, shortfalls, and all sampled PMIDs

#### Pipeline Architecture

Two-script pipeline with intermediate JSONL as the contract:

```
[1. Download]           →  raw JSONL   →  [2. Filter & Sample]  →  sampled JSONL + audit log
 (download_hf.py)          (data/raw/)     (sample.py)              (data/processed/)
```

- **download_hf.py**: Downloads from NLM FTP baseline (.xml.gz files), parses MedlineCitation XML, extracts 10 fields (PMID, title, abstract, authors, pub_date, MeSH terms, keywords, pub_types, language, journal), outputs raw JSONL
- **sample.py**: Reads raw JSONL, applies filters (year, language, abstract), year-stratified sampling with MeSH minimum coverage, outputs final JSONL + audit log
- **config.yaml**: Single config controls all parameters (n_max, seed, years, categories, paths)

#### Ingestion to Vector DB

```
sampled.jsonl (100k)
  → loader: parse JSONL to Article models
  → chunker: 1 abstract = 1 chunk (Title + Abstract + MeSH terms)
  → embedder: text-embedding-3-small, batch 100, 1536 dims
  → Milvus: HNSW index (dense) + BM25 index (sparse) + metadata fields
```

- 1 abstract = 1 chunk rationale: PubMed abstracts are ~200-300 words (median 263), semantically self-contained — splitting would break coherence (ADR-0001)
- Milvus schema supports both dense (HNSW/COSINE) and sparse (BM25) indexes on the same collection

#### Scalability Path

The pipeline is designed to scale without code changes:
- Change `n_max: 100000` → `300000` or remove cap entirely
- Swap data source (HuggingFace → NLM FTP baseline) by replacing download script; sample.py unchanged
- Same ingestion pipeline handles any corpus size (streaming, batched embedding)

### A5: Test Coverage Details

- Unit tests: 38 files
  - All 8 agents
  - All 6 API routes
  - RAG chain (sync + stream)
  - Guardrails (input + output)
  - Retrieval (search, reranker, query_expander)
  - Ingestion (loader, chunker, embedder, pipeline)
  - Shared (config, LLM, MeSH DB, models, prompt_loader)
  - Eval metrics
- Integration tests: Milvus connection
- Eval tests: DeepEval suite (8 metrics, dataset-driven)
