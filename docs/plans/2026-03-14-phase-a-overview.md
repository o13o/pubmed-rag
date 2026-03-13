# Phase A: Data Foundation + Basic RAG - Overview

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working CLI that takes a natural language query and returns a cited answer from PubMed abstracts.

**Architecture:** Modular monolith in `capstone/backend/src/` with 5 modules (shared, ingestion, retrieval, rag, guardrails). Modules communicate via Pydantic models in `shared/models.py`. Milvus 2.5+ for vector storage, LiteLLM for LLM abstraction, DuckDB for MeSH lookups.

**Tech Stack:** Python 3.11+, uv, Milvus 2.5+ (Docker), text-embedding-3-small, LiteLLM (GPT-4o-mini default), DuckDB, Pydantic v2, pytest

**Spec:** [2026-03-14-pubmed-rag-system-design.md](../specs/2026-03-14-pubmed-rag-system-design.md)

---

## Execution Order

```
Phase A-1: Project scaffold & Milvus setup       ─┐
Phase A-2: Ingestion pipeline (JSONL → Milvus)    ├─ PARALLEL (separate plan files)
Phase A-3: shared/ (models, config, LiteLLM, MeSH)─┘
         ↓ A-1, A-2, A-3 done → merge branches
Phase A-4: Retrieval                               ─┐
Phase A-5: RAG Chain                                ├─ SEQUENTIAL (single plan file)
Phase A-6: E2E Validation                           ─┘
```

## Plan Files

| Plan | Tasks | Can Run In Parallel | File |
|---|---|---|---|
| A-1: Scaffold & Milvus | Project setup, Docker Compose, Milvus collection | Yes | [phase-a1-scaffold.md](./2026-03-14-phase-a1-scaffold.md) |
| A-2: Ingestion Pipeline | loader, chunker, embedder | Yes | [phase-a2-ingestion.md](./2026-03-14-phase-a2-ingestion.md) |
| A-3: Shared Module | models, config, llm, mesh_db | Yes | [phase-a3-shared.md](./2026-03-14-phase-a3-shared.md) |
| A-4/5/6: Retrieval + RAG + E2E | search, query expansion, chain, CLI | Sequential (after A-1,2,3) | [phase-a456-retrieval-rag.md](./2026-03-14-phase-a456-retrieval-rag.md) |

## Dependencies

- **A-1** has no dependencies (standalone scaffold)
- **A-2** depends on `shared/models.py` from A-3 for `Article` and `Chunk` types. **However**, A-2 can start with stub types and integrate later. So A-2 and A-3 can run in parallel if A-2 defines its own temporary types or uses dicts until merge.
- **A-3** has no dependencies (standalone shared module)
- **A-4** depends on A-1 (Milvus running) + A-3 (shared models, mesh_db)
- **A-5** depends on A-4 (retrieval module) + A-3 (LiteLLM wrapper)
- **A-6** depends on A-5 (full pipeline)

## Integration Strategy

After A-1, A-2, A-3 complete in parallel:
1. Merge all three into a single branch
2. Resolve any interface mismatches (A-2's loader/chunker should use A-3's Pydantic models)
3. Run `pytest` across all modules to verify
4. Then proceed with A-4/5/6 sequentially

## JSONL Input Format (from playground pipeline)

Each line in `sampled.jsonl` is:
```json
{
  "pmid": "12345678",
  "title": "Article title...",
  "abstract": "Abstract text...",
  "authors": ["First Last", "First Last"],
  "publication_date": "2023",
  "mesh_terms": ["Neoplasms", "Cardiovascular Diseases"],
  "keywords": ["keyword1", "keyword2"],
  "publication_types": ["Journal Article", "Randomized Controlled Trial"],
  "language": "eng",
  "journal": "Journal of Medicine"
}
```
