# PubMed RAG — Q&A Preparation

Anticipated questions from panel evaluators, organized by category. Each includes the likely question, a concise answer, and references to supporting evidence in the repository.

---

## 1. Scope & Requirements Coverage

### Q1: Did you do performance testing for scalability across 36M+ abstracts?

We built Locust-based load tests in `loadtest/` — both Milvus-level (`loadtest/milvus/`) and API-level (`loadtest/api/`). These validate latency and throughput on our 100k-record corpus. Full 36M-scale testing was scoped out for this PoC (ADR-0002), but the pipeline scales without code changes — adjust `n_max` in `data_pipeline/config.yaml` and re-run.

**References:** `loadtest/README.md`, `docs/adr/0002-data-scope.md`, `docs/specs/2026-03-17-milvus-loadtest-design.md`

### Q2: The spec mentions "clinical trial stage" filtering. Is that implemented?

Publication type filtering covers "Clinical Trial" and "Randomized Controlled Trial" as selectable presets. However, granular phase-level filtering (Phase I/II/III/IV) is not implemented because PubMed metadata does not systematically encode trial phase as a structured field — it appears inconsistently in free-text. We chose to support what the data reliably provides.

**References:** `backend/src/retrieval/search.py` (build_filter_expression), `frontend/src/components/FilterPanel.tsx` (PUBLICATION_TYPE_PRESETS)

### Q3: Did you implement Agent-to-Agent (A2A) communication?

We chose independent parallel execution over A2A (ADR-0005). Each agent receives the same search results and analyzes them from a different perspective. This avoids serial latency and single-agent failure cascading to others. The Review Synthesizer (`src/agents/review_synthesizer.py`) provides A2A-like capability by synthesizing outputs across all agents into a unified literature review.

**References:** `docs/adr/0005-agent-orchestration.md`, `backend/src/agents/pipeline.py`, `backend/src/agents/review_synthesizer.py`

### Q4: What about "handoff to full-text retrieval"?

Scoped to abstract-level search for this PoC (Slide 10). Full-text retrieval via PubMed Central API is positioned as future work. Users can still access full text through PMID links displayed in the citation panel.

**References:** `docs/adr/0002-data-scope.md`, presentation Slide 10

---

## 2. Technical Deep Dives

### Q5: Why Milvus over Pinecone, Qdrant, or Weaviate?

Milvus 2.5 is the only open-source vector database with native BM25 + dense hybrid search on a single collection. Pinecone is managed (vendor lock-in, no self-hosting). Qdrant lacks native BM25. Weaviate supports hybrid but with a different architecture. Self-hosting gives us full transparency and no per-query costs.

**References:** `docs/adr/0006-vector-database-selection.md`, Appendix A2 in presentation

### Q6: Why is gpt-4o-mini sufficient? What about accuracy?

gpt-4o-mini costs ~1/30 of gpt-4o while performing well on abstract summarization and classification tasks. Quality is validated through DeepEval with 8 metrics (3 standard + 5 custom). LiteLLM abstraction allows model swaps without code changes — upgrading to gpt-4o or Claude requires only a config change.

**References:** `docs/adr/0013-llm-provider-resilience.md`, `docs/evaluation-results.md`, Appendix A2/A3 in presentation

### Q7: Isn't "1 abstract = 1 chunk" too coarse for chunking?

PubMed abstracts have a median length of 263 words — well within embedding model input limits. Splitting would break semantic coherence of self-contained abstracts. This is validated by our retrieval metrics (Contextual Relevancy in DeepEval). For future full-text support, section-level chunking would be introduced.

**References:** `docs/adr/0001-chunking-and-embedding.md`

### Q8: Tell me about the CrossEncoder reranker — model size, accuracy, cost?

We use `ms-marco-MiniLM-L-6-v2` (~80MB). It runs locally on CPU with zero API cost. If reranking quality is insufficient for a query, an LLM-based fallback is designed (ADR-0009). The local model keeps per-query cost at $0 for the reranking step.

**References:** `docs/adr/0009-reranking-strategy.md`, `backend/src/retrieval/reranker.py`

### Q9: How does MeSH Query Expansion work in detail?

Two-step pipeline: (1) LLM extracts medical keywords from the natural language query, (2) DuckDB lookups against the MeSH hierarchy resolve each keyword to its descriptor, synonyms, and child terms. For example, "pancreatic cancer" expands to "Pancreatic Neoplasms" plus subordinate MeSH terms. The expanded terms improve BM25 recall in hybrid search.

**References:** `docs/adr/0010-mesh-query-expansion.md`, `backend/src/retrieval/query_expander.py`

---

## 3. Quality & Safety

### Q10: How do you ensure the system doesn't produce harmful medical misinformation?

We cannot guarantee zero misinformation, but we mitigate risk through two layers. The input guardrail uses LLM-as-judge to warn on off-topic (non-medical) queries. The output guardrail runs four checks: (1) citation grounding verification, (2) hallucination detection, (3) MeSH terminology validation, (4) automatic medical disclaimer attachment. Guardrails are soft warnings (non-blocking) — we prioritize transparency over suppression.

**References:** `docs/adr/0008-guardrail-implementation.md`, `backend/src/guardrails/`

### Q11: What are the DeepEval results? Specific scores?

Evaluation results are documented in `docs/evaluation-results.md` and `docs/evaluation-results-basic.md`. We run 8 metrics: Faithfulness, Answer Relevancy, Contextual Relevancy (standard), plus Citation Presence, Medical Disclaimer, Methodology Quality, Statistical Validity, and Clinical Relevance (custom). The custom metrics reuse our agent implementations as DeepEval metric classes.

**References:** `docs/evaluation-results.md`, `docs/evaluation-results-basic.md`, `backend/src/eval/`

### Q12: What is your test coverage?

43 test files with 185 unit tests covering all modules: 8 agents, 6 API routes, RAG chain (sync + streaming), guardrails (input + output), retrieval (search, reranker, query expander), ingestion (loader, chunker, embedder, pipeline), shared utilities, and eval metrics. Plus integration tests for Milvus connectivity and DeepEval end-to-end evaluation suite.

**References:** `backend/tests/unit/`, `backend/tests/integration/`, `backend/tests/eval/`

---

## 4. Operations & Cost

### Q13: What is the per-query cost and response time?

| Scenario | LLM Calls | Estimated Cost | Response Time |
|----------|-----------|---------------|---------------|
| Ask only | 4 (expand + generate + input guard + output guard) | ~$0.002-0.005 | 2-5s |
| Ask + 8 Agents | 12 | ~$0.005-0.01 | 10-15s |

CrossEncoder reranker and DuckDB MeSH lookups are local — zero API cost. Embedding costs ~$0.0001/query. Voice input via Whisper adds ~$0.006/min.

**References:** Appendix A3 in presentation, `docs/adr/0004-token-usage-tracking.md`

### Q14: What would be needed for production deployment?

Five areas: (1) Circuit breaker for external service resilience (ADR-0015, designed but not implemented), (2) Local LLM fallback for offline operation (ADR-0013, designed), (3) Authentication and authorization, (4) Rate limiting, (5) Full corpus ingestion. The architecture supports all of these — Protocol-based DI enables swapping components without structural changes.

**References:** `docs/production-architecture.md`, `docs/adr/0013-llm-provider-resilience.md`, `docs/adr/0015-retry-and-circuit-breaker.md`

### Q15: Is Docker Compose suitable for production?

The current 5-service Docker Compose (etcd + MinIO + Milvus + Backend + Frontend) is a development configuration. Production would require Milvus cluster mode, horizontal scaling of the backend, CDN delivery for the frontend, and managed object storage replacing MinIO. This is documented in `docs/production-architecture.md`.

**References:** `docs/production-architecture.md`, `docker-compose.yml`
