# Phase B: Guardrails + Search Quality — Design Spec

**Date:** 2026-03-14
**Owner:** Yasuhiro Okamoto
**Status:** Approved
**Parent Spec:** [2026-03-14-pubmed-rag-system-design.md](2026-03-14-pubmed-rag-system-design.md)

## 1. Goal

Enhance the Phase A RAG system with output guardrails, hybrid search (BM25 + Dense), cross-encoder reranking, and a DeepEval-based evaluation pipeline.

## 2. Scope

| Task | Description | Parallelism |
|------|-------------|-------------|
| B-1 | Output guardrails (citation grounding, hallucination detection, medical terminology validation, disclaimer, treatment recommendation guard) | B-1 and B-2 parallel |
| B-2 | Hybrid search via Milvus 2.5+ BM25 Function + RRF fusion | B-1 and B-2 parallel |
| B-3 | Cross-encoder reranker with Protocol abstraction for provider swapping | After B-2 |
| B-4 | Evaluation pipeline using DeepEval with extensible custom metrics | After B-1 and B-3 |

## 3. New Dependencies

| Package | Purpose | Dependency Group |
|---------|---------|-----------------|
| `sentence-transformers` | Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) | main |
| `deepeval` | Evaluation framework with pytest integration | `eval` optional group |

## 4. B-1: Output Guardrails

### 4.1 Architecture

Dependencies (`LLMClient`, `MeSHDatabase`) are injected at construction time via a `GuardrailValidator` class, keeping the module boundary clean:

```python
class GuardrailValidator:
    def __init__(self, llm: LLMClient, mesh_db: MeSHDatabase): ...
    def validate(self, response: RAGResponse, search_results: list[SearchResult]) -> ValidatedResponse: ...
```

`src/guardrails/__init__.py` exports `GuardrailValidator`, `ValidatedResponse`, `GuardrailWarning`.

Three internal components:

1. **LLM Validator** — Single LLM call checks citation grounding + hallucination detection + treatment recommendation guard. Receives the RAGResponse answer and SearchResults list. Returns structured JSON with per-check results.
2. **MeSH Validator** — Uses existing `MeSHDatabase.validate_term()` to verify medical entities in the answer against MeSH controlled vocabulary. Rule-based, no LLM cost.
3. **Disclaimer** — Always appends fixed medical disclaimer text.

### 4.2 Output Model

```python
class GuardrailWarning(BaseModel):
    check: str          # "citation_grounding" | "hallucination" | "terminology" | "treatment_recommendation"
    severity: str       # "error" | "warning"
    message: str
    span: str = ""      # The problematic text span, if applicable

class ValidatedResponse(BaseModel):
    answer: str                       # Original or rewritten answer
    citations: list[Citation]
    query: str
    warnings: list[GuardrailWarning]  # All detected issues
    disclaimer: str                   # Always present
    is_grounded: bool                 # True if all claims are grounded
```

### 4.3 LLM Validator Prompt Strategy

One LLM call requesting JSON output. The system prompt instructs the LLM to return a JSON array of issues. The guardrails module parses the JSON string response from `LLMClient.complete()` (no structured output mode needed — the prompt engineering ensures JSON compliance, with fallback parsing for malformed responses).

The prompt instructs:
- For each sentence in the answer, check if it is supported by a cited abstract
- Flag ungrounded claims (claims not supported by any retrieved abstract)
- Flag potential hallucinations (factual statements like drug names, statistics not in source material)
- Flag unqualified treatment recommendations (definitive recommendations without hedging)
- Return JSON array of issues

### 4.4 Input Guardrail (Lightweight)

`src/guardrails/input.py` — LLM-based one-shot medical topic classification. Returns a relevance flag (soft warning, does not block processing).

### 4.5 Integration Point

RAG chain calls `validate_output()` after LLM generation, before returning to the user. The `ask()` function in `rag/chain.py` will be updated to optionally run guardrails (default: enabled).

## 5. B-2: Hybrid Search

### 5.1 Milvus Schema Changes

The existing `pubmed_abstracts` collection needs a new schema to support BM25:

- Add a `SPARSE_FLOAT_VECTOR` field (`chunk_text_sparse`) for BM25 output
- Add a `Function(FunctionType.BM25)` mapping `chunk_text` (VARCHAR input) → `chunk_text_sparse` (sparse vector output)
- Add a sparse vector index on `chunk_text_sparse`

This requires **recreating the collection** and **re-ingesting data** (BM25 tokenization happens at insert time).

### 5.2 Updated Collection Setup and Migration

`src/ingestion/milvus_setup.py` will be updated with the new schema including the BM25 Function.

**Migration strategy:** `create_collection()` gains a `--recreate` flag (default: False). When True, drops the existing collection and recreates with the new schema. When False, the function detects whether the existing collection has the `chunk_text_sparse` field — if missing, it logs a warning advising the user to run with `--recreate` and re-ingest data.

The `chunk_text` FieldSchema must include `enable_analyzer=True` to support the BM25 Function.

**Re-ingestion:** After recreating the collection, the full ingestion pipeline must be re-run. BM25 tokenization happens at insert time, so existing data cannot be migrated in-place.

### 5.3 Search Modes

`search_mode` parameter in `SearchFilters`:

| Mode | Behavior |
|------|----------|
| `dense` (default) | Existing cosine similarity search. No BM25. |
| `hybrid` | Two `AnnSearchRequest`s (dense + BM25 sparse) → `Collection.hybrid_search()` → `RRFRanker` fusion |

### 5.4 Hybrid Search Flow

```
User Query
  → embed_query() → dense vector
  → Build AnnSearchRequest for dense (cosine) on "embedding" field
  → Build AnnSearchRequest for BM25 sparse on "chunk_text_sparse" field
  → Collection.hybrid_search(reqs=[dense_req, bm25_req], rerank=RRFRanker(k=60))
  → Parse results → list[SearchResult]
```

**AnnSearchRequest construction:**

```python
# Dense vector search
dense_req = AnnSearchRequest(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 128}},
    limit=top_k,
    expr=filter_expr if filter_expr else None,
)

# BM25 sparse search (Milvus tokenizes the raw text via the BM25 Function)
sparse_req = AnnSearchRequest(
    data=[query_text],
    anns_field="chunk_text_sparse",
    param={"metric_type": "BM25"},
    limit=top_k,
    expr=filter_expr if filter_expr else None,
)

# Fuse results with Reciprocal Rank Fusion
results = collection.hybrid_search(
    reqs=[dense_req, sparse_req],
    rerank=RRFRanker(k=60),
    limit=top_k,
    output_fields=[...],
)
```

### 5.5 Configuration

Add to `Settings`:
- `search_mode: str = "dense"` — default preserves Phase A behavior

Add to `SearchFilters`:
- `search_mode: str | None = None` — per-query override (falls back to config default)

## 6. B-3: Reranker

### 6.1 Architecture

Protocol-based abstraction for swappable reranker implementations.

```python
class BaseReranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...
```

Implementations:

| Class | Backend | Use Case |
|-------|---------|----------|
| `NoOpReranker` | Passthrough | Phase A behavior, testing |
| `CrossEncoderReranker` | sentence-transformers (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | Default, CPU-friendly, ~80MB model |
| `LLMReranker` | LiteLLM (GPT-4o-mini) | Fallback, no model download needed |

### 6.2 CrossEncoderReranker

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU, ~80MB)
- Input: list of (query, abstract_text) pairs
- Output: relevance scores → sort descending → return top_k
- Lazy model loading (loaded on first call, cached)
- Note: This is a general-purpose model trained on MS MARCO (web queries). It is chosen for its small size and CPU compatibility. A medical-domain fine-tuned cross-encoder can be substituted later via the Protocol abstraction.

### 6.3 LLMReranker

- Uses existing `LLMClient` for pointwise scoring
- Prompt: "Rate the relevance of this abstract to the query on a scale of 0-10"
- Parses numeric scores → sort → return top_k
- Slower but requires no additional dependencies

### 6.4 Pipeline Integration

Note: Phase A's `chain.py` does **not** currently call `reranker.rerank()`. The existing reranker stub is unused. Phase B adds the reranker as a new pipeline step between search and prompt building.

Search pipeline becomes: `search(top_k * multiplier) → rerank(top_k) → build prompt → LLM → guardrails → return`

The reranker receives an expanded candidate set (3x top_k by default) and narrows to the final top_k. The existing `reranker.rerank(results, query)` function signature will be replaced by the Protocol-based classes with signature `rerank(query, results, top_k)`.

### 6.5 Configuration

Add to `Settings`:
- `reranker_type: str = "cross_encoder"` — options: `"none"`, `"cross_encoder"`, `"llm"`
- `reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"` — model name for cross-encoder
- `reranker_top_k_multiplier: int = 3` — how many candidates to fetch before reranking

## 7. B-4: Evaluation Pipeline

### 7.1 Architecture

`tests/eval/` directory with pytest-compatible evaluation suite powered by DeepEval.

**Traceability note:** DeepEval is listed under Requirement 2 (Advanced) in `statements.md`. It is pulled forward into Phase B because: (1) it provides immediate, measurable feedback on the quality improvements from B-1 (guardrails), B-2 (hybrid search), and B-3 (reranker); (2) the DeepEval framework is extensible, so domain-specific medical metrics (clinical relevance, evidence quality, study design assessment) will be layered on in Phase D when Requirement 2 is fully addressed.

### 7.2 Metrics

**DeepEval Built-in Metrics:**

| Metric | What It Measures |
|--------|------------------|
| Faithfulness | Is the answer grounded in the retrieved context? |
| Answer Relevancy | Is the answer relevant to the query? |
| Contextual Precision | Are relevant documents ranked higher? |
| Contextual Recall | Are all relevant documents retrieved? |

### 7.3 Custom Metrics Extensibility

Metrics are defined in `tests/eval/metrics/` as separate modules. Each custom metric inherits from `deepeval.metrics.BaseMetric`. Structure:

```
tests/eval/
├── __init__.py
├── conftest.py              # DeepEval test configuration
├── dataset.json             # Evaluation dataset (20-30 test cases)
├── test_rag_evaluation.py   # Main evaluation test file
└── metrics/
    ├── __init__.py
    └── custom.py            # Custom metrics (extensible)
```

Adding a new metric: create a class inheriting `BaseMetric` in `metrics/`, add it to the test file's metric list.

### 7.4 Evaluation Dataset

`tests/eval/dataset.json` format:

```json
[
  {
    "query": "What are the latest treatments for breast cancer?",
    "expected_output_keywords": ["immunotherapy", "targeted therapy"],
    "relevant_pmids": ["12345678", "23456789"],
    "context_required": true
  }
]
```

### 7.5 Execution

```bash
# Run evaluation suite
cd capstone/backend
uv run deepeval test run tests/eval/test_rag_evaluation.py

# Or via pytest
uv run pytest tests/eval/ -v
```

### 7.6 Dependencies

Add `deepeval` to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
eval = [
    "deepeval>=1.5,<2.0",
]
```

## 8. File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `src/guardrails/output.py` | Output validation (LLM + MeSH + disclaimer) |
| `src/guardrails/input.py` | Input topic classification (lightweight) |
| `tests/unit/test_guardrails_output.py` | Unit tests for output guardrails |
| `tests/unit/test_guardrails_input.py` | Unit tests for input guardrails |
| `tests/unit/test_reranker.py` | Unit tests for reranker implementations |
| `tests/eval/__init__.py` | Eval package |
| `tests/eval/conftest.py` | DeepEval configuration |
| `tests/eval/dataset.json` | Evaluation test cases |
| `tests/eval/test_rag_evaluation.py` | Main evaluation tests |
| `tests/eval/metrics/__init__.py` | Custom metrics package |
| `tests/eval/metrics/custom.py` | Custom metric implementations |

### Modified Files

| File | Changes |
|------|---------|
| `src/shared/models.py` | Add `GuardrailWarning`, `ValidatedResponse`, `search_mode` to `SearchFilters` |
| `src/shared/config.py` | Add `search_mode`, `reranker_type`, `reranker_model`, `reranker_top_k_multiplier` |
| `src/ingestion/milvus_setup.py` | Add BM25 Function + sparse vector field to schema |
| `src/retrieval/search.py` | Add hybrid search mode with `hybrid_search()` + `RRFRanker` |
| `src/retrieval/reranker.py` | Replace passthrough with Protocol + CrossEncoderReranker + LLMReranker |
| `src/rag/chain.py` | Integrate reranker + guardrails into pipeline |
| `src/rag/__init__.py` | Update exports |
| `src/guardrails/__init__.py` | Add public interface |
| `src/cli.py` | Add `--search-mode`, `--reranker` flags |
| `pyproject.toml` | Add `sentence-transformers`, `eval` optional group with `deepeval` |
