ADR: Reranking Strategy — Cross-Encoder with LLM Fallback

Status: Accepted
Date: 2026-03-16
Owner: Yasuhiro Okamoto

## Context

Vector search retrieves candidates based on embedding similarity, but embedding-based ranking is approximate. Cross-encoder reranking passes each (query, document) pair through a model that jointly attends to both, producing more accurate relevance scores. The requirements specify "Reranking using cross-encoder models for improved research relevance" (Requirement 2).

The question is which reranker to use, and how to make the choice configurable.

## Decision

Implement three reranker backends behind a common `BaseReranker` Protocol, selectable via the `RERANKER_TYPE` environment variable:

| Type | Class | Use Case |
|------|-------|----------|
| `none` | `NoOpReranker` | Passthrough — Phase A behavior, baseline comparison |
| `cross_encoder` | `CrossEncoderReranker` | Default — best precision/latency trade-off |
| `llm` | `LLMReranker` | Fallback — when cross-encoder cannot be installed |

Default: `cross_encoder` with `cross-encoder/ms-marco-MiniLM-L-6-v2`.

## Cross-Encoder Model Selection

`cross-encoder/ms-marco-MiniLM-L-6-v2` was chosen for:

- **Size:** ~80MB — CPU-friendly, no GPU required
- **Latency:** ~200ms for 10 query-document pairs on CPU
- **Quality:** Trained on MS MARCO passage ranking; generalizes well to medical text despite not being domain-specific
- **Availability:** Popular model on HuggingFace, well-tested with sentence-transformers

### Why Not a Biomedical Cross-Encoder?

Biomedical-specific cross-encoders (e.g., trained on BioASQ or PubMedQA) would likely improve precision for medical queries. However:

- Few pre-trained biomedical cross-encoders are publicly available with permissive licenses
- Training a custom one requires labeled relevance data we don't have
- The general-purpose model performs well enough for the PoC — the primary precision gain comes from cross-encoding itself (joint query-document attention), not domain specialization

This can be revisited if evaluation shows medical-specific weaknesses.

## Implementation

**File:** `retrieval/reranker.py`

### Protocol

```python
class BaseReranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...
```

All reranker implementations conform to this Protocol. The RAG chain (`rag/chain.py`) depends only on the Protocol, not on any concrete class.

### CrossEncoderReranker

- Model loaded lazily on first call (avoids import-time overhead)
- Cached after first load (singleton per process)
- Pairs: `[(query, abstract_text) for each result]`
- Scores: `model.predict(pairs)` → sorted descending → top_k returned
- Each result's `score` is updated to the cross-encoder score

### LLMReranker

- Pointwise scoring: each abstract gets a separate LLM call asking for a 0-10 relevance score
- Slower and more expensive than cross-encoder (N LLM calls vs. 1 model inference)
- Useful when `sentence-transformers` cannot be installed (e.g., restricted environments)

### NoOpReranker

- Returns `results[:top_k]` unchanged
- Used for Phase A baseline and A/B comparison

### Factory

```python
reranker = get_reranker(reranker_type="cross_encoder", model_name="...", llm=...)
```

The factory is called once at app startup in `api/main.py` and injected via FastAPI dependency injection.

## Pipeline Integration

The reranker sits between search and LLM generation in the RAG chain:

```
Query Expansion → Search (top_k * multiplier) → Rerank (top_k) → LLM Generation
```

`reranker_top_k_multiplier` (default: 3) controls over-retrieval: search fetches `top_k * 3` candidates, then the reranker selects the best `top_k`. This gives the reranker a wider candidate pool to work with.

## Alternatives Considered

### Cohere Rerank API

- High quality, easy to use
- SaaS dependency, additional cost per query
- Not self-contained

**Rejected:** Same reason as Pinecone — the system must be locally runnable.

### ColBERT (late interaction)

- Efficient retrieval + reranking in one step
- Requires custom index format; not compatible with Milvus pipeline
- More complex to integrate

**Rejected:** Overkill for PoC. Cross-encoder is simpler and the retrieval layer (Milvus) is already separate.

## Consequences

### Positive

- Cross-encoder significantly improves precision over embedding-only ranking
- Protocol pattern makes the reranker swappable without touching the RAG chain
- Three options cover different deployment constraints (full, no-GPU, no-dependencies)
- Lazy loading avoids startup penalty when reranker is set to `none`

### Trade-offs

- Cross-encoder adds ~200ms per query (10 documents on CPU)
- `sentence-transformers` is a heavy dependency (~500MB with PyTorch)
- LLM reranker is 10-20x slower and more expensive than cross-encoder
- The general-purpose model may rank some medical-specific terms less accurately than a domain-specific one
