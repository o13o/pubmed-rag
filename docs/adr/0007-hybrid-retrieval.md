ADR: Hybrid Retrieval vs Semantic-Only Retrieval

Status: Accepted
Date: 2026-03-16
Owner: Yasuhiro Okamoto

## Context

The system was initially built with dense-only (semantic) search (Phase A). The requirements call for "Hybrid search combining vector similarity and keyword retrieval" (Requirement 2). The question is how to implement hybrid search and when to use it vs. dense-only.

Medical literature search has specific characteristics that affect this decision:

- **Exact terminology matters.** Drug names ("FOLFIRINOX"), gene names ("BRCA1"), and disease codes require exact matching that dense embeddings may miss.
- **Semantic understanding is essential.** Queries like "non-invasive therapy for knee arthritis" need semantic similarity to match abstracts that use different terms ("conservative treatment", "physical therapy").
- Both capabilities are needed — neither keyword nor semantic search alone is sufficient.

## Decision

Implement hybrid search using **Milvus's native `hybrid_search()`** with **RRF (Reciprocal Rank Fusion)** to combine dense vector similarity and BM25 keyword matching. Make it switchable via a `search_mode` configuration parameter.

## How It Works

### Dense Search (Phase A / `search_mode=dense`)

```
Query → OpenAI embedding → HNSW cosine search → top_k results
```

- Uses `text-embedding-3-small` (1536-dim) to embed the query
- Searches the `embedding` field via HNSW index with cosine similarity
- Good for semantic matching; may miss exact keyword matches

### Hybrid Search (Phase B / `search_mode=hybrid`)

```
Query → [Dense path] OpenAI embedding → HNSW cosine search → candidates
      → [Sparse path] BM25 text match → keyword search → candidates
      → RRF fusion (k=60) → merged top_k results
```

- Dense path: same as above
- Sparse path: Milvus BM25 function on `chunk_text_sparse` field matches keywords
- Fusion: `RRFRanker(k=60)` merges both ranked lists using reciprocal rank scoring

### RRF Fusion

RRF score for each document = sum over all lists of `1 / (k + rank)`, where `k=60` is the smoothing constant. This balances contributions from both retrieval methods without requiring score normalization.

`k=60` was chosen as the standard default from the original RRF paper (Cormack et al., 2009). It works well when the two input lists have similar quality — which is the case here since both dense and BM25 are strong retrieval methods for medical text.

## Configuration

```python
# Per-system default (config.py / .env)
SEARCH_MODE=dense          # "dense" or "hybrid"

# Per-query override (API request)
{"query": "...", "search_mode": "hybrid"}
```

Resolution order: per-query override > system default. This allows the frontend to let users choose, while the backend defaults to a sensible mode.

## Implementation

`retrieval/search.py`:

- `_dense_search()` — dense-only path using `collection.search()`
- `_hybrid_search()` — dual-path using `collection.hybrid_search()` with `AnnSearchRequest` for each field
- `search()` — entry point that resolves `search_mode` and dispatches

Both paths share `build_filter_expression()` for metadata filtering and `parse_search_results()` for output normalization.

## Why Not Other Fusion Methods

| Method | Pros | Cons |
|--------|------|------|
| **RRF** (chosen) | Simple, no hyperparameters beyond k, robust | Cannot weight one method over another |
| Weighted linear combination | Can tune dense vs. sparse weight | Requires score normalization; sensitive to tuning |
| Learn-to-rank | Optimal fusion | Needs training data; overkill for PoC |

RRF was chosen for its simplicity and robustness. It requires no score normalization (which is tricky since cosine scores and BM25 scores are on different scales) and has a single well-understood parameter.

## Consequences

### Positive

- Hybrid search captures both semantic similarity and exact keyword matches
- Particularly beneficial for medical queries with specific drug/gene/disease names
- Switchable at runtime — no reindexing needed to change modes
- Single Milvus call handles both paths (no external BM25 engine)

### Trade-offs

- Hybrid search is slightly slower than dense-only (~10-20% more latency due to two-path retrieval + fusion)
- BM25 field (`chunk_text_sparse`) adds storage overhead in Milvus
- RRF treats both paths equally — no way to boost one over the other without switching to weighted fusion
