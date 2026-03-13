# Phase B-2: Hybrid Search (BM25 + Dense)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add BM25 full-text search to Milvus collection and implement hybrid search (dense + BM25 via RRF fusion) as an alternative to dense-only search.

**Architecture:** Update Milvus collection schema to include BM25 Function + sparse vector field. Extend `search.py` with `hybrid_search` mode that creates two `AnnSearchRequest`s and fuses via `RRFRanker`. Default `search_mode` remains `dense` for backward compatibility.

**Tech Stack:** pymilvus (Function, FunctionType.BM25, AnnSearchRequest, RRFRanker, hybrid_search)

**Spec:** [2026-03-14-phase-b-design.md](../specs/2026-03-14-phase-b-design.md) — Section 5

**Dependency:** B-S (shared models + config) must be merged first. Uses `search_mode` field from `SearchFilters`.

---

## Chunk 1: Milvus Schema + Hybrid Search

### Task 1: Update Milvus Collection Schema with BM25

**Files:**
- Modify: `capstone/backend/src/ingestion/milvus_setup.py`

- [ ] **Step 1: Update `get_schema()` to include BM25 Function**

Replace the existing `get_schema()` and `create_collection()` functions:

```python
# src/ingestion/milvus_setup.py
"""Create the pubmed_abstracts collection in Milvus."""

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    connections,
    utility,
)

COLLECTION_NAME = "pubmed_abstracts"
EMBEDDING_DIM = 1536


def get_schema() -> CollectionSchema:
    """Define the pubmed_abstracts collection schema per spec Section 5.

    Includes BM25 Function for hybrid search (Phase B).
    """
    fields = [
        FieldSchema(name="pmid", dtype=DataType.VARCHAR, is_primary=True, max_length=20),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="abstract_text", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(
            name="chunk_text", dtype=DataType.VARCHAR, max_length=12000,
            enable_analyzer=True, enable_match=True,
        ),
        FieldSchema(name="chunk_text_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="year", dtype=DataType.INT16),
        FieldSchema(name="journal", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="authors", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="mesh_terms", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="publication_types", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=5000),
    ]

    schema = CollectionSchema(fields, description="PubMed abstracts for RAG")

    # BM25 Function: chunk_text (VARCHAR) → chunk_text_sparse (SPARSE_FLOAT_VECTOR)
    bm25_function = Function(
        name="text_bm25",
        function_type=FunctionType.BM25,
        input_field_names=["chunk_text"],
        output_field_names=["chunk_text_sparse"],
    )
    schema.add_function(bm25_function)

    return schema


def create_collection(
    host: str = "localhost", port: str = "19530", recreate: bool = False,
) -> Collection:
    """Create the collection and indexes. Idempotent unless recreate=True.

    Args:
        recreate: If True, drops existing collection and recreates with new schema.
                  Required when upgrading from Phase A schema (no BM25) to Phase B.
    """
    connections.connect("default", host=host, port=port)

    if utility.has_collection(COLLECTION_NAME):
        if recreate:
            Collection(COLLECTION_NAME).drop()
        else:
            col = Collection(COLLECTION_NAME)
            field_names = [f.name for f in col.schema.fields]
            if "chunk_text_sparse" not in field_names:
                import logging
                logging.getLogger(__name__).warning(
                    "Collection '%s' exists but lacks BM25 fields. "
                    "Run with recreate=True and re-ingest data for hybrid search.",
                    COLLECTION_NAME,
                )
            return col

    schema = get_schema()
    collection = Collection(COLLECTION_NAME, schema)

    # Dense vector index (HNSW)
    collection.create_index(
        "embedding",
        {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 256}},
    )

    # Sparse vector index (BM25)
    collection.create_index(
        "chunk_text_sparse",
        {"metric_type": "BM25", "index_type": "AUTOINDEX"},
    )

    return collection


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    args = parser.parse_args()
    col = create_collection(recreate=args.recreate)
    print(f"Collection '{col.name}' ready. Fields: {[f.name for f in col.schema.fields]}")
```

- [ ] **Step 2: Commit**

```bash
git add capstone/backend/src/ingestion/milvus_setup.py
git commit -m "feat(milvus): add BM25 Function + sparse vector field to collection schema"
```

---

### Task 2: Add Hybrid Search to search.py

**Files:**
- Modify: `capstone/backend/src/retrieval/search.py`
- Modify: `capstone/backend/tests/unit/test_search.py`

- [ ] **Step 1: Write failing tests for hybrid search**

Add to `tests/unit/test_search.py`:

```python
from src.retrieval.search import _resolve_search_mode


def test_resolve_search_mode_from_filters():
    filters = SearchFilters(search_mode="hybrid")
    assert _resolve_search_mode(filters) == "hybrid"


def test_resolve_search_mode_default_from_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    filters = SearchFilters()  # search_mode is None
    # Should fall back to config default ("dense")
    mode = _resolve_search_mode(filters)
    assert mode == "dense"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd capstone/backend
uv run pytest tests/unit/test_search.py::test_resolve_search_mode_from_filters -v
```

Expected: FAIL — `_resolve_search_mode` not found.

- [ ] **Step 3: Implement hybrid search in search.py**

Replace the full `search.py`:

```python
# src/retrieval/search.py
"""Milvus vector search with metadata filtering.

Supports dense search (Phase A) and hybrid search (Phase B via search_mode parameter).
"""

import json
import logging

from openai import OpenAI
from pymilvus import AnnSearchRequest, Collection, RRFRanker

from src.shared.config import get_settings
from src.shared.models import SearchFilters, SearchResult

logger = logging.getLogger(__name__)

OUTPUT_FIELDS = ["pmid", "title", "abstract_text", "year", "journal", "mesh_terms"]


def _get_openai_client() -> OpenAI:
    """Lazy OpenAI client initialization (avoids import-time OPENAI_API_KEY check)."""
    return OpenAI()


def build_filter_expression(filters: SearchFilters) -> str:
    """Build a Milvus boolean filter expression from SearchFilters."""
    conditions = []

    if filters.year_min is not None:
        conditions.append(f"year >= {filters.year_min}")
    if filters.year_max is not None:
        conditions.append(f"year <= {filters.year_max}")
    if filters.journals:
        journals_str = json.dumps(filters.journals)
        conditions.append(f"journal in {journals_str}")

    return " and ".join(conditions)


def embed_query(query: str) -> list[float]:
    """Embed a query string using the configured embedding model."""
    settings = get_settings()
    client = _get_openai_client()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=[query],
    )
    return response.data[0].embedding


def parse_search_results(hits: list) -> list[SearchResult]:
    """Convert raw Milvus hits into SearchResult models.

    Milvus COSINE metric: distance = cosine similarity (0-1, higher = more similar).
    For hybrid search, scores are RRF-fused (higher = more relevant).
    """
    results = []
    for hit in hits:
        mesh_raw = hit.entity.get("mesh_terms")
        mesh_terms = json.loads(mesh_raw) if isinstance(mesh_raw, str) else mesh_raw

        results.append(
            SearchResult(
                pmid=hit.entity.get("pmid"),
                title=hit.entity.get("title"),
                abstract_text=hit.entity.get("abstract_text"),
                score=hit.distance,
                year=hit.entity.get("year"),
                journal=hit.entity.get("journal"),
                mesh_terms=mesh_terms if mesh_terms else [],
            )
        )
    return results


def _resolve_search_mode(filters: SearchFilters) -> str:
    """Resolve search mode: per-query override > config default."""
    if filters.search_mode is not None:
        return filters.search_mode
    return get_settings().search_mode


def _dense_search(
    query_embedding: list[float],
    collection: Collection,
    filters: SearchFilters,
    filter_expr: str,
) -> list[SearchResult]:
    """Execute dense-only vector search."""
    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

    collection.load()
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=filters.top_k,
        expr=filter_expr if filter_expr else None,
        output_fields=OUTPUT_FIELDS,
    )

    if not results or len(results) == 0:
        return []

    return parse_search_results(results[0])


def _hybrid_search(
    query: str,
    query_embedding: list[float],
    collection: Collection,
    filters: SearchFilters,
    filter_expr: str,
) -> list[SearchResult]:
    """Execute hybrid search (dense + BM25 via RRF fusion)."""
    dense_req = AnnSearchRequest(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=filters.top_k,
        expr=filter_expr if filter_expr else None,
    )

    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field="chunk_text_sparse",
        param={"metric_type": "BM25"},
        limit=filters.top_k,
        expr=filter_expr if filter_expr else None,
    )

    collection.load()
    results = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=RRFRanker(k=60),
        limit=filters.top_k,
        output_fields=OUTPUT_FIELDS,
    )

    if not results or len(results) == 0:
        return []

    return parse_search_results(results[0])


def search(
    query: str,
    collection: Collection,
    filters: SearchFilters | None = None,
) -> list[SearchResult]:
    """Execute vector search against Milvus.

    Args:
        query: Natural language query (will be embedded).
        collection: Milvus collection to search.
        filters: Optional metadata filters. search_mode controls dense vs hybrid.

    Returns: List of SearchResult sorted by relevance.
    """
    if filters is None:
        filters = SearchFilters()

    query_embedding = embed_query(query)
    filter_expr = build_filter_expression(filters)
    mode = _resolve_search_mode(filters)

    if mode == "hybrid":
        logger.info("Executing hybrid search (dense + BM25)")
        return _hybrid_search(query, query_embedding, collection, filters, filter_expr)
    else:
        logger.info("Executing dense search")
        return _dense_search(query_embedding, collection, filters, filter_expr)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd capstone/backend
uv run pytest tests/unit/test_search.py -v
```

Expected: All 8 tests PASS (6 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add capstone/backend/src/retrieval/search.py capstone/backend/tests/unit/test_search.py
git commit -m "feat(retrieval): add hybrid search mode (dense + BM25 via RRF fusion)"
```
