# Phase A-4/5/6: Retrieval, RAG Chain, E2E Validation

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the retrieval layer (vector search + MeSH query expansion), RAG chain (retrieve → prompt → LLM → cited answer), and validate end-to-end via CLI.

**Architecture:** `retrieval/` handles search and query expansion. `rag/` orchestrates the full chain. Both consume shared models and services. CLI provides E2E entry point.

**Tech Stack:** pymilvus, LiteLLM, DuckDB (MeSH), pytest

**Spec:** [2026-03-14-pubmed-rag-system-design.md](../specs/2026-03-14-pubmed-rag-system-design.md) - Sections 4.2, 3.3

**Prerequisites:** A-1 (Milvus running + collection), A-2 (ingestion), A-3 (shared models, config, LLM, MeSH DB) all merged.

---

## Chunk 1: Retrieval Module

### Task 1: Vector Search

**Files:**
- Create: `capstone/backend/src/retrieval/search.py`
- Create: `capstone/backend/tests/unit/test_search.py`

- [ ] **Step 1: Write failing tests for search**

```python
# tests/unit/test_search.py
"""Tests for Milvus vector search."""

from unittest.mock import MagicMock, patch

from src.shared.models import SearchFilters, SearchResult
from src.retrieval.search import build_filter_expression, parse_search_results


def test_build_filter_no_filters():
    filters = SearchFilters()
    expr = build_filter_expression(filters)
    assert expr == ""


def test_build_filter_year_range():
    filters = SearchFilters(year_min=2022, year_max=2024)
    expr = build_filter_expression(filters)
    assert "year >= 2022" in expr
    assert "year <= 2024" in expr


def test_build_filter_year_min_only():
    filters = SearchFilters(year_min=2023)
    expr = build_filter_expression(filters)
    assert "year >= 2023" in expr
    assert "year <=" not in expr


def test_build_filter_journals():
    filters = SearchFilters(journals=["Nature", "Science"])
    expr = build_filter_expression(filters)
    assert 'journal in ["Nature", "Science"]' in expr


def test_build_filter_combined():
    filters = SearchFilters(year_min=2022, journals=["Nature"])
    expr = build_filter_expression(filters)
    assert "year >= 2022" in expr
    assert "journal" in expr
    assert " and " in expr


def test_parse_search_results():
    """Parse raw Milvus search results into SearchResult models.

    Note: Milvus COSINE metric returns similarity (0-1, higher is better) as distance.
    """
    entity_data = {
        "pmid": "123",
        "title": "Test Title",
        "abstract_text": "Test abstract",
        "year": 2023,
        "journal": "Nature",
        "mesh_terms": '["Neoplasms"]',
    }
    mock_hit = MagicMock()
    mock_hit.entity.get = lambda k: entity_data[k]
    mock_hit.distance = 0.95  # Milvus COSINE: higher = more similar

    results = parse_search_results([mock_hit])
    assert len(results) == 1
    assert results[0].pmid == "123"
    assert results[0].score == 0.95
    assert results[0].mesh_terms == ["Neoplasms"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd capstone/backend
uv run pytest tests/unit/test_search.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement search module**

```python
# src/retrieval/search.py
"""Milvus vector search with metadata filtering.

Supports dense search (Phase A) and hybrid search (Phase B via search_mode parameter).
"""

import json
import logging

from openai import OpenAI
from pymilvus import Collection

from src.shared.config import get_settings
from src.shared.models import SearchFilters, SearchResult

logger = logging.getLogger(__name__)


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


def search(
    query: str,
    collection: Collection,
    filters: SearchFilters | None = None,
) -> list[SearchResult]:
    """Execute vector search against Milvus.

    Args:
        query: Natural language query (will be embedded).
        collection: Milvus collection to search.
        filters: Optional metadata filters.

    Returns: List of SearchResult sorted by relevance (cosine similarity).
    """
    if filters is None:
        filters = SearchFilters()

    query_embedding = embed_query(query)
    filter_expr = build_filter_expression(filters)

    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

    collection.load()
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=filters.top_k,
        expr=filter_expr if filter_expr else None,
        output_fields=["pmid", "title", "abstract_text", "year", "journal", "mesh_terms"],
    )

    if not results or len(results) == 0:
        return []

    return parse_search_results(results[0])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd capstone/backend
uv run pytest tests/unit/test_search.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add capstone/backend/src/retrieval/search.py capstone/backend/tests/unit/test_search.py
git commit -m "feat(retrieval): add Milvus vector search with metadata filtering"
```

---

### Task 2: MeSH Query Expansion

**Files:**
- Create: `capstone/backend/src/retrieval/query_expander.py`
- Create: `capstone/backend/tests/unit/test_query_expander.py`

- [ ] **Step 1: Write failing tests for query expander**

```python
# tests/unit/test_query_expander.py
"""Tests for MeSH-based query expansion."""

from unittest.mock import MagicMock, patch

import duckdb
import pytest

from src.shared.mesh_db import MeSHDatabase
from src.retrieval.query_expander import QueryExpander


@pytest.fixture
def mesh_db():
    db = MeSHDatabase(":memory:")
    db._init_schema()
    db.conn.execute("""
        INSERT INTO mesh_descriptors VALUES
        ('D009369', 'Neoplasms', ['C04']),
        ('D001943', 'Breast Neoplasms', ['C04.588.180']),
        ('D020370', 'Osteoarthritis, Knee', ['C05.550.114.606'])
    """)
    db.conn.execute("""
        INSERT INTO mesh_synonyms VALUES
        ('Cancer', 'D009369'),
        ('Breast Cancer', 'D001943'),
        ('Knee Osteoarthritis', 'D020370'),
        ('Knee Pain', 'D020370')
    """)
    return db


def test_expand_with_mesh_terms(mesh_db):
    """When LLM extracts keywords that match MeSH, expand with descriptors and children."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '["cancer", "treatment"]'

    expander = QueryExpander(llm=mock_llm, mesh_db=mesh_db)
    result = expander.expand("What are the latest cancer treatments?")

    assert "cancer" in result.original_query.lower()
    assert "Neoplasms" in result.mesh_terms
    # Breast Neoplasms is a child of Neoplasms (C04 → C04.588.180)
    assert "Breast Neoplasms" in result.child_terms
    assert result.expanded_query != result.original_query


def test_expand_no_mesh_match(mesh_db):
    """When keywords don't match MeSH, return original query unchanged."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '["completely_unknown_term"]'

    expander = QueryExpander(llm=mock_llm, mesh_db=mesh_db)
    result = expander.expand("completely unknown medical query")

    assert result.mesh_terms == []
    assert result.expanded_query == result.original_query


def test_expand_query_format(mesh_db):
    """Expanded query should include original + MeSH terms."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '["knee pain"]'

    expander = QueryExpander(llm=mock_llm, mesh_db=mesh_db)
    result = expander.expand("knee pain treatment options")

    assert "knee pain treatment options" in result.expanded_query
    assert "Osteoarthritis, Knee" in result.expanded_query
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd capstone/backend
uv run pytest tests/unit/test_query_expander.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement query expander**

```python
# src/retrieval/query_expander.py
"""MeSH-based query expansion: LLM keyword extraction + DuckDB MeSH hierarchy lookup.

Flow (per spec Section 4.2):
1. LLM extracts medical keywords from natural language
2. DuckDB MeSH lookup: keywords → descriptors + synonyms + child terms
3. Build expanded query (original + MeSH terms)
"""

import json
import logging

from pydantic import BaseModel, Field

from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase

logger = logging.getLogger(__name__)

KEYWORD_EXTRACTION_PROMPT = """Extract medical/biomedical keywords from the user's query.
Return ONLY a JSON array of keyword strings, no explanation.
Focus on diseases, conditions, treatments, drugs, anatomical terms.

Example:
Query: "What are the latest treatments for knee osteoarthritis in elderly patients?"
Output: ["knee osteoarthritis", "treatment", "elderly"]

Query: "{query}"
Output:"""


class ExpandedQuery(BaseModel):
    """Result of query expansion."""

    original_query: str
    keywords: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    child_terms: list[str] = Field(default_factory=list)
    expanded_query: str = ""


class QueryExpander:
    """Expand queries using LLM keyword extraction + MeSH hierarchy lookup."""

    def __init__(self, llm: LLMClient, mesh_db: MeSHDatabase):
        self.llm = llm
        self.mesh_db = mesh_db

    def _extract_keywords(self, query: str) -> list[str]:
        """Use LLM to extract medical keywords from the query."""
        prompt = KEYWORD_EXTRACTION_PROMPT.format(query=query)
        response = self.llm.complete(
            system_prompt="You extract medical keywords from queries. Return only a JSON array.",
            user_prompt=prompt,
        )
        try:
            keywords = json.loads(response.strip())
            if isinstance(keywords, list):
                return [str(k).strip() for k in keywords if k]
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM keyword response: %s", response)
        return []

    def expand(self, query: str) -> ExpandedQuery:
        """Expand a query with MeSH terms.

        1. Extract keywords via LLM
        2. Look up each keyword in MeSH (name or synonym)
        3. Get child terms for matched descriptors
        4. Build expanded query string
        """
        keywords = self._extract_keywords(query)
        logger.debug("Extracted keywords: %s", keywords)

        mesh_terms = []
        child_terms = []
        seen = set()

        for keyword in keywords:
            result = self.mesh_db.lookup(keyword)
            if result is None:
                continue

            name = result["name"]
            if name not in seen:
                mesh_terms.append(name)
                seen.add(name)

            # Get child terms via tree number prefix match
            for tree_num in result.get("tree_numbers", []):
                children = self.mesh_db.get_children(tree_num)
                for child in children:
                    if child["name"] not in seen:
                        child_terms.append(child["name"])
                        seen.add(child["name"])

        # Build expanded query
        all_terms = mesh_terms + child_terms
        if all_terms:
            expanded = f"{query} ({'; '.join(all_terms)})"
        else:
            expanded = query

        result = ExpandedQuery(
            original_query=query,
            keywords=keywords,
            mesh_terms=mesh_terms,
            child_terms=child_terms,
            expanded_query=expanded,
        )
        logger.info("Query expanded: %d MeSH terms, %d children", len(mesh_terms), len(child_terms))
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd capstone/backend
uv run pytest tests/unit/test_query_expander.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add capstone/backend/src/retrieval/query_expander.py capstone/backend/tests/unit/test_query_expander.py
git commit -m "feat(retrieval): add MeSH-based query expansion (LLM + DuckDB)"
```

---

### Task 3: Retrieval Public Interface + Reranker Stub

**Files:**
- Create: `capstone/backend/src/retrieval/reranker.py`
- Modify: `capstone/backend/src/retrieval/__init__.py`

- [ ] **Step 1: Create reranker stub**

```python
# src/retrieval/reranker.py
"""Reranker stub for Phase B (cross-encoder)."""

from src.shared.models import SearchResult


def rerank(results: list[SearchResult], query: str) -> list[SearchResult]:
    """Rerank search results. Phase A: passthrough. Phase B: cross-encoder."""
    return results
```

- [ ] **Step 2: Update `__init__.py`**

```python
# src/retrieval/__init__.py
"""Retrieval module - public interface."""

from src.retrieval.query_expander import QueryExpander
from src.retrieval.search import search

__all__ = ["QueryExpander", "search"]
```

- [ ] **Step 3: Commit**

```bash
git add capstone/backend/src/retrieval/
git commit -m "feat(retrieval): add reranker stub and public interface"
```

---

## Chunk 2: RAG Chain

### Task 4: Prompt Templates

**Files:**
- Create: `capstone/backend/src/rag/prompts.py`
- Create: `capstone/backend/tests/unit/test_prompts.py`

- [ ] **Step 1: Write failing tests for prompts**

```python
# tests/unit/test_prompts.py
"""Tests for prompt templates."""

from src.rag.prompts import build_system_prompt, build_user_prompt
from src.shared.models import SearchResult


def test_system_prompt_contains_instructions():
    prompt = build_system_prompt()
    assert "cite" in prompt.lower() or "citation" in prompt.lower()
    assert "PMID" in prompt


def test_user_prompt_includes_query_and_abstracts():
    results = [
        SearchResult(
            pmid="111", title="Title 1", abstract_text="Abstract 1",
            score=0.95, year=2023, journal="J1", mesh_terms=["Neoplasms"],
        ),
        SearchResult(
            pmid="222", title="Title 2", abstract_text="Abstract 2",
            score=0.90, year=2024, journal="J2", mesh_terms=[],
        ),
    ]
    prompt = build_user_prompt("test query", results)
    assert "test query" in prompt
    assert "PMID: 111" in prompt
    assert "Title 1" in prompt
    assert "Abstract 1" in prompt
    assert "PMID: 222" in prompt


def test_user_prompt_empty_results():
    prompt = build_user_prompt("test query", [])
    assert "test query" in prompt
    assert "no relevant" in prompt.lower() or "no abstracts" in prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd capstone/backend
uv run pytest tests/unit/test_prompts.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement prompts**

```python
# src/rag/prompts.py
"""Prompt templates for the RAG chain."""

from src.shared.models import SearchResult

SYSTEM_PROMPT = """You are a medical research assistant that answers questions based on PubMed abstracts.

Rules:
1. ONLY use information from the provided abstracts to answer the question.
2. ALWAYS cite your sources using PMID numbers in the format [PMID: 12345678].
3. If the abstracts don't contain enough information, say so explicitly.
4. Be precise and use appropriate medical terminology.
5. Do NOT provide medical advice or treatment recommendations without qualifying language.
6. Structure your answer clearly with relevant findings from the literature."""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_user_prompt(query: str, results: list[SearchResult]) -> str:
    """Build the user prompt with the query and retrieved abstracts."""
    if not results:
        return f"""Question: {query}

No relevant abstracts were found. Please inform the user that no relevant research was found for their query."""

    abstracts_text = []
    for i, r in enumerate(results, 1):
        abstracts_text.append(
            f"[{i}] PMID: {r.pmid}\n"
            f"Title: {r.title}\n"
            f"Journal: {r.journal} ({r.year})\n"
            f"Abstract: {r.abstract_text}\n"
        )

    return f"""Question: {query}

Relevant abstracts:

{"\n---\n".join(abstracts_text)}

Based on the abstracts above, provide a comprehensive answer to the question. Cite each claim with the relevant PMID."""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd capstone/backend
uv run pytest tests/unit/test_prompts.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add capstone/backend/src/rag/prompts.py capstone/backend/tests/unit/test_prompts.py
git commit -m "feat(rag): add prompt templates with citation instructions"
```

---

### Task 5: RAG Chain

**Files:**
- Create: `capstone/backend/src/rag/chain.py`
- Modify: `capstone/backend/src/rag/__init__.py`
- Create: `capstone/backend/tests/unit/test_chain.py`

- [ ] **Step 1: Write failing tests for RAG chain**

```python
# tests/unit/test_chain.py
"""Tests for RAG chain."""

import re
from unittest.mock import MagicMock, patch

from src.shared.models import Citation, RAGResponse, SearchFilters, SearchResult
from src.rag.chain import ask


def _mock_search_results():
    return [
        SearchResult(
            pmid="111", title="Title 1", abstract_text="Abstract about cancer treatment.",
            score=0.95, year=2023, journal="Nature", mesh_terms=["Neoplasms"],
        ),
    ]


@patch("src.rag.chain.search")
@patch("src.rag.chain.QueryExpander")
def test_ask_returns_rag_response(mock_expander_cls, mock_search):
    mock_search.return_value = _mock_search_results()
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="cancer treatment")
    mock_expander_cls.return_value = mock_expander

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "Based on PMID: 111, cancer treatment shows..."

    response = ask(
        query="cancer treatment",
        collection=MagicMock(),
        llm=mock_llm,
        mesh_db=MagicMock(),
    )

    assert isinstance(response, RAGResponse)
    assert response.query == "cancer treatment"
    assert len(response.answer) > 0
    assert len(response.citations) == 1


@patch("src.rag.chain.search")
@patch("src.rag.chain.QueryExpander")
def test_ask_with_no_results(mock_expander_cls, mock_search):
    mock_search.return_value = []
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="unknown query")
    mock_expander_cls.return_value = mock_expander

    mock_llm = MagicMock()
    mock_llm.complete.return_value = "No relevant research was found."

    response = ask(
        query="unknown query",
        collection=MagicMock(),
        llm=mock_llm,
        mesh_db=MagicMock(),
    )

    assert isinstance(response, RAGResponse)
    assert response.citations == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd capstone/backend
uv run pytest tests/unit/test_chain.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement RAG chain**

```python
# src/rag/chain.py
"""RAG chain: retrieve → expand → prompt → LLM → cited response.

Orchestrates the full retrieval-augmented generation pipeline.
"""

import logging

from pymilvus import Collection

from src.rag.prompts import build_system_prompt, build_user_prompt
from src.retrieval.query_expander import QueryExpander
from src.retrieval.search import search
from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase
from src.shared.models import Citation, RAGResponse, SearchFilters, SearchResult

logger = logging.getLogger(__name__)


def ask(
    query: str,
    collection: Collection,
    llm: LLMClient,
    mesh_db: MeSHDatabase,
    filters: SearchFilters | None = None,
) -> RAGResponse:
    """Execute the full RAG pipeline.

    1. Expand query with MeSH terms
    2. Search Milvus for relevant abstracts
    3. Build prompt with query + retrieved abstracts
    4. Call LLM for answer generation
    5. Package response with citations
    """
    if filters is None:
        filters = SearchFilters()

    # 1. Query expansion
    expander = QueryExpander(llm=llm, mesh_db=mesh_db)
    expanded = expander.expand(query)
    logger.info("Expanded query: '%s' → '%s'", query, expanded.expanded_query)

    # 2. Search
    results = search(expanded.expanded_query, collection, filters)
    logger.info("Retrieved %d results", len(results))

    # 3. Build prompt
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(query, results)

    # 4. Generate answer
    answer = llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)

    # 5. Build citations from search results
    citations = [
        Citation(
            pmid=r.pmid,
            title=r.title,
            journal=r.journal,
            year=r.year,
            relevance_score=r.score,
        )
        for r in results
    ]

    return RAGResponse(
        answer=answer,
        citations=citations,
        query=query,
    )
```

- [ ] **Step 4: Update `__init__.py`**

```python
# src/rag/__init__.py
"""RAG module - public interface."""

from src.rag.chain import ask

__all__ = ["ask"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd capstone/backend
uv run pytest tests/unit/test_chain.py -v
```

Expected: All 2 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add capstone/backend/src/rag/ capstone/backend/tests/unit/test_chain.py
git commit -m "feat(rag): add RAG chain (retrieve → expand → prompt → LLM → response)"
```

---

## Chunk 3: CLI and E2E Validation

### Task 6: CLI Entry Point

**Files:**
- Create: `capstone/backend/src/cli.py`

- [ ] **Step 1: Implement CLI**

```python
# src/cli.py
"""CLI for PubMed RAG system.

Usage:
    uv run python -m src.cli "What are the latest treatments for breast cancer?"
    uv run python -m src.cli "knee pain treatment" --year-min 2023 --top-k 5
"""

import argparse
import json
import logging
import sys

from pymilvus import Collection, connections

from src.rag.chain import ask
from src.shared.config import get_settings
from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase
from src.shared.models import SearchFilters


def main():
    parser = argparse.ArgumentParser(description="PubMed RAG - Ask questions about medical research")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--year-min", type=int, default=None, help="Minimum publication year")
    parser.add_argument("--year-max", type=int, default=None, help="Maximum publication year")
    parser.add_argument("--journals", nargs="*", default=[], help="Filter by journal names")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument("--model", default=None, help="LLM model override (default: gpt-4o-mini)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()

    # Connect to Milvus
    connections.connect("default", host=settings.milvus_host, port=str(settings.milvus_port))
    collection = Collection(settings.milvus_collection)

    # Initialize services
    model = args.model or settings.llm_model
    llm = LLMClient(model=model, timeout=settings.llm_timeout)
    mesh_db = MeSHDatabase(settings.mesh_db_path)

    # Build filters
    filters = SearchFilters(
        year_min=args.year_min,
        year_max=args.year_max,
        journals=args.journals,
        top_k=args.top_k,
    )

    # Execute RAG
    response = ask(
        query=args.query,
        collection=collection,
        llm=llm,
        mesh_db=mesh_db,
        filters=filters,
    )

    # Output
    if args.json:
        print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*60}")
        print(f"Query: {response.query}")
        print(f"{'='*60}\n")
        print(response.answer)
        print(f"\n{'='*60}")
        print(f"Citations ({len(response.citations)}):")
        print(f"{'='*60}")
        for c in response.citations:
            print(f"  PMID: {c.pmid} | {c.title}")
            print(f"       {c.journal} ({c.year}) | Score: {c.relevance_score:.3f}")

    # Cleanup
    mesh_db.close()
    connections.disconnect("default")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add capstone/backend/src/cli.py
git commit -m "feat: add CLI entry point for PubMed RAG"
```

---

### Task 7: E2E Validation

**Prerequisite:** Milvus running, data ingested, MeSH DuckDB built.

- [ ] **Step 1: Ingest sample data**

```bash
cd capstone/backend
# Assuming sampled.jsonl exists from playground pipeline
uv run python -c "
from pymilvus import connections
from src.ingestion.milvus_setup import create_collection
from src.ingestion.pipeline import ingest
from pathlib import Path

connections.connect('default', host='localhost', port='19530')
collection = create_collection()
report = ingest(Path('../playground/pubmed_pipeline/data/processed/sampled.jsonl'), collection)
print(report)
"
```

Expected: IngestReport showing articles loaded and upserted.

- [ ] **Step 2: Run E2E query via CLI**

```bash
cd capstone/backend
uv run python -m src.cli "What are the recent advances in breast cancer immunotherapy?" --top-k 5 --year-min 2023
```

Expected: A coherent answer citing specific PMIDs, with relevant abstracts.

- [ ] **Step 3: Run E2E query with JSON output**

```bash
cd capstone/backend
uv run python -m src.cli "treatment options for knee osteoarthritis" --top-k 3 --json
```

Expected: JSON output with `answer`, `citations`, and `query` fields.

- [ ] **Step 4: Run E2E with metadata filter**

```bash
cd capstone/backend
uv run python -m src.cli "cardiovascular risk factors" --year-min 2022 --year-max 2024 --top-k 5
```

Expected: Results filtered to 2022-2024 only.

- [ ] **Step 5: Run full test suite**

```bash
cd capstone/backend
uv run pytest tests/ -v --tb=short
```

Expected: All unit tests PASS. Integration tests PASS if Milvus is running.

- [ ] **Step 6: Commit**

```bash
git commit -m "test: verify E2E RAG pipeline via CLI"
```
