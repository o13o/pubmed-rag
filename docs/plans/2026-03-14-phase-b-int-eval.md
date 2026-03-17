# Phase B-INT + B-4: Integration & Evaluation Pipeline

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate reranker + guardrails into the RAG chain, update CLI with new flags, and build the DeepEval evaluation pipeline with extensible custom metrics.

**Architecture:** `chain.py` gains reranker step (between search and prompt) and guardrails step (after LLM). CLI gains `--search-mode`, `--reranker`, `--no-guardrails` flags. Evaluation pipeline lives in `tests/eval/` with DeepEval metrics.

**Tech Stack:** pymilvus, LiteLLM, sentence-transformers, DeepEval

**Spec:** [2026-03-14-phase-b-design.md](../specs/2026-03-14-phase-b-design.md) — Sections 6.4, 4.5, 7

**Prerequisites:** B-S, B-1, B-2, B-3 all merged.

---

## Chunk 1: Chain Integration + CLI

### Task 1: Integrate Reranker + Guardrails into RAG Chain

**Files:**
- Modify: `backend/src/rag/chain.py`
- Modify: `backend/tests/unit/test_chain.py`

- [ ] **Step 1: Write failing tests for updated chain**

Replace `tests/unit/test_chain.py` with:

```python
# tests/unit/test_chain.py
"""Tests for RAG chain with reranker and guardrails."""

from unittest.mock import MagicMock, patch

from src.shared.models import (
    Citation, GuardrailWarning, RAGResponse, SearchFilters,
    SearchResult, ValidatedResponse,
)
from src.rag.chain import ask


def _mock_search_results():
    return [
        SearchResult(
            pmid="111", title="Title 1",
            abstract_text="Abstract about cancer treatment.",
            score=0.95, year=2023, journal="Nature", mesh_terms=["Neoplasms"],
        ),
    ]


@patch("src.rag.chain.search")
@patch("src.rag.chain.QueryExpander")
def test_ask_returns_validated_response(mock_expander_cls, mock_search):
    """With guardrails enabled, ask() should return ValidatedResponse."""
    mock_search.return_value = _mock_search_results()
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="cancer treatment")
    mock_expander_cls.return_value = mock_expander

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        "Based on PMID: 111, cancer treatment shows...",  # RAG answer
        "[]",  # Guardrail validation (no issues)
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = _mock_search_results()

    response = ask(
        query="cancer treatment",
        collection=MagicMock(),
        llm=mock_llm,
        mesh_db=MagicMock(),
        reranker=mock_reranker,
        guardrails_enabled=True,
    )

    assert isinstance(response, ValidatedResponse)
    assert response.query == "cancer treatment"
    assert len(response.answer) > 0
    assert len(response.citations) == 1
    assert response.disclaimer != ""
    mock_reranker.rerank.assert_called_once()


@patch("src.rag.chain.search")
@patch("src.rag.chain.QueryExpander")
def test_ask_without_guardrails(mock_expander_cls, mock_search):
    """With guardrails disabled, ask() should return RAGResponse."""
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
        guardrails_enabled=False,
    )

    assert isinstance(response, RAGResponse)
    assert not isinstance(response, ValidatedResponse)


@patch("src.rag.chain.search")
@patch("src.rag.chain.QueryExpander")
def test_ask_with_no_results(mock_expander_cls, mock_search):
    """Empty search results should still produce a response."""
    mock_search.return_value = []
    mock_expander = MagicMock()
    mock_expander.expand.return_value = MagicMock(expanded_query="unknown query")
    mock_expander_cls.return_value = mock_expander

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        "No relevant research was found.",  # RAG answer
        "[]",  # Guardrail validation
    ]

    response = ask(
        query="unknown query",
        collection=MagicMock(),
        llm=mock_llm,
        mesh_db=MagicMock(),
        guardrails_enabled=True,
    )

    assert isinstance(response, ValidatedResponse)
    assert response.citations == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/unit/test_chain.py -v
```

Expected: FAIL — new parameters not accepted.

- [ ] **Step 3: Update chain.py**

```python
# src/rag/chain.py
"""RAG chain: retrieve → expand → rerank → prompt → LLM → guardrails → response.

Orchestrates the full retrieval-augmented generation pipeline.
"""

import logging

from pymilvus import Collection

from src.guardrails.output import GuardrailValidator
from src.rag.prompts import build_system_prompt, build_user_prompt
from src.retrieval.query_expander import QueryExpander
from src.retrieval.reranker import BaseReranker, NoOpReranker
from src.retrieval.search import search
from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase
from src.shared.models import (
    Citation, RAGResponse, SearchFilters, SearchResult, ValidatedResponse,
)

logger = logging.getLogger(__name__)


def ask(
    query: str,
    collection: Collection,
    llm: LLMClient,
    mesh_db: MeSHDatabase,
    filters: SearchFilters | None = None,
    reranker: BaseReranker | None = None,
    guardrails_enabled: bool = True,
) -> RAGResponse | ValidatedResponse:
    """Execute the full RAG pipeline.

    1. Expand query with MeSH terms
    2. Search Milvus for relevant abstracts
    3. Rerank results (if reranker provided)
    4. Build prompt with query + retrieved abstracts
    5. Call LLM for answer generation
    6. Run guardrails (if enabled)
    7. Package response with citations
    """
    if filters is None:
        filters = SearchFilters()
    if reranker is None:
        reranker = NoOpReranker()

    # 1. Query expansion
    expander = QueryExpander(llm=llm, mesh_db=mesh_db)
    expanded = expander.expand(query)
    logger.info("Expanded query: '%s' → '%s'", query, expanded.expanded_query)

    # 2. Search
    results = search(expanded.expanded_query, collection, filters)
    logger.info("Retrieved %d results", len(results))

    # 3. Rerank
    results = reranker.rerank(query, results, top_k=filters.top_k)
    logger.info("After reranking: %d results", len(results))

    # 4. Build prompt
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(query, results)

    # 5. Generate answer
    answer = llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)

    # 6. Build citations from search results
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

    rag_response = RAGResponse(
        answer=answer,
        citations=citations,
        query=query,
    )

    # 7. Guardrails
    if guardrails_enabled:
        validator = GuardrailValidator(llm=llm, mesh_db=mesh_db)
        return validator.validate(rag_response, results)

    return rag_response
```

- [ ] **Step 4: Update `rag/__init__.py`**

```python
# src/rag/__init__.py
"""RAG module - public interface."""

from src.rag.chain import ask

__all__ = ["ask"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/unit/test_chain.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/src/rag/chain.py backend/src/rag/__init__.py backend/tests/unit/test_chain.py
git commit -m "feat(rag): integrate reranker and guardrails into chain pipeline"
```

---

### Task 2: Update CLI with Phase B Flags

**Files:**
- Modify: `backend/src/cli.py`

- [ ] **Step 1: Update CLI with new arguments**

```python
# src/cli.py
"""CLI for PubMed RAG system.

Usage:
    uv run python -m src.cli "What are the latest treatments for breast cancer?"
    uv run python -m src.cli "knee pain treatment" --year-min 2023 --top-k 5
    uv run python -m src.cli "cancer therapy" --search-mode hybrid --reranker cross_encoder
"""

import argparse
import json
import logging
import sys

from pymilvus import Collection, connections

from src.rag.chain import ask
from src.retrieval.reranker import get_reranker
from src.shared.config import get_settings
from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase
from src.shared.models import SearchFilters, ValidatedResponse


def main():
    parser = argparse.ArgumentParser(description="PubMed RAG - Ask questions about medical research")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--year-min", type=int, default=None, help="Minimum publication year")
    parser.add_argument("--year-max", type=int, default=None, help="Maximum publication year")
    parser.add_argument("--journals", nargs="*", default=[], help="Filter by journal names")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument("--model", default=None, help="LLM model override (default: gpt-4o-mini)")
    parser.add_argument("--search-mode", default=None, choices=["dense", "hybrid"],
                        help="Search mode (default: from config)")
    parser.add_argument("--reranker", default=None, choices=["none", "cross_encoder", "llm"],
                        help="Reranker type (default: from config)")
    parser.add_argument("--no-guardrails", action="store_true", help="Disable output guardrails")
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

    # Reranker
    reranker_type = args.reranker or settings.reranker_type
    reranker = get_reranker(
        reranker_type=reranker_type,
        model_name=settings.reranker_model,
        llm=llm if reranker_type == "llm" else None,
    )

    # Build filters
    filters = SearchFilters(
        year_min=args.year_min,
        year_max=args.year_max,
        journals=args.journals,
        top_k=args.top_k,
        search_mode=args.search_mode,
    )

    # Execute RAG
    response = ask(
        query=args.query,
        collection=collection,
        llm=llm,
        mesh_db=mesh_db,
        filters=filters,
        reranker=reranker,
        guardrails_enabled=not args.no_guardrails,
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

        if isinstance(response, ValidatedResponse):
            if response.warnings:
                print(f"\n{'='*60}")
                print(f"Warnings ({len(response.warnings)}):")
                print(f"{'='*60}")
                for w in response.warnings:
                    print(f"  [{w.severity}] {w.check}: {w.message}")
            print(f"\n{response.disclaimer}")

    # Cleanup
    mesh_db.close()
    connections.disconnect("default")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add backend/src/cli.py
git commit -m "feat(cli): add --search-mode, --reranker, --no-guardrails flags"
```

---

## Chunk 2: Evaluation Pipeline

### Task 3: DeepEval Evaluation Pipeline

**Files:**
- Create: `backend/tests/eval/__init__.py`
- Create: `backend/tests/eval/conftest.py`
- Create: `backend/tests/eval/dataset.json`
- Create: `backend/tests/eval/metrics/__init__.py`
- Create: `backend/tests/eval/metrics/custom.py`
- Create: `backend/tests/eval/test_rag_evaluation.py`

- [ ] **Step 1: Create eval directory structure**

```bash
cd backend
mkdir -p tests/eval/metrics
```

- [ ] **Step 2: Create `__init__.py` files**

```python
# tests/eval/__init__.py
# (empty)
```

```python
# tests/eval/metrics/__init__.py
# (empty)
```

- [ ] **Step 3: Create conftest.py**

```python
# tests/eval/conftest.py
"""DeepEval test configuration."""

import os
import pytest


@pytest.fixture(autouse=True)
def eval_env(monkeypatch):
    """Ensure OPENAI_API_KEY is set for evaluation runs."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping evaluation tests")
```

- [ ] **Step 4: Create evaluation dataset**

```json
[
  {
    "query": "What are the latest treatments for breast cancer?",
    "expected_output_keywords": ["immunotherapy", "targeted therapy", "chemotherapy"],
    "relevant_pmids": [],
    "notes": "General breast cancer treatment query"
  },
  {
    "query": "How effective are mRNA vaccines against COVID-19 variants?",
    "expected_output_keywords": ["mRNA", "vaccine", "efficacy", "variant"],
    "relevant_pmids": [],
    "notes": "COVID-19 vaccine effectiveness"
  },
  {
    "query": "What are the risk factors for cardiovascular disease?",
    "expected_output_keywords": ["hypertension", "diabetes", "obesity", "smoking"],
    "relevant_pmids": [],
    "notes": "CVD risk factors"
  },
  {
    "query": "Non-invasive treatment options for knee osteoarthritis",
    "expected_output_keywords": ["physical therapy", "exercise", "NSAIDs"],
    "relevant_pmids": [],
    "notes": "Knee OA non-surgical treatments"
  },
  {
    "query": "What is the role of gut microbiome in mental health?",
    "expected_output_keywords": ["microbiome", "depression", "anxiety", "gut-brain axis"],
    "relevant_pmids": [],
    "notes": "Gut-brain connection"
  }
]
```

Save to `tests/eval/dataset.json`.

- [ ] **Step 5: Create custom metrics module**

```python
# tests/eval/metrics/custom.py
"""Custom DeepEval metrics for PubMed RAG evaluation.

Extensible: add new metrics by inheriting from BaseMetric.
"""

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class CitationPresenceMetric(BaseMetric):
    """Check if the answer contains PMID citations."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        if not test_case.actual_output:
            self.score = 0.0
            self.reason = "No output generated"
            return self.score

        import re
        pmid_pattern = r"PMID[:\s]+\d+"
        citations = re.findall(pmid_pattern, test_case.actual_output)
        self.score = 1.0 if len(citations) > 0 else 0.0
        self.reason = f"Found {len(citations)} PMID citation(s)"
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Citation Presence"


class MedicalDisclaimerMetric(BaseMetric):
    """Check if the response includes a medical disclaimer."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        if not test_case.actual_output:
            self.score = 0.0
            self.reason = "No output generated"
            return self.score

        disclaimer_keywords = ["not medical advice", "consult", "healthcare professional", "disclaimer"]
        found = any(kw.lower() in test_case.actual_output.lower() for kw in disclaimer_keywords)
        self.score = 1.0 if found else 0.0
        self.reason = "Disclaimer present" if found else "No disclaimer found"
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Medical Disclaimer"
```

- [ ] **Step 6: Create main evaluation test file**

```python
# tests/eval/test_rag_evaluation.py
"""DeepEval evaluation suite for PubMed RAG.

Run with: uv run pytest tests/eval/test_rag_evaluation.py -v
Requires: OPENAI_API_KEY set, Milvus running, data ingested.
"""

import json
from pathlib import Path

import pytest

from deepeval import assert_test
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

from tests.eval.metrics.custom import CitationPresenceMetric, MedicalDisclaimerMetric


DATASET_PATH = Path(__file__).parent / "dataset.json"

# Metrics to run on every test case
METRICS = [
    FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini"),
    AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini"),
    CitationPresenceMetric(threshold=0.5),
]


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def _run_rag_query(query: str) -> tuple[str, list[str]]:
    """Execute RAG pipeline and return (answer, context_list).

    Requires Milvus running and data ingested.
    """
    from pymilvus import Collection, connections
    from src.rag.chain import ask
    from src.retrieval.reranker import get_reranker
    from src.shared.config import get_settings
    from src.shared.llm import LLMClient
    from src.shared.mesh_db import MeSHDatabase
    from src.shared.models import SearchFilters, ValidatedResponse

    settings = get_settings()
    connections.connect("default", host=settings.milvus_host, port=str(settings.milvus_port))
    collection = Collection(settings.milvus_collection)
    llm = LLMClient(model=settings.llm_model, timeout=settings.llm_timeout)
    mesh_db = MeSHDatabase(settings.mesh_db_path)
    reranker = get_reranker(
        reranker_type=settings.reranker_type,
        model_name=settings.reranker_model,
        llm=llm if settings.reranker_type == "llm" else None,
    )

    response = ask(
        query=query,
        collection=collection,
        llm=llm,
        mesh_db=mesh_db,
        reranker=reranker,
        guardrails_enabled=True,
    )

    # Extract context from citations
    answer = response.answer
    if isinstance(response, ValidatedResponse):
        answer = f"{response.answer}\n\n{response.disclaimer}"

    # Re-fetch abstracts for context
    from src.shared.models import SearchFilters as SF
    from src.retrieval.search import search, embed_query
    results = search(query, collection, SF(top_k=settings.top_k))
    context = [f"PMID: {r.pmid}\n{r.title}\n{r.abstract_text}" for r in results]

    mesh_db.close()
    connections.disconnect("default")

    return answer, context


@pytest.fixture(params=load_dataset(), ids=lambda d: d["query"][:50])
def eval_case(request):
    return request.param


def test_rag_quality(eval_case):
    """Run evaluation metrics on each dataset query."""
    query = eval_case["query"]
    answer, context = _run_rag_query(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=context,
    )

    for metric in METRICS:
        assert_test(test_case, [metric])
```

- [ ] **Step 7: Commit**

```bash
git add backend/tests/eval/
git commit -m "feat(eval): add DeepEval evaluation pipeline with custom metrics"
```

---

### Task 4: Run Full Test Suite

- [ ] **Step 1: Run all unit tests**

```bash
cd backend
uv run pytest tests/unit/ -v
```

Expected: All unit tests PASS (45 from Phase A + new Phase B tests).

- [ ] **Step 2: Commit if needed**

```bash
git commit -m "test: verify all Phase B unit tests pass"
```
