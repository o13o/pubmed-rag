# Phase B-3: Reranker (Cross-Encoder + Protocol)

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Phase A reranker passthrough with a Protocol-based abstraction supporting CrossEncoderReranker (sentence-transformers, default), LLMReranker (LiteLLM fallback), and NoOpReranker (passthrough).

**Architecture:** `BaseReranker` Protocol defines the interface. Three implementations in `reranker.py`. Config `reranker_type` selects which to use. Factory function `get_reranker()` creates the appropriate instance.

**Tech Stack:** sentence-transformers (cross-encoder/ms-marco-MiniLM-L-6-v2), LiteLLM

**Spec:** [2026-03-14-phase-b-design.md](../specs/2026-03-14-phase-b-design.md) — Section 6

**Dependency:** B-S (shared models + config) must be merged first. Uses `reranker_type`, `reranker_model` from config. `sentence-transformers` must be installed (B-S Task 3).

---

## Chunk 1: Reranker Implementations

### Task 1: Replace Reranker Stub with Protocol + Implementations

**Files:**
- Modify: `backend/src/retrieval/reranker.py`
- Create: `backend/tests/unit/test_reranker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_reranker.py
"""Tests for reranker implementations."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.shared.models import SearchResult
from src.retrieval.reranker import (
    NoOpReranker,
    CrossEncoderReranker,
    LLMReranker,
    get_reranker,
)


def _make_results(n: int = 5) -> list[SearchResult]:
    return [
        SearchResult(
            pmid=str(i), title=f"Title {i}", abstract_text=f"Abstract text {i}",
            score=0.9 - i * 0.1, year=2023, journal="J", mesh_terms=[],
        )
        for i in range(n)
    ]


class TestNoOpReranker:
    def test_returns_same_results(self):
        results = _make_results(3)
        reranker = NoOpReranker()
        reranked = reranker.rerank("test query", results, top_k=3)
        assert reranked == results

    def test_respects_top_k(self):
        results = _make_results(5)
        reranker = NoOpReranker()
        reranked = reranker.rerank("test query", results, top_k=2)
        assert len(reranked) == 2


class TestCrossEncoderReranker:
    @patch("src.retrieval.reranker.CrossEncoder")
    def test_reranks_by_cross_encoder_scores(self, mock_ce_cls):
        mock_ce = MagicMock()
        # Return scores in reverse order so reranker must re-sort
        mock_ce.predict.return_value = [0.1, 0.5, 0.9]
        mock_ce_cls.return_value = mock_ce

        results = _make_results(3)
        reranker = CrossEncoderReranker(model_name="test-model")
        reranked = reranker.rerank("test query", results, top_k=2)

        assert len(reranked) == 2
        # Highest score (0.9) was the 3rd result (index 2, pmid="2")
        assert reranked[0].pmid == "2"
        assert reranked[0].score == 0.9

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_lazy_model_loading(self, mock_ce_cls):
        reranker = CrossEncoderReranker(model_name="test-model")
        # Model not loaded yet
        mock_ce_cls.assert_not_called()

        results = _make_results(2)
        reranker.rerank("query", results, top_k=2)
        # Now loaded
        mock_ce_cls.assert_called_once_with("test-model")


class TestLLMReranker:
    def test_reranks_by_llm_scores(self):
        mock_llm = MagicMock()
        # Return scores as strings (LLM output)
        mock_llm.complete.side_effect = ["3", "8", "5"]

        results = _make_results(3)
        reranker = LLMReranker(llm=mock_llm)
        reranked = reranker.rerank("test query", results, top_k=2)

        assert len(reranked) == 2
        # Highest score (8) was result at index 1 (pmid="1")
        assert reranked[0].pmid == "1"

    def test_handles_malformed_llm_response(self):
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = ["not a number", "7", "3"]

        results = _make_results(3)
        reranker = LLMReranker(llm=mock_llm)
        reranked = reranker.rerank("test query", results, top_k=3)

        # Malformed response gets score 0, so it should be last
        assert len(reranked) == 3


class TestGetReranker:
    def test_get_noop_reranker(self):
        reranker = get_reranker(reranker_type="none")
        assert isinstance(reranker, NoOpReranker)

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_get_cross_encoder_reranker(self, mock_ce_cls):
        reranker = get_reranker(reranker_type="cross_encoder", model_name="test")
        assert isinstance(reranker, CrossEncoderReranker)

    def test_get_llm_reranker(self):
        mock_llm = MagicMock()
        reranker = get_reranker(reranker_type="llm", llm=mock_llm)
        assert isinstance(reranker, LLMReranker)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown reranker type"):
            get_reranker(reranker_type="unknown")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/unit/test_reranker.py -v
```

Expected: FAIL — imports fail.

- [ ] **Step 3: Implement reranker module**

```python
# src/retrieval/reranker.py
"""Reranker implementations with Protocol abstraction.

Supports: NoOp (passthrough), CrossEncoder (sentence-transformers), LLM (LiteLLM).
"""

import logging
from typing import Protocol, runtime_checkable

from src.shared.models import SearchResult

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseReranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...


class NoOpReranker:
    """Passthrough reranker (Phase A behavior)."""

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        return results[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers.

    Model is loaded lazily on first call and cached.
    Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (CPU-friendly, ~80MB).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []

        model = self._get_model()
        pairs = [(query, r.abstract_text) for r in results]
        scores = model.predict(pairs)

        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for result, score in scored[:top_k]:
            reranked.append(result.model_copy(update={"score": float(score)}))
        return reranked


class LLMReranker:
    """LLM-based pointwise reranker. Fallback when cross-encoder is unavailable."""

    SCORING_PROMPT = """Rate the relevance of this abstract to the query on a scale of 0-10.
Return ONLY a single number.

Query: {query}
Abstract: {abstract}
Score:"""

    def __init__(self, llm):
        self.llm = llm

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []

        scored = []
        for r in results:
            try:
                response = self.llm.complete(
                    system_prompt="You rate document relevance. Return only a number 0-10.",
                    user_prompt=self.SCORING_PROMPT.format(query=query, abstract=r.abstract_text),
                )
                score = float(response.strip())
            except (ValueError, Exception) as e:
                logger.warning("LLM scoring failed for PMID %s: %s", r.pmid, e)
                score = 0.0
            scored.append((r, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [r.model_copy(update={"score": s}) for r, s in scored[:top_k]]


def get_reranker(
    reranker_type: str = "cross_encoder",
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    llm=None,
) -> BaseReranker:
    """Factory function to create a reranker based on type."""
    if reranker_type == "none":
        return NoOpReranker()
    elif reranker_type == "cross_encoder":
        return CrossEncoderReranker(model_name=model_name)
    elif reranker_type == "llm":
        if llm is None:
            raise ValueError("LLMReranker requires an llm argument")
        return LLMReranker(llm=llm)
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/unit/test_reranker.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Update retrieval `__init__.py`**

```python
# src/retrieval/__init__.py
"""Retrieval module - public interface."""

from src.retrieval.query_expander import QueryExpander
from src.retrieval.reranker import get_reranker
from src.retrieval.search import search

__all__ = ["QueryExpander", "get_reranker", "search"]
```

- [ ] **Step 6: Commit**

```bash
git add backend/src/retrieval/reranker.py backend/src/retrieval/__init__.py backend/tests/unit/test_reranker.py
git commit -m "feat(retrieval): add Protocol-based reranker (CrossEncoder + LLM + NoOp)"
```
