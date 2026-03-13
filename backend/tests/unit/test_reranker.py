"""Tests for reranker implementations."""

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
    @patch("sentence_transformers.CrossEncoder")
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

    @patch("sentence_transformers.CrossEncoder")
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

    def test_get_cross_encoder_reranker(self):
        reranker = get_reranker(reranker_type="cross_encoder", model_name="test")
        assert isinstance(reranker, CrossEncoderReranker)

    def test_get_llm_reranker(self):
        mock_llm = MagicMock()
        reranker = get_reranker(reranker_type="llm", llm=mock_llm)
        assert isinstance(reranker, LLMReranker)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown reranker type"):
            get_reranker(reranker_type="unknown")
