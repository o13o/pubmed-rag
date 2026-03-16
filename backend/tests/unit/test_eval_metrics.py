"""Tests for agent-based DeepEval custom metrics."""

import json
from unittest.mock import MagicMock, patch

import pytest

deepeval = pytest.importorskip("deepeval", reason="deepeval not available")
from deepeval.test_case import LLMTestCase  # noqa: E402


def test_methodology_quality_metric():
    from tests.eval.metrics.custom import MethodologyQualityMetric

    with patch("tests.eval.metrics.custom.LLMClient") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps({
            "summary": "Adequate methodology.",
            "findings": [],
            "confidence": 0.8,
            "score": 7,
        })
        mock_llm_cls.return_value = mock_llm

        metric = MethodologyQualityMetric(threshold=0.5)
        test_case = LLMTestCase(
            input="cancer treatment",
            actual_output="Drug X is effective.",
            retrieval_context=[
                "PMID: 111\nRCT of Drug X\nA randomized trial of 500 patients.",
            ],
        )
        score = metric.measure(test_case)
        assert score == 0.7  # 7/10
        assert metric.is_successful()


def test_clinical_relevance_metric():
    from tests.eval.metrics.custom import ClinicalRelevanceMetric

    with patch("tests.eval.metrics.custom.LLMClient") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = json.dumps({
            "summary": "Highly applicable.",
            "findings": [],
            "confidence": 0.9,
            "score": 9,
        })
        mock_llm_cls.return_value = mock_llm

        metric = ClinicalRelevanceMetric(threshold=0.5)
        test_case = LLMTestCase(
            input="cancer treatment",
            actual_output="Drug X is effective.",
            retrieval_context=[
                "PMID: 111\nRCT of Drug X\nA randomized trial.",
            ],
        )
        score = metric.measure(test_case)
        assert score == 0.9  # 9/10
        assert metric.is_successful()


def test_metric_with_no_context():
    from tests.eval.metrics.custom import MethodologyQualityMetric

    with patch("tests.eval.metrics.custom.LLMClient"):
        metric = MethodologyQualityMetric(threshold=0.5)
        test_case = LLMTestCase(
            input="cancer treatment",
            actual_output="Some answer.",
            retrieval_context=[],
        )
        score = metric.measure(test_case)
        assert score == 0.0
        assert not metric.is_successful()
