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
