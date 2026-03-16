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
    from src.retrieval.client import LocalSearchClient
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
    search_client = LocalSearchClient(collection)

    response = ask(
        query=query,
        search_client=search_client,
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
    from src.retrieval.search import search
    results = search(query, collection, SearchFilters(top_k=settings.top_k))
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
