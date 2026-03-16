"""RAG chain: retrieve → expand → rerank → prompt → LLM → guardrails → response.

Orchestrates the full retrieval-augmented generation pipeline.
Uses SearchClient and GuardrailClient protocols so the same code works
in both monolith (direct calls) and microservice (HTTP calls) deployments.
"""

import logging
from collections.abc import Generator

from src.guardrails.client import GuardrailClient, LocalGuardrailClient
from src.rag.prompts import build_system_prompt, build_user_prompt
from src.retrieval.client import SearchClient
from src.retrieval.query_expander import QueryExpander
from src.retrieval.reranker import BaseReranker, NoOpReranker
from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase
from src.shared.models import (
    Citation, RAGResponse, SearchFilters, SearchResult, ValidatedResponse,
)

logger = logging.getLogger(__name__)


def ask(
    query: str,
    search_client: SearchClient,
    llm: LLMClient,
    mesh_db: MeSHDatabase,
    filters: SearchFilters | None = None,
    reranker: BaseReranker | None = None,
    guardrails_enabled: bool = True,
    guardrail_client: GuardrailClient | None = None,
) -> RAGResponse | ValidatedResponse:
    """Execute the full RAG pipeline.

    1. Expand query with MeSH terms
    2. Search for relevant abstracts (via SearchClient)
    3. Rerank results (if reranker provided)
    4. Build prompt with query + retrieved abstracts
    5. Call LLM for answer generation
    6. Run guardrails (via GuardrailClient, if enabled)
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
    results = search_client.search(expanded.expanded_query, filters)
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
        if guardrail_client is None:
            guardrail_client = LocalGuardrailClient(llm=llm, mesh_db=mesh_db)
        return guardrail_client.validate(rag_response, results)

    return rag_response


def ask_stream(
    query: str,
    search_client: SearchClient,
    llm: LLMClient,
    mesh_db: MeSHDatabase,
    filters: SearchFilters | None = None,
    reranker: BaseReranker | None = None,
    guardrails_enabled: bool = True,
    guardrail_client: GuardrailClient | None = None,
) -> Generator[dict, None, None]:
    """Execute the RAG pipeline with streaming LLM output.

    Yields dicts with 'event' and 'data' keys:
    - {"event": "token", "data": {"text": "..."}} for each LLM chunk
    - {"event": "done", "data": {...}} with citations and guardrail results
    - {"event": "error", "data": {"message": "..."}} on failure
    """
    try:
        if filters is None:
            filters = SearchFilters()
        if reranker is None:
            reranker = NoOpReranker()

        # 1. Query expansion
        expander = QueryExpander(llm=llm, mesh_db=mesh_db)
        expanded = expander.expand(query)
        logger.info("Expanded query: '%s' → '%s'", query, expanded.expanded_query)

        # 2. Search
        results = search_client.search(expanded.expanded_query, filters)
        logger.info("Retrieved %d results", len(results))

        # 3. Rerank
        results = reranker.rerank(query, results, top_k=filters.top_k)
        logger.info("After reranking: %d results", len(results))

        # 3.5 Emit citations early (before LLM generation)
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
        yield {
            "event": "citations",
            "data": {
                "citations": [c.model_dump() for c in citations],
                "search_results": [r.model_dump() for r in results],
            },
        }

        # 4. Build prompt
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(query, results)

        # 5. Stream LLM tokens
        full_answer = ""
        for chunk in llm.complete_stream(system_prompt=system_prompt, user_prompt=user_prompt):
            full_answer += chunk
            yield {"event": "token", "data": {"text": chunk}}

        # 6. Guardrails
        warnings = []
        disclaimer = ""
        is_grounded = True

        if guardrails_enabled:
            if guardrail_client is None:
                guardrail_client = LocalGuardrailClient(llm=llm, mesh_db=mesh_db)
            rag_response = RAGResponse(answer=full_answer, citations=citations, query=query)
            validated = guardrail_client.validate(rag_response, results)
            warnings = [w.model_dump() for w in validated.warnings]
            disclaimer = validated.disclaimer
            is_grounded = validated.is_grounded

        yield {
            "event": "done",
            "data": {
                "citations": [c.model_dump() for c in citations],
                "warnings": warnings,
                "disclaimer": disclaimer,
                "is_grounded": is_grounded,
            },
        }

    except Exception as e:
        logger.exception("Error in ask_stream")
        yield {"event": "error", "data": {"message": str(e)}}
