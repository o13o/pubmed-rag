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
