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
from src.retrieval.client import LocalSearchClient
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

    search_client = LocalSearchClient(collection)

    # Execute RAG
    response = ask(
        query=args.query,
        search_client=search_client,
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
