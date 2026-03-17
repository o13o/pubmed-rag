"""Prompt templates for the RAG chain."""

from src.shared.models import SearchResult
from src.shared.prompt_loader import load_prompt

_PROMPT = load_prompt("rag/system")


def build_system_prompt() -> str:
    return _PROMPT["system"]


def build_user_prompt(query: str, results: list[SearchResult]) -> str:
    """Build the user prompt with the query and retrieved abstracts."""
    if not results:
        return _PROMPT["no_results_template"].format(query=query)

    abstracts_text = []
    for i, r in enumerate(results, 1):
        abstracts_text.append(
            f"[{i}] PMID: {r.pmid}\n"
            f"Title: {r.title}\n"
            f"Journal: {r.journal} ({r.year})\n"
            f"Abstract: {r.abstract_text}\n"
        )

    joined = "\n---\n".join(abstracts_text)
    return _PROMPT["user_template"].format(query=query, abstracts=joined)
