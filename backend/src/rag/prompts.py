"""Prompt templates for the RAG chain."""

from src.shared.models import SearchResult

SYSTEM_PROMPT = """You are a medical research assistant that answers questions based on PubMed abstracts.

Rules:
1. ONLY use information from the provided abstracts to answer the question.
2. ALWAYS cite your sources using PMID numbers in the format [PMID: 12345678].
3. If the abstracts don't contain enough information, say so explicitly.
4. Be precise and use appropriate medical terminology.
5. Do NOT provide medical advice or treatment recommendations without qualifying language.
6. Structure your answer clearly with relevant findings from the literature."""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_user_prompt(query: str, results: list[SearchResult]) -> str:
    """Build the user prompt with the query and retrieved abstracts."""
    if not results:
        return f"""Question: {query}

No relevant abstracts were found. Please inform the user that no relevant research was found for their query."""

    abstracts_text = []
    for i, r in enumerate(results, 1):
        abstracts_text.append(
            f"[{i}] PMID: {r.pmid}\n"
            f"Title: {r.title}\n"
            f"Journal: {r.journal} ({r.year})\n"
            f"Abstract: {r.abstract_text}\n"
        )

    joined = "\n---\n".join(abstracts_text)
    return f"""Question: {query}

Relevant abstracts:

{joined}

Based on the abstracts above, provide a comprehensive answer to the question. Cite each claim with the relevant PMID."""
