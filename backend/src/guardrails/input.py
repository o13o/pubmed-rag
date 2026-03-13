"""Lightweight input classification: is the query medical/biomedical?"""

import logging

from pydantic import BaseModel

from src.shared.llm import LLMClient

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """Is the following query related to medical or biomedical research?
Answer with only "yes" or "no".

Query: "{query}"
Answer:"""


class RelevanceResult(BaseModel):
    is_relevant: bool
    warning: str = ""


def classify_medical_relevance(query: str, llm: LLMClient) -> RelevanceResult:
    """Classify whether a query is medical/biomedical (soft warning, does not block)."""
    try:
        response = llm.complete(
            system_prompt="You classify queries as medical or non-medical. Answer only yes or no.",
            user_prompt=CLASSIFICATION_PROMPT.format(query=query),
        )
        answer = response.strip().lower()
        if answer.startswith("no"):
            return RelevanceResult(
                is_relevant=False,
                warning="This query may not be related to medical research. Results may be less relevant.",
            )
        return RelevanceResult(is_relevant=True)
    except Exception as e:
        logger.warning("Input classification failed: %s", e)
        return RelevanceResult(is_relevant=True)
