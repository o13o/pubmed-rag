"""Lightweight input classification: is the query medical/biomedical?"""

import logging

from pydantic import BaseModel

from src.shared.llm import LLMClient
from src.shared.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_PROMPT = load_prompt("guardrails/input")


class RelevanceResult(BaseModel):
    is_relevant: bool
    warning: str = ""


def classify_medical_relevance(query: str, llm: LLMClient) -> RelevanceResult:
    """Classify whether a query is medical/biomedical (soft warning, does not block)."""
    try:
        response = llm.complete(
            system_prompt=_PROMPT["system"],
            user_prompt=_PROMPT["user_template"].format(query=query),
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
