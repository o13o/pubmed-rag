"""Output validation: citation grounding, hallucination detection, terminology check, disclaimer."""

import json
import logging
import re

from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase
from src.shared.models import (
    GuardrailWarning,
    RAGResponse,
    SearchResult,
    ValidatedResponse,
)
from src.shared.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_PROMPT = load_prompt("guardrails/output")
MEDICAL_DISCLAIMER = _PROMPT["disclaimer"]


class GuardrailValidator:
    """Output guardrails with dependency injection."""

    def __init__(self, llm: LLMClient, mesh_db: MeSHDatabase):
        self.llm = llm
        self.mesh_db = mesh_db

    def validate(
        self, response: RAGResponse, search_results: list[SearchResult]
    ) -> ValidatedResponse:
        """Run all output validation checks."""
        warnings: list[GuardrailWarning] = []

        # 1. LLM-based validation (grounding + hallucination + treatment)
        llm_warnings = self._llm_validate(response, search_results)
        warnings.extend(llm_warnings)

        # 2. MeSH terminology validation
        mesh_warnings = self._mesh_validate(response.answer)
        warnings.extend(mesh_warnings)

        # Determine grounding status
        has_grounding_errors = any(
            w.check == "citation_grounding" and w.severity == "error"
            for w in warnings
        )

        return ValidatedResponse(
            answer=response.answer,
            citations=response.citations,
            query=response.query,
            warnings=warnings,
            disclaimer=_PROMPT["disclaimer"],
            is_grounded=not has_grounding_errors,
        )

    def _llm_validate(
        self, response: RAGResponse, search_results: list[SearchResult]
    ) -> list[GuardrailWarning]:
        """Use LLM to check grounding, hallucination, and treatment recommendations."""
        abstracts_text = "\n\n".join(
            f"PMID: {r.pmid}\nTitle: {r.title}\nAbstract: {r.abstract_text}"
            for r in search_results
        )

        user_prompt = f"""Answer to validate:
{response.answer}

Source abstracts:
{abstracts_text}

Check the answer against the source abstracts and return a JSON array of issues."""

        try:
            result = self.llm.complete(
                system_prompt=_PROMPT["system"],
                user_prompt=user_prompt,
            )
            issues = json.loads(result.strip())
            if not isinstance(issues, list):
                return []
            return [
                GuardrailWarning(
                    check=issue.get("check", "unknown"),
                    severity=issue.get("severity", "warning"),
                    message=issue.get("message", ""),
                    span=issue.get("span", ""),
                )
                for issue in issues
            ]
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("LLM validation failed: %s", e)
            return []

    def _mesh_validate(self, answer: str) -> list[GuardrailWarning]:
        """Check medical terms in the answer against MeSH vocabulary."""
        # Extract capitalized multi-word terms that look medical
        terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", answer)
        # Deduplicate and filter short/common words
        seen = set()
        not_found = []
        for term in terms:
            if term in seen or len(term) < 4:
                continue
            seen.add(term)
            if not self.mesh_db.validate_term(term):
                not_found.append(term)
        if not not_found:
            return []
        return [
            GuardrailWarning(
                check="terminology",
                severity="warning",
                message=f"Not found in MeSH: {', '.join(not_found)}",
                span=", ".join(not_found),
            )
        ]
