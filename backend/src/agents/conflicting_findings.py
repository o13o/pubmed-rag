"""Conflicting Findings Agent — identifies contradictory conclusions across studies."""

import json
import logging

from src.shared.llm import LLMClient
from src.shared.models import AgentResult, Finding, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical research contradiction analyst. Analyze the provided research abstracts and identify conflicting or contradictory findings between studies.

Focus on:
- Directly contradictory conclusions about the same treatment or intervention
- Conflicting statistical outcomes (e.g., one study shows benefit, another shows no effect)
- Disagreements about risk factors or mechanisms
- Contradictions in recommended dosages or treatment protocols

Return your analysis as a JSON object with these exact fields:
{
  "summary": "1-2 sentence overview of conflicts found",
  "findings": [
    {"label": "short label", "detail": "explanation of the conflict", "severity": "info|warning|critical"}
  ],
  "confidence": 0.0-1.0,
  "conflicts": [
    {"pmid_a": "...", "pmid_b": "...", "topic": "...", "description": "..."}
  ]
}

If no conflicts are found, return an empty conflicts array and note this in the summary.
Return ONLY the JSON object, no explanation."""


class ConflictingFindingsAgent:
    name = "conflicting_findings"
    description = "Identifies contradictory conclusions across research studies"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, query: str, results: list[SearchResult]) -> AgentResult:
        abstracts_text = "\n\n".join(
            f"PMID: {r.pmid}\nTitle: {r.title}\nAbstract: {r.abstract_text}"
            for r in results
        )
        user_prompt = f"Query: {query}\n\nResearch abstracts to analyze for conflicts:\n{abstracts_text}"

        try:
            raw = self.llm.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
            data = json.loads(raw.strip())
            return AgentResult(
                agent_name=self.name,
                summary=data.get("summary", ""),
                findings=[Finding(**f) for f in data.get("findings", [])],
                confidence=data.get("confidence", 0.0),
                score=None,
                details={"conflicts": data.get("conflicts", [])},
            )
        except Exception as e:
            logger.warning("ConflictingFindingsAgent failed: %s", e)
            return AgentResult(
                agent_name=self.name,
                summary=f"Analysis failed: {e}",
                findings=[],
                confidence=0.0,
                score=None,
            )
