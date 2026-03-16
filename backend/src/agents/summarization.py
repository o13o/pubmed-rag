"""Summarization Agent — synthesizes insights across multiple research studies."""

import json
import logging

from src.shared.llm import LLMClient
from src.shared.models import AgentResult, Finding, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical research synthesis expert. Analyze the provided research abstracts and produce a comprehensive synthesis of findings.

Your synthesis should:
- Identify consensus findings across studies
- Highlight conflicting or contradictory results
- Note gaps in the evidence base
- Identify emerging trends or promising directions
- Assess the overall strength of evidence

Return your analysis as a JSON object with these exact fields:
{
  "summary": "1-2 sentence synthesis of key insights",
  "findings": [
    {"label": "short label", "detail": "explanation", "severity": "info|warning|critical"}
  ],
  "confidence": 0.0-1.0
}

Note: Do NOT include a "score" field. Summarization does not score.
Return ONLY the JSON object, no explanation."""


class SummarizationAgent:
    name = "summarization"
    description = "Synthesizes insights across multiple research studies"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, query: str, results: list[SearchResult]) -> AgentResult:
        abstracts_text = "\n\n".join(
            f"PMID: {r.pmid}\nTitle: {r.title}\nAbstract: {r.abstract_text}"
            for r in results
        )
        user_prompt = f"Query: {query}\n\nResearch abstracts to synthesize:\n{abstracts_text}"

        try:
            raw = self.llm.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
            data = json.loads(raw.strip())
            return AgentResult(
                agent_name=self.name,
                summary=data.get("summary", ""),
                findings=[Finding(**f) for f in data.get("findings", [])],
                confidence=data.get("confidence", 0.0),
                score=None,
            )
        except Exception as e:
            logger.warning("SummarizationAgent failed: %s", e)
            return AgentResult(
                agent_name=self.name,
                summary=f"Analysis failed: {e}",
                findings=[],
                confidence=0.0,
                score=None,
            )
