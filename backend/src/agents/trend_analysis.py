"""Trend Analysis Agent — detects emerging research trends and directions."""

import logging

from src.agents import parse_llm_json
from src.shared.llm import LLMClient
from src.shared.models import AgentResult, Finding, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical research trend analyst. Analyze the provided research abstracts and identify emerging trends, shifts in research focus, and promising new directions.

Consider:
- Publication years to detect temporal patterns
- MeSH terms and keywords to identify growing research areas
- Shifts in treatment approaches (e.g., from surgical to immunotherapy)
- Emerging technologies or methodologies gaining traction
- Declining interest in previously popular approaches

Return your analysis as a JSON object with these exact fields:
{
  "summary": "1-2 sentence overview of key trends",
  "findings": [
    {"label": "short label", "detail": "explanation of the trend", "severity": "info|warning|critical"}
  ],
  "confidence": 0.0-1.0,
  "trends": [
    {"topic": "...", "direction": "increasing|decreasing|stable", "period": "e.g. 2020-2024", "evidence_count": 5}
  ]
}

Return ONLY the JSON object, no explanation."""


class TrendAnalysisAgent:
    name = "trend_analysis"
    description = "Detects emerging research trends and promising directions"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, query: str, results: list[SearchResult]) -> AgentResult:
        abstracts_text = "\n\n".join(
            f"PMID: {r.pmid}\nTitle: {r.title}\nYear: {r.year}\nMeSH: {', '.join(r.mesh_terms)}\nAbstract: {r.abstract_text}"
            for r in results
        )
        user_prompt = f"Query: {query}\n\nResearch abstracts to analyze for trends:\n{abstracts_text}"

        try:
            raw = self.llm.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
            data = parse_llm_json(raw)
            return AgentResult(
                agent_name=self.name,
                summary=data.get("summary", ""),
                findings=[Finding(**f) for f in data.get("findings", [])],
                confidence=data.get("confidence", 0.0),
                score=None,
                details={"trends": data.get("trends", [])},
            )
        except Exception as e:
            logger.warning("TrendAnalysisAgent failed: %s", e)
            return AgentResult(
                agent_name=self.name,
                summary=f"Analysis failed: {e}",
                findings=[],
                confidence=0.0,
                score=None,
            )
