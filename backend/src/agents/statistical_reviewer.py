"""Statistical Reviewer Agent — analyzes statistical methods and validity."""

import json
import logging

from src.shared.llm import LLMClient
from src.shared.models import AgentResult, Finding, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a biostatistics expert. Analyze the provided research abstracts and evaluate their statistical methods and validity.

For each abstract, evaluate:
- Statistical methods used (t-test, chi-square, regression, survival analysis, etc.)
- Sample size adequacy and power
- P-values and confidence intervals reported
- Effect size magnitude and clinical significance
- Potential statistical biases or methodological flaws

Return your analysis as a JSON object with these exact fields:
{
  "summary": "1-2 sentence overall assessment",
  "findings": [
    {"label": "short label", "detail": "explanation", "severity": "info|warning|critical"}
  ],
  "confidence": 0.0-1.0,
  "score": 1-10
}

Score guide: 1-3 = poor statistical rigor, 4-6 = moderate, 7-9 = strong, 10 = exemplary.
Return ONLY the JSON object, no explanation."""


class StatisticalReviewerAgent:
    name = "statistical_reviewer"
    description = "Analyzes statistical methods, significance, and sample sizes"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, query: str, results: list[SearchResult]) -> AgentResult:
        abstracts_text = "\n\n".join(
            f"PMID: {r.pmid}\nTitle: {r.title}\nAbstract: {r.abstract_text}"
            for r in results
        )
        user_prompt = f"Query: {query}\n\nResearch abstracts to evaluate:\n{abstracts_text}"

        try:
            raw = self.llm.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
            data = json.loads(raw.strip())
            return AgentResult(
                agent_name=self.name,
                summary=data.get("summary", ""),
                findings=[Finding(**f) for f in data.get("findings", [])],
                confidence=data.get("confidence", 0.0),
                score=data.get("score"),
            )
        except Exception as e:
            logger.warning("StatisticalReviewerAgent failed: %s", e)
            return AgentResult(
                agent_name=self.name,
                summary=f"Analysis failed: {e}",
                findings=[],
                confidence=0.0,
                score=None,
            )
