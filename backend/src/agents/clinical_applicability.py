"""Clinical Applicability Agent — assesses real-world clinical relevance."""

import json
import logging

from src.shared.llm import LLMClient
from src.shared.models import AgentResult, Finding, SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical medicine expert. Analyze the provided research abstracts and assess their real-world clinical applicability.

For each abstract, evaluate:
- Target patient population (age, condition severity, comorbidities)
- Clinical setting applicability (primary care, specialist, hospital)
- Treatment feasibility (availability, cost, implementation complexity)
- Generalizability of findings to broader patient populations
- Safety considerations and contraindications mentioned

Return your analysis as a JSON object with these exact fields:
{
  "summary": "1-2 sentence overall assessment",
  "findings": [
    {"label": "short label", "detail": "explanation", "severity": "info|warning|critical"}
  ],
  "confidence": 0.0-1.0,
  "score": 1-10
}

Score guide: 1-3 = low applicability, 4-6 = moderate, 7-9 = high, 10 = directly actionable.
Return ONLY the JSON object, no explanation."""


class ClinicalApplicabilityAgent:
    name = "clinical_applicability"
    description = "Assesses real-world clinical relevance and applicability"

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
            logger.warning("ClinicalApplicabilityAgent failed: %s", e)
            return AgentResult(
                agent_name=self.name,
                summary=f"Analysis failed: {e}",
                findings=[],
                confidence=0.0,
                score=None,
            )
