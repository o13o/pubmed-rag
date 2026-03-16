"""Custom DeepEval metrics for PubMed RAG evaluation.

Extensible: add new metrics by inheriting from BaseMetric.
"""

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class CitationPresenceMetric(BaseMetric):
    """Check if the answer contains PMID citations."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        if not test_case.actual_output:
            self.score = 0.0
            self.reason = "No output generated"
            return self.score

        import re
        pmid_pattern = r"PMID[:\s]+\d+"
        citations = re.findall(pmid_pattern, test_case.actual_output)
        self.score = 1.0 if len(citations) > 0 else 0.0
        self.reason = f"Found {len(citations)} PMID citation(s)"
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Citation Presence"


class MedicalDisclaimerMetric(BaseMetric):
    """Check if the response includes a medical disclaimer."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        if not test_case.actual_output:
            self.score = 0.0
            self.reason = "No output generated"
            return self.score

        disclaimer_keywords = ["not medical advice", "consult", "healthcare professional", "disclaimer"]
        found = any(kw.lower() in test_case.actual_output.lower() for kw in disclaimer_keywords)
        self.score = 1.0 if found else 0.0
        self.reason = "Disclaimer present" if found else "No disclaimer found"
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Medical Disclaimer"


from src.agents.methodology_critic import MethodologyCriticAgent
from src.agents.statistical_reviewer import StatisticalReviewerAgent
from src.agents.clinical_applicability import ClinicalApplicabilityAgent
from src.shared.llm import LLMClient
from src.shared.models import SearchResult


def _parse_retrieval_context(test_case: LLMTestCase) -> list[SearchResult]:
    """Convert DeepEval retrieval_context strings to SearchResult objects."""
    results = []
    for i, ctx in enumerate(test_case.retrieval_context or []):
        lines = ctx.strip().split("\n")
        pmid = lines[0].replace("PMID: ", "").strip() if lines else str(i)
        title = lines[1].strip() if len(lines) > 1 else ""
        abstract = "\n".join(lines[2:]).strip() if len(lines) > 2 else ctx
        results.append(SearchResult(
            pmid=pmid, title=title, abstract_text=abstract,
            score=1.0, year=2023, journal="", mesh_terms=[],
        ))
    return results


class MethodologyQualityMetric(BaseMetric):
    """Evaluate study methodology quality using MethodologyCriticAgent."""

    def __init__(self, threshold: float = 0.5, llm_model: str = "gpt-4o-mini"):
        self.threshold = threshold
        self.llm = LLMClient(model=llm_model)

    def measure(self, test_case: LLMTestCase) -> float:
        results = _parse_retrieval_context(test_case)
        if not results:
            self.score = 0.0
            self.reason = "No retrieval context provided"
            return self.score

        agent = MethodologyCriticAgent(llm=self.llm)
        result = agent.run(query=test_case.input, results=results)
        self.score = (result.score or 0) / 10
        self.reason = result.summary
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Methodology Quality"


class StatisticalValidityMetric(BaseMetric):
    """Evaluate statistical validity using StatisticalReviewerAgent."""

    def __init__(self, threshold: float = 0.5, llm_model: str = "gpt-4o-mini"):
        self.threshold = threshold
        self.llm = LLMClient(model=llm_model)

    def measure(self, test_case: LLMTestCase) -> float:
        results = _parse_retrieval_context(test_case)
        if not results:
            self.score = 0.0
            self.reason = "No retrieval context provided"
            return self.score

        agent = StatisticalReviewerAgent(llm=self.llm)
        result = agent.run(query=test_case.input, results=results)
        self.score = (result.score or 0) / 10
        self.reason = result.summary
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Statistical Validity"


class ClinicalRelevanceMetric(BaseMetric):
    """Evaluate clinical relevance using ClinicalApplicabilityAgent."""

    def __init__(self, threshold: float = 0.5, llm_model: str = "gpt-4o-mini"):
        self.threshold = threshold
        self.llm = LLMClient(model=llm_model)

    def measure(self, test_case: LLMTestCase) -> float:
        results = _parse_retrieval_context(test_case)
        if not results:
            self.score = 0.0
            self.reason = "No retrieval context provided"
            return self.score

        agent = ClinicalApplicabilityAgent(llm=self.llm)
        result = agent.run(query=test_case.input, results=results)
        self.score = (result.score or 0) / 10
        self.reason = result.summary
        return self.score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Clinical Relevance"
