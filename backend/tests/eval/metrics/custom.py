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
