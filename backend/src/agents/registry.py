"""Agent registry — maps agent names to classes.

Registry is built lazily inside get_agents() to avoid import errors
when individual agent modules don't exist yet during development.
"""

from src.agents.base import BaseAgent
from src.shared.llm import LLMClient


def get_agents(llm: LLMClient, names: list[str] | None = None) -> list[BaseAgent]:
    """Return agent instances. If names is None, return all."""
    from src.agents.clinical_applicability import ClinicalApplicabilityAgent
    from src.agents.methodology_critic import MethodologyCriticAgent
    from src.agents.retrieval import RetrievalAgent
    from src.agents.statistical_reviewer import StatisticalReviewerAgent
    from src.agents.summarization import SummarizationAgent

    registry: dict[str, type[BaseAgent]] = {
        "retrieval": RetrievalAgent,
        "methodology_critic": MethodologyCriticAgent,
        "statistical_reviewer": StatisticalReviewerAgent,
        "clinical_applicability": ClinicalApplicabilityAgent,
        "summarization": SummarizationAgent,
    }

    if names is not None:
        registry = {k: v for k, v in registry.items() if k in names}
    return [cls(llm=llm) for cls in registry.values()]
