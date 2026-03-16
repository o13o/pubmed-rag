"""POST /analyze — multi-agent research analysis endpoint."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.agents.registry import get_agents
from src.api.dependencies import get_llm
from src.shared.llm import LLMClient
from src.shared.models import AgentResult, SearchResult

router = APIRouter()


class AnalyzeRequest(BaseModel):
    query: str
    results: list[SearchResult]
    agents: list[str] | None = None


class AnalyzeResponse(BaseModel):
    query: str
    agent_results: list[AgentResult]


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(
    req: AnalyzeRequest,
    llm: LLMClient = Depends(get_llm),
):
    agents = get_agents(llm=llm, names=req.agents)
    agent_results = [agent.run(req.query, req.results) for agent in agents]
    return AnalyzeResponse(query=req.query, agent_results=agent_results)
