ADR: Agent Orchestration — Independent Execution Over A2A Communication

Status: Accepted
Date: 2026-03-16
Owner: Yasuhiro Okamoto

## Context

The requirements describe several advanced agent capabilities:

- Agent-to-Agent (A2A) communication for systematic review collaboration
- Handoff to full-text retrieval and meta-analysis agents
- Automated literature review generation for complex medical queries

The current system has 8 independent agents (retrieval, methodology_critic, statistical_reviewer, clinical_applicability, summarization, conflicting_findings, trend_analysis, knowledge_graph), each receiving `(query, list[SearchResult])` and returning `AgentResult`. They run in a simple loop with no inter-agent communication.

## Decision

Keep agents as independent, stateless functions. Do not implement A2A communication or handoff protocols at this stage.

## Rationale

- **8 agents is not enough to justify orchestration overhead.** A2A communication adds complexity (message formats, routing, error propagation, ordering) that pays off at scale (20+ agents with dependencies). With 8 independent agents, a `for` loop is simpler, faster, and easier to debug.
- **No agent has dependencies on another agent's output.** Each agent analyzes the same `(query, results)` input. There is no case where one agent needs another's findings to do its work. Adding handoff without real data dependencies would be architecture theater.
- **Parallel execution is the immediate win.** The current bottleneck is sequential LLM calls. Running agents concurrently (e.g., `asyncio.gather` or thread pool) gives a ~8x speedup with zero architectural changes. This is a better investment than orchestration.

## Future Direction: Agent Router

When the system grows to warrant orchestration, the natural evolution is an **Agent Router** — not full A2A communication:

```
User Query
  → Agent Router (LLM-based or rule-based)
  → selects relevant subset of agents
  → runs selected agents in parallel
  → aggregates results
```

This pattern:
- **Reduces cost** — not all 8 agents need to run for every query
- **Scales to more agents** — adding an agent means registering it with the router, not wiring it to every other agent
- **Avoids A2A complexity** — the router is the single coordination point; agents remain stateless

### When to Introduce the Router

- When agent count exceeds ~15 and running all of them per query becomes too expensive
- When agents emerge that only apply to specific query types (e.g., a pharmacogenomics agent that should only run for drug-gene queries)
- When latency requirements demand selective execution

### When to Introduce A2A / Handoff

- When a concrete workflow requires chained reasoning (e.g., methodology_critic identifies a flawed study → meta_analysis agent excludes it from synthesis)
- When implementing automated systematic review generation, where agents must build on each other's outputs in a defined sequence

Until these conditions are met, the current architecture is the right one.

## Consequences

### Positive

- Agents are trivial to add, test, and debug independently
- No orchestration bugs, no message format versioning, no routing failures
- Can be parallelized with a one-line change (`asyncio.gather`)

### Trade-offs

- All agents run on every `/analyze` call (no selective execution)
- No agent can build on another agent's output
- Literature review generation requires a separate implementation path when needed
