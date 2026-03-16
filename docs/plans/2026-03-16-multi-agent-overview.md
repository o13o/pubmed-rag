# Multi-Agent Research Analysis Layer — Execution Overview

**Spec:** [../specs/2026-03-16-multi-agent-design.md](../specs/2026-03-16-multi-agent-design.md)

## Execution Order

```
Plan S: Shared (Models + BaseAgent + Registry)       ─── Sequential (first)
         ↓ Plan S done
Plan A: Must-Have Agents (3 agents)                  ─┐
Plan B: Nice-to-Have Agents (2 agents + registry)    ├─ Parallel
Plan C: Frontend (types, API, component, wiring)     ─┘
         ↓ Plan A, B, C done
Plan INT: Integration (API endpoint + DeepEval + build verify) ─── Sequential (last)
```

## Plan Files

| Plan | Tasks | Can Run In Parallel | File |
|------|-------|---------------------|------|
| S | Models, BaseAgent, Registry | Run first | [2026-03-16-multi-agent-s-shared.md](./2026-03-16-multi-agent-s-shared.md) |
| A | MethodologyCritic, ClinicalApplicability, Summarization | Parallel with B, C | [2026-03-16-multi-agent-a-agents-must.md](./2026-03-16-multi-agent-a-agents-must.md) |
| B | Retrieval, StatisticalReviewer, Registry tests | Parallel with A, C | [2026-03-16-multi-agent-b-agents-nice.md](./2026-03-16-multi-agent-b-agents-nice.md) |
| C | Frontend types, API client, AgentResultsPanel, App.tsx | Parallel with A, B | [2026-03-16-multi-agent-c-frontend.md](./2026-03-16-multi-agent-c-frontend.md) |
| INT | POST /analyze endpoint, DeepEval metrics, build verify | After A, B, C | [2026-03-16-multi-agent-int.md](./2026-03-16-multi-agent-int.md) |

## Dependencies

- **Plan S** has no dependencies (standalone)
- **Plan A, B, C** each depend on Plan S (models + base + registry must exist)
- **Plan INT** depends on Plan A + Plan B + Plan C (all agents must exist for DeepEval metrics)
- Plan B is required for full build — the `StatisticalValidityMetric` imports `StatisticalReviewerAgent`

## Integration Strategy

After Plan A, B, C complete:
1. Merge all branches into main
2. Run `uv run pytest tests/unit/ -v` across all modules
3. Then proceed with Plan INT (API endpoint, DeepEval, build verify)
