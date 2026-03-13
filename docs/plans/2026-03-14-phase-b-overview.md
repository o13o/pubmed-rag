# Phase B: Execution Overview

**Spec:** [2026-03-14-phase-b-design.md](../specs/2026-03-14-phase-b-design.md)

## Execution Order

```
B-S: Shared models + config updates        ─── Sequential (first)
         ↓ B-S done
B-1: Output guardrails                     ─┐
B-2: Hybrid search (BM25 + RRF)            ├─ Parallel
B-3: Reranker (cross-encoder + Protocol)   ─┘
         ↓ B-1, B-2, B-3 done
B-INT: Integration (chain.py + cli.py)     ─── Sequential
         ↓ B-INT done
B-4: Evaluation pipeline (DeepEval)        ─── Sequential
```

## Plan Files

| Plan | File | Parallelism |
|------|------|-------------|
| B-S | `2026-03-14-phase-b-shared.md` | Run first |
| B-1 | `2026-03-14-phase-b1-guardrails.md` | Parallel with B-2, B-3 |
| B-2 | `2026-03-14-phase-b2-hybrid-search.md` | Parallel with B-1, B-3 |
| B-3 | `2026-03-14-phase-b3-reranker.md` | Parallel with B-1, B-2 |
| B-INT + B-4 | `2026-03-14-phase-b-int-eval.md` | After B-1, B-2, B-3 |
