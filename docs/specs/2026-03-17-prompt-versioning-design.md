# Prompt Versioning Design

## Problem

LLM prompts are hardcoded as Python string constants across 15+ files in the backend codebase. This makes it difficult to:

- Track prompt changes independently from code changes in git history
- Correlate prompt versions with evaluation results (DeepEval)
- Experiment with prompt variations (A/B testing)

## Goals

1. **Change tracking (primary)** — Prompt changes appear as isolated git diffs, separate from code logic changes
2. **Evaluation linkage (primary)** — Each prompt carries version metadata that can be logged alongside evaluation results
3. **A/B testing (future)** — The structure supports loading alternate prompt versions at runtime

## Design

### File Format: YAML

Each prompt is a YAML file containing metadata and prompt text:

```yaml
version: "1.0"
description: "RAG main system prompt for medical research assistant"

system: |
  You are a medical research assistant...

user_template: |
  Question: {query}
  Relevant abstracts:
  {abstracts}
```

Fields:
- `version` — Semantic version string, bumped when prompt content changes
- `description` — Human-readable purpose of this prompt
- `system` — System prompt text (optional, some prompts only have user_template)
- `user_template` — User prompt template with `{variable}` placeholders (optional)

Variable substitution uses Python `str.format()`.

### Directory Structure

```
capstone/backend/prompts/
├── rag/
│   └── system.yaml
├── guardrails/
│   ├── input.yaml
│   └── output.yaml
├── retrieval/
│   ├── query_expander.yaml
│   └── reranker.yaml
├── agents/
│   ├── methodology_critic.yaml
│   ├── statistical_reviewer.yaml
│   ├── clinical_applicability.yaml
│   ├── retrieval.yaml
│   ├── summarization.yaml
│   ├── conflicting_findings.yaml
│   ├── trend_analysis.yaml
│   └── knowledge_graph.yaml
└── transcribe/
    ├── image.yaml
    └── document.yaml
```

Mirrors the `src/` module structure for easy navigation.

### Loader

A single function in `src/shared/prompt_loader.py`:

```python
def load_prompt(name: str) -> dict:
    """Load a prompt YAML file by path relative to prompts/.

    Example: load_prompt("agents/methodology_critic")
    Returns: {"version": "1.0", "description": "...", "system": "...", ...}
    """
```

- Resolves paths relative to `capstone/backend/prompts/`
- Caches loaded prompts in a module-level dict (prompts don't change at runtime)
- Raises `FileNotFoundError` with a clear message if the YAML file is missing

### Migration

Each Python module that currently defines prompt constants will be updated:

**Before:**
```python
SYSTEM_PROMPT = """You are a medical research assistant..."""

def build_system_prompt() -> str:
    return SYSTEM_PROMPT
```

**After:**
```python
from src.shared.prompt_loader import load_prompt

_PROMPT = load_prompt("rag/system")

def build_system_prompt() -> str:
    return _PROMPT["system"]
```

Modules with inline system prompts passed directly to `llm.complete()` will also be updated to load from YAML.

### Evaluation Linkage

Prompt version metadata can be accessed via `load_prompt("rag/system")["version"]` and logged alongside DeepEval results. This enables correlating prompt changes with accuracy/latency metrics over time.

## Out of Scope

- Runtime prompt switching (A/B testing UI) — future work
- Jinja2 templating — `str.format()` is sufficient for current variable substitution needs
- Prompt registry/database — file-based YAML with git versioning is sufficient
