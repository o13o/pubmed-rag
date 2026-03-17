# Phase B-S: Shared Models & Config Updates

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Phase B models (`GuardrailWarning`, `ValidatedResponse`) and config settings (`search_mode`, `reranker_type`, etc.) to shared modules. Also update `pyproject.toml` with new dependencies.

**Architecture:** Extends existing `shared/models.py` and `shared/config.py` with new types and settings needed by B-1, B-2, B-3. All changes are additive and backward-compatible.

**Tech Stack:** pydantic, pydantic-settings

**Spec:** [2026-03-14-phase-b-design.md](../specs/2026-03-14-phase-b-design.md) — Sections 4.2, 5.5, 6.5

---

## Chunk 1: Shared Updates

### Task 1: Add Phase B Models

**Files:**
- Modify: `backend/src/shared/models.py`
- Modify: `backend/tests/unit/test_models.py`

- [ ] **Step 1: Write failing tests for new models**

Add to `tests/unit/test_models.py`:

```python
def test_guardrail_warning():
    from src.shared.models import GuardrailWarning
    w = GuardrailWarning(
        check="citation_grounding",
        severity="error",
        message="Claim not supported by any abstract",
        span="Drug X cures cancer",
    )
    assert w.check == "citation_grounding"
    assert w.severity == "error"
    assert w.span == "Drug X cures cancer"


def test_guardrail_warning_defaults():
    from src.shared.models import GuardrailWarning
    w = GuardrailWarning(check="hallucination", severity="warning", message="test")
    assert w.span == ""


def test_validated_response():
    from src.shared.models import ValidatedResponse, Citation, GuardrailWarning
    vr = ValidatedResponse(
        answer="Test answer",
        citations=[Citation(pmid="123", title="T")],
        query="test query",
        warnings=[GuardrailWarning(check="hallucination", severity="warning", message="Possible hallucination")],
        disclaimer="This is not medical advice.",
        is_grounded=True,
    )
    assert vr.is_grounded is True
    assert len(vr.warnings) == 1
    assert vr.disclaimer == "This is not medical advice."


def test_validated_response_defaults():
    from src.shared.models import ValidatedResponse
    vr = ValidatedResponse(
        answer="a", citations=[], query="q",
        warnings=[], disclaimer="d", is_grounded=True,
    )
    assert vr.warnings == []


def test_search_filters_search_mode():
    from src.shared.models import SearchFilters
    f = SearchFilters(search_mode="hybrid")
    assert f.search_mode == "hybrid"


def test_search_filters_search_mode_default():
    from src.shared.models import SearchFilters
    f = SearchFilters()
    assert f.search_mode is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/unit/test_models.py -v
```

Expected: FAIL — `GuardrailWarning`, `ValidatedResponse` not found, `search_mode` not a field.

- [ ] **Step 3: Add models to `shared/models.py`**

Add after the `RAGResponse` class:

```python
class GuardrailWarning(BaseModel):
    check: str
    severity: str
    message: str
    span: str = ""


class ValidatedResponse(BaseModel):
    answer: str
    citations: list[Citation]
    query: str
    warnings: list["GuardrailWarning"]
    disclaimer: str
    is_grounded: bool
```

Add `search_mode` field to `SearchFilters`:

```python
class SearchFilters(BaseModel):
    year_min: int | None = None
    year_max: int | None = None
    journals: list[str] = Field(default_factory=list)
    mesh_categories: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    top_k: int = 10
    search_mode: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/unit/test_models.py -v
```

Expected: All tests PASS (original 8 + 6 new = 14 total).

- [ ] **Step 5: Commit**

```bash
git add backend/src/shared/models.py backend/tests/unit/test_models.py
git commit -m "feat(shared): add GuardrailWarning, ValidatedResponse models and search_mode filter"
```

---

### Task 2: Add Phase B Config Settings

**Files:**
- Modify: `backend/src/shared/config.py`
- Modify: `backend/tests/unit/test_config.py`

- [ ] **Step 1: Write failing tests for new settings**

Add to `tests/unit/test_config.py`:

```python
def test_phase_b_settings_defaults(mock_env):
    from src.shared.config import Settings
    s = Settings()
    assert s.search_mode == "dense"
    assert s.reranker_type == "cross_encoder"
    assert s.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert s.reranker_top_k_multiplier == 3
    assert s.guardrails_enabled is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
uv run pytest tests/unit/test_config.py -v
```

Expected: FAIL — new settings not defined.

- [ ] **Step 3: Add settings to `shared/config.py`**

Add to the `Settings` class:

```python
    # Phase B: Hybrid Search
    search_mode: str = "dense"

    # Phase B: Reranker
    reranker_type: str = "cross_encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k_multiplier: int = 3

    # Phase B: Guardrails
    guardrails_enabled: bool = True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
uv run pytest tests/unit/test_config.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/shared/config.py backend/tests/unit/test_config.py
git commit -m "feat(shared): add Phase B config settings (search_mode, reranker, guardrails)"
```

---

### Task 3: Update pyproject.toml Dependencies

**Files:**
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Add sentence-transformers to main dependencies and deepeval to eval group**

```toml
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pymilvus>=2.5.0",
    "litellm>=1.0",
    "openai>=1.0",
    "duckdb>=1.0",
    "python-dotenv>=1.0",
    "sentence-transformers>=3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
eval = [
    "deepeval>=1.5,<2.0",
]
```

- [ ] **Step 2: Sync dependencies**

```bash
cd backend
uv sync
```

Expected: `sentence-transformers` installed. (Note: `deepeval` is in the `eval` group, install separately with `uv sync --extra eval` when needed.)

- [ ] **Step 3: Verify import works**

```bash
cd backend
uv run python -c "from sentence_transformers import CrossEncoder; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Run full test suite to verify no regressions**

```bash
cd backend
uv run pytest tests/unit/ -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock
git commit -m "deps: add sentence-transformers and deepeval (eval group)"
```
