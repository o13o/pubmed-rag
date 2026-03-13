"""DeepEval test configuration."""

import os
import pytest


@pytest.fixture(autouse=True)
def eval_env(monkeypatch):
    """Ensure OPENAI_API_KEY is set for evaluation runs."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping evaluation tests")
