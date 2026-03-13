"""Tests for configuration management."""

import os
from unittest.mock import patch

from src.shared.config import Settings


def test_default_settings():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        s = Settings()
    assert s.milvus_host == "localhost"
    assert s.milvus_port == 19530
    assert s.llm_model == "gpt-4o-mini"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.embedding_dim == 1536
    assert s.embedding_batch_size == 100
    assert s.top_k == 10


def test_settings_from_env():
    env = {
        "OPENAI_API_KEY": "sk-test",
        "MILVUS_HOST": "milvus-server",
        "MILVUS_PORT": "29530",
        "LLM_MODEL": "gpt-4o",
        "TOP_K": "20",
    }
    with patch.dict(os.environ, env, clear=False):
        s = Settings()
    assert s.milvus_host == "milvus-server"
    assert s.milvus_port == 29530
    assert s.llm_model == "gpt-4o"
    assert s.top_k == 20


def test_phase_b_settings_defaults():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        s = Settings()
    assert s.search_mode == "dense"
    assert s.reranker_type == "cross_encoder"
    assert s.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert s.reranker_top_k_multiplier == 3
    assert s.guardrails_enabled is True
