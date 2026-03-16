"""Application settings via pydantic-settings (env vars + .env)."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "pubmed_abstracts"
    llm_model: str = "gpt-4o-mini"
    llm_timeout: int = 30
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_batch_size: int = 100
    top_k: int = 10
    mesh_db_path: str = "data/mesh.duckdb"

    # Phase B: Hybrid Search
    search_mode: str = "dense"

    # Phase B: Reranker
    reranker_type: str = "cross_encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k_multiplier: int = 3

    # Phase B: Guardrails
    guardrails_enabled: bool = True

    # Deployment mode: "monolith" (default) or "microservice"
    deploy_mode: Literal["monolith", "microservice"] = "monolith"
    search_service_url: str = "http://localhost:8001"

    # Observability: LangFuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
