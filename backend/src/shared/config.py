"""Application settings via pydantic-settings (env vars + .env)."""

from functools import lru_cache

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
