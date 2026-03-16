"""FastAPI application factory with lifespan-managed services."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymilvus import Collection, connections

from src.api.routes import analyze, ask, health, search
from src.retrieval.client import LocalSearchClient, RemoteSearchClient
from src.retrieval.reranker import get_reranker
from src.shared.config import get_settings
from src.shared.llm import LLMClient
from src.shared.mesh_db import MeSHDatabase

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown shared services."""
    settings = get_settings()

    # Startup
    llm = LLMClient(model=settings.llm_model, timeout=settings.llm_timeout)
    mesh_db = MeSHDatabase(settings.mesh_db_path)
    reranker = get_reranker(
        reranker_type=settings.reranker_type,
        model_name=settings.reranker_model,
        llm=llm if settings.reranker_type == "llm" else None,
    )

    if settings.deploy_mode == "microservice":
        search_client = RemoteSearchClient(settings.search_service_url)
        collection = None
    else:
        connections.connect("default", host=settings.milvus_host, port=str(settings.milvus_port))
        collection = Collection(settings.milvus_collection)
        collection.load()
        search_client = LocalSearchClient(collection)

    app.state.collection = collection
    app.state.llm = llm
    app.state.mesh_db = mesh_db
    app.state.reranker = reranker
    app.state.search_client = search_client
    app.state.settings = settings

    logger.info("API started: collection=%s", settings.milvus_collection)
    yield

    # Shutdown
    mesh_db.close()
    if settings.deploy_mode != "microservice":
        connections.disconnect("default")
    logger.info("API shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="PubMed RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(ask.router)
    app.include_router(search.router)
    app.include_router(analyze.router)

    return app


app = create_app()
