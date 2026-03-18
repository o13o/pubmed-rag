"""Standalone Search Service — exposes Milvus search as an HTTP API.

Used in microservice mode: the main backend sets DEPLOY_MODE=microservice
and sends search requests here via RemoteSearchClient.

Usage:
    uvicorn src.search_service.main:app --host 0.0.0.0 --port 8001
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pymilvus import Collection, connections

from src.retrieval.search import search
from src.shared.config import get_settings
from src.shared.logging_config import setup_logging
from src.shared.models import SearchFilters

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    connections.connect("default", host=settings.milvus_host, port=str(settings.milvus_port))
    collection = Collection(settings.milvus_collection)
    collection.load()
    app.state.collection = collection
    logger.info("Search Service started: collection=%s", settings.milvus_collection)
    yield
    connections.disconnect("default")
    logger.info("Search Service shutdown")


app = FastAPI(title="PubMed RAG Search Service", version="0.1.0", lifespan=lifespan)


class SearchRequest(SearchFilters):
    query: str


@app.get("/health")
def health():
    collection = app.state.collection
    return {"status": "ok", "collection_count": collection.num_entities}


@app.post("/search")
def search_endpoint(req: SearchRequest):
    filters = SearchFilters(
        year_min=req.year_min,
        year_max=req.year_max,
        journals=req.journals,
        publication_types=req.publication_types,
        mesh_categories=req.mesh_categories,
        top_k=req.top_k,
        search_mode=req.search_mode,
    )
    results = search(req.query, app.state.collection, filters)
    return {"results": [r.model_dump() for r in results], "total": len(results)}
