"""Health check endpoint."""

import logging

from fastapi import APIRouter, Depends
from pymilvus import Collection

from src.api.dependencies import get_collection

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health_check(collection: Collection = Depends(get_collection)):
    try:
        count = collection.num_entities
        return {"status": "ok", "milvus_connected": True, "collection_count": count}
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {"status": "degraded", "milvus_connected": False, "error": str(e)}
