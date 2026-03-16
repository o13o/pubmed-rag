"""Health check endpoint."""

import logging

from fastapi import APIRouter, Depends

from src.api.dependencies import get_collection

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health_check(collection=Depends(get_collection)):
    if collection is None:
        return {"status": "ok", "milvus_connected": False, "deploy_mode": "microservice"}
    try:
        count = collection.num_entities
        return {"status": "ok", "milvus_connected": True, "collection_count": count}
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {"status": "degraded", "milvus_connected": False, "error": str(e)}
