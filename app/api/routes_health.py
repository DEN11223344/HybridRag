try:
    from fastapi import APIRouter
except ImportError:
    class APIRouter:
        def __init__(self, *args, **kwargs):
            pass
        def get(self, *args, **kwargs):
            return lambda fn: fn
        def post(self, *args, **kwargs):
            return lambda fn: fn
        def delete(self, *args, **kwargs):
            return lambda fn: fn

from app.database.qdrant_client import QdrantManager
from app.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check():
    qdrant = QdrantManager.get_instance(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection,
    )
    client = qdrant.get_client()
    qdrant_connected = client is not None and not qdrant._use_in_memory_fallback
    status = "healthy" if qdrant_connected else "degraded (in-memory mode)"

    return {
        "status": status,
        "qdrant_connected": qdrant_connected,
        "qdrant_host": settings.qdrant_host,
        "qdrant_port": settings.qdrant_port,
        "collection": settings.qdrant_collection,
        "dense_model": settings.dense_model,
        "splade_model": settings.splade_model,
        "reranker_model": settings.reranker_model,
        "llm_provider": settings.llm_provider,
    }
