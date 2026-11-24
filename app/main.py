import os
import logging

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    class FastAPI:
        def __init__(self, *args, **kwargs):
            self.routes = []
        def include_router(self, router):
            pass
        def add_middleware(self, *args, **kwargs):
            pass
        def on_event(self, event_name):
            return lambda fn: fn

from app.config import settings
from app.api.routes_health import router as health_router
from app.api.routes_documents import router as documents_router
from app.api.routes_search import router as search_router

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hybrid_rag")

app = FastAPI(
    title="Hybrid RAG for Any PDF (CPU-Only)",
    description=(
        "Production-oriented Local Hybrid Retrieval-Augmented Generation (RAG) system for universal PDFs.\n"
        "Features: Dense (Jina-v2-512), BM25 Sparse, SPLADE Sparse, Reciprocal Rank Fusion (RRF), "
        "BAAI BGE Cross-Encoder Reranking, Table Extraction, and Page-Level Citations on CPU."
    ),
    version="1.0.0",
)

if HAS_FASTAPI:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(search_router)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Hybrid RAG Application...")
    logger.info(f"Storage Upload Dir: {settings.upload_dir}")
    logger.info(f"Qdrant Target: {settings.qdrant_host}:{settings.qdrant_port} (collection: {settings.qdrant_collection})")
    logger.info(f"Dense Model: {settings.dense_model} (dim={settings.dense_dim})")
    logger.info(f"SPLADE Model: {settings.splade_model}")
    logger.info(f"Reranker Model: {settings.reranker_model}")
    logger.info(f"LLM Provider: {settings.llm_provider}")

if __name__ == "__main__":
    if HAS_FASTAPI:
        import uvicorn
        uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=settings.debug)
