import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

try:
    from fastapi import APIRouter, HTTPException
except ImportError:
    class HTTPException(Exception):
        def __init__(self, status_code=400, detail=""):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)
    class APIRouter:
        def __init__(self, *args, **kwargs):
            pass
        def get(self, *args, **kwargs):
            return lambda fn: fn
        def post(self, *args, **kwargs):
            return lambda fn: fn
        def delete(self, *args, **kwargs):
            return lambda fn: fn

from app.config import settings
from app.schemas.search import SearchRequest, SearchResponse, SearchResultChunk
from app.schemas.answer import AskRequest, AskResponse, Citation
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.splade_retriever import SpladeRetriever
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.reranker import CrossEncoderReranker
from app.generation.answer_generator import AnswerGenerator
from app.database.qdrant_client import QdrantManager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Search & Question Answering"])

qdrant = QdrantManager.get_instance(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    collection_name=settings.qdrant_collection,
    dense_dim=settings.dense_dim,
)

dense_retriever = DenseRetriever(qdrant_manager=qdrant)
bm25_retriever = BM25Retriever(qdrant_manager=qdrant)
splade_retriever = SpladeRetriever(qdrant_manager=qdrant)
fusion = ReciprocalRankFusion(k=settings.rrf_k)
reranker = CrossEncoderReranker.get_instance(model_name=settings.reranker_model)
generator = AnswerGenerator(
    provider=settings.llm_provider,
    model=settings.llm_model,
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
)

def execute_retrieval(
    query: str,
    mode: str = "hybrid",
    top_k: int = 5,
    document_ids: Optional[List[str]] = None,
    apply_rerank: Optional[bool] = None,
    score_threshold: Optional[float] = None,
) -> SearchResponse:
    start_time = time.time()
    candidates: List[SearchResultChunk] = []

    retrieval_k = max(top_k * 4, settings.dense_top_k)
    mode = (mode or "hybrid").lower()

    if mode == "dense":
        candidates = dense_retriever.retrieve(query, top_k=retrieval_k, document_ids=document_ids)
    elif mode == "bm25":
        candidates = bm25_retriever.retrieve(query, top_k=retrieval_k, document_ids=document_ids)
    elif mode == "splade":
        candidates = splade_retriever.retrieve(query, top_k=retrieval_k, document_ids=document_ids)
    elif mode in ("bm25_dense", "dense_bm25"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            f_bm25 = executor.submit(bm25_retriever.retrieve, query, retrieval_k, document_ids)
            f_dense = executor.submit(dense_retriever.retrieve, query, retrieval_k, document_ids)
            candidates = fusion.fuse([f_bm25.result(), f_dense.result()], top_k=retrieval_k)
    elif mode in ("bm25_splade", "splade_bm25"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            f_bm25 = executor.submit(bm25_retriever.retrieve, query, retrieval_k, document_ids)
            f_splade = executor.submit(splade_retriever.retrieve, query, retrieval_k, document_ids)
            candidates = fusion.fuse([f_bm25.result(), f_splade.result()], top_k=retrieval_k)
    elif mode == "hybrid":
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_dense = executor.submit(dense_retriever.retrieve, query, retrieval_k, document_ids)
            f_bm25 = executor.submit(bm25_retriever.retrieve, query, retrieval_k, document_ids)
            f_splade = executor.submit(splade_retriever.retrieve, query, retrieval_k, document_ids)
            candidates = fusion.fuse([f_bm25.result(), f_splade.result(), f_dense.result()], top_k=retrieval_k)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid retrieval mode '{mode}'. Valid modes: hybrid, dense, bm25, splade, bm25_dense, bm25_splade",
        )

    retrieval_latency = (time.time() - start_time) * 1000.0
    rerank_latency = 0.0
    final_chunks = candidates

    if apply_rerank is None:
        apply_rerank = settings.enable_reranker

    if apply_rerank and candidates:
        r_start = time.time()
        final_chunks = reranker.rerank(query, candidates, top_k=top_k)
        rerank_latency = (time.time() - r_start) * 1000.0
    else:
        final_chunks = candidates[:top_k]

    if score_threshold is not None:
        final_chunks = [c for c in final_chunks if c.score >= score_threshold]

    return SearchResponse(
        query=query,
        mode=mode,
        total_retrieved=len(final_chunks),
        chunks=final_chunks,
        retrieval_latency_ms=round(retrieval_latency, 2),
        rerank_latency_ms=round(rerank_latency, 2),
    )

@router.post("/search")
def search_documents(request: SearchRequest):
    return execute_retrieval(
        query=request.query,
        mode=request.mode,
        top_k=request.top_k,
        document_ids=request.document_ids,
        apply_rerank=request.rerank,
        score_threshold=request.score_threshold,
    )

@router.post("/ask")
def ask_question(request: AskRequest):
    t_start = time.time()
    search_resp = execute_retrieval(
        query=request.query,
        mode=request.mode,
        top_k=request.top_k,
        document_ids=request.document_ids,
        apply_rerank=request.rerank,
    )

    answer, citations = generator.generate_answer(
        query=request.query,
        context_chunks=search_resp.chunks,
    )

    latency_ms = (time.time() - t_start) * 1000.0

    return AskResponse(
        query=request.query,
        answer=answer,
        citations=citations,
        retrieved_chunks=search_resp.chunks,
        latency_ms=round(latency_ms, 2),
        llm_provider=generator.provider,
    )
