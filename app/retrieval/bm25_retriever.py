from typing import List, Optional
from app.embeddings.bm25_embeddings import BM25SparseEmbedder
from app.database.qdrant_client import QdrantManager
from app.schemas.search import SearchResultChunk

class BM25Retriever:
    def __init__(self, qdrant_manager: Optional[QdrantManager] = None, embedder: Optional[BM25SparseEmbedder] = None):
        self.qdrant = qdrant_manager or QdrantManager.get_instance()
        self.embedder = embedder or BM25SparseEmbedder()

    def retrieve(self, query: str, top_k: int = 20, document_ids: Optional[List[str]] = None) -> List[SearchResultChunk]:
        query_sparse = self.embedder.encode_query(query)
        results = self.qdrant.search_sparse("bm25", query_sparse, top_k=top_k, document_ids=document_ids)
        for r in results:
            r.retrieval_method = "bm25"
        return results
