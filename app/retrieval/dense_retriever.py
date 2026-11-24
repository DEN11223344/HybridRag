from typing import List, Optional
from app.embeddings.dense_embeddings import DenseEmbedder
from app.database.qdrant_client import QdrantManager
from app.schemas.search import SearchResultChunk

class DenseRetriever:
    def __init__(self, qdrant_manager: Optional[QdrantManager] = None, embedder: Optional[DenseEmbedder] = None):
        self.qdrant = qdrant_manager or QdrantManager.get_instance()
        self.embedder = embedder or DenseEmbedder.get_instance()

    def retrieve(self, query: str, top_k: int = 20, document_ids: Optional[List[str]] = None) -> List[SearchResultChunk]:
        query_vec = self.embedder.embed_query(query)
        results = self.qdrant.search_dense(query_vec, top_k=top_k, document_ids=document_ids)
        for r in results:
            r.retrieval_method = "dense"
        return results
