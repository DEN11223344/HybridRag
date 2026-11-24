from typing import List, Optional
from app.embeddings.splade_embeddings import SpladeEmbedder
from app.database.qdrant_client import QdrantManager
from app.schemas.search import SearchResultChunk

class SpladeRetriever:
    def __init__(self, qdrant_manager: Optional[QdrantManager] = None, embedder: Optional[SpladeEmbedder] = None):
        self.qdrant = qdrant_manager or QdrantManager.get_instance()
        self.embedder = embedder or SpladeEmbedder.get_instance()

    def retrieve(self, query: str, top_k: int = 20, document_ids: Optional[List[str]] = None) -> List[SearchResultChunk]:
        query_sparse = self.embedder.encode_text(query)
        results = self.qdrant.search_sparse("splade", query_sparse, top_k=top_k, document_ids=document_ids)
        for r in results:
            r.retrieval_method = "splade"
        return results
