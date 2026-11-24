from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

if HAS_PYDANTIC:
    class SearchRequest(BaseModel):
        query: str
        document_ids: Optional[List[str]] = None
        top_k: int = 5
        mode: str = "hybrid"  # "hybrid", "dense", "bm25", "splade", "bm25_dense", "bm25_splade"
        rerank: bool = True
        score_threshold: Optional[float] = None

    class SearchResultChunk(BaseModel):
        chunk_id: str
        document_id: str
        document_name: str
        page_number: int
        section_title: Optional[str] = None
        chunk_text: str
        token_count: int = 0
        content_type: str = "text"
        table_id: Optional[str] = None
        image_references: List[str] = Field(default_factory=list)
        score: float
        retrieval_method: str = "unknown"

    class SearchResponse(BaseModel):
        query: str
        mode: str
        total_retrieved: int
        chunks: List[SearchResultChunk]
        retrieval_latency_ms: float
        rerank_latency_ms: float = 0.0
else:
    class SearchRequest:
        def __init__(self, **kwargs):
            self.query = kwargs.get("query", "")
            self.document_ids = kwargs.get("document_ids")
            self.top_k = kwargs.get("top_k", 5)
            self.mode = kwargs.get("mode", "hybrid")
            self.rerank = kwargs.get("rerank", True)
            self.score_threshold = kwargs.get("score_threshold")

    class SearchResultChunk:
        def __init__(
            self,
            chunk_id: str = "",
            document_id: str = "",
            document_name: str = "",
            page_number: int = 1,
            chunk_text: str = "",
            score: float = 0.0,
            section_title: Optional[str] = None,
            token_count: int = 0,
            content_type: str = "text",
            table_id: Optional[str] = None,
            image_references: Optional[List[str]] = None,
            retrieval_method: str = "unknown",
            **kwargs
        ):
            self.chunk_id = chunk_id or kwargs.get("chunk_id", "")
            self.document_id = document_id or kwargs.get("document_id", "")
            self.document_name = document_name or kwargs.get("document_name", "")
            self.page_number = page_number or kwargs.get("page_number", 1)
            self.chunk_text = chunk_text or kwargs.get("chunk_text", "")
            self.score = score if score is not None else kwargs.get("score", 0.0)
            self.section_title = section_title if section_title is not None else kwargs.get("section_title")
            self.token_count = token_count or kwargs.get("token_count", 0)
            self.content_type = content_type or kwargs.get("content_type", "text")
            self.table_id = table_id if table_id is not None else kwargs.get("table_id")
            self.image_references = image_references if image_references is not None else kwargs.get("image_references", [])
            self.retrieval_method = retrieval_method or kwargs.get("retrieval_method", "unknown")

        def to_dict(self):
            return {
                "chunk_id": self.chunk_id,
                "document_id": self.document_id,
                "document_name": self.document_name,
                "page_number": self.page_number,
                "section_title": self.section_title,
                "chunk_text": self.chunk_text,
                "token_count": self.token_count,
                "content_type": self.content_type,
                "table_id": self.table_id,
                "image_references": self.image_references,
                "score": self.score,
                "retrieval_method": self.retrieval_method,
            }

    class SearchResponse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
