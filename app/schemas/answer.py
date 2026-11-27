from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

if HAS_PYDANTIC:
    class Citation(BaseModel):
        document_name: str
        document_id: str
        page_number: int
        section_title: Optional[str] = None
        chunk_id: str
        snippet: str

    class AskRequest(BaseModel):
        query: str
        document_ids: Optional[List[str]] = None
        top_k: int = 5
        mode: str = "hybrid"
        rerank: bool = False

    class AskResponse(BaseModel):
        query: str
        answer: str
        citations: List[Citation]
        retrieved_chunks: List[Any]
        latency_ms: float
        llm_provider: str
else:
    class Citation:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def to_dict(self):
            return self.__dict__

    class AskRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class AskResponse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def to_dict(self):
            return self.__dict__
