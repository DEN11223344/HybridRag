from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

if HAS_PYDANTIC:
    class DocumentMetadata(BaseModel):
        document_id: str
        document_name: str
        total_pages: int
        file_size_bytes: int
        mime_type: str = "application/pdf"
        created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
        status: str = "uploaded"  # uploaded, processing, indexed, failed
        error_message: Optional[str] = None
        chunk_count: int = 0
        table_count: int = 0
        image_count: int = 0
        title: Optional[str] = None
        author: Optional[str] = None

    class DocumentListResponse(BaseModel):
        documents: List[DocumentMetadata]
        total_count: int

    class DocumentDeleteResponse(BaseModel):
        document_id: str
        document_name: str
        deleted_chunks: int
        success: bool
        message: str
else:
    class DocumentMetadata:
        def __init__(self, **kwargs):
            self.document_id = kwargs.get("document_id", "")
            self.document_name = kwargs.get("document_name", "")
            self.total_pages = kwargs.get("total_pages", 0)
            self.file_size_bytes = kwargs.get("file_size_bytes", 0)
            self.mime_type = kwargs.get("mime_type", "application/pdf")
            self.created_at = kwargs.get("created_at", datetime.utcnow().isoformat())
            self.status = kwargs.get("status", "uploaded")
            self.error_message = kwargs.get("error_message")
            self.chunk_count = kwargs.get("chunk_count", 0)
            self.table_count = kwargs.get("table_count", 0)
            self.image_count = kwargs.get("image_count", 0)
            self.title = kwargs.get("title")
            self.author = kwargs.get("author")

        def to_dict(self):
            return self.__dict__

    class DocumentListResponse:
        def __init__(self, documents: List[Any], total_count: int):
            self.documents = documents
            self.total_count = total_count

    class DocumentDeleteResponse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
