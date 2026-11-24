from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def dict(self):
            return self.to_dict()
    def Field(default=None, default_factory=None):
        if default_factory is not None:
            return default_factory()
        return default

if HAS_PYDANTIC:
    class ChunkMetadata(BaseModel):
        chunk_id: str
        document_id: str
        document_name: str
        page_number: int
        section_title: Optional[str] = None
        chunk_text: str
        token_count: int = 0
        content_type: str = "text"  # "text", "table", "image_ocr", "caption"
        table_id: Optional[str] = None
        image_references: List[str] = Field(default_factory=list)

        def to_dict(self) -> Dict[str, Any]:
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
            }
else:
    class ChunkMetadata:
        def __init__(
            self,
            chunk_id: str,
            document_id: str,
            document_name: str,
            page_number: int,
            chunk_text: str,
            section_title: Optional[str] = None,
            token_count: int = 0,
            content_type: str = "text",
            table_id: Optional[str] = None,
            image_references: Optional[List[str]] = None,
        ):
            self.chunk_id = chunk_id
            self.document_id = document_id
            self.document_name = document_name
            self.page_number = page_number
            self.section_title = section_title
            self.chunk_text = chunk_text
            self.token_count = token_count
            self.content_type = content_type
            self.table_id = table_id
            self.image_references = image_references if image_references is not None else []

        def to_dict(self) -> Dict[str, Any]:
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
            }
