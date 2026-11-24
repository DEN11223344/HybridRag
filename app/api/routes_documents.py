import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Any

try:
    from fastapi import APIRouter, UploadFile, File, HTTPException
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
    class UploadFile:
        def __init__(self, filename="file.pdf", file=None):
            self.filename = filename
            self.file = file
    def File(*args, **kwargs):
        return None

from app.config import settings
from app.schemas.document import DocumentMetadata, DocumentListResponse, DocumentDeleteResponse
from app.ingestion.pdf_validator import PDFValidator, PDFValidationError
from app.ingestion.pdf_extractor import PDFExtractor
from app.ingestion.chunker import StructureAwareChunker
from app.embeddings.dense_embeddings import DenseEmbedder
from app.embeddings.bm25_embeddings import BM25SparseEmbedder
from app.embeddings.splade_embeddings import SpladeEmbedder
from app.database.qdrant_client import QdrantManager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Documents"])

validator = PDFValidator(upload_dir=settings.upload_dir)
extractor = PDFExtractor(extracted_dir=settings.extracted_dir, char_threshold=settings.ocr_char_threshold)
chunker = StructureAwareChunker(
    target_chunk_size=settings.target_chunk_size,
    chunk_overlap=settings.chunk_overlap,
    min_chunk_size=settings.min_chunk_size,
    max_chunk_size=settings.max_chunk_size,
)
dense_embedder = DenseEmbedder.get_instance(model_name=settings.dense_model, dim=settings.dense_dim, batch_size=settings.batch_size)
bm25_embedder = BM25SparseEmbedder()
splade_embedder = SpladeEmbedder.get_instance(model_name=settings.splade_model)
qdrant = QdrantManager.get_instance(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    collection_name=settings.qdrant_collection,
    dense_dim=settings.dense_dim,
)

doc_registry = {}

@router.post("/upload")
async def upload_documents(files: List[Any] = None):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results: List[DocumentMetadata] = []

    for file in files:
        filename = getattr(file, "filename", "uploaded.pdf")
        logger.info(f"Processing upload for: {filename}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            if hasattr(file, "file") and file.file is not None:
                shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        try:
            doc_id, saved_path, val_meta = validator.save_and_register(tmp_path, filename)
            
            doc_meta = DocumentMetadata(
                document_id=doc_id,
                document_name=filename,
                total_pages=val_meta.get("total_pages", 1),
                file_size_bytes=val_meta.get("file_size_bytes", 0),
                status="processing",
                title=val_meta.get("title"),
                author=val_meta.get("author"),
            )
            doc_registry[doc_id] = doc_meta

            extracted = extractor.extract_document(saved_path, doc_id, filename)
            doc_meta.table_count = extracted.get("total_tables", 0)
            doc_meta.image_count = extracted.get("total_images", 0)

            chunks = chunker.chunk_extracted_document(extracted)
            if not chunks:
                doc_meta.status = "indexed"
                doc_meta.chunk_count = 0
                results.append(doc_meta)
                continue

            doc_meta.chunk_count = len(chunks)
            chunk_texts = [c.chunk_text for c in chunks]

            dense_vectors = dense_embedder.embed_texts(chunk_texts)
            bm25_vectors = bm25_embedder.encode_documents_batch(chunk_texts)
            splade_vectors = splade_embedder.encode_batch(chunk_texts)

            qdrant.upsert_chunks(chunks, dense_vectors, bm25_vectors, splade_vectors)

            doc_meta.status = "indexed"
            results.append(doc_meta)
        except PDFValidationError as pve:
            logger.error(f"Validation failed for {filename}: {pve}")
            failed_doc = DocumentMetadata(
                document_id="error",
                document_name=filename,
                total_pages=0,
                file_size_bytes=0,
                status="failed",
                error_message=str(pve),
            )
            results.append(failed_doc)
        except Exception as e:
            logger.error(f"Ingestion failed for {filename}: {e}", exc_info=True)
            failed_doc = DocumentMetadata(
                document_id="error",
                document_name=filename,
                total_pages=0,
                file_size_bytes=0,
                status="failed",
                error_message=str(e),
            )
            results.append(failed_doc)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return results

@router.get("/documents")
def list_documents():
    q_docs = qdrant.list_documents()
    for qd in q_docs:
        did = qd.get("document_id")
        if did and did not in doc_registry:
            doc_registry[did] = DocumentMetadata(
                document_id=did,
                document_name=qd.get("document_name", "unknown.pdf"),
                total_pages=qd.get("total_pages", 1),
                file_size_bytes=0,
                status="indexed",
                chunk_count=qd.get("chunk_count", 0),
            )

    docs = list(doc_registry.values())
    return DocumentListResponse(documents=docs, total_count=len(docs))

@router.get("/documents/{document_id}")
def get_document(document_id: str):
    if document_id in doc_registry:
        return doc_registry[document_id]

    q_docs = qdrant.list_documents()
    for qd in q_docs:
        if qd.get("document_id") == document_id:
            meta = DocumentMetadata(
                document_id=document_id,
                document_name=qd.get("document_name", "unknown.pdf"),
                total_pages=qd.get("total_pages", 1),
                file_size_bytes=0,
                status="indexed",
                chunk_count=qd.get("chunk_count", 0),
            )
            doc_registry[document_id] = meta
            return meta

    raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    deleted = qdrant.delete_document(document_id)
    doc_name = "unknown"
    if document_id in doc_registry:
        doc_name = doc_registry[document_id].document_name
        del doc_registry[document_id]

    return DocumentDeleteResponse(
        document_id=document_id,
        document_name=doc_name,
        deleted_chunks=deleted,
        success=True,
        message=f"Document {document_id} and associated indexed vectors removed.",
    )
