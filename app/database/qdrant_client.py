import os
import uuid
import logging
from typing import List, Dict, Any, Optional

from app.schemas.chunk import ChunkMetadata
from app.schemas.search import SearchResultChunk

logger = logging.getLogger(__name__)

class QdrantManager:
    _instance: Optional["QdrantManager"] = None

    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "pdf_hybrid_rag", dense_dim: int = 384):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dense_dim = dense_dim
        self._client = None
        self._use_in_memory_fallback = False
        self._fallback_store: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_instance(cls, host: str = "localhost", port: int = 6333, collection_name: str = "pdf_hybrid_rag", dense_dim: int = 384) -> "QdrantManager":
        if cls._instance is None:
            cls._instance = cls(host=host, port=port, collection_name=collection_name, dense_dim=dense_dim)
        return cls._instance

    def get_client(self):
        if self._client is not None or self._use_in_memory_fallback:
            return self._client

        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host=self.host, port=self.port, timeout=5.0)
            client.get_collections()
            self._client = client
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            self.ensure_collection()
            return self._client
        except Exception as e:
            logger.warning(
                f"Could not connect to Qdrant service at {self.host}:{self.port} ({e}). "
                "Enabling in-memory Qdrant simulation store."
            )
            self._use_in_memory_fallback = True
            return None

    def ensure_collection(self):
        client = self.get_client()
        if self._use_in_memory_fallback:
            return

        try:
            from qdrant_client.http import models
            collections = [c.name for c in client.get_collections().collections]
            if self.collection_name not in collections:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(size=self.dense_dim, distance=models.Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "bm25": models.SparseVectorParams(),
                        "splade": models.SparseVectorParams(),
                    }
                )
                logger.info(f"Collection {self.collection_name} created with dense ({self.dense_dim}, cosine) and sparse (bm25, splade) vectors.")
        except Exception as e:
            logger.error(f"Error ensuring collection {self.collection_name}: {e}")
            self._use_in_memory_fallback = True

    def upsert_chunks(
        self,
        chunks: List[ChunkMetadata],
        dense_vectors: List[List[float]],
        bm25_vectors: List[Dict[str, Any]],
        splade_vectors: List[Dict[str, Any]],
    ) -> int:
        client = self.get_client()

        if self._use_in_memory_fallback or client is None:
            count = 0
            for chunk, d_vec, b_vec, s_vec in zip(chunks, dense_vectors, bm25_vectors, splade_vectors):
                self._fallback_store[chunk.chunk_id] = {
                    "chunk": chunk,
                    "dense": d_vec,
                    "bm25": b_vec,
                    "splade": s_vec,
                    "payload": chunk.to_dict(),
                }
                count += 1
            return count

        try:
            from qdrant_client.http import models
            points = []
            for chunk, d_vec, b_vec, s_vec in zip(chunks, dense_vectors, bm25_vectors, splade_vectors):
                point_id = str(uuid.UUID(chunk.chunk_id)) if self._is_valid_uuid(chunk.chunk_id) else str(uuid.uuid4())
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "dense": d_vec,
                            "bm25": models.SparseVector(indices=b_vec.get("indices", []), values=b_vec.get("values", [])),
                            "splade": models.SparseVector(indices=s_vec.get("indices", []), values=s_vec.get("values", [])),
                        },
                        payload=chunk.to_dict(),
                    )
                )

            batch_size = 64
            for i in range(0, len(points), batch_size):
                client.upsert(
                    collection_name=self.collection_name,
                    points=points[i:i + batch_size],
                    wait=True
                )
            return len(points)
        except Exception as e:
            logger.error(f"Qdrant upsert failed: {e}. Falling back to memory store.")
            self._use_in_memory_fallback = True
            return self.upsert_chunks(chunks, dense_vectors, bm25_vectors, splade_vectors)

    def search_dense(self, query_vec: List[float], top_k: int = 20, document_ids: Optional[List[str]] = None) -> List[SearchResultChunk]:
        client = self.get_client()

        if self._use_in_memory_fallback or client is None:
            return self._fallback_search_dense(query_vec, top_k, document_ids)

        try:
            from qdrant_client.http import models
            query_filter = None
            if document_ids:
                query_filter = models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids))]
                )

            hits = client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_vec),
                limit=top_k,
                query_filter=query_filter,
            )

            results = []
            for hit in hits:
                p = hit.payload
                results.append(
                    SearchResultChunk(
                        chunk_id=p["chunk_id"],
                        document_id=p["document_id"],
                        document_name=p["document_name"],
                        page_number=p["page_number"],
                        section_title=p.get("section_title"),
                        chunk_text=p["chunk_text"],
                        token_count=p.get("token_count", 0),
                        content_type=p.get("content_type", "text"),
                        table_id=p.get("table_id"),
                        image_references=p.get("image_references", []),
                        score=float(hit.score),
                        retrieval_method="dense",
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Error in search_dense: {e}")
            return self._fallback_search_dense(query_vec, top_k, document_ids)

    def search_sparse(self, vector_name: str, sparse_vec: Dict[str, Any], top_k: int = 20, document_ids: Optional[List[str]] = None) -> List[SearchResultChunk]:
        client = self.get_client()

        if self._use_in_memory_fallback or client is None:
            return self._fallback_search_sparse(vector_name, sparse_vec, top_k, document_ids)

        try:
            from qdrant_client.http import models
            query_filter = None
            if document_ids:
                query_filter = models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids))]
                )

            named_sparse_vec = models.NamedSparseVector(
                name=vector_name,
                vector=models.SparseVector(
                    indices=sparse_vec.get("indices", []),
                    values=sparse_vec.get("values", []),
                )
            )

            hits = client.search(
                collection_name=self.collection_name,
                query_vector=named_sparse_vec,
                limit=top_k,
                query_filter=query_filter,
            )

            results = []
            for hit in hits:
                p = hit.payload
                results.append(
                    SearchResultChunk(
                        chunk_id=p["chunk_id"],
                        document_id=p["document_id"],
                        document_name=p["document_name"],
                        page_number=p["page_number"],
                        section_title=p.get("section_title"),
                        chunk_text=p["chunk_text"],
                        token_count=p.get("token_count", 0),
                        content_type=p.get("content_type", "text"),
                        table_id=p.get("table_id"),
                        image_references=p.get("image_references", []),
                        score=float(hit.score),
                        retrieval_method=vector_name,
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Error in search_sparse ({vector_name}): {e}")
            return self._fallback_search_sparse(vector_name, sparse_vec, top_k, document_ids)

    def delete_document(self, document_id: str) -> int:
        client = self.get_client()

        if self._use_in_memory_fallback or client is None:
            to_del = [cid for cid, v in self._fallback_store.items() if v["payload"].get("document_id") == document_id]
            for cid in to_del:
                del self._fallback_store[cid]
            return len(to_del)

        try:
            from qdrant_client.http import models
            client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                    )
                ),
            )
            return 1
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return 0

    def list_documents(self) -> List[Dict[str, Any]]:
        client = self.get_client()

        if self._use_in_memory_fallback or client is None:
            docs = {}
            for item in self._fallback_store.values():
                p = item["payload"]
                did = p.get("document_id")
                if did not in docs:
                    docs[did] = {
                        "document_id": did,
                        "document_name": p.get("document_name", "unknown"),
                        "chunk_count": 0,
                        "total_pages": 0,
                    }
                docs[did]["chunk_count"] += 1
                docs[did]["total_pages"] = max(docs[did]["total_pages"], p.get("page_number", 1))
            return list(docs.values())

        try:
            records, _ = client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=["document_id", "document_name", "page_number"],
            )
            docs = {}
            for r in records:
                p = r.payload
                did = p.get("document_id")
                if not did:
                    continue
                if did not in docs:
                    docs[did] = {
                        "document_id": did,
                        "document_name": p.get("document_name", "unknown"),
                        "chunk_count": 0,
                        "total_pages": 0,
                    }
                docs[did]["chunk_count"] += 1
                docs[did]["total_pages"] = max(docs[did]["total_pages"], p.get("page_number", 1))
            return list(docs.values())
        except Exception as e:
            logger.error(f"Error listing documents in Qdrant: {e}")
            return []

    def _fallback_search_dense(self, query_vec: List[float], top_k: int, document_ids: Optional[List[str]]) -> List[SearchResultChunk]:
        import math
        scored = []
        for cid, item in self._fallback_store.items():
            p = item["payload"]
            if document_ids and p.get("document_id") not in document_ids:
                continue
            d_vec = item["dense"]
            dot = sum(a * b for a, b in zip(query_vec, d_vec))
            norm_q = math.sqrt(sum(a * a for a in query_vec)) or 1.0
            norm_d = math.sqrt(sum(b * b for b in d_vec)) or 1.0
            score = dot / (norm_q * norm_d)
            scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResultChunk(
                chunk_id=p["chunk_id"],
                document_id=p["document_id"],
                document_name=p["document_name"],
                page_number=p["page_number"],
                section_title=p.get("section_title"),
                chunk_text=p["chunk_text"],
                token_count=p.get("token_count", 0),
                content_type=p.get("content_type", "text"),
                table_id=p.get("table_id"),
                image_references=p.get("image_references", []),
                score=float(score),
                retrieval_method="dense",
            )
            for score, p in scored[:top_k]
        ]

    def _fallback_search_sparse(self, vector_name: str, sparse_vec: Dict[str, Any], top_k: int, document_ids: Optional[List[str]]) -> List[SearchResultChunk]:
        q_map = dict(zip(sparse_vec.get("indices", []), sparse_vec.get("values", [])))
        scored = []
        for cid, item in self._fallback_store.items():
            p = item["payload"]
            if document_ids and p.get("document_id") not in document_ids:
                continue
            s_vec = item[vector_name]
            d_map = dict(zip(s_vec.get("indices", []), s_vec.get("values", [])))
            dot = sum(val * d_map[idx] for idx, val in q_map.items() if idx in d_map)
            if dot > 0 or len(scored) < top_k:
                scored.append((dot, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResultChunk(
                chunk_id=p["chunk_id"],
                document_id=p["document_id"],
                document_name=p["document_name"],
                page_number=p["page_number"],
                section_title=p.get("section_title"),
                chunk_text=p["chunk_text"],
                token_count=p.get("token_count", 0),
                content_type=p.get("content_type", "text"),
                table_id=p.get("table_id"),
                image_references=p.get("image_references", []),
                score=float(score),
                retrieval_method=vector_name,
            )
            for score, p in scored[:top_k]
        ]

    @staticmethod
    def _is_valid_uuid(val: str) -> bool:
        try:
            uuid.UUID(str(val))
            return True
        except ValueError:
            return False
