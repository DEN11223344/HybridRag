import unittest
from app.schemas.chunk import ChunkMetadata
from app.embeddings.dense_embeddings import DenseEmbedder
from app.embeddings.bm25_embeddings import BM25SparseEmbedder
from app.embeddings.splade_embeddings import SpladeEmbedder
from app.database.qdrant_client import QdrantManager
from app.api.routes_search import execute_retrieval

class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.qdrant = QdrantManager.get_instance()
        self.dense = DenseEmbedder.get_instance()
        self.bm25 = BM25SparseEmbedder()
        self.splade = SpladeEmbedder.get_instance()

        # Ingest 3 sample chunks
        self.chunks = [
            ChunkMetadata(
                chunk_id="chk_1",
                document_id="doc_ai_guide",
                document_name="deep_learning_guide.pdf",
                page_number=1,
                section_title="Transformers",
                chunk_text="Transformers use multi-head self-attention mechanisms to process tokens in parallel.",
                token_count=12,
                content_type="text",
            ),
            ChunkMetadata(
                chunk_id="chk_2",
                document_id="doc_ai_guide",
                document_name="deep_learning_guide.pdf",
                page_number=3,
                section_title="Dense Embeddings",
                chunk_text="Dense embeddings project semantic text into a continuous 512-dimensional vector space.",
                token_count=13,
                content_type="text",
            ),
            ChunkMetadata(
                chunk_id="chk_3",
                document_id="doc_database",
                document_name="qdrant_manual.pdf",
                page_number=2,
                section_title="Vector Search",
                chunk_text="Qdrant supports hybrid search by combining dense vectors with named sparse vectors such as BM25 and SPLADE.",
                token_count=18,
                content_type="text",
            ),
        ]

        texts = [c.chunk_text for c in self.chunks]
        d_vecs = self.dense.embed_texts(texts)
        b_vecs = self.bm25.encode_documents_batch(texts)
        s_vecs = self.splade.encode_batch(texts)

        self.qdrant.upsert_chunks(self.chunks, d_vecs, b_vecs, s_vecs)

    def test_search_modes(self):
        # 1. Test BM25 retrieval
        resp_bm25 = execute_retrieval("self-attention mechanisms", mode="bm25", top_k=2)
        self.assertTrue(len(resp_bm25.chunks) > 0)
        self.assertEqual(resp_bm25.chunks[0].chunk_id, "chk_1")

        # 2. Test SPLADE retrieval
        resp_splade = execute_retrieval("vector space embeddings", mode="splade", top_k=2)
        self.assertTrue(len(resp_splade.chunks) > 0)

        # 3. Test Dense retrieval
        resp_dense = execute_retrieval("continuous vector dimension", mode="dense", top_k=2)
        self.assertTrue(len(resp_dense.chunks) > 0)

        # 4. Test Hybrid with RRF and Rerank
        resp_hybrid = execute_retrieval("Qdrant hybrid sparse vectors", mode="hybrid", top_k=2, apply_rerank=True)
        self.assertTrue(len(resp_hybrid.chunks) > 0)
        self.assertEqual(resp_hybrid.chunks[0].chunk_id, "chk_3")

        # 5. Test Document ID filtering
        resp_filtered = execute_retrieval("Transformers", mode="hybrid", top_k=2, document_ids=["doc_database"])
        for c in resp_filtered.chunks:
            self.assertEqual(c.document_id, "doc_database")

    def test_document_deletion(self):
        del_count = self.qdrant.delete_document("doc_database")
        self.assertTrue(del_count >= 1)
        # Verify it cannot be retrieved
        resp = execute_retrieval("Qdrant supports hybrid search", mode="dense", top_k=3, document_ids=["doc_database"])
        self.assertEqual(len(resp.chunks), 0)

if __name__ == "__main__":
    unittest.main()
