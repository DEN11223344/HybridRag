import unittest
from app.embeddings.dense_embeddings import DenseEmbedder
from app.embeddings.bm25_embeddings import BM25SparseEmbedder
from app.embeddings.splade_embeddings import SpladeEmbedder

class TestEmbeddings(unittest.TestCase):
    def test_dense_embedder(self):
        embedder = DenseEmbedder(model_name="BAAI/bge-small-en-v1.5", dim=384)
        texts = ["Machine learning with hybrid RAG", "Qdrant vector database on CPU"]
        vecs = embedder.embed_texts(texts)
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), 384)
        self.assertEqual(len(vecs[1]), 384)

        q_vec = embedder.embed_query("hybrid rag")
        self.assertEqual(len(q_vec), 384)

    def test_bm25_sparse_embedder(self):
        bm25 = BM25SparseEmbedder()
        doc = "FastAPI with Qdrant and PyMuPDF version 1.24.0 table extraction."
        encoded_doc = bm25.encode_document(doc)
        
        self.assertIn("indices", encoded_doc)
        self.assertIn("values", encoded_doc)
        self.assertEqual(len(encoded_doc["indices"]), len(encoded_doc["values"]))
        self.assertTrue(len(encoded_doc["indices"]) > 0)

        query = "PyMuPDF table extraction"
        encoded_q = bm25.encode_query(query)
        self.assertTrue(len(encoded_q["indices"]) > 0)
        
        for idx in encoded_doc["indices"]:
            self.assertIsInstance(idx, int)
            self.assertTrue(idx > 0)

    def test_splade_sparse_embedder(self):
        splade = SpladeEmbedder()
        text = "Reciprocal rank fusion combines BM25 and SPLADE."
        encoded = splade.encode_text(text)
        
        self.assertIn("indices", encoded)
        self.assertIn("values", encoded)
        self.assertEqual(len(encoded["indices"]), len(encoded["values"]))
        self.assertTrue(len(encoded["indices"]) > 0)

if __name__ == "__main__":
    unittest.main()
