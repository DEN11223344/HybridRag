import unittest
from app.schemas.search import SearchResultChunk
from app.retrieval.reranker import CrossEncoderReranker

class TestReranker(unittest.TestCase):
    def setUp(self):
        self.reranker = CrossEncoderReranker()

    def test_reranker_ordering(self):
        query = "What is the operating revenue?"
        c1 = SearchResultChunk(
            chunk_id="c1",
            document_id="d1",
            document_name="report.pdf",
            page_number=1,
            chunk_text="The company's weather forecast for tomorrow is sunny.",
            score=0.5,
            retrieval_method="dense"
        )
        c2 = SearchResultChunk(
            chunk_id="c2",
            document_id="d1",
            document_name="report.pdf",
            page_number=2,
            chunk_text="Total operating revenue reached $450 million in fiscal year 2024.",
            score=0.4,
            retrieval_method="bm25"
        )

        results = self.reranker.rerank(query, [c1, c2], top_k=2)
        self.assertEqual(len(results), 2)
        # c2 has high lexical & semantic relevance for query tokens
        self.assertEqual(results[0].chunk_id, "c2")
        self.assertTrue(results[0].score > results[1].score)

if __name__ == "__main__":
    unittest.main()
