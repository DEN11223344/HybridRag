import unittest
from app.schemas.search import SearchResultChunk
from app.retrieval.fusion import ReciprocalRankFusion

class TestReciprocalRankFusion(unittest.TestCase):
    def setUp(self):
        self.rrf = ReciprocalRankFusion(k=60)

    def test_rrf_scoring_and_deduplication(self):
        # Candidate 1 appears in both lists
        c1 = SearchResultChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            document_name="paper.pdf",
            page_number=1,
            chunk_text="Dense retrieval using neural networks",
            score=0.9,
            retrieval_method="dense"
        )
        # Candidate 2 appears only in list 1
        c2 = SearchResultChunk(
            chunk_id="chunk_2",
            document_id="doc_1",
            document_name="paper.pdf",
            page_number=2,
            chunk_text="BM25 lexical search ranking",
            score=0.8,
            retrieval_method="dense"
        )
        # Candidate 3 appears first in list 2
        c3 = SearchResultChunk(
            chunk_id="chunk_3",
            document_id="doc_2",
            document_name="manual.pdf",
            page_number=5,
            chunk_text="SPLADE sparse semantic retrieval",
            score=12.5,
            retrieval_method="splade"
        )
        c1_in_bm25 = SearchResultChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            document_name="paper.pdf",
            page_number=1,
            chunk_text="Dense retrieval using neural networks",
            score=10.0,
            retrieval_method="bm25"
        )

        list_dense = [c1, c2]
        list_bm25 = [c3, c1_in_bm25]

        fused = self.rrf.fuse([list_dense, list_bm25], top_k=5)
        self.assertEqual(len(fused), 3)

        # chunk_1 is rank 1 in list_dense (1/(60+1) = 0.01639) and rank 2 in list_bm25 (1/(60+2) = 0.01613)
        # total score ~ 0.03252
        # chunk_3 is rank 1 in list_bm25 (1/61 = 0.01639)
        # chunk_2 is rank 2 in list_dense (1/62 = 0.01613)
        # Therefore, chunk_1 must be ranked #1
        self.assertEqual(fused[0].chunk_id, "chunk_1")
        self.assertTrue("rrf(" in fused[0].retrieval_method)
        self.assertTrue(fused[0].score > fused[1].score)

if __name__ == "__main__":
    unittest.main()
