import unittest
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.table_extractor import TableExtractor

class TestChunkerAndTables(unittest.TestCase):
    def setUp(self):
        self.chunker = StructureAwareChunker(
            target_chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=20,
            max_chunk_size=200
        )

    def test_token_estimation(self):
        text = "This is a simple sentence for token counting."
        tokens = self.chunker.estimate_token_count(text)
        self.assertTrue(tokens >= 7)

    def test_table_splitter_repeats_headers(self):
        headers = ["Month", "Revenue", "Expenses"]
        rows = [[f"Month {i}", f"${i*1000}", f"${i*500}"] for i in range(1, 15)]
        parts = TableExtractor.split_table_rows_with_header(headers, rows, max_rows=5)
        self.assertEqual(len(parts), 3)
        for part in parts:
            self.assertIn("| Month | Revenue | Expenses |", part)
            self.assertIn("| --- | --- | --- |", part)

    def test_chunk_extracted_document(self):
        mock_doc = {
            "document_id": "test-doc-123",
            "document_name": "annual_report.pdf",
            "pages": [
                {
                    "page_number": 1,
                    "text_blocks": [
                        {"text": "1. Executive Summary", "is_heading": True, "section": "1. Executive Summary"},
                        {"text": "The company achieved record revenue in 2025 across all primary product lines.", "is_heading": False, "section": "1. Executive Summary"},
                        {"text": "2. Financial Highlights", "is_heading": True, "section": "2. Financial Highlights"},
                        {"text": "Operating margin improved by 14% year over year.", "is_heading": False, "section": "2. Financial Highlights"}
                    ],
                    "tables": [
                        {
                            "table_id": "tbl_test_1",
                            "headers": ["Metric", "2024", "2025"],
                            "rows": [["Revenue", "100M", "150M"]],
                            "searchable_text": "| Metric | 2024 | 2025 |\n|---|---|---|\n| Revenue | 100M | 150M |"
                        }
                    ],
                    "images": [
                        {
                            "image_id": "img_1",
                            "file_path": "/path/img1.png",
                            "caption": "Figure 1: Revenue breakdown",
                            "description": "[Figure 1: Revenue breakdown]"
                        }
                    ]
                }
            ]
        }
        chunks = self.chunker.chunk_extracted_document(mock_doc)
        self.assertTrue(len(chunks) >= 3)
        
        # Verify schema conformance
        types = {c.content_type for c in chunks}
        self.assertIn("table", types)
        self.assertIn("caption", types)
        self.assertIn("text", types)

        for c in chunks:
            self.assertEqual(c.document_id, "test-doc-123")
            self.assertEqual(c.document_name, "annual_report.pdf")
            self.assertEqual(c.page_number, 1)
            self.assertTrue(hasattr(c, "chunk_id"))
            self.assertTrue(hasattr(c, "chunk_text"))
            self.assertTrue(hasattr(c, "token_count"))
            self.assertTrue(hasattr(c, "image_references"))

    def test_chunk_overlap_across_blocks_and_sentences(self):
        chunker = StructureAwareChunker(
            target_chunk_size=30,
            chunk_overlap=15,
            min_chunk_size=15,
            max_chunk_size=45,
        )
        # Multi-block within the same section
        doc = {
            "document_id": "overlap-doc",
            "document_name": "overlap.pdf",
            "pages": [
                {
                    "page_number": 1,
                    "text_blocks": [
                        {"text": "First sentence has several words here. Second sentence is the overlap tail.", "section": "Sec 1"},
                        {"text": "Third sentence starts a new block with more words to trigger flush.", "section": "Sec 1"},
                        {"text": "Fourth block also has enough words to force another chunk boundary.", "section": "Sec 1"},
                    ],
                    "tables": [],
                    "images": [],
                }
            ],
        }
        chunks = chunker.chunk_extracted_document(doc)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("Third sentence starts a new block with more words to trigger flush.", chunks[0].chunk_text)
        self.assertIn("Third sentence starts a new block with more words to trigger flush.", chunks[1].chunk_text)

        # Single oversized block split by sentences
        long_text = (
            "Sentence one has enough tokens to fill up space. "
            "Sentence two is the shared boundary sentence. "
            "Sentence three continues the paragraph with additional detail. "
            "Sentence four finishes the oversized block cleanly."
        )
        sent_chunks = chunker._split_by_sentences(long_text)
        self.assertGreaterEqual(len(sent_chunks), 2)
        self.assertIn("Sentence three continues the paragraph with additional detail.", sent_chunks[0])
        self.assertIn("Sentence three continues the paragraph with additional detail.", sent_chunks[1])

if __name__ == "__main__":
    unittest.main()
