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

if __name__ == "__main__":
    unittest.main()
