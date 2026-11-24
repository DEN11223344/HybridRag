import unittest
from app.schemas.search import SearchResultChunk
from app.generation.answer_generator import AnswerGenerator

class TestAnswerGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = AnswerGenerator(provider="mock")

    def test_insufficient_information_response(self):
        answer, citations = self.generator.generate_answer("What is the capital of Mars?", [])
        self.assertIn("do not contain sufficient information", answer)
        self.assertEqual(len(citations), 0)

    def test_extractive_answer_with_citations(self):
        chunks = [
            SearchResultChunk(
                chunk_id="c1",
                document_id="doc_financial_2025",
                document_name="annual_financial_report_2025.pdf",
                page_number=4,
                section_title="Operating Results",
                chunk_text="The company generated $520 million in net profit during fiscal year 2025.",
                content_type="text",
                score=0.95
            ),
            SearchResultChunk(
                chunk_id="c2",
                document_id="doc_financial_2025",
                document_name="annual_financial_report_2025.pdf",
                page_number=7,
                section_title="Balance Sheet",
                chunk_text="| Metric | 2024 | 2025 |\n|---|---|---|\n| Net Profit | $410M | $520M |",
                content_type="table",
                table_id="tbl_1",
                score=0.90
            )
        ]

        query = "What was the net profit in 2025?"
        answer, citations = self.generator.generate_answer(query, chunks)

        self.assertIn("$520 million", answer)
        self.assertTrue(len(citations) >= 1)
        # Check citation fields
        self.assertEqual(citations[0].document_name, "annual_financial_report_2025.pdf")
        self.assertEqual(citations[0].page_number, 4)

if __name__ == "__main__":
    unittest.main()
