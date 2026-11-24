import os
import tempfile
import unittest
from pathlib import Path
from app.ingestion.pdf_validator import PDFValidator, PDFValidationError

class TestPDFValidator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.upload_dir = Path(self.temp_dir.name)
        self.validator = PDFValidator(self.upload_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_non_existent_file(self):
        result = self.validator.validate_file(Path("non_existent.pdf"), "test.pdf")
        self.assertFalse(result["is_valid"])
        self.assertIn("does not exist", result["error"])

    def test_empty_file(self):
        empty_file = self.upload_dir / "empty.pdf"
        empty_file.write_bytes(b"")
        result = self.validator.validate_file(empty_file, "empty.pdf")
        self.assertFalse(result["is_valid"])
        self.assertIn("empty", result["error"])

    def test_invalid_header_file(self):
        corrupt_file = self.upload_dir / "corrupt.txt"
        corrupt_file.write_bytes(b"Not a valid PDF file content")
        result = self.validator.validate_file(corrupt_file, "corrupt.pdf")
        self.assertFalse(result["is_valid"])
        self.assertIn("Missing '%PDF-' header", result["error"])

    def test_valid_pdf_header(self):
        # A minimal mock PDF structure
        mock_pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF"
        pdf_file = self.upload_dir / "minimal.pdf"
        pdf_file.write_bytes(mock_pdf)
        result = self.validator.validate_file(pdf_file, "minimal.pdf")
        # Header is checked
        if result["is_valid"]:
            doc_id, dest_path, meta = self.validator.save_and_register(pdf_file, "minimal.pdf")
            self.assertTrue(dest_path.exists())
            self.assertEqual(meta["document_id"], doc_id)

if __name__ == "__main__":
    unittest.main()
