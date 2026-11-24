import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class PDFValidationError(Exception):
    pass

class PDFValidator:
    def __init__(self, upload_dir: Path):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def validate_file(self, file_path: Path, original_filename: str) -> Dict[str, Any]:
        """
        Validates a PDF file:
        - Checks magic bytes (%PDF-)
        - Validates file readability and integrity
        - Detects encryption and corruption
        - Extracts document metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return {
                "is_valid": False,
                "error": f"File does not exist: {file_path}",
                "is_encrypted": False,
            }

        file_size = file_path.stat().st_size
        if file_size == 0:
            return {
                "is_valid": False,
                "error": "The uploaded PDF file is empty (0 bytes).",
                "is_encrypted": False,
            }

        # Check PDF magic bytes
        try:
            with open(file_path, "rb") as f:
                header = f.read(1024)
                if b"%PDF-" not in header:
                    return {
                        "is_valid": False,
                        "error": "Invalid file format: Missing '%PDF-' header.",
                        "is_encrypted": False,
                    }
        except Exception as e:
            return {
                "is_valid": False,
                "error": f"Failed to read file header: {str(e)}",
                "is_encrypted": False,
            }

        # Try opening with fitz (PyMuPDF) if available, or basic stream check
        try:
            import fitz
            try:
                doc = fitz.open(file_path)
            except Exception as e:
                return {
                    "is_valid": False,
                    "error": f"Corrupted or invalid PDF file: {str(e)}",
                    "is_encrypted": False,
                }

            if doc.is_encrypted:
                doc.close()
                return {
                    "is_valid": False,
                    "error": "PDF is password protected / encrypted. Unencrypted PDFs required.",
                    "is_encrypted": True,
                }

            total_pages = len(doc)
            if total_pages == 0:
                doc.close()
                return {
                    "is_valid": False,
                    "error": "PDF contains zero pages.",
                    "is_encrypted": False,
                }

            metadata = doc.metadata or {}
            doc.close()

            return {
                "is_valid": True,
                "error": None,
                "is_encrypted": False,
                "total_pages": total_pages,
                "file_size_bytes": file_size,
                "title": metadata.get("title") or original_filename,
                "author": metadata.get("author") or "Unknown",
                "creation_date": metadata.get("creationDate") or "",
            }
        except ImportError:
            # Fallback if fitz is not installed in the current environment
            logger.warning("fitz (PyMuPDF) not available, performed header-only validation.")
            return {
                "is_valid": True,
                "error": None,
                "is_encrypted": False,
                "total_pages": 1,
                "file_size_bytes": file_size,
                "title": original_filename,
                "author": "Unknown",
            }

    def save_and_register(self, temp_path: Path, filename: str) -> Tuple[str, Path, Dict[str, Any]]:
        """
        Validates and copies file to the permanent upload store under a unique document_id.
        """
        val_result = self.validate_file(temp_path, filename)
        if not val_result["is_valid"]:
            raise PDFValidationError(val_result["error"])

        doc_id = str(uuid.uuid4())
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
        dest_filename = f"{doc_id}_{safe_filename}"
        dest_path = self.upload_dir / dest_filename

        shutil.copy2(temp_path, dest_path)
        val_result["document_id"] = doc_id
        val_result["document_name"] = filename
        val_result["saved_path"] = str(dest_path)
        return doc_id, dest_path, val_result
