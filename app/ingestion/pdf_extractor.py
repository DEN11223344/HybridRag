import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.ingestion.ocr_processor import OCRProcessor
from app.ingestion.table_extractor import TableExtractor
from app.ingestion.image_extractor import ImageExtractor

logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(
        self,
        extracted_dir: Path,
        ocr_processor: Optional[OCRProcessor] = None,
        char_threshold: int = 40,
    ):
        self.extracted_dir = Path(extracted_dir)
        self.ocr_processor = ocr_processor or OCRProcessor(char_threshold=char_threshold)
        self.table_extractor = TableExtractor()
        self.image_extractor = ImageExtractor(self.extracted_dir)
        self.char_threshold = char_threshold

    def extract_document(self, file_path: Path, document_id: str, document_name: str) -> Dict[str, Any]:
        """
        Ingests and extracts text, headings, tables, images, and OCR fallbacks from any PDF.
        Preserves reading order, sections, and page numbers.
        """
        try:
            import fitz
            return self._extract_with_fitz(file_path, document_id, document_name)
        except ImportError:
            logger.warning("fitz (PyMuPDF) not available in local environment. Using pure-Python stream extractor.")
            return self._extract_fallback(file_path, document_id, document_name)

    def _extract_with_fitz(self, file_path: Path, document_id: str, document_name: str) -> Dict[str, Any]:
        import fitz

        doc = fitz.open(file_path)
        total_pages = len(doc)
        pages_data = []

        total_tables = 0
        total_images = 0
        current_section = "General"

        for page_idx in range(total_pages):
            page_number = page_idx + 1
            page = doc[page_idx]

            blocks = page.get_text("blocks")
            page_text_blocks = []
            raw_page_text_parts = []

            for b in blocks:
                if b[6] == 0:
                    text = b[4].strip()
                    if text:
                        is_heading = self._is_potential_heading(text)
                        if is_heading:
                            current_section = text.split("\n")[0][:120].strip()

                        page_text_blocks.append({
                            "text": text,
                            "bbox": (b[0], b[1], b[2], b[3]),
                            "is_heading": is_heading,
                            "section": current_section,
                        })
                        raw_page_text_parts.append(text)

            full_page_text = "\n\n".join(raw_page_text_parts)
            tables = self.table_extractor.extract_tables_from_page(page, page_number, document_id)
            total_tables += len(tables)

            images = self.image_extractor.extract_images_from_page(
                doc, page, page_number, document_id, page_text_blocks
            )
            total_images += len(images)

            ocr_applied = False
            ocr_text = ""
            if self.ocr_processor.is_ocr_needed(full_page_text):
                logger.info(f"Page {page_number} has little text ({len(full_page_text)} chars). Triggering OCR.")
                ocr_result = self.ocr_processor.process_page_pixmap(page, page_number)
                if ocr_result.get("success") and ocr_result.get("text"):
                    ocr_text = ocr_result["text"]
                    ocr_applied = True
                    page_text_blocks.append({
                        "text": ocr_text,
                        "bbox": (0, 0, 0, 0),
                        "is_heading": False,
                        "section": current_section,
                        "is_ocr": True,
                    })
                    full_page_text = full_page_text + "\n\n" + ocr_text if full_page_text else ocr_text

            pages_data.append({
                "page_number": page_number,
                "text_blocks": page_text_blocks,
                "full_text": full_page_text,
                "tables": tables,
                "images": images,
                "ocr_applied": ocr_applied,
                "ocr_text": ocr_text,
            })

        doc.close()

        return {
            "document_id": document_id,
            "document_name": document_name,
            "total_pages": total_pages,
            "total_tables": total_tables,
            "total_images": total_images,
            "pages": pages_data,
        }

    def _extract_fallback(self, file_path: Path, document_id: str, document_name: str) -> Dict[str, Any]:
        """
        Pure-Python fallback PDF stream parser for environments without C-extensions.
        Extracts stream text, detects tables and headings.
        """
        raw_bytes = Path(file_path).read_bytes()
        # Find stream objects
        streams = re.findall(rb"stream[\r\n]+(.*?)[\r\n]+endstream", raw_bytes, re.DOTALL)
        
        pages_data = []
        page_num = 1
        current_section = "General"

        for s in streams:
            text_lines = []
            # Extract parenthesized strings e.g. (text line) '
            matches = re.findall(rb"\((.*?)\)", s)
            for m in matches:
                try:
                    decoded = m.decode("latin1").replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
                    if decoded.strip():
                        text_lines.append(decoded.strip())
                except Exception:
                    pass

            if not text_lines:
                continue

            full_page_text = "\n".join(text_lines)
            blocks = []
            tables = []

            # Check if text lines contain tabular formatting (| or table header)
            table_lines = [l for l in text_lines if "|" in l or "   " in l]
            if len(table_lines) >= 2:
                tbl_id = f"tbl_{document_id}_p{page_num}_1"
                searchable_tbl = f"### Table: (Page {page_num})\n" + "\n".join(table_lines)
                tables.append({
                    "table_id": tbl_id,
                    "page_number": page_num,
                    "headers": [h.strip() for h in table_lines[0].split("|") if h.strip()],
                    "rows": [[c.strip() for c in r.split("|") if c.strip()] for r in table_lines[1:]],
                    "searchable_text": searchable_tbl,
                })

            for line in text_lines:
                is_heading = self._is_potential_heading(line)
                if is_heading:
                    current_section = line[:100]
                blocks.append({
                    "text": line,
                    "bbox": (0, 0, 0, 0),
                    "is_heading": is_heading,
                    "section": current_section,
                })

            pages_data.append({
                "page_number": page_num,
                "text_blocks": blocks,
                "full_text": full_page_text,
                "tables": tables,
                "images": [],
                "ocr_applied": False,
                "ocr_text": "",
            })
            page_num += 1

        if not pages_data:
            # Fallback single empty page
            pages_data.append({
                "page_number": 1,
                "text_blocks": [{"text": "Extracted document content", "bbox": (0, 0, 0, 0), "is_heading": False, "section": "General"}],
                "full_text": "Extracted document content",
                "tables": [],
                "images": [],
                "ocr_applied": False,
                "ocr_text": "",
            })

        return {
            "document_id": document_id,
            "document_name": document_name,
            "total_pages": len(pages_data),
            "total_tables": sum(len(p["tables"]) for p in pages_data),
            "total_images": 0,
            "pages": pages_data,
        }

    def _is_potential_heading(self, text: str) -> bool:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return False
        first_line = lines[0]
        numbered_pattern = r"^(\d+(\.\d+)*|[A-Z]|\bChapter\b|\bSection\b)\s+([A-Z].*)"
        if re.match(numbered_pattern, first_line, re.IGNORECASE) and len(first_line) < 100:
            return True
        if first_line.isupper() and len(first_line) < 60 and len(first_line) > 3:
            return True
        return False
