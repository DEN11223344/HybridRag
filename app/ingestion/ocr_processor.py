import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, char_threshold: int = 40, lang: str = "eng", dpi: int = 150):
        self.char_threshold = char_threshold
        self.lang = lang
        self.dpi = dpi
        self._tesseract_available: Optional[bool] = None

    def is_ocr_needed(self, extracted_text: str, has_images: bool = False) -> bool:
        """
        Determines whether OCR fallback should be executed:
        If page text is shorter than char_threshold or mostly whitespace.
        """
        clean_text = extracted_text.strip()
        if len(clean_text) < self.char_threshold:
            return True
        return False

    def check_tesseract(self) -> bool:
        if self._tesseract_available is not None:
            return self._tesseract_available
        try:
            import pytesseract
            # Test getting tesseract version
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
        except Exception as e:
            logger.warning(f"Tesseract OCR is not installed or not in PATH: {e}. OCR fallback disabled.")
            self._tesseract_available = False
        return self._tesseract_available

    def process_page_pixmap(self, page, page_number: int) -> Dict[str, Any]:
        """
        Renders a PyMuPDF page at configured DPI and performs OCR using Tesseract.
        """
        if not self.check_tesseract():
            return {
                "text": "",
                "page_number": page_number,
                "content_type": "image_ocr",
                "success": False,
                "error": "Tesseract OCR engine unavailable",
            }

        try:
            import fitz
            import pytesseract
            from PIL import Image
            import io

            # Render page to pixmap
            pix = page.get_pixmap(dpi=self.dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            
            ocr_text = pytesseract.image_to_string(img, lang=self.lang)
            clean_ocr = ocr_text.strip()
            
            return {
                "text": clean_ocr,
                "page_number": page_number,
                "content_type": "image_ocr",
                "success": True,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error executing OCR on page {page_number}: {e}")
            return {
                "text": "",
                "page_number": page_number,
                "content_type": "image_ocr",
                "success": False,
                "error": str(e),
            }

    def process_image_file(self, image_path: Path) -> str:
        """
        Extract text from an extracted image file if needed.
        """
        if not self.check_tesseract():
            return ""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            return pytesseract.image_to_string(img, lang=self.lang).strip()
        except Exception as e:
            logger.debug(f"Failed to OCR image {image_path}: {e}")
            return ""
