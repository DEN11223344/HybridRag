import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ImageExtractor:
    def __init__(self, extracted_dir: Path):
        self.extracted_dir = Path(extracted_dir)

    def extract_images_from_page(
        self, doc, page, page_number: int, doc_id: str, text_blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extracts images from a page, saves them to disk, detects nearby figure captions,
        and annotates limitation for complex charts/visual diagrams.
        """
        image_results = []
        try:
            image_list = page.get_images(full=True)
            if not image_list:
                return []

            target_img_dir = self.extracted_dir / doc_id / "images"
            target_img_dir.mkdir(parents=True, exist_ok=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image.get("image")
                image_ext = base_image.get("ext", "png")
                img_filename = f"img_{doc_id}_p{page_number}_{img_index+1}.{image_ext}"
                img_path = target_img_dir / img_filename

                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                # Caption detection: search nearby text blocks for Figure / Chart / Diagram / Table
                caption = self._find_nearby_caption(page, xref, text_blocks)
                
                # Limitation note for visual graphs/charts
                limitation_note = (
                    "Visual element (diagram/chart/image). "
                    "Note: Pure visual data (trends, colors, diagram arrows) requires manual review or multimodal inspection."
                )

                searchable_desc = f"[Image/Figure on Page {page_number}: {caption if caption else 'Visual Figure'}]\n{limitation_note}"

                image_results.append({
                    "image_id": f"img_{doc_id}_p{page_number}_{img_index+1}",
                    "file_path": str(img_path),
                    "filename": img_filename,
                    "page_number": page_number,
                    "caption": caption,
                    "description": searchable_desc,
                })
        except Exception as e:
            logger.warning(f"Error extracting images on page {page_number}: {e}")

        return image_results

    def _find_nearby_caption(self, page, xref: int, text_blocks: List[Dict[str, Any]]) -> Optional[str]:
        """
        Scans text blocks on page to find captions matching Figure / Fig / Chart / Diagram.
        """
        caption_patterns = [
            r"^(?:Figure|Fig\.?|Chart|Diagram|Illustration|Map)\s+\d+[:\.\-]?\s*.+",
            r"^(?:Figure|Fig\.?)\s+[A-Za-z0-9\.\-]+.*",
        ]
        
        for block in text_blocks:
            text = block.get("text", "").strip()
            for pattern in caption_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    return text[:300]
        return None
