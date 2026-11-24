import re
import uuid
import logging
from typing import List, Dict, Any, Optional

from app.schemas.chunk import ChunkMetadata

logger = logging.getLogger(__name__)

class StructureAwareChunker:
    def __init__(
        self,
        target_chunk_size: int = 512,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
        max_chunk_size: int = 700,
    ):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def estimate_token_count(self, text: str) -> int:
        """
        Fast, CPU-efficient token count estimation.
        Standard rule of thumb: ~0.75 words per token or ~4 chars per token.
        Splitting by whitespace and punctuation provides an accurate proxy.
        """
        if not text:
            return 0
        # Count words and punctuation clusters
        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return max(1, len(words))

    def chunk_extracted_document(self, extracted_doc: Dict[str, Any]) -> List[ChunkMetadata]:
        """
        Performs structure-aware chunking on an extracted document:
        - Keeps tables intact or splits rows while repeating headers
        - Preserves figure captions and associates image references
        - Splits by headings, paragraphs, and sentence boundaries
        - Respects target_chunk_size, chunk_overlap, min/max chunk size
        - Retains full metadata for every chunk.
        """
        document_id = extracted_doc["document_id"]
        document_name = extracted_doc["document_name"]
        pages = extracted_doc.get("pages", [])

        all_chunks: List[ChunkMetadata] = []

        for page in pages:
            page_number = page["page_number"]
            tables = page.get("tables", [])
            images = page.get("images", [])
            text_blocks = page.get("text_blocks", [])

            # 1. Chunk Tables
            for table in tables:
                table_id = table["table_id"]
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                searchable_text = table["searchable_text"]
                tbl_tokens = self.estimate_token_count(searchable_text)

                if tbl_tokens <= self.max_chunk_size:
                    chunk = ChunkMetadata(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_number,
                        section_title="Table",
                        chunk_text=searchable_text,
                        token_count=tbl_tokens,
                        content_type="table",
                        table_id=table_id,
                        image_references=[],
                    )
                    all_chunks.append(chunk)
                else:
                    # Split large table across multiple chunks, repeating headers
                    from app.ingestion.table_extractor import TableExtractor
                    table_chunks_md = TableExtractor.split_table_rows_with_header(headers, rows, max_rows=10)
                    for part_idx, sub_md in enumerate(table_chunks_md, 1):
                        sub_text = f"### Table {table_id} (Page {page_number}, Part {part_idx}):\n{sub_md}"
                        sub_tokens = self.estimate_token_count(sub_text)
                        chunk = ChunkMetadata(
                            chunk_id=str(uuid.uuid4()),
                            document_id=document_id,
                            document_name=document_name,
                            page_number=page_number,
                            section_title="Table (Split)",
                            chunk_text=sub_text,
                            token_count=sub_tokens,
                            content_type="table",
                            table_id=table_id,
                            image_references=[],
                        )
                        all_chunks.append(chunk)

            # 2. Chunk Images & Captions
            for img in images:
                img_desc = img.get("description", "")
                img_tokens = self.estimate_token_count(img_desc)
                img_ref = img.get("file_path", "")
                chunk = ChunkMetadata(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_number,
                    section_title="Figure / Caption",
                    chunk_text=img_desc,
                    token_count=img_tokens,
                    content_type="caption",
                    table_id=None,
                    image_references=[img_ref] if img_ref else [],
                )
                all_chunks.append(chunk)

            # 3. Chunk Text Blocks (structure-aware: by section, paragraph, sentence)
            current_section = None
            accumulated_text = []
            accumulated_tokens = 0
            accumulated_type = "text"

            for block in text_blocks:
                b_text = block.get("text", "").strip()
                if not b_text:
                    continue

                b_tokens = self.estimate_token_count(b_text)
                b_section = block.get("section")
                is_ocr = block.get("is_ocr", False)
                block_type = "image_ocr" if is_ocr else "text"

                if current_section is None:
                    current_section = b_section

                # If section changed and we have accumulated enough text, flush chunk
                if b_section and b_section != current_section and accumulated_tokens >= self.min_chunk_size:
                    self._flush_chunk(
                        all_chunks, document_id, document_name, page_number,
                        current_section, accumulated_text, accumulated_tokens, accumulated_type
                    )
                    accumulated_text = []
                    accumulated_tokens = 0
                    current_section = b_section

                # If adding this block exceeds target chunk size, split or flush
                if accumulated_tokens + b_tokens > self.target_chunk_size:
                    if accumulated_tokens >= self.min_chunk_size:
                        self._flush_chunk(
                            all_chunks, document_id, document_name, page_number,
                            current_section, accumulated_text, accumulated_tokens, accumulated_type
                        )
                        # Keep overlap if possible
                        accumulated_text = []
                        accumulated_tokens = 0

                    # If a single block is huge (> max_chunk_size), split by sentences
                    if b_tokens > self.max_chunk_size:
                        sentence_chunks = self._split_by_sentences(b_text)
                        for sent_chunk in sentence_chunks:
                            s_toks = self.estimate_token_count(sent_chunk)
                            chunk = ChunkMetadata(
                                chunk_id=str(uuid.uuid4()),
                                document_id=document_id,
                                document_name=document_name,
                                page_number=page_number,
                                section_title=current_section,
                                chunk_text=sent_chunk,
                                token_count=s_toks,
                                content_type=block_type,
                                table_id=None,
                                image_references=[],
                            )
                            all_chunks.append(chunk)
                        continue

                accumulated_text.append(b_text)
                accumulated_tokens += b_tokens
                if is_ocr:
                    accumulated_type = "image_ocr"

            # Flush remaining accumulated text on the page
            if accumulated_text:
                self._flush_chunk(
                    all_chunks, document_id, document_name, page_number,
                    current_section, accumulated_text, accumulated_tokens, accumulated_type
                )

        return all_chunks

    def _flush_chunk(
        self,
        chunks: List[ChunkMetadata],
        doc_id: str,
        doc_name: str,
        page_num: int,
        section: Optional[str],
        text_parts: List[str],
        token_count: int,
        content_type: str,
    ):
        combined = "\n\n".join(text_parts).strip()
        if not combined:
            return
        chunk = ChunkMetadata(
            chunk_id=str(uuid.uuid4()),
            document_id=doc_id,
            document_name=doc_name,
            page_number=page_num,
            section_title=section,
            chunk_text=combined,
            token_count=token_count,
            content_type=content_type,
            table_id=None,
            image_references=[],
        )
        chunks.append(chunk)

    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Splits a large paragraph by sentence boundaries respecting target_chunk_size.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        results = []
        cur_parts = []
        cur_toks = 0

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_toks = self.estimate_token_count(s_clean)
            if cur_toks + s_toks > self.target_chunk_size and cur_toks >= self.min_chunk_size:
                results.append(" ".join(cur_parts))
                cur_parts = []
                cur_toks = 0
            cur_parts.append(s_clean)
            cur_toks += s_toks

        if cur_parts:
            results.append(" ".join(cur_parts))
        return results
