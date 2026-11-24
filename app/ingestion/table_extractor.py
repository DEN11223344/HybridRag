import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class TableExtractor:
    def __init__(self):
        pass

    def extract_tables_from_page(self, page, page_number: int, doc_id: str) -> List[Dict[str, Any]]:
        """
        Extracts structured tables from a PyMuPDF page using page.find_tables().
        Converts extracted tables into Markdown and structured key-value text.
        """
        extracted_tables = []
        try:
            tabs = page.find_tables()
            if not tabs or len(tabs.tables) == 0:
                return []

            for idx, tab in enumerate(tabs.tables):
                table_id = f"tbl_{doc_id}_p{page_number}_{idx+1}"
                df = tab.extract()  # Returns list of rows (each row is list of cell strings)
                if not df or len(df) == 0:
                    continue

                # Filter out completely empty rows and clean strings
                cleaned_rows = []
                for row in df:
                    cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                    if any(cleaned_row):
                        cleaned_rows.append(cleaned_row)

                if len(cleaned_rows) < 1:
                    continue

                headers = cleaned_rows[0]
                data_rows = cleaned_rows[1:] if len(cleaned_rows) > 1 else []

                # Format as Markdown table
                md_lines = []
                md_lines.append("| " + " | ".join(headers) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for r in data_rows:
                    # Pad or truncate row to match headers count
                    padded = r + [""] * max(0, len(headers) - len(r))
                    md_lines.append("| " + " | ".join(padded[:len(headers)]) + " |")
                markdown_table = "\n".join(md_lines)

                # Format as key-value row statements to guarantee value-column associations
                row_descriptions = []
                for r_idx, r in enumerate(data_rows, start=1):
                    items = []
                    for h, val in zip(headers, r):
                        if h and val:
                            items.append(f"{h}: {val}")
                        elif val:
                            items.append(val)
                    if items:
                        row_descriptions.append(f"Row {r_idx} -> " + ", ".join(items))
                row_desc_text = "\n".join(row_descriptions)

                full_searchable_text = f"### Table: (Page {page_number})\n{markdown_table}\n\nStructured Table Content:\n{row_desc_text}"

                extracted_tables.append({
                    "table_id": table_id,
                    "page_number": page_number,
                    "headers": headers,
                    "rows": data_rows,
                    "markdown": markdown_table,
                    "searchable_text": full_searchable_text,
                    "bbox": tab.bbox,
                })
        except Exception as e:
            logger.warning(f"PyMuPDF find_tables encountered error on page {page_number}: {e}")

        return extracted_tables

    @staticmethod
    def split_table_rows_with_header(headers: List[str], rows: List[List[str]], max_rows: int = 15) -> List[str]:
        """
        Splits large table into multiple chunks, repeating the table header in each chunk.
        """
        if not rows:
            return ["| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |"]

        chunks = []
        header_block = "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |"
        for i in range(0, len(rows), max_rows):
            batch = rows[i:i + max_rows]
            batch_lines = [header_block]
            for r in batch:
                padded = r + [""] * max(0, len(headers) - len(r))
                batch_lines.append("| " + " | ".join(padded[:len(headers)]) + " |")
            chunks.append("\n".join(batch_lines))
        return chunks
