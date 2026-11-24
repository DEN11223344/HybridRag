import os
import json
from pathlib import Path
from typing import List, Dict, Any

from app.schemas.chunk import ChunkMetadata

class SampleDataGenerator:
    """
    Generates realistic evaluation datasets covering 5 distinct document archetypes:
    1. Text-Heavy Research Document (AI architectures)
    2. Table-Heavy Financial Report (Tables, metrics, columns)
    3. Technical Specification Manual (Exact part codes, BM25 keywords, numbers)
    4. Diagram & Chart Report (Figure captions, image references)
    5. Historical Scanned Memo (OCR extracted content)
    """

    @staticmethod
    def get_test_documents() -> List[Dict[str, Any]]:
        return [
            {
                "document_id": "doc_research_01",
                "document_name": "ai_research_transformers.pdf",
                "chunks": [
                    {
                        "chunk_id": "res_c1",
                        "page_number": 1,
                        "section_title": "1. Introduction to Attention Mechanisms",
                        "chunk_text": "The Transformer architecture replaces recurrent neural networks with multi-head self-attention mechanisms, computing scaled dot-product attention across all input tokens simultaneously.",
                        "content_type": "text",
                    },
                    {
                        "chunk_id": "res_c2",
                        "page_number": 2,
                        "section_title": "2. Positional Encodings",
                        "chunk_text": "Since self-attention contains no inherent recurrence or convolution, positional encodings using sinusoidal functions are injected into input embeddings to preserve token order.",
                        "content_type": "text",
                    },
                    {
                        "chunk_id": "res_c3",
                        "page_number": 3,
                        "section_title": "3. Computational Complexity",
                        "chunk_text": "Standard self-attention exhibits O(N^2) time and memory complexity with respect to sequence length N, making linear attention approximations critical for long sequences.",
                        "content_type": "text",
                    }
                ]
            },
            {
                "document_id": "doc_finance_02",
                "document_name": "quarterly_financial_report_q4.pdf",
                "chunks": [
                    {
                        "chunk_id": "fin_c1",
                        "page_number": 2,
                        "section_title": "Executive Financial Highlights",
                        "chunk_text": "Consolidated net revenue for Q4 reached $840 million, representing a 22% increase year-over-year compared to $688 million in the prior year.",
                        "content_type": "text",
                    },
                    {
                        "chunk_id": "fin_c2",
                        "page_number": 4,
                        "section_title": "Segment Revenue Table",
                        "chunk_text": "### Table: (Page 4)\n| Business Segment | Q4 Revenue | Q4 Operating Income | YoY Growth |\n|---|---|---|---|\n| Cloud Infrastructure | $420M | $135M | +34% |\n| Enterprise Software | $280M | $90M | +15% |\n| Hardware Systems | $140M | $25M | +6% |",
                        "content_type": "table",
                        "table_id": "tbl_q4_rev",
                    },
                    {
                        "chunk_id": "fin_c3",
                        "page_number": 5,
                        "section_title": "Capital Expenditures",
                        "chunk_text": "Free cash flow for the trailing twelve months totaled $310 million after deducting $160 million in datacenter capital expenditures.",
                        "content_type": "text",
                    }
                ]
            },
            {
                "document_id": "doc_specs_03",
                "document_name": "industrial_robotics_manual_v3.pdf",
                "chunks": [
                    {
                        "chunk_id": "spec_c1",
                        "page_number": 1,
                        "section_title": "Actuator Model TX-9040",
                        "chunk_text": "The industrial robotic arm employs actuator model TX-9040 with peak torque output of 450 Nm and continuous operating voltage of 48V DC.",
                        "content_type": "text",
                    },
                    {
                        "chunk_id": "spec_c2",
                        "page_number": 2,
                        "section_title": "Error Code Diagnostics",
                        "chunk_text": "Diagnostic code ERR-0x4F indicates thermal overload shutdown on Axis-3. Technicians must cycle power and verify coolant flow through valve V-12B.",
                        "content_type": "text",
                    }
                ]
            },
            {
                "document_id": "doc_diagrams_04",
                "document_name": "aerospace_turbine_diagrams.pdf",
                "chunks": [
                    {
                        "chunk_id": "diag_c1",
                        "page_number": 3,
                        "section_title": "Figure 4: Combustion Chamber Flow",
                        "chunk_text": "[Image/Figure on Page 3: Figure 4: Dual-stage combustor airflow]\nVisual element (diagram/chart/image). Note: Pure visual data (trends, colors, diagram arrows) requires manual review or multimodal inspection. Caption indicates primary airflow enters combustor liner at Mach 0.25.",
                        "content_type": "caption",
                        "image_references": ["data/extracted/aerospace_turbine_diagrams/images/img_p3_1.png"],
                    }
                ]
            },
            {
                "document_id": "doc_ocr_05",
                "document_name": "scanned_archive_memorandum_1988.pdf",
                "chunks": [
                    {
                        "chunk_id": "ocr_c1",
                        "page_number": 1,
                        "section_title": "Scanned Historical Archive",
                        "chunk_text": "MEMORANDUM DATED OCTOBER 14, 1988: Project Falcon flight telemetry tests successfully logged 4,200 continuous hours without avionics bus degradation.",
                        "content_type": "image_ocr",
                    }
                ]
            }
        ]

    @staticmethod
    def get_evaluation_queries() -> List[Dict[str, Any]]:
        """
        Gold evaluation queries mapped to expected ground truth chunk IDs.
        """
        return [
            {
                "query": "How do Transformers compute attention across input tokens?",
                "expected_chunks": ["res_c1"],
                "target_type": "semantic_concept",
            },
            {
                "query": "What is the computational complexity of self-attention?",
                "expected_chunks": ["res_c3"],
                "target_type": "technical_formula",
            },
            {
                "query": "What was the Q4 revenue and YoY growth for Cloud Infrastructure?",
                "expected_chunks": ["fin_c2"],
                "target_type": "tabular_value",
            },
            {
                "query": "How much free cash flow was generated for the trailing twelve months?",
                "expected_chunks": ["fin_c3"],
                "target_type": "financial_metric",
            },
            {
                "query": "What does diagnostic code ERR-0x4F mean and which valve must be checked?",
                "expected_chunks": ["spec_c2"],
                "target_type": "exact_code_bm25",
            },
            {
                "query": "What is the peak torque and operating voltage of actuator model TX-9040?",
                "expected_chunks": ["spec_c1"],
                "target_type": "technical_spec_bm25",
            },
            {
                "query": "What is the primary combustor airflow speed shown in Figure 4?",
                "expected_chunks": ["diag_c1"],
                "target_type": "diagram_caption",
            },
            {
                "query": "How many hours of flight telemetry were logged in the October 1988 Project Falcon memo?",
                "expected_chunks": ["ocr_c1"],
                "target_type": "ocr_scanned_data",
            }
        ]
