import os
from pathlib import Path

def create_simple_pdf(output_path: Path, title: str, pages_text: list):
    """
    Creates a compliant PDF 1.4 document purely in Python without external binary dependencies.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    objects = []
    
    # 1: Catalog
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    
    # 2: Pages container (placeholder, will be updated)
    page_count = len(pages_text)
    page_obj_refs = " ".join([f"{3 + i*2} 0 R" for i in range(page_count)])
    pages_obj = f"<< /Type /Pages /Kids [ {page_obj_refs} ] /Count {page_count} >>".encode("ascii")
    objects.append(pages_obj)
    
    for i, p_text in enumerate(pages_text):
        content_idx = 4 + i*2
        # Page object
        page_dict = (
            f"<< /Type /Page /Parent 2 0 R "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
            f"/MediaBox [ 0 0 612 792 ] "
            f"/Contents {content_idx} 0 R >>"
        ).encode("ascii")
        objects.append(page_dict)
        
        # Content stream
        # Escape parenthesis in text
        stream_lines = ["BT", "/F1 12 Tf", "50 720 Td", "16 TL"]
        # Split text into lines
        for line in p_text.split("\n"):
            clean_l = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_lines.append(f"({clean_l}) '")
        stream_lines.append("ET")
        stream_content = "\n".join(stream_lines).encode("latin1")
        
        stream_obj = (
            f"<< /Length {len(stream_content)} >>\nstream\n".encode("ascii")
            + stream_content
            + b"\nendstream"
        )
        objects.append(stream_obj)
    
    # Assemble PDF with cross-reference table
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    
    for idx, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{idx} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
        
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        output.extend(f"{off:010d} 00000 n \n".encode("ascii"))
        
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R "
        f"/Info << /Title ({title}) >> >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    output.extend(trailer)
    
    with open(output_path, "wb") as f:
        f.write(output)
    print(f"Generated PDF: {output_path} ({len(output)} bytes)")

def generate_all_samples():
    sample_dir = Path("data/sample_pdfs")
    
    # PDF 1: AI Transformers Research
    p1_texts = [
        "1. Executive Summary and Architecture Overview\n"
        "The Transformer architecture replaces recurrent neural networks with multi-head self-attention mechanisms.\n"
        "This allows computing scaled dot-product attention across all input tokens simultaneously in parallel.\n"
        "Dense vector representations capture semantic nuances in high-dimensional continuous space.\n"
        "ALiBi and Rotary Positional Embeddings provide strong length extrapolation for large context windows.",
        
        "2. Computational Complexity and Attention Variations\n"
        "Standard multi-head self-attention exhibits O(N^2) computational time and memory complexity.\n"
        "When sequence length N scales to 32k tokens, linear attention approximations and FlashAttention become essential.\n"
        "Cross-encoders are utilized in downstream reranking stages to accurately score query-document relevance."
    ]
    create_simple_pdf(sample_dir / "doc1_ai_transformers.pdf", "AI Transformers Research", p1_texts)

    # PDF 2: Financial Report Q4
    p2_texts = [
        "1. Consolidated Financial Statement Q4 2025\n"
        "Consolidated net revenue for Q4 reached $840 million, an increase of 22% year-over-year.\n"
        "Operating profit expanded to $250 million with gross margin reaching 68% across all core sectors.\n"
        "Free cash flow for the trailing twelve months totaled $310 million after deducting $160 million in Capex.",

        "2. Business Segment Financial Breakdown\n"
        "Financial Performance Table:\n"
        "Segment               Revenue   Operating_Income  Growth\n"
        "Cloud Infrastructure  $420M     $135M             +34%\n"
        "Enterprise Software   $280M     $90M              +15%\n"
        "Hardware Systems      $140M     $25M              +6%\n\n"
        "The Cloud Infrastructure division represented the largest growth vector driven by AI workload deployments."
    ]
    create_simple_pdf(sample_dir / "doc2_quarterly_financials.pdf", "Quarterly Financials Q4 2025", p2_texts)

    # PDF 3: Technical Specifications & Robotics
    p3_texts = [
        "Robotic Arm Model TX-9040 Technical Manual\n"
        "The six-axis industrial manipulator TX-9040 provides 450 Nm peak torque output.\n"
        "Continuous power requirements specify 48V DC with maximum current consumption of 35 Amperes.\n"
        "The control unit operates on deterministic real-time Linux with cycle latency below 250 microseconds.",

        "Maintenance and Troubleshooting Diagnostics\n"
        "Critical Error Code ERR-0x4F:\n"
        "Indicates thermal overload shutdown on Axis-3 joint.\n"
        "Required Action: Cycle power immediately and inspect coolant valve V-12B for obstruction.\n"
        "Secondary Error Code ERR-0x12 indicates encoder synchronization timeout on Axis-1."
    ]
    create_simple_pdf(sample_dir / "doc3_robotics_manual.pdf", "TX-9040 Technical Manual", p3_texts)

    # PDF 4: Aerospace Propulsion & Diagrams
    p4_texts = [
        "Aerospace Turbine Combustor Engineering Report\n"
        "Dual-stage combustion chambers improve thermodynamic efficiency while reducing NOx emissions.\n"
        "Figure 4: Dual-stage combustor airflow diagram.\n"
        "Caption: Figure 4 illustrates primary combustor liner airflow entering at Mach 0.25.\n"
        "Secondary bypass ducting maintains cooling envelope along inner wall boundary layer.",

        "Turbine Blade Temperature Distribution\n"
        "High-pressure turbine inlet temperature reaches 1,650 degrees Celsius under maximum takeoff thrust.\n"
        "Thermal barrier coating consisting of yttria-stabilized zirconia protects nickel-based superalloy substrate."
    ]
    create_simple_pdf(sample_dir / "doc4_aerospace_propulsion.pdf", "Turbine Combustor Engineering", p4_texts)

    # PDF 5: Scanned Archive Memorandum
    p5_texts = [
        "PROJECT FALCON DECLASSIFIED ARCHIVE MEMORANDUM\n"
        "DATE: OCTOBER 14, 1988\n"
        "SUBJECT: LONG-DURATION AVIONICS TELEMETRY FLIGHT TEST RESULTS\n\n"
        "Flight telemetry testing for Project Falcon completed 4,200 continuous hours without avionics bus degradation.\n"
        "All telemetry frequencies were maintained within 0.05 kHz tolerance.\n"
        "Signed: Chief Test Director, Flight Operations Division."
    ]
    create_simple_pdf(sample_dir / "doc5_project_falcon_memo.pdf", "Project Falcon Memo 1988", p5_texts)

if __name__ == "__main__":
    generate_all_samples()
