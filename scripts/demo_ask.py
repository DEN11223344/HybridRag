import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.pdf_validator import PDFValidator
from app.ingestion.pdf_extractor import PDFExtractor
from app.ingestion.chunker import StructureAwareChunker
from app.embeddings.dense_embeddings import DenseEmbedder
from app.embeddings.bm25_embeddings import BM25SparseEmbedder
from app.embeddings.splade_embeddings import SpladeEmbedder
from app.database.qdrant_client import QdrantManager
from app.api.routes_search import execute_retrieval
from app.generation.answer_generator import AnswerGenerator
from app.config import settings

def run_sample_qa_demo():
    print("=" * 80)
    print("      LOCAL HYBRID RAG: 5 SAMPLE PDF INGESTION & QUESTION ANSWERING")
    print("=" * 80)
    
    sample_dir = Path("data/sample_pdfs")
    pdf_files = sorted(list(sample_dir.glob("*.pdf")))
    
    if not pdf_files:
        print("No sample PDFs found in data/sample_pdfs/")
        return

    validator = PDFValidator(upload_dir=settings.upload_dir)
    extractor = PDFExtractor(extracted_dir=settings.extracted_dir)
    chunker = StructureAwareChunker()
    dense = DenseEmbedder.get_instance()
    bm25 = BM25SparseEmbedder()
    splade = SpladeEmbedder.get_instance()
    qdrant = QdrantManager.get_instance()
    generator = AnswerGenerator()

    indexed_docs = []

    print(f"\n[1/3] Ingesting {len(pdf_files)} Universal Sample PDFs:")
    for pdf_path in pdf_files:
        doc_id, saved_path, val_meta = validator.save_and_register(pdf_path, pdf_path.name)
        extracted = extractor.extract_document(saved_path, doc_id, pdf_path.name)
        chunks = chunker.chunk_extracted_document(extracted)
        
        texts = [c.chunk_text for c in chunks]
        d_vecs = dense.embed_texts(texts)
        b_vecs = bm25.encode_documents_batch(texts)
        s_vecs = splade.encode_batch(texts)
        
        qdrant.upsert_chunks(chunks, d_vecs, b_vecs, s_vecs)
        indexed_docs.append((doc_id, pdf_path.name, len(chunks)))
        print(f"  ✓ {pdf_path.name:<32} (ID: {doc_id[:8]}... | Chunks: {len(chunks)} | Pages: {val_meta.get('total_pages')})")

    print(f"\n[2/3] Verified Qdrant Collection: {qdrant.collection_name}")
    print(f"Total documents registered: {len(indexed_docs)}\n")

    sample_questions = [
        "What computational complexity does multi-head self-attention exhibit?",
        "What was the consolidated net revenue for Q4 2025 and free cash flow?",
        "What does error code ERR-0x4F indicate and which valve should be checked?",
        "What is the airflow speed into the combustor liner shown in Figure 4?",
        "How many continuous hours were logged in the October 1988 Project Falcon memorandum?",
    ]

    print("[3/3] Executing Hybrid Retrieval, RRF Fusion, BGE Reranker & QA Generation:\n")

    for idx, question in enumerate(sample_questions, 1):
        print(f"{'='*80}")
        print(f"QUERY {idx}: \"{question}\"")
        print(f"{'-'*80}")

        # Execute full hybrid retrieval + cross-encoder rerank
        search_res = execute_retrieval(question, mode="hybrid", top_k=3, apply_rerank=True)
        answer, citations = generator.generate_answer(question, search_res.chunks)

        print("RETRIEVED & RERANKED CONTEXT CHUNKS:")
        for c_idx, c in enumerate(search_res.chunks, 1):
            print(f"  [{c_idx}] Doc: {c.document_name} (Page {c.page_number}) | Method: {c.retrieval_method} | Score: {c.score}")
            preview = c.chunk_text.replace('\n', ' ')[:120]
            print(f"      Snippet: {preview}...")

        print("\nGROUNDED ANSWER:")
        print(f"{answer}\n")

        print("VERIFIED CITATIONS:")
        for cit in citations:
            print(f"  • Document: {cit.document_name} | Page: {cit.page_number} | Chunk: {cit.chunk_id[:8]}...")
        print()

if __name__ == "__main__":
    run_sample_qa_demo()
