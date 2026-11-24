#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.pdf_validator import PDFValidator, PDFValidationError
from app.ingestion.pdf_extractor import PDFExtractor
from app.ingestion.chunker import StructureAwareChunker
from app.embeddings.dense_embeddings import DenseEmbedder
from app.embeddings.bm25_embeddings import BM25SparseEmbedder
from app.embeddings.splade_embeddings import SpladeEmbedder
from app.database.qdrant_client import QdrantManager
from app.api.routes_search import execute_retrieval
from app.generation.answer_generator import AnswerGenerator
from app.config import settings

def ingest_pdf(pdf_path: Path):
    """
    Ingests a single PDF into the local Hybrid RAG system.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"❌ Error: File not found: {pdf_path}")
        return False

    validator = PDFValidator(upload_dir=settings.upload_dir)
    extractor = PDFExtractor(extracted_dir=settings.extracted_dir)
    chunker = StructureAwareChunker()
    dense = DenseEmbedder.get_instance()
    bm25 = BM25SparseEmbedder()
    splade = SpladeEmbedder.get_instance()
    qdrant = QdrantManager.get_instance()

    print(f"📄 Validating & processing: {pdf_path.name}...")
    try:
        doc_id, saved_path, val_meta = validator.save_and_register(pdf_path, pdf_path.name)
        extracted = extractor.extract_document(saved_path, doc_id, pdf_path.name)
        chunks = chunker.chunk_extracted_document(extracted)
        
        if not chunks:
            print(f"⚠️  No text/content extracted from {pdf_path.name}.")
            return False

        texts = [c.chunk_text for c in chunks]
        d_vecs = dense.embed_texts(texts)
        b_vecs = bm25.encode_documents_batch(texts)
        s_vecs = splade.encode_batch(texts)
        
        qdrant.upsert_chunks(chunks, d_vecs, b_vecs, s_vecs)
        print(f"✅ Successfully indexed '{pdf_path.name}' (ID: {doc_id[:8]}... | Chunks: {len(chunks)} | Pages: {val_meta.get('total_pages')})")
        return True
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return False

def ensure_documents_indexed():
    """
    Ensures at least the sample PDFs are indexed if the collection is empty.
    """
    qdrant = QdrantManager.get_instance()
    docs = qdrant.list_documents()
    if not docs:
        sample_dir = Path("data/sample_pdfs")
        pdf_files = sorted(list(sample_dir.glob("*.pdf")))
        if pdf_files:
            print("📦 Initializing knowledge base with sample PDFs...")
            for p in pdf_files:
                ingest_pdf(p)
            print()

def ask_cli(query: str, mode: str = "hybrid", top_k: int = 4, rerank: bool = True, show_chunks: bool = False):
    """
    Executes retrieval and answer generation, printing clean terminal output.
    """
    print("\n" + "=" * 80)
    print(f"❓ QUESTION: {query}")
    print("=" * 80)

    generator = AnswerGenerator()

    # Execute Hybrid retrieval + Reranking
    search_res = execute_retrieval(query, mode=mode, top_k=top_k, apply_rerank=rerank)
    answer, citations = generator.generate_answer(query, search_res.chunks)

    print("\n💡 GROUNDED ANSWER:")
    print("-" * 80)
    print(answer)
    print("-" * 80)

    print("\n📌 VERIFIED CITATIONS:")
    if citations:
        for idx, cit in enumerate(citations, 1):
            sec = f" | Section: {cit.section_title}" if cit.section_title else ""
            print(f"  [{idx}] 📄 Document: {cit.document_name} | Page: {cit.page_number}{sec}")
            print(f"      Snippet: {cit.snippet}...")
    else:
        print("  (No direct citations found)")

    if show_chunks:
        print("\n🔍 RETRIEVED & RERANKED CONTEXT CHUNKS:")
        for idx, c in enumerate(search_res.chunks, 1):
            print(f"  [{idx}] Doc: {c.document_name} (Page {c.page_number}) | Method: {c.retrieval_method} | Score: {c.score}")
            preview = c.chunk_text.replace("\n", " ")[:150]
            print(f"      Text: {preview}...")

    print(f"\n⏱️  Retrieval Latency: {search_res.retrieval_latency_ms:.2f} ms | Rerank Latency: {search_res.rerank_latency_ms:.2f} ms")
    print("=" * 80 + "\n")

def interactive_loop(mode: str = "hybrid", top_k: int = 4, rerank: bool = True, show_chunks: bool = False):
    """
    Runs an interactive chat loop in the CLI.
    """
    print("=" * 80)
    print("          LOCAL HYBRID RAG: INTERACTIVE CLI TERMINAL (CPU)")
    print("=" * 80)
    print("Commands:")
    print("  • Type your question and press Enter to get an answer with citations.")
    print("  • Type 'ingest <path/to/file.pdf>' to index a new PDF on the fly.")
    print("  • Type 'docs' to list currently indexed documents.")
    print("  • Type 'chunks' to toggle showing retrieved context chunks.")
    print("  • Type 'exit' or 'quit' to close the CLI.")
    print("-" * 80)

    ensure_documents_indexed()

    while True:
        try:
            user_input = input("\nrag-cli > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            if user_input.lower() == "docs":
                qdrant = QdrantManager.get_instance()
                docs = qdrant.list_documents()
                print(f"\n📚 Currently Indexed Documents ({len(docs)}):")
                for d in docs:
                    print(f"  • {d.get('document_name', 'unknown')} (ID: {d.get('document_id', '')[:8]}... | Chunks: {d.get('chunk_count', 0)})")
                continue

            if user_input.lower() == "chunks":
                show_chunks = not show_chunks
                status = "ENABLED" if show_chunks else "DISABLED"
                print(f"Context chunk display {status}.")
                continue

            if user_input.lower().startswith("ingest "):
                pdf_target = user_input[7:].strip().strip("'\"")
                ingest_pdf(Path(pdf_target))
                continue

            # Regular question
            ask_cli(user_input, mode=mode, top_k=top_k, rerank=rerank, show_chunks=show_chunks)

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

def main():
    parser = argparse.ArgumentParser(description="Ask questions against uploaded PDFs in your CLI.")
    parser.add_argument("query", nargs="?", help="Question to ask. If omitted, starts interactive chat mode.")
    parser.add_argument("--ingest", help="Path to a PDF to ingest before querying.")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25", "splade", "bm25_dense", "bm25_splade"], help="Retrieval mode.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of context chunks to use.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking.")
    parser.add_argument("--show-chunks", action="store_true", help="Display retrieved chunks details.")

    args = parser.parse_args()

    if args.ingest:
        ingest_pdf(Path(args.ingest))

    ensure_documents_indexed()

    if args.query:
        # One-shot CLI question
        ask_cli(
            query=args.query,
            mode=args.mode,
            top_k=args.top_k,
            rerank=not args.no_rerank,
            show_chunks=args.show_chunks,
        )
    else:
        # Interactive chat loop
        interactive_loop(
            mode=args.mode,
            top_k=args.top_k,
            rerank=not args.no_rerank,
            show_chunks=args.show_chunks,
        )

if __name__ == "__main__":
    main()
