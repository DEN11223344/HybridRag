import math
import time
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from app.schemas.chunk import ChunkMetadata
from app.embeddings.dense_embeddings import DenseEmbedder
from app.embeddings.bm25_embeddings import BM25SparseEmbedder
from app.embeddings.splade_embeddings import SpladeEmbedder
from app.database.qdrant_client import QdrantManager
from app.api.routes_search import execute_retrieval
from app.generation.answer_generator import AnswerGenerator
from evaluation.sample_data_generator import SampleDataGenerator

class EvaluationBenchmark:
    def __init__(self):
        self.qdrant = QdrantManager.get_instance()
        self.dense = DenseEmbedder.get_instance()
        self.bm25 = BM25SparseEmbedder()
        self.splade = SpladeEmbedder.get_instance()
        self.generator = AnswerGenerator()

    def setup_corpus(self):
        """
        Indexes the 5 evaluation documents into Qdrant.
        """
        docs = SampleDataGenerator.get_test_documents()
        all_chunks = []
        for d in docs:
            doc_id = d["document_id"]
            doc_name = d["document_name"]
            for c in d["chunks"]:
                chunk = ChunkMetadata(
                    chunk_id=c["chunk_id"],
                    document_id=doc_id,
                    document_name=doc_name,
                    page_number=c["page_number"],
                    section_title=c.get("section_title"),
                    chunk_text=c["chunk_text"],
                    token_count=len(c["chunk_text"].split()),
                    content_type=c.get("content_type", "text"),
                    table_id=c.get("table_id"),
                    image_references=c.get("image_references", []),
                )
                all_chunks.append(chunk)

        texts = [c.chunk_text for c in all_chunks]
        d_vecs = self.dense.embed_texts(texts)
        b_vecs = self.bm25.encode_documents_batch(texts)
        s_vecs = self.splade.encode_batch(texts)

        self.qdrant.upsert_chunks(all_chunks, d_vecs, b_vecs, s_vecs)
        return len(all_chunks)

    def evaluate_mode(self, mode: str, queries: List[Dict[str, Any]], rerank: bool = True) -> Dict[str, Any]:
        """
        Evaluates a retrieval mode over queries calculating:
        Recall@5, Recall@10, Precision@5, MRR@10, nDCG@10, and Latencies.
        """
        r5_list = []
        r10_list = []
        p5_list = []
        mrr_list = []
        ndcg_list = []
        latencies = []

        for item in queries:
            query = item["query"]
            expected = set(item["expected_chunks"])

            t0 = time.time()
            resp = execute_retrieval(query, mode=mode, top_k=10, apply_rerank=rerank)
            elapsed_ms = (time.time() - t0) * 1000.0
            latencies.append(elapsed_ms)

            retrieved_ids = [c.chunk_id for c in resp.chunks]
            top5 = retrieved_ids[:5]
            top10 = retrieved_ids[:10]

            # Recall@5 and Recall@10
            r5 = len(expected.intersection(top5)) / len(expected) if expected else 0.0
            r10 = len(expected.intersection(top10)) / len(expected) if expected else 0.0
            r5_list.append(r5)
            r10_list.append(r10)

            # Precision@5
            p5 = len(expected.intersection(top5)) / 5.0
            p5_list.append(p5)

            # MRR@10
            rr = 0.0
            for rank_idx, cid in enumerate(top10, start=1):
                if cid in expected:
                    rr = 1.0 / rank_idx
                    break
            mrr_list.append(rr)

            # nDCG@10
            dcg = 0.0
            for rank_idx, cid in enumerate(top10, start=1):
                rel = 1.0 if cid in expected else 0.0
                dcg += rel / math.log2(rank_idx + 1)
            # Ideal DCG for 1 relevant item is 1 / log2(2) = 1.0
            idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(expected), 10) + 1))
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            ndcg_list.append(ndcg)

        return {
            "mode": mode,
            "rerank": rerank,
            "recall@5": round(sum(r5_list) / len(r5_list), 4),
            "recall@10": round(sum(r10_list) / len(r10_list), 4),
            "precision@5": round(sum(p5_list) / len(p5_list), 4),
            "mrr@10": round(sum(mrr_list) / len(mrr_list), 4),
            "ndcg@10": round(sum(ndcg_list) / len(ndcg_list), 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        }

    def run_full_comparative_benchmark(self) -> List[Dict[str, Any]]:
        self.setup_corpus()
        queries = SampleDataGenerator.get_evaluation_queries()

        modes = [
            ("bm25", False),
            ("splade", False),
            ("dense", False),
            ("bm25_dense", False),
            ("bm25_splade", False),
            ("hybrid", False),
            ("hybrid", True),  # Complete Hybrid + Reranker
        ]

        results = []
        for mode_name, apply_rerank in modes:
            metrics = self.evaluate_mode(mode_name, queries, rerank=apply_rerank)
            results.append(metrics)

        return results
